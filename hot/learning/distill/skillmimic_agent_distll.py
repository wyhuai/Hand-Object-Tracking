# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import learning.skillmimic_agent as skillmimic_agent

from tensorboardX import SummaryWriter

class SkillMimicAgentDistill(skillmimic_agent.SkillMimicAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.expert_loss_coef = config['expert_loss_coef']
        self.entropy_coef = config['entropy_coef']
        self.ev_ma            = 0.0   # running avg explained‑variance
        self.critic_win_streak = 0    # consecutive windows EV ≥ threshold
        self.actor_update_num = 0
        self.policy_game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.expert_game_rewards = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        return

    def clear_stats(self):
        super().clear_stats()
        self.policy_game_rewards.clear()
        self.expert_game_rewards.clear()

    def init_tensors(self):
        super().init_tensors()
        batch_shape = self.experience_buffer.obs_base_shape
        action_size = 51
        self.experience_buffer.tensor_dict['expert_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['expert'] = torch.zeros((*batch_shape, action_size), dtype=torch.float32, device=self.ppo_device) #51 is action size
        self.tensor_list += ['amp_obs', 'rand_action_mask', 'expert', 'expert_mask']
        return

    def train(self):
        if self.resume_from != 'None':
            self.restore(self.resume_from)
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0

        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        
        model_output_file = os.path.join(self.nn_dir, self.config['name'])
        
        if self.multi_gpu:
            self.hvd.setup_algo(self)

        self._init_train()

        # 用于存储最佳模型的列表，每个元素为 (reward, filepath)
        best_models = []
        best_reward = -float('inf')  # Initialize the best reward
        
        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch() # core

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame
            if self.multi_gpu:
                self.hvd.sync_stats(self)

            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    fps_step = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    beta_t = max(1 - max((self.epoch_num - 500) / 5000, 0), 0)
                    res_dict = self.get_action_values(self.refined_obs, self._rand_action_probs, beta_t, self.expert['actions'].to(self.ppo_device))
                    expert_mask = res_dict['expert_mask']
                    num_expert = expert_mask.sum()
                    print("epoch_num:{}".format(epoch_num), "mean_rewards:{}".format(self._get_mean_rewards()), "mean_policy_rewards:{}".format(self._get_mean_policy_rewards()), "mean_expert_rewards:{}".format(self._get_mean_expert_rewards()), f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}', f'num experts {num_expert}')

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                self._log_train_info(train_info, frame)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                
                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()
                    mean_policy_rewards = self._get_mean_policy_rewards()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('policy_reward/iter'.format(i), mean_policy_rewards[i], epoch_num)
                        self.writer.add_scalar('expert_reward/iter'.format(i), mean_policy_rewards[i], epoch_num)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)
                
                local_rank = int(os.getenv('LOCAL_RANK', '0'))
                if local_rank == 0:
                    multi_gpu_save = True
                else:
                    multi_gpu_save = True
                if self.save_freq > 0 and multi_gpu_save:
                    if (epoch_num % self.save_freq == 0):
                        self.save(model_output_file)

                        if (self._save_intermediate):
                            int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)
                            self.save(int_model_output_file)

                if epoch_num > self.max_epochs and multi_gpu_save:
                    self.save(model_output_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num
                
                if epoch_num % (2500) == 0 and multi_gpu_save: #self.max_epochs // 4
                    self.save(f"{model_output_file}_e{epoch_num}")
                    print(f'Checkpoint saved at epoch {epoch_num}')

                # 检查是否出现新的最佳奖励
                mean_rewards = self._get_mean_rewards()
                current_reward = mean_rewards[0]
                if current_reward > best_reward:
                    best_reward = current_reward
                    new_best_model_file = f"{model_output_file}_e{epoch_num}_r{current_reward:.4f}"
                    
                    # 保存新的最佳模型
                    self.save(new_best_model_file)
                    
                    # 将新的最佳模型添加到列表
                    best_models.append((best_reward, new_best_model_file))
                    # 按奖励从大到小排序
                    best_models.sort(key=lambda x: x[0], reverse=True)

                    # 如果超过2个，删除最差的模型文件
                    while len(best_models) > 2:
                        worst_model = best_models.pop()  # 移除最末的(最差)模型
                        if os.path.exists(worst_model[1] + '.pth'):
                            os.remove(worst_model[1] + '.pth')

                update_time = 0
        return


    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        # Initialize DAgger beta coefficient
        beta_t = max(1 - max((self.epoch_num - 500) / 5000, 0), 0)

        for n in range(self.horizon_length):

            self.obs, self.expert, self.refined_obs = self.env_reset(self.done_indices)
            self.experience_buffer.update_data('obses', n, self.refined_obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.refined_obs, masks)
            else:
                res_dict = self.get_action_values(self.refined_obs, self._rand_action_probs, beta_t, self.expert['actions'].to(self.ppo_device))
            self.experience_buffer.update_data('expert', n, self.expert['mus'].to(self.ppo_device))

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos, self.expert, self.refined_obs = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.refined_obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.refined_obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            self.done_indices = all_done_indices[::self.num_agents]
  
            if len(self.done_indices) > 0:
                expert_mask = res_dict['expert_mask']
                # Get expert mask for done environments
                done_envs = self.done_indices[:, 0]  # Get env indices
                done_expert_mask = expert_mask[done_envs]  # Vectorized lookup
                
                # Create boolean mask for policy actions (expert_mask == 0)
                policy_action_mask = (done_expert_mask == 0)
                
                # Apply mask to get policy-only done indices
                policy_done_indices = self.done_indices[policy_action_mask]

                non_expert_mask = res_dict['non_expert_mask']
                # Get expert mask for done environments
                done_non_expert_mask = non_expert_mask[done_envs]  # Vectorized lookup
                
                # Create boolean mask for policy actions (expert_mask == 0)
                expert_action_mask = (done_non_expert_mask == 0)
                
                # Apply mask to get policy-only done indices
                expert_done_indices = self.done_indices[expert_action_mask]

            else:
                policy_done_indices = torch.tensor([], device=self.ppo_device)
            self.game_rewards.update(self.current_rewards[self.done_indices])
            self.policy_game_rewards.update(self.current_rewards[policy_done_indices])
            self.expert_game_rewards.update(self.current_rewards[expert_done_indices])
            self.game_lengths.update(self.current_lengths[self.done_indices])
            self.algo_observer.process_infos(infos, self.done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            self.done_indices = self.done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        return batch_dict


    def get_action_values(self, obs_dict, rand_action_probs, use_experts=0.0, expert=None):
        res_dict = super().get_action_values(obs_dict, rand_action_probs)
        num_envs = self.vec_env.env.task.num_envs
        expert_action_probs = to_torch([use_experts for _ in range(num_envs)], dtype=torch.float32, device=self.ppo_device)
        expert_action_probs = torch.bernoulli(expert_action_probs)
        det_action_mask = expert_action_probs == 1.0
        res_dict['actions'][det_action_mask] = expert[det_action_mask]
        res_dict['expert_mask'] = expert_action_probs
        non_expert_mask = (expert_action_probs == 0.)  
        res_dict['non_expert_mask'] = non_expert_mask
        return res_dict
    

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        expert = batch_dict['expert']
        expert_mask = batch_dict['expert_mask']
        self.dataset.values_dict['expert'] = expert
        self.dataset.values_dict['expert_mask'] = expert_mask
        return


    def _supervise_loss(self, student, teacher):
        e_loss = (student - teacher)**2

        info = {
            'expert_loss': e_loss.sum(dim=-1)
        }
        return info
    
    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, expert, refined_obs = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos, expert, self.obs_to_tensors(refined_obs)#.to(self.ppo_device)
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos, expert, self.obs_to_tensors(refined_obs) #.to(self.ppo_device)
        
    def env_reset(self, env_ids=None):
        obs, expert, refined_obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        refined_obs = self.obs_to_tensors(refined_obs)
        return obs, expert, refined_obs


    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        expert_mus = input_dict['expert']
        obs_batch = self._preproc_obs(obs_batch)


        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            if rand_action_sum > 0:
                a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
                a_loss = a_info['actor_loss']
                a_clipped = a_info['actor_clipped'].float()

                c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                c_loss = c_info['critic_loss']
                if self.epoch_num > 7000:
                    returns_var = return_batch.var(unbiased=False) + 1e-8  # avoid divide‑by‑0
                    errors_var = (return_batch - values).var(unbiased=False)
                    ev = 1.0 - errors_var / returns_var
                    self.ev_ma = 0.99 * self.ev_ma + 0.01 * ev.item()
                    if self.ev_ma >= 0.6:
                        self.critic_win_streak += 1
                    else:
                        self.critic_win_streak = 0
                        
                b_loss = self.bound_loss(mu)
                
                c_loss = torch.mean(c_loss)
                e_info = self._supervise_loss(mu, expert_mus)
                e_loss = e_info['expert_loss']
                a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
                entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
                b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
                a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum
                e_loss = torch.mean(e_loss)
                if self.epoch_num > 6000 and self.critic_win_streak >= 3:
                    loss = a_loss * min((self.actor_update_num / 4000), 1) + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss + self.expert_loss_coef * e_loss * max(1 - (self.actor_update_num / 4000), 0.1)
                    self.actor_update_num += 1
                elif self.epoch_num > 5000:
                    loss = min(((self.epoch_num - 5000) / 1000), 1) * self.critic_coef * c_loss + self.expert_loss_coef * e_loss + 0 * a_loss
                else:
                    loss = self.expert_loss_coef * e_loss + 0 * a_loss + 0 * b_loss  + 0 * c_loss
            else:
                e_info = self._supervise_loss(mu, expert_mus)
                e_loss = e_info['expert_loss']
                e_loss = torch.mean(e_loss)
                loss = self.expert_loss_coef * e_loss + 0 * a_loss + 0 * b_loss  + 0 * c_loss
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(e_info)
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        self.writer.add_scalar('losses/e_loss', torch_ext.mean_list(train_info['expert_loss']).item(), frame)

        return

    
    def _get_mean_policy_rewards(self):
        mean = self.policy_game_rewards.get_mean()
        return np.expand_dims(mean, axis=0) if mean.ndim == 0 else mean

    def _get_mean_expert_rewards(self):
        mean = self.expert_game_rewards.get_mean()
        return np.expand_dims(mean, axis=0) if mean.ndim == 0 else mean



        