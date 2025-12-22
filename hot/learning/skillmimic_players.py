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

import torch, time 

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import os
import learning.common_player as common_player
from metric.metric_factory import create_metric
from metric.metric_manager import MetricManager

# from utils import fid #V1
# import physhoi.learning.fid as fid

METRIC = False

class SkillMimicPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        
        super().__init__(config)

        if config.get('dual', False):
            print("-----------Dual Policy-----------")
        
        
        metric_kwargs = {}
        if hasattr(self.env.task, 'layup_target'):
            metric_kwargs.update({"layup_target":self.env.task.layup_target})  # 如果需要额外的参数，可以从 cfg 中获取
        if hasattr(self.env.task, 'switch_skill_name'):
            metric_kwargs.update({"switch_skill_name":self.env.task.switch_skill_name})  # 如果需要额外的参数，可以从 cfg 中获取
        metric = create_metric(self.env.task.skill_name, self.env.task.num_envs, self.env.task.device, self.env.task.max_episode_length, **metric_kwargs) # 使用工厂函数创建 Metric
        # 初始化 Metric 管理器
        if metric:
            self.metric_manager = MetricManager([metric])
        else:
            self.metric_manager = None
        
        return

    def run(self):

        n_games = self.games_num #Z
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0

        sum_steps = 0
        sum_error_step = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        sum_accuracy = 0 #metric
        sum_mpjpe_b = 0
        sum_mpjpe_o = 0
        sum_cg_error = 0
        sum_succ = 0

        total_E_op = 0
        total_E_or = 0
        total_E_h = 0
        total_reward = 0
        total_error_step = 0 
        total_metric = 0
        rounds = 5
        if self.env.task._save_refined_data:
            rounds = 10000

        for episode in range(rounds): #n_games
            # if games_played >= n_games: #Z
            #     break
            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False
            cum_error_steps=0
            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cE_op = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cE_or = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cE_h = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            # culmulate for current episode
            cum_E_op = 0.0
            cum_E_or =  0.0
            cum_E_h =  0.0
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            error_steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            all_error_done_indices = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            hidden_sim = [] #fid
            hidden_ref = []

            cum_accuracy = torch.zeros(batch_size, dtype=torch.float32, device=self.device) #metric
            cum_mpjpe_b = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cum_mpjpe_o = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            cum_cg_error = torch.zeros(batch_size, dtype=torch.float32, device=self.device)


            print_game_res = False
            error_print = False
            done_indices = []

            if self.env.task.play_dataset:
                # play dataset
                while True: #Z
                    all_obs = []
                    motid = torch.randint(0, self.env.task._motion_data.num_motions, (1,)).item()
                    length = self.env.task._motion_data.motion_lengths[motid]
                    for t in range(length): 
                        obs_buf = self.env.task.play_dataset_step(t, motid, length)
                        all_obs.append(obs_buf.clone())
            elif self.env.task.postproc_hopdata:
                self.env.task.play_dataset_postproc_hopdata()
                exit()


            else:
                # try:
                # inference
                for n in range(self.env.task.max_episode_length): #fid self.max_steps
                    # print(n)
                    obs_dict = self.env_reset(done_indices)
                    if self.env.task.cfg['name'] != 'OpenLoop':
                        if has_masks:
                            masks = self.env.get_action_mask()
                            action = self.get_masked_action(obs_dict, masks, is_determenistic)
                        else:
                            action = self.get_action(obs_dict, is_determenistic)

                        # a_out = fid.g_a_out #fid #V1
                        # hidden_sim.append(a_out)

                        obs_dict, r, done, info =  self.env_step(self.env, action)
                    else:
                        obs_dict, r, done, info =  self.env.task.open_loop_step()

                    cr += r
                    cE_op += info['metrics']['E_op']
                    cE_or += info['metrics']['E_or']
                    cE_h += info['metrics']['E_h']
                    steps += 1
                    error_steps += 1
                    self._post_step(info)

                    if METRIC: #metric
                        accuracy = info["accuracy"]
                        mpjpe_b = info["mpjpe_b"]
                        mpjpe_o = info["mpjpe_o"]
                        cg_error = info["cg_error"]

                        cum_accuracy += accuracy
                        cum_mpjpe_b += mpjpe_b
                        cum_mpjpe_o += mpjpe_o
                        cum_cg_error += cg_error

                    # print(done)
                    ''' #fid
                    if not torch.all(done == 0):
                        os.kill(os.getpid(), signal.SIGINT)
                    '''
                    if render:
                        self.env.render(mode = 'human')
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[::self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count
                    if self.env.task._save_refined_data:
                        metics_indices = info["cal_metrics"].nonzero(as_tuple=False)[::self.num_agents]
                        save_metics_indices = info["save_metrics"].nonzero(as_tuple=False)[::self.num_agents]
                    if (self.metric_manager and not self.env.task._save_refined_data) or (self.env.task._save_refined_data and len(metics_indices) > 0):
                        state = self.env.task.get_state_for_metric()
                        state['progress'] = n
                        self.metric_manager.update(state)

                    error_done_indices = ((info['metrics']['error_done'] | done) & ~ all_error_done_indices).int().nonzero(as_tuple=False).clone() # only store those done in this round as done
                    all_error_done_indices = all_error_done_indices | info['metrics']['error_done'] | done

                    error_done_count = len(error_done_indices)
                    if error_done_count > 0 and error_print == False:
                        cur_E_op = cE_op[error_done_indices].sum().item()
                        cur_E_or = cE_or[error_done_indices].sum().item()
                        cur_E_h = cE_h[error_done_indices].sum().item()
                        
                        curr_error_steps = error_steps[error_done_indices].sum().item()
                        cum_error_steps += curr_error_steps
                        # to play save
                        error_steps = error_steps * (1.0 - all_error_done_indices.float())
                        cE_op = cE_op * (1.0 - all_error_done_indices.float())
                        cE_or = cE_or * (1.0 - all_error_done_indices.float())
                        cE_h = cE_h * (1.0 - all_error_done_indices.float())
                        cum_E_op += cur_E_op
                        cum_E_or+= cur_E_or
                        cum_E_h += cur_E_h

                        total_E_op += cur_E_op
                        total_E_or+= cur_E_or
                        total_E_h += cur_E_h
                        total_error_step += curr_error_steps
                        
                    if self.env.task._save_refined_data and len(save_metics_indices) > 0:
                        succ_envs_id = self.metric_manager.get_succ_envs()
                        for metric_name, value in succ_envs_id.items():
                            save_done_flat = save_metics_indices.squeeze(0)
                            # Find the comment indices (present in all_done_flat but not in value)
                            mask = torch.isin(save_done_flat, value)
                            comment_indices = save_done_flat[mask]
                            print(comment_indices)
                            self.env.task.save_refined_data(comment_indices.tolist())

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()
                        cr = cr * (1.0 - done.float())

                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        if METRIC: #metric
                            ca = cum_accuracy[done_indices].sum().item() 
                            cb = cum_mpjpe_b[done_indices].sum().item()
                            co = cum_mpjpe_o[done_indices].sum().item()
                            cc = cum_cg_error[done_indices].sum().item()

                            cum_accuracy *= (1.0 - done.float())
                            cum_mpjpe_b *= (1.0 - done.float())
                            cum_mpjpe_o *= (1.0 - done.float())
                            cum_cg_error *= (1.0 - done.float())

                            sum_accuracy += ca
                            sum_mpjpe_b += cb
                            sum_mpjpe_o += co
                            sum_cg_error += cc

                            sum_succ += info["succ"][done_indices].sum().item()
                        #



                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)             
                            else:
                                if METRIC: #metric
                                    # print('acc', ca/cur_steps, 'err_b', cb/cur_steps, 'err_o', co/cur_steps, 'err_cg', cc/cur_steps,
                                    #       f'rpf: {cur_rewards*100.0/cur_steps:.2f}%', 'reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 
                                    #       )
                                    pass
                                else:
                                    total_reward += cur_rewards/done_count
                                    print(f'rpf: {cur_rewards*100.0/cur_steps:.2f}%', 'reward:', cur_rewards/done_count, 'avg reward:', total_reward/(episode+1), 'steps:', cur_steps/done_count)
                                    print(f'ObjPosError [{cum_E_op/cum_error_steps:.2f}]', f'ObjRotError [{cum_E_or/cum_error_steps:.2f}]', f'KeyPosError [{cum_E_h/cum_error_steps:.2f}]')
                                    error_print = True

                        sum_game_res += game_res
                        if batch_size//self.num_agents == 1 : #ZC3 or games_played >= n_games
                            break
                    
                    done_indices = done_indices[:, 0]

                if METRIC:
                    print("succ_rate_till_now", sum_succ / games_played * n_game_life)
                    if episode == 0:
                        games_played = 0
                        sum_succ = 0
                
                if self.metric_manager:
                    results = self.metric_manager.compute()
                    for metric_name, value in results.items():
                        total_metric += value
                        print(f"{metric_name}: {value}")
                    self.metric_manager.reset(done_indices)
                if self.env.task._save_refined_data and games_played >= self.env.task._motion_data.num_motions:
                    break
        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            if METRIC: #metric
                print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life,
                      'ACC', sum_accuracy/sum_steps, 'ERR_b', sum_mpjpe_b/sum_steps, 'ERR_o', sum_mpjpe_o/sum_steps, 'ERR_cg', sum_cg_error/sum_steps,
                      'SUCC', sum_succ / games_played * n_game_life)
            else:
                print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)
            print(f'av ObjPosError [{total_E_op/total_error_step:.2f}], ObjRotError [{total_E_or/total_error_step:.2f}], KeyPosError [{total_E_h/total_error_step:.2f}]')
            if self.metric_manager:
                print(f'av {metric_name} {total_metric/rounds}')

        return
    
    def load_dual(self, fn): #ZC9
        networks = [self.model.a2c_network.network1, self.model.a2c_network.network2]
        dual_model_cp = [fn] * 2
        for network, cp in zip(networks, dual_model_cp):
            checkpoint = torch.load(cp, map_location=self.device)
            model_checkpoint = {x[len('a2c_network.'):]: y for x, y in checkpoint['model'].items()}

            model_state_dict = network.state_dict()
            missing_checkpoint_keys = [key for key in model_state_dict if key not in model_checkpoint]
            print(f'loading model from EmbodyPose checkpoint {cp} ...')
            print('Shared keys in model and checkpoint:', [key for key in model_state_dict if key in model_checkpoint])
            print('Keys not found in current model:', [key for key in model_checkpoint if key not in model_state_dict])
            print('Keys not found in checkpoint:', missing_checkpoint_keys)

            discard_pretrained_sigma = self.config.get('discard_pretrained_sigma', False)
            if discard_pretrained_sigma:
                checkpoint_keys = list(model_checkpoint)
                for key in checkpoint_keys:
                    if 'sigma' in key:
                        model_checkpoint.pop(key)
            
            # If model has larger obs dim
            network_obs_dim = network.actor_mlp[0].weight.data.shape[-1]
            weight_obs_dim = model_checkpoint['actor_mlp.0.weight'].shape[-1]
            if network_obs_dim > weight_obs_dim:
                network.actor_mlp[0].weight.data.zero_()
                network.actor_mlp[0].weight.data[:, :weight_obs_dim] = model_checkpoint['actor_mlp.0.weight']
                del model_checkpoint['actor_mlp.0.weight']
                network.critic_mlp[0].weight.data.zero_()
                network.critic_mlp[0].weight.data[:, :weight_obs_dim] = model_checkpoint['critic_mlp.0.weight']
                del model_checkpoint['critic_mlp.0.weight']

            network.load_state_dict(model_checkpoint, strict=False)

            load_rlgame_norm = len([key for key in missing_checkpoint_keys if 'running_obs' in key]) > 0
            if load_rlgame_norm:
                if 'running_mean_std' in checkpoint:
                    print('loading running_obs from rl game running_mean_std')
                    obs_len = checkpoint['running_mean_std']['running_mean'].shape[0]
                    network.running_obs.n = checkpoint['running_mean_std']['count'].long()
                    network.running_obs.mean[:obs_len] = checkpoint['running_mean_std']['running_mean'].float()
                    network.running_obs.var[:obs_len] = checkpoint['running_mean_std']['running_var'].float()
                    network.running_obs.std[:obs_len] = torch.sqrt(network.running_obs.var[:obs_len])
                elif 'running_obs.running_mean' in model_checkpoint:
                    print('loading running_obs from rl game in model')
                    obs_len = model_checkpoint['running_obs.running_mean'].shape[0]
                    network.running_obs.n = model_checkpoint['running_obs.count'].long()
                    network.running_obs.mean[:obs_len] = model_checkpoint['running_obs.running_mean'].float()
                    network.running_obs.var[:obs_len] = model_checkpoint['running_obs.running_var'].float()
                    network.running_obs.std[:obs_len] = torch.sqrt(network.running_obs.var[:obs_len])
    
    def restore(self, fn):
        if self.config.get('dual', False): #ZC9
            self.load_dual(fn)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(torch_ext.load_checkpoint(fn)['running_mean_std'])
            return

        if (fn != 'Base'):
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return

    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config

    def _amp_debug(self, info):
        return


