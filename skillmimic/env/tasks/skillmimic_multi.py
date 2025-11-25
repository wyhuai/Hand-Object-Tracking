from enum import Enum
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn.utils.rnn as rnn_utils
from typing import Tuple
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import time as pyton_time# changed by me
from pathlib import Path
from datetime import datetime

from utils import torch_utils
from utils.multi_motion_data_handler import MultiMotionDataHandler

from env.tasks.skillmimic import SkillMimicBallPlay


class MultiSkillMimicBallPlay(SkillMimicBallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                        sim_params=sim_params,
                        physics_engine=physics_engine,
                        device_type=device_type,
                        device_id=device_id,
                        headless= headless)


    def _load_motion(self, motion_file):
        self.skill_name = motion_file.split('/')[-1] #metric
        self.max_episode_length = 60 #60 by meeee
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]
        
        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)

        self._motion_data = MultiMotionDataHandler(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                            self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset,
                                            map_env_2_object=self._map_env_2_object_tensor)
    
    def _reset_random_ref_state_init(self, env_ids): #Z11
        motion_ids = self._motion_data.sample_motions(env_ids)
        motion_times = self._motion_data.sample_time(motion_ids)

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids], \
        self.init_obj2_pos[env_ids], self.init_obj2_pos_vel[env_ids], self.init_obj2_rot[env_ids], self.init_obj2_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)
        
        skill_label = self._motion_data.motion_class[motion_ids.cpu().numpy()]
        self.skill_labels[env_ids] = torch.from_numpy(skill_label).to(self.device)

        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        motion_ids = self._motion_data.sample_motions(env_ids)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids], \
        self.init_obj2_pos[env_ids], self.init_obj2_pos_vel[env_ids], self.init_obj2_rot[env_ids], self.init_obj2_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)
        
        skill_label = self._motion_data.motion_class[motion_ids.cpu().numpy()]
        self.skill_labels[env_ids] = torch.from_numpy(skill_label).to(self.device)

        return motion_ids, motion_times

        return


class MultiSkillMimic2BallPlayReweight(MultiSkillMimicBallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.reweight = cfg['env']['reweight']
        self.reweight_alpha = cfg['env']['reweight_alpha']
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        ############## Reweighting Mechanism Initialization ##############
        # Initialize tensors to track motion IDs and times for all environments
        self.motion_ids_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_times_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # Calculate total frames across all motions for setting reweighting intervals
        self.reweight_interval = 20000  # Determines how often reweighting occurs
        
        # Initialize reward tracking tensors
        self.envs_reward = torch.zeros(self.num_envs, self.max_episode_length, device=self.device, dtype=torch.float64)
        self.average_rewards = torch.zeros(self._motion_data.num_motions, device=self.device, dtype=torch.float64)
        
        # Initialize motion time sequence rewards using a tensor for efficient indexing
        self.motion_time_seqreward = torch.zeros(
            (self._motion_data.num_motions, self._motion_data.motion_lengths.max() - 3),
            device=self.device,
            dtype=torch.float64
        )
        #######################################################################
    
    def _load_motion(self, motion_file):
        self.skill_name = motion_file.split('/')[-1] #metric
        self.max_episode_length = 60 #60 by meeee
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]
        
        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
        self._motion_data = MultiMotionDataHandler(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                                    self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset,
                                                    reweight=self.reweight, reweight_alpha=self.reweight_alpha)
        return
        
    def _compute_reset(self):
        # self.max_episode_length = self._motion_data.motion_lengths[self.motion_ids_total.item()] - 2
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._curr_ref_obs, self._curr_obs, self._motion_data.envid2episode_lengths,
                                                   self.isTest, self.cfg["env"]["episodeLength"],
                                                   self.extras["metrics"]['E_op'], self.extras["metrics"]['E_or'], self.extras["metrics"]['E_h']
                                                   )
        
        # Identify environments that need reweighting
        reset_env_ids = torch.nonzero(self.reset_buf == 1, as_tuple=False).squeeze(-1)
        
        # Perform reweighting on the identified environments
        self._reweight_motion(reset_env_ids)
        return

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
        
        # Batch update motion_ids_total and motion_times_total for reset environments
        self.motion_ids_total[env_ids] = self.motion_ids.to(self.device)
        self.motion_times_total[env_ids] = self.motion_times.to(self.device)
        return

    def _reweight_motion(self, reset_env_ids):
        """
        Reweights motion sampling probabilities based on accumulated rewards.
        This method is optimized to utilize vectorized tensor operations for efficiency.
        """
        if not self.cfg['env']['reweight']:
            return  # Reweighting is disabled
        
        # Record rewards for reset environments
        self.record_motion_time_reward(reset_env_ids)
        
        # Perform reweighting at specified intervals
        if (self.progress_buf_total % self.reweight_interval == 0) and (self.progress_buf_total > 0):
            if self._motion_data.num_motions > 1:
                self.average_rewards = torch.mean(self.motion_time_seqreward, dim=1)
                # Debugging information
                print('##### Reweighting Motion Sampling Rates #####')
                print('Class Average Reward:', self.average_rewards.cpu().numpy())
                # Update motion sampling weights based on average_rewards
                # 调用修改后的 clip 级别 reweight 函数（内含「类先分配 -> clip 再分配」逻辑）
                self._motion_data._reweight_clip_sampling_rate_vectorized(self.average_rewards)
            
            # Reweight motion time sampling rates
            if not self.cfg['env']['disable_time_reweight']:
                self._motion_data._reweight_time_sampling_rate_vectorized(self.motion_time_seqreward)

        return

    def record_motion_time_reward(self, reset_env_ids):
        """
        Records rewards for each motion clip at each time step.
        Optimized to use batch processing and vectorized tensor operations.
        """
        if reset_env_ids.numel() == 0:
            return  # No environments to process
        
        # Gather relevant data for reset environments
        motion_ids_reset = self.motion_ids_total[reset_env_ids]  # Shape: (num_reset_envs,)
        motion_times_reset = self.motion_times_total[reset_env_ids]  # Shape: (num_reset_envs,)
        
        self.envs_reward[torch.arange(self.num_envs), self.progress_buf] = self.rew_buf.double()

        non_zero_mask = self.envs_reward[reset_env_ids] != 0  # Shape: (num_reset_envs, max_episode_length)
        non_zero_sum = torch.sum(self.envs_reward[reset_env_ids] * non_zero_mask.double(), dim=1)  # Shape: (num_reset_envs,)
        non_zero_count = torch.clamp(torch.sum(non_zero_mask, dim=1), min=1.0)  # Shape: (num_reset_envs,)
        non_zero_mean = non_zero_sum / non_zero_count  # Shape: (num_reset_envs,)

        # (1) clamp lower bound
        valid_motion_times = motion_times_reset - 2
        valid_motion_times = torch.clamp(valid_motion_times, min=0)
        # (2) clamp upper bound
        max_tensor = (self._motion_data.motion_lengths[motion_ids_reset] - 3).long()
        valid_motion_times = torch.min(valid_motion_times, max_tensor)

        self.motion_time_seqreward[motion_ids_reset, valid_motion_times] = (
            self.motion_time_seqreward[motion_ids_reset, valid_motion_times] + non_zero_mean
        ) / 2.0
        
        # Reset envs_reward for the reset environments
        self.envs_reward[reset_env_ids] = 0.0
        return


class MultiSkillMimic2BallPlayRandInd(MultiSkillMimic2BallPlayReweight):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                                sim_params=sim_params,
                                physics_engine=physics_engine,
                                device_type=device_type,
                                device_id=device_id,
                                headless=headless)

    def p_noise(self, shape, p, scale):
        mask = torch.rand(shape[0]).to('cuda') < p
        noise = (torch.rand(shape).to('cuda') * 2 - 1) * scale
        return mask.unsqueeze(1) * noise

    def p_noise_rotate_noncontact(self, init_obj_rot, max_radians):
        # 生成在 [-max_radians, max_radians] 之间的随机旋转角度 (X, Y, Z)
        rand_angles = self.p_noise(init_obj_rot.shape, p=self.cfg['env']['state_noise_prob'], scale=max_radians)
        # rand_angles = rand_angles.to(init_obj_rot.device)
        roll_noise, pitch_noise, yaw_noise = rand_angles[:,0], rand_angles[:,1], rand_angles[:,2]
        # 获取当前四元数的欧拉角
        current_euler = torch_utils.quat_to_euler(init_obj_rot)  # (N, 3)
        roll_cur, pitch_cur, yaw_cur = current_euler[:,0], current_euler[:,1], current_euler[:,2]
        # 施加随机旋转
        roll_new, pitch_new, yaw_new = roll_cur + roll_noise, pitch_cur + pitch_noise, yaw_cur + yaw_noise
        # 转回四元数
        new_quat = torch_utils.quat_from_euler_xyz(roll_new, pitch_new, yaw_new)
        return new_quat
    
    def p_noise_rotate_contact(self, init_obj_rot, max_radians):
        # 生成在 [-max_radians, max_radians] 之间的随机旋转角度 (X, Y, Z)
        rand_angles = self.p_noise(init_obj_rot.shape, p=self.cfg['env']['state_noise_prob'], scale=max_radians)
        # rand_angles = rand_angles.to(init_obj_rot.device)
        roll_noise, pitch_noise, yaw_noise = rand_angles[:,0], rand_angles[:,1], rand_angles[:,2]
        # 获取当前四元数的欧拉角
        current_euler = torch_utils.quat_to_euler(init_obj_rot)  # (N, 3)
        roll_cur, pitch_cur, yaw_cur = current_euler[:,0], current_euler[:,1], current_euler[:,2]
        # 施加z轴随机旋转
        yaw_new = yaw_cur + yaw_noise
        # 转回四元数
        new_quat = torch_utils.quat_from_euler_xyz(roll_cur, pitch_cur, yaw_new)
        return new_quat
    
    def _reset_state_init(self, env_ids):
        super()._reset_state_init(env_ids)
        
        self.state_random_flags = [False for _ in env_ids]
        if self.cfg['env']['state_noise_prob'] > 0:
            self._init_with_random_noise(env_ids)

    def _init_with_random_noise(self, env_ids): 
        self.init_dof_pos[env_ids, 6:] += self.p_noise(self.init_dof_pos[env_ids, 6:].shape, 
                                                       p=self.cfg['env']['state_noise_prob'], 
                                                       scale=torch.pi/8)
        self.init_dof_pos_vel[env_ids, 6:] += self.p_noise(self.init_dof_pos_vel[env_ids, 6:].shape, 
                                                          p=self.cfg['env']['state_noise_prob'], 
                                                          scale=0.1)
        self.init_obj_pos_vel[env_ids] += self.p_noise(self.init_obj_pos_vel[env_ids].shape, 
                                                      p=self.cfg['env']['state_noise_prob'], scale=0.02)
        self.init_obj_rot_vel[env_ids] += self.p_noise(self.init_obj_rot_vel[env_ids].shape, 
                                                      p=self.cfg['env']['state_noise_prob'], 
                                                      scale=0.02)
        
        # contact info
        init_contact_info = self.hoi_data_batch[env_ids,0,-2:-1]
        # ObjPos-Rot noise for non-contact frames
        ncf_obj_pos_noise = self.p_noise(self.init_obj_pos[env_ids].shape, 
                                                  p=self.cfg['env']['state_noise_prob'], 
                                                  scale=0.05)
        ncf_obj_pos_noise[..., 2] = torch.abs(ncf_obj_pos_noise[..., 2])
        ncf_obj_rot_noise = self.p_noise_rotate_noncontact(self.init_obj_rot[env_ids], torch.pi/5)
        # ObjPos-Rot noise for contact frames
        cf_obj_pos_noise = self.p_noise(self.init_obj_pos[env_ids].shape, 
                                                  p=self.cfg['env']['state_noise_prob'], 
                                                  scale=0.02)
        cf_obj_rot_noise = self.p_noise_rotate_contact(self.init_obj_rot[env_ids], torch.pi/8)
        # condition decision
        self.init_obj_pos[env_ids] = torch.where(init_contact_info == 0,
                                                self.init_obj_pos[env_ids] + ncf_obj_pos_noise, # non-contact noise
                                                self.init_obj_pos[env_ids] + cf_obj_pos_noise) # contact noise
        self.init_obj_rot[env_ids] = torch.where(init_contact_info == 0, ncf_obj_rot_noise, cf_obj_rot_noise)

        if self.isTest:
            print(f"Random noise added to initial state for env {env_ids}")
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths,
                           isTest, maxEpisodeLength, E_op, E_or, E_h):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, bool, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    
    if enable_early_termination:
        # if ObjPosError > 10, or ObjRotError > 100, or KeyPosError > 20, terminate the env
        has_failed = ((E_op>10) | (E_or>100) | (E_h>20)) & (progress_buf>5)
        has_contact = (contact_buf != 0).any(dim=-1).any(dim=-1) # [num_envs]
        terminated = torch.where(has_failed & ~has_contact, torch.ones_like(reset_buf), terminated)
    
    if isTest and maxEpisodeLength > 0 :
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    else:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    return reset, terminated