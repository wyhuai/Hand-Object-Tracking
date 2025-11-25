import torch
import torch.nn.functional as F
from collections import defaultdict
import copy

from env.tasks.skillmimic_blender import SkillMimicBallPlayBlender

from utils.motion_data_handler import MotionDataHandler

class SkillMimic2BallPlayReweight(SkillMimicBallPlayBlender): 
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
        total_frames = sum([self._motion_data.motion_lengths[motion_id] for motion_id in self._motion_data.hoi_data_dict])
        self.reweight_interval = 5 * total_frames  # Determines how often reweighting occurs
        
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
        self._motion_data = MotionDataHandler(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                            self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset,
                                            reweight=self.reweight, reweight_alpha=self.reweight_alpha)
        return
        
    def _compute_reset(self):
        super()._compute_reset()
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

    ################### Optimized Reweighting Methods ###################
    def _reweight_motion(self, reset_env_ids):
        """
        Reweights motion sampling probabilities based on accumulated rewards.
        This method is optimized to utilize vectorized tensor operations for efficiency.
        """
        if not self.cfg['env']['reweight']:
            return  # Reweighting is disabled
        
        # # Debug
        # old_time = self._motion_data.time_sample_rate_tensor.clone()
        # old_clip = self._motion_data._motion_weights_tensor.clone()

        # Record rewards for reset environments
        self.record_motion_time_reward(reset_env_ids)
        
        # Perform reweighting at specified intervals
        if (self.progress_buf_total % self.reweight_interval == 0) and (self.progress_buf_total > 0):
            if self._motion_data.num_motions > 1:
                self.average_rewards = torch.mean(self.motion_time_seqreward, dim=1)
                # Debugging information
                print('##### Reweighting Motion Sampling Rates #####')
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
