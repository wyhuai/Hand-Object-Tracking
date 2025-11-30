from enum import Enum
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import time as pyton_time# changed by me
from pathlib import Path
from datetime import datetime
import math 

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObjectPlane


class SkillMimicBallPlay(HumanoidWholeBodyWithObjectPlane): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = str(cfg["env"]["stateInit"])
        if state_init.lower() == "random":
            self._state_init = -1
            print("Random Reference State Init (RRSI)")
        else:
            self._state_init = int(state_init)
            print(f"Deterministic Reference State Init from {self._state_init}")

        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.postproc_unihotdata = cfg['env']['postproc_unihotdata']
        self.robot_type = cfg["env"]["asset"]["assetFileName"]
        self.reward_weights_default = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.save_images_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.init_vel = cfg['env']['initVel']
        self.isTest = cfg['args'].test
        self._enable_rand_target_obs = cfg["env"]["enable_rand_target_obs"]
        self._enable_future_target_obs = cfg["env"]["enable_future_target_obs"]
        self._enable_text_obs = cfg["env"]["enable_text_obs"]
        self._enable_nearest_vector = cfg["env"]["enable_nearest_vector"]
        self._enable_obj_keypoints = cfg["env"]["enable_obj_keypoints"]
        self._enable_ig_scale = cfg["env"]["enable_ig_scale"]
        self._enable_ig_plus_reward = cfg["env"]["enable_ig_plus_reward"]
        self.show_current_traj = cfg["env"]["show_current_traj"]
        self._use_delta_action = cfg["env"]["use_delta_action"]
        self._use_res_action = cfg["env"]["use_res_action"]
        self._obj_rand_scale = cfg["env"]["obj_rand_scale"]
        self.apply_disturbance = cfg['env']['applyDisturbance']
        self._enable_disgravity = cfg['env']['enableDisgravity']
        self._enable_dof_obs = cfg['env']['enableDofObs']
        self._enable_dense_obj = cfg["env"]["enable_dense_obj"]

        self._enable_wrist_local_obs = cfg["env"]["enable_wrist_local_obs"]
        self.hand_model = cfg['env']['hand_model'] if cfg['env']['hand_model'] is not None else: "mano"
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless= headless)
        # 323 = root_pos (3) + root_rot (4)+ dof_pos (51*3)+ dof_pos_vel (51*3)+ obj_pos (3) + obj_rot (4)+ obj_pos_vel (3)
        # +6 is what????
        #self.ref_hoi_obs_size = 323 + len(self.cfg["env"]["keyBodies"])*3 + 6 -282#V1 #changed by me warning don't what is 323 so I just -276
        # 119 = root_pos (3) + root_rot (4)+ dof_pos (51)+ dof_pos_vel (51)+ obj_pos (3) + obj_rot (4)+ obj_pos_vel (3)
        self.ref_hoi_obs_size = 119 + (len(self.cfg["env"]["keyBodies"]))*3 +2 #V1 #changed by me warning don't what is 323 so I just -276 +1 is contact
        if self.hand_model == "mano":
            # 119 = root_pos (3) + root_rot (4)+ dof_pos (51)+ dof_pos_vel (51)+ obj_pos (3) + obj_rot (4)+ obj_pos_vel (3)
            self.ref_hoi_obs_size = 119 + (len(cfg["env"]["keyBodies"]))*3 +2 #V1 #changed by me warning don't what is 323 so I just -276 +1 is contact
            self.condition_size = 52
        elif self.hand_model == "shadow":
            # 73 = root_pos (3) + root_rot (4)+ dof_pos (28)+ dof_pos_vel (28)+ obj_pos (3) + obj_rot (4)+ obj_pos_vel (3)
            self.ref_hoi_obs_size = 73 + (len(cfg["env"]["keyBodies"]))*3+2 #V1 #changed by me warning don't what is 323 so I just -276 +1 is contact
            self.condition_size = 76 # the dimension of target (dim of target_obj_pos(3) + target_obj_quat(4) + target key body pos(23*3))
        elif self.hand_model == "allegro":
            # 61 = root_pos (3) + root_rot (4)+ dof_pos (22)+ dof_pos_vel (22)+ obj_pos (3) + obj_rot (4)+ obj_pos_vel (3)
            self.ref_hoi_obs_size = 61 + (len(cfg["env"]["keyBodies"]))*3+2 #V1 #changed by me warning don't what is 323 so I just -276 +1 is contact
            self.condition_size = 58 # the dimension of target (dim of target_obj_pos(3) + target_obj_quat(4) + target key body pos(17*3))

        self._load_motion(self.motion_file) #ZC1
        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        self.progress_buf_total = 0
        self.hoi_data_label_batch = torch.zeros([self.num_envs, self.condition_size], device=self.device, dtype=torch.float)
        self.motion_ids_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_times_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self._subscribe_events_for_change_condition()

        self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) #{}

        self.show_motion_test = False
        # self.init_from_frame_test = 0 #2 #ZC3
        self.motion_id_test = 0
        # self.options = [i for i in range(6) if i != 2]
        self.succ_pos = []
        self.fail_pos = []
        self.reached_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.int) #metric torch.bool

        self.show_abnorm = [0] * self.num_envs
        self.skill_labels = torch.ones(self.num_envs, device=self.device, dtype=torch.long) * 100
        self.obj_force = {'Sword': 12, 'Bottle': 6}

        return

    def get_state_for_metric(self):
        # 提供 Metric 计算所需的状态
        return {
            'obj_pos': self._target_states[..., 0:3],
            'obj_pos_vel': self._target_states[..., 7:10],
            'wrist_pos': self._rigid_body_pos[:, self._key_body_ids[-1], :],
            'progress': self.progress_buf,
            'key_pos_error': self.extras["metrics"]['E_h'],
            'obj_pos_error': self.extras["metrics"]['E_op'],
            'obj_rot_error': self.extras["metrics"]['E_or'],
        }
    

    def post_physics_step(self):
        self._update_condition()
        ######## Test ##########
        if self.isTest and not self.headless: 
            if not self.show_current_traj:
                self._set_traj()
            elif self.show_current_traj:
                self._set_traj_current()
        if self.apply_disturbance:
            self._apply_disturbance_forces()
        if self._enable_disgravity:
            self._disgravity()
        # extra calc of self._curr_obs, for imitation reward
        self._compute_hoi_observations()
        super().post_physics_step()

        self._update_hist_hoi_obs()
        return

    def _apply_disturbance_forces(self):
        # Generate random forces
        force_offset = 0.0
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        ## Apply forces at the target's current position
        positions = self._rigid_body_state[..., 0:3].clone().reshape(self.num_envs, bodies_per_env, 3)
        positions[:, self.num_bodies, 1] += force_offset
        positions = positions.reshape(1, -1, 3)
        forces = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float).reshape(self.num_envs, bodies_per_env, 3)
        torques = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        
        # Create a mask based on obj_contact
        # contact_mask = (self._curr_ref_obs[:, -2] == 1)
        # [just for place task] add random force for previous 50 frames
        contact_mask = (self.motion_times_total + self.progress_buf < 50) 
        random_forces = torch.zeros((self.num_envs, 3), device=self.device)
        random_forces[contact_mask] = torch.randn((contact_mask.sum(), 3),  # 只生成需要力的数量
                                                    device=self.device) * self.disturbance_force_scale
        # Apply forces only where obj_contact is True
        forces[:, self.num_bodies, :] = random_forces.reshape(-1, 3)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        # Draw force vectors as red lines
        self.gym.clear_lines(self.viewer)
        if self.cfg["headless"] == False:
            for env_idx in range(self.num_envs):
                # Get the force and position for the object in this environment
                force = forces[env_idx, self.num_bodies, :]
                pos_idx = env_idx * bodies_per_env + self.num_bodies
                pos = positions.reshape(-1, 3)[pos_idx]
                
                # Convert to numpy arrays for visualization
                pos_np = pos.cpu().numpy()
                force_np = force.cpu().numpy()
                
                # Scale the force for better visualization
                scale = 0.01  # Adjust this value to change line length
                end_pos = pos_np + force_np * scale
                
                # Draw line from object position to force direction
                vertices = np.array(
                    [pos_np[0], pos_np[1], pos_np[2], end_pos[0], end_pos[1], end_pos[2]],
                    dtype=np.float32
                )
                colors = np.array(
                    [[0.85, 0.1, 0.1]],  # 红色
                    dtype=np.float32
                )
                self.gym.add_lines(self.viewer, self.envs[env_idx], 1, vertices, colors)

    def _disgravity(self):
        # Generate random forces
        force_offset = 0.0
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        ## Apply forces at the target's current position
        positions = self._rigid_body_state[..., 0:3].clone().reshape(self.num_envs, bodies_per_env, 3)
        positions[:, self.num_bodies, 1] += force_offset
        positions = positions.reshape(1, -1, 3)
        forces = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float).reshape(self.num_envs, bodies_per_env, 3)
        torques = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        
        # Create a mask based on obj_contact
        z_forces = torch.zeros((self.num_envs, 3), device=self.device)
        z_forces[..., -1] = torch.ones((self.num_envs), device=self.device) * self.obj_force[self.object_names[0]]
        # Apply forces only where obj_contact is True
        forces[:, self.num_bodies, :] = z_forces.reshape(-1, 3)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        # Draw force vectors as red lines
        self.gym.clear_lines(self.viewer)
        if self.cfg["headless"] == False:
            for env_idx in range(self.num_envs):
                # Get the force and position for the object in this environment
                force = forces[env_idx, self.num_bodies, :]
                pos_idx = env_idx * bodies_per_env + self.num_bodies
                pos = positions.reshape(-1, 3)[pos_idx]
                
                # Convert to numpy arrays for visualization
                pos_np = pos.cpu().numpy()
                force_np = force.cpu().numpy()
                
                # Scale the force for better visualization
                scale = 0.01  # Adjust this value to change line length
                end_pos = pos_np + force_np * scale
                
                # Draw line from object position to force direction
                vertices = np.array(
                    [pos_np[0], pos_np[1], pos_np[2], end_pos[0], end_pos[1], end_pos[2]],
                    dtype=np.float32
                )
                colors = np.array(
                    [[0.85, 0.1, 0.1]],  # 红色
                    dtype=np.float32
                )
                self.gym.add_lines(self.viewer, self.envs[env_idx], 1, vertices, colors)


                
    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        
        obs_size += self.condition_size
        return obs_size

    def get_task_obs_size(self):
        return 0
    
    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None

        if self._enable_wrist_local_obs:
            humanoid_obs = self._compute_humanoid_local_obs(env_ids)
            obj_obs = self._compute_obj_local_obs(env_ids)
        else:
            humanoid_obs = self._compute_humanoid_obs(env_ids)
            obj_obs = self._compute_obj_obs(env_ids)
        if self.skill_labels[env_ids].shape != torch.Size([1]):
            obj_obs_cond = (self.skill_labels[env_ids]!=9).squeeze(0).unsqueeze(1)
        else:
            obj_obs_cond = self.skill_labels[env_ids]!=9
        obj_obs = torch.where(obj_obs_cond, obj_obs, torch.zeros_like(obj_obs))

        obs = humanoid_obs
        obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)

        if (env_ids is None): #Z
            env_ids = torch.arange(self.num_envs)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone() #ZC0
        if self.hand_model == "mano":
            ref_tar_pos = self._curr_ref_obs[:, 109:109+3].clone()
            ref_tar_rot = self._curr_ref_obs[:, 112:112+4].clone()
        elif self.hand_model == "shadow":        
            ref_tar_pos = self._curr_ref_obs[:, 63:63+3].clone()
            ref_tar_rot = self._curr_ref_obs[:, 66:66+4].clone()
        elif self.hand_model == "allegro":        
            ref_tar_pos = self._curr_ref_obs[:, 51:51+3].clone()
            ref_tar_rot = self._curr_ref_obs[:, 54:54+4].clone()

        
        self._ref_target_keypoints_per_epoch, self._ref_target_keypoints_vectors_per_epoch = self.extract_keypoints_per_epoch(ref_tar_pos, ref_tar_rot)
        mts = self.motion_times_total[env_ids]
        num_key = len(self._key_body_ids)
        if self.hand_model == "mano":
            next_target_obj_pos = self.hoi_data_batch[env_ids,ts][:,109:109+3].clone()
            next_target_obj_quat = self.hoi_data_batch[env_ids,ts][:,112:112+4].clone()
            next_target_key_pos = self.hoi_data_batch[env_ids,ts][:,119:119+num_key*3].clone()
            next_target_wrist_pos_vel = self.hoi_data_batch[env_ids,ts][:,58:58+3].clone()
        elif self.hand_model == "shadow":        
            next_target_obj_pos = self.hoi_data_batch[env_ids,ts][:,63:63+3].clone().clone()
            next_target_obj_quat = self.hoi_data_batch[env_ids,ts][:,66:66+4].clone().clone()
            next_target_key_pos = self.hoi_data_batch[env_ids,ts][:,73:73+num_key*3].clone()
            next_target_wrist_pos_vel = self.hoi_data_batch[env_ids,ts][:,35:35+3].clone()
        elif self.hand_model == "allegro":        
            next_target_obj_pos = self.hoi_data_batch[env_ids,ts][:,51:51+3].clone()
            next_target_obj_quat = self.hoi_data_batch[env_ids,ts][:,54:54+4].clone()
            next_target_key_pos = self.hoi_data_batch[env_ids,ts][:,61:61+num_key*3].clone()
            next_target_wrist_pos_vel = self.hoi_data_batch[env_ids,ts][:,29:29+3].clone()

        next_target_contact = self.hoi_data_batch[env_ids,ts][:, -2:-1].clone()

        wrist_pos = self._rigid_body_pos[env_ids, self._key_body_ids[-1], :].clone()
        wrist_rot = self._rigid_body_rot[env_ids, self._key_body_ids[-1], :].clone()
        current_obj_pos = self._target_states[env_ids, :3].clone()
        current_obj_quat = self._target_states[env_ids, 3:7].clone()
        current_key_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :].clone()
        next_target_obj_pos, next_target_obj_pos_residual, next_target_obj_quat, next_target_obj_quat_residual, \
        next_target_key_pos, next_target_key_pos_residual, next_target_wrist_pos_vel = \
            compute_local_next_target(wrist_pos, wrist_rot, num_key, current_key_pos, current_obj_pos, current_obj_quat,
                                        next_target_obj_pos, next_target_obj_quat, next_target_key_pos, next_target_wrist_pos_vel)

        tracking_obs = torch.cat((next_target_obj_pos, next_target_obj_quat, 
                                  next_target_key_pos, next_target_key_pos_residual, next_target_wrist_pos_vel,
                                  next_target_obj_pos_residual, next_target_obj_quat_residual), dim=-1)
        
        ############## Test ######################
        if self._enable_dof_obs:
            next_target_dof_pos = self.hoi_data_batch[env_ids,ts][:,7:7+self.num_joints].clone()
            next_target_dof_pos_residual = next_target_dof_pos - self._dof_pos[env_ids].clone()
            # 将差值调整到[-π, π]范围内
            next_target_dof_pos_residual = (next_target_dof_pos_residual + torch.pi) % (2 * torch.pi) - torch.pi
            tracking_obs = torch.cat((tracking_obs, next_target_contact, next_target_dof_pos, next_target_dof_pos_residual), dim=-1)
        ##########################################
        
        if self._enable_future_target_obs:
            key_frame_ids = torch.tensor([10, 20, 30, 40, 50], device=self.device).repeat(len(env_ids), 1)
            # key_frame_ids = torch.ones_like(key_frame_ids)
            num_key_frames = key_frame_ids.shape[1]
            key_frame_times = key_frame_ids + mts.unsqueeze(-1) + ts.unsqueeze(-1) # (num_envs, 6)

            # Ensure key_frame_tims smaller than motion_lengths
            ml = self._motion_data.motion_lengths[self.motion_ids_total[env_ids]].unsqueeze(-1).clone()
            key_frame_times = torch.where(key_frame_times >= ml-1, ml-1, key_frame_times)

            ref_motion = [self._motion_data.hoi_data_dict[mid.item()]['hoi_data'][key_frame_times[idx]].clone()
                                  for idx, mid in enumerate(self.motion_ids_total[env_ids])]
            key_ref_motion = torch.stack(ref_motion, dim=0) # (num_envs, 5, dim)
            if self.hand_model == "mano":
                seq_target_obj_pos = key_ref_motion[:,:,109:109+3].clone()
                seq_target_key_pos = key_ref_motion[:,:,119:119+num_key*3].clone()
            elif self.hand_model == "shadow":                
                seq_target_obj_pos = key_ref_motion[:,:,63:63+3].clone()
                seq_target_key_pos = key_ref_motion[:,:,73:73+num_key*3].clone()
            elif self.hand_model == "allegro":                
                seq_target_obj_pos = key_ref_motion[:,:,51:51+3].clone()
                seq_target_key_pos = key_ref_motion[:,:,61:61+num_key*3].clone() 

            seq_target_obj_pos, seq_target_key_pos, seq_target_key_pos_residual = \
            compute_local_future_target(wrist_pos, wrist_rot, num_key, num_key_frames,
                                        seq_target_obj_pos, seq_target_key_pos, current_key_pos)

            seq_target_obj_pos = seq_target_obj_pos.reshape(-1,num_key_frames*3) # (num_envs, 5*3)
            seq_target_key_pos = seq_target_key_pos.reshape(-1,num_key_frames*num_key*3) # (num_envs, 5*45/48)
            seq_target_key_pos_residual = seq_target_key_pos_residual.reshape(-1,num_key_frames*num_key*3) # (num_envs, 5*45/48)
            key_frame_times = key_frame_times - ts.unsqueeze(-1) - mts.unsqueeze(-1)
            key_frame_times = key_frame_times.float() / 50 # normalize to 0-1

            tracking_obs = torch.cat((tracking_obs, seq_target_obj_pos, seq_target_key_pos, seq_target_key_pos_residual, key_frame_times), dim=-1)

        obs = torch.cat((obs,tracking_obs),dim=-1)

        if self._enable_text_obs:
            textemb_batch = self.hoi_data_label_batch[env_ids].clone()
            obs = torch.cat((obs, textemb_batch), dim=-1)

        self.obs_buf[env_ids] = obs
        return

    def _compute_reset(self):
        step = self.motion_times_total + self.progress_buf
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, step,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._curr_ref_obs, self._curr_obs, self._motion_data.envid2episode_lengths,
                                                   self.isTest, self.cfg["env"]["episodeLength"],
                                                   self.extras["metrics"]['E_op'], self.extras["metrics"]['E_or'], self.extras["metrics"]['E_h']
                                                   )
        return
    
    def _compute_reward(self):
        # self.full_contact_forces
        self.rew_buf[:] = compute_humanoid_reward(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._hist_obs,
                                                  self._contact_forces,
                                                  self._tar_contact_forces,
                                                  self._tar2_contact_forces,
                                                  len(self._key_body_ids),
                                                  self._motion_data.reward_weights,
                                                  self.skill_labels,
                                                  self._enable_ig_scale,
                                                  self._ref_target_keypoints_per_epoch,
                                                  self._obs_target_keypoints_per_epoch,
                                                  self._enable_ig_plus_reward,
                                                  self.hand_model
                                                  )
        return
    
    def _compute_metrics(self):
        # self.full_contact_forces
        self.extras["metrics"] = {}
        self.extras["metrics"]['E_op'], self.extras["metrics"]['E_or'], self.extras["metrics"]['E_h'], self.extras["metrics"]['error_done'] \
            = compute_metrics(
                                                  self._curr_ref_obs,
                                                  self._curr_obs,
                                                  self._hist_obs,
                                                  self._contact_forces,
                                                  self._tar_contact_forces,
                                                  self._tar2_contact_forces,
                                                  len(self._key_body_ids),
                                                  self._motion_data.reward_weights,
                                                  self.skill_labels,
                                                  self._enable_ig_scale,
                                                  self._ref_target_keypoints_per_epoch,
                                                  self._obs_target_keypoints_per_epoch,
                                                  self._enable_ig_plus_reward,
                                                  self.hand_model
                                                  )
        return
    
    def _load_motion(self, motion_file):
        self.skill_name = motion_file.split('/')[-1] #metric
        self.max_episode_length = 60 #60 by meeee
        if self.cfg["env"]["episodeLength"] > 0:
            self.max_episode_length =  self.cfg["env"]["episodeLength"]
        
        self.hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)

        self._motion_data = MotionDataHandler(motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                            self.max_episode_length, self.reward_weights_default, self.init_vel, self.play_dataset)
        
        return
    


    def _subscribe_events_for_change_condition(self):
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "011") # dribble left
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "012") # dribble right
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "013") # dribble forward
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "001") # pick up
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "002") # shot
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "031") # layup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_X, "032") #
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "033") #
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "034") # turnaround layup
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "035") #
        
        return
    

    def _reset_envs(self, env_ids):
        if(len(env_ids)>0): #metric
            self.reached_target[env_ids] = 0
        
        super()._reset_envs(env_ids)

        return

    def _reset_env_tensors(self, env_ids): #Z10
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def _reset_state_init(self, env_ids):
        if self._state_init == -1:
            self.motion_ids, self.motion_times = self._reset_random_ref_state_init(env_ids) #V1 Random Ref State Init (RRSI)
        elif self._state_init >= 2:
            self.motion_ids, self.motion_times = self._reset_deterministic_ref_state_init(env_ids)
        else:
            assert(False), f"Unsupported state initialization from: {self._state_init}"
        return

    def after_reset_actors(self, env_ids):
        skill_label = self._motion_data.motion_class_tensor[self.motion_ids]
        self.hoi_data_label_batch[env_ids] = F.one_hot(skill_label, num_classes=self.condition_size).float()
        self.motion_ids_total[env_ids] = self.motion_ids.to(self.device).clone() # update motion_ids_total
        self.motion_times_total[env_ids] = self.motion_times.to(self.device).clone() # update motion_times_total
        total_env_ids = torch.arange(self.num_envs, device=self.device)
        self._target_keypoints = self.extract_keypoints(env_ids=total_env_ids)
  
        pass

    def _reset_actors(self, env_ids):
        self._reset_state_init(env_ids)

        super()._reset_actors(env_ids)

        self.after_reset_actors(env_ids)
        return
    
    def _reset_humanoid(self, env_ids):
        self._humanoid_root_states[env_ids, 0:3] = self.init_root_pos[env_ids]
        self._humanoid_root_states[env_ids, 3:7] = self.init_root_rot[env_ids]
        # self._humanoid_root_states[env_ids, 7:10] = self.init_root_pos_vel[env_ids]
        # self._humanoid_root_states[env_ids, 10:13] = self.init_root_rot_vel[env_ids]
        
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids]
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]
        return
    
    def _set_traj(self):
        super()._set_traj()
        motion_lengths = self._motion_data.motion_lengths.to(self.device).clone()
        
        # 计算有效索引
        max_indices = motion_lengths[self.motion_ids] - 1  # 减1防越界
        ts = self.progress_buf.clone()
        target_indices = torch.where((ts + 1) < max_indices, ts + 1, max_indices)
        ts_data = self.hoi_data_batch[torch.arange(self.num_envs, device=self.device), target_indices].clone()
        
        if self.hand_model == "mano":
            self._traj_states[:, :7] = ts_data[...,109:116]
            self._traj_states[:, 7:] = 0.0

            self._keypose_traj_states[..., :3] = ts_data[..., 119:119+3*16].reshape(self.num_envs, 16, 3)
        elif self.hand_model == "shadow":
            self._traj_states[:, :7] = ts_data[...,63:70]
            self._traj_states[:, 7:] = 0.0

            self._keypose_traj_states[..., :3] = ts_data[..., 73:73+3*23].reshape(self.num_envs, 23, 3)
        elif self.hand_model == "allegro":
            self._traj_states[:, :7] = ts_data[...,51:58]
            self._traj_states[:, 7:] = 0.0
            self._keypose_traj_states[..., :3] = ts_data[..., 61:61+3*17].reshape(self.num_envs, 17, 3)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._traj_actor_ids),
            len(self._traj_actor_ids)
        )
    
    def _set_traj_current(self):
        super()._set_traj_current()
        ts = self.progress_buf.clone()
        ts_data = self.hoi_data_batch[torch.arange(self.num_envs, device=self.device), ts].clone()
        if self.hand_model == "mano":
            self._traj_states[:, :7] = ts_data[..., 109:116]
            self._traj_states[:, 7:] = 0.0
            self._keypose_traj_states[..., :3] = ts_data[..., 119:119+3*16].reshape(self.num_envs, 16, 3)
        elif self.hand_model == "shadow":       
            self._traj_states[:, :7] = ts_data[..., 63:70]
            self._traj_states[:, 7:] = 0.0
            self._keypose_traj_states[..., :3] = ts_data[..., 73:73+3*23].reshape(self.num_envs, 23, 3)
        elif self.hand_model == "allegro":                    
            self._traj_states[:, :7] = ts_data[..., 51:58]
            self._traj_states[:, 7:] = 0.0
            self._keypose_traj_states[..., :3] = ts_data[..., 61:61+3*17].reshape(self.num_envs, 17, 3)         

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._traj_actor_ids),
            len(self._traj_actor_ids)
        )
    
    def _set_traj_play_dataset(self, motids):
        super()._set_traj_play_dataset(motids)
        motids = torch.tensor([motids], device=self.device).repeat(self.num_envs) if isinstance(motids, int) else motids
        motion_length = self._motion_data.motion_lengths.to(self.device).clone()
        self.progress_buf = (self.progress_buf + 1) % motion_length[motids]
        hoi_data = [self._motion_data.hoi_data_dict[motid.item()]['hoi_data'].clone() for motid in motids]
        hoi_data = torch.stack(hoi_data, dim=0)

        # 计算有效索引
        max_indices = motion_length[motids] - 1  # 减1防越界
        target_indices = torch.where(
            (self.progress_buf + 1) < max_indices,
            self.progress_buf + 1,
            max_indices
        )    
        ts_data = hoi_data[torch.arange(self.num_envs, device=self.device), target_indices].clone()

        if self.hand_model == "mano":
            self._traj_states[:, :7] = ts_data[...,109:116]
            self._traj_states[:, 7:] = 0.0

            self._keypose_traj_states[..., :3] = ts_data[..., 119:119+3*16].reshape(self.num_envs, 16, 3)
        elif self.hand_model == "shadow":
            self._traj_states[:, :7] = ts_data[...,63:70]
            self._traj_states[:, 7:] = 0.0

            self._keypose_traj_states[..., :3] = ts_data[..., 73:73+3*23].reshape(self.num_envs, 23, 3)
        elif self.hand_model == "allegro":
            self._traj_states[:, :7] = ts_data[...,51:58]
            self._traj_states[:, 7:] = 0.0

            self._keypose_traj_states[..., :3] = ts_data[..., 61:61+3*17].reshape(self.num_envs, 17, 3)
        self._target_traj_actor_ids = torch.cat([self._tar_actor_ids, self._traj_actor_ids], dim=-1)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._target_traj_actor_ids),
            len(self._target_traj_actor_ids)
        )
        
    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
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
        num_envs = env_ids.shape[0]

        ######## Test ##########
        motion_ids = self._motion_data.sample_motions(num_envs)
        # motion_ids = torch.arange(num_envs, device=self.device)
        # print(self._motion_data.hoi_data_dict[motion_ids.item()]['hoi_data_path'])
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

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :].clone()
        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :].clone(), 
                                                               self._rigid_body_rot[:, 0, :].clone(), 
                                                               self._rigid_body_vel[:, 0, :].clone(), 
                                                               self._rigid_body_ang_vel[:, 0, :].clone(), 
                                                               self._dof_pos.clone(), self._dof_vel.clone(), key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                            #    self._target2_states,
                                                               self._hist_obs,
                                                               self.progress_buf)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :].clone(), 
                                                                   self._rigid_body_rot[env_ids][:, 0, :].clone(), 
                                                                   self._rigid_body_vel[env_ids][:, 0, :].clone(), 
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :].clone(),
                                                                   self._dof_pos[env_ids].clone(), self._dof_vel[env_ids].clone(), key_body_pos[env_ids].clone(),
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._target_states[env_ids],
                                                                #    self._target2_states[env_ids],
                                                                   self._hist_obs[env_ids],
                                                                   self.progress_buf[env_ids])


        
        return
    
    def _update_condition(self):
        for evt in self.evts:
            if evt.action.isdigit() and evt.value > 0:
                self.hoi_data_label_batch = torch.nn.functional.one_hot(torch.tensor(int(evt.action)).to("cuda"), num_classes=self.condition_size).repeat(self.hoi_data_label_batch.shape[0],1)


    def play_dataset_postproc_unihotdata(self): #Z12
        
        # for each raw data
        for motid in range(self._motion_data.num_motions):

            self.datalength = self._motion_data.hoi_data_dict[motid]['dof_pos'].shape[0]

            for time in range(self.datalength):

                ### update object ###
                self._target_states[:, 0:3] = self._motion_data.hoi_data_dict[motid]['obj_pos'][time,:]
                self._target_states[:, 3:7] = self._motion_data.hoi_data_dict[motid]['obj_rot'][time,:]          
                self._target_states[:, 7:10] = torch.zeros_like(self._target_states[:, 7:10])
                self._target_states[:, 10:13] = torch.zeros_like(self._target_states[:, 10:13])

                ### update subject ###   
                _humanoid_root_pos = self._motion_data.hoi_data_dict[motid]['root_pos'][time,:].clone()
                _humanoid_root_rot = self._motion_data.hoi_data_dict[motid]['root_rot'][time,:].clone()
                self._humanoid_root_states[:, 0:3] = _humanoid_root_pos
                self._humanoid_root_states[:, 3:7] = _humanoid_root_rot               
                self._dof_pos[:, :] = self._motion_data.hoi_data_dict[motid]['dof_pos'][time,:].clone()
                self._dof_vel[:] = torch.zeros_like(self._dof_vel[:])

                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
                self._refresh_sim_tensors()     
                self.render(t=time)
                self.gym.simulate(self.sim)

                # record keybody pos
                if time == 0: 
                    self.keybodies = self._rigid_body_pos[:1, self._key_body_ids, :]
                else:
                    self.keybodies = torch.cat((self.keybodies, self._rigid_body_pos[:1, self._key_body_ids, :]),dim=0)      

                if time>=(self.datalength-1):
                    hoi_data = torch.cat((
                        self._motion_data.hoi_data_dict[motid]['root_pos'][:self.datalength-1].clone(),
                        self._motion_data.hoi_data_dict[motid]['root_rot'][:self.datalength-1].clone(),
                        self._motion_data.hoi_data_dict[motid]['dof_pos'][:self.datalength-1].clone(),
                        self.keybodies[1:].reshape(-1,len(self._key_body_ids)*3),
                        self._motion_data.hoi_data_dict[motid]['obj_pos'][:self.datalength-1].clone(),
                        self._motion_data.hoi_data_dict[motid]['obj_rot'][:self.datalength-1].clone(),
                        self._motion_data.hoi_data_dict[motid]['obj2_pos'][:self.datalength-1].clone(),
                        self._motion_data.hoi_data_dict[motid]['obj2_rot'][:self.datalength-1].clone(),
                        self._motion_data.hoi_data_dict[motid]['contact1'][:self.datalength-1].clone(),
                        self._motion_data.hoi_data_dict[motid]['contact2'][:self.datalength-1].clone(),
                        ),dim=-1)

                    original_path = self._motion_data.hoi_data_dict[motid]['hoi_data_path']
                    path_obj = Path(original_path)
                    parent_dir = path_obj.parent
                    base_dir = path_obj.parent.name
                    filename = path_obj.name

                    new_dir_name = f"{base_dir}_kp"
                    new_dir_path = parent_dir.parent/new_dir_name
                    os.makedirs(new_dir_path,exist_ok=True)

                    new_file_path = new_dir_path/filename

                    save_hoi_data = hoi_data.clone()
                    torch.save(save_hoi_data,str(new_file_path))
                    print("save to",new_file_path)
                    # import sys
                    # sys.exit(0)

        return self.obs_buf


    def play_dataset_step(self, time, motid=None, length=None): #Z12
        self._target_keypoints = torch.stack([
                self.extract_keypoints(motid=motid).to(self.device).squeeze(0)
                for i in range(self.num_envs)
            ])
        t = time
        if t == 3:
            print(self._motion_data.hoi_data_dict[motid]['hoi_data_path'])
        for env_id, env_ptr in enumerate(self.envs):
            if motid is None:
                motid = self._motion_data.envid2motid[env_id].item() # if not self.play_dataset_switch[env_id] else 1
            t = t % self._motion_data.motion_lengths[motid]

            ### update object ###
            self._target_states[env_id, 0:3] = self._motion_data.hoi_data_dict[motid]['obj_pos'][t,:]
            self._target_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj_rot'][t,:]          
            self._target_states[env_id, 7:10] = torch.zeros_like(self._target_states[env_id, 7:10])
            self._target_states[env_id, 10:13] = torch.zeros_like(self._target_states[env_id, 10:13])

            self._target2_states[env_id, 0:3] = self._motion_data.hoi_data_dict[motid]['obj2_pos'][t,:]
            self._target2_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj2_rot'][t,:]   
            
            # ### update subject ###   
            _humanoid_root_pos = self._motion_data.hoi_data_dict[motid]['root_pos'][t,:].clone()
            _humanoid_root_rot = self._motion_data.hoi_data_dict[motid]['root_rot'][t,:].clone()
            self._humanoid_root_states[env_id, 0:3] = _humanoid_root_pos
            self._humanoid_root_states[env_id, 3:7] = _humanoid_root_rot               
            
            ref_wrist_pose = self._motion_data.hoi_data_dict[motid]['dof_pos'][t,:6].clone()
            self._dof_pos[env_id, :] = self._motion_data.hoi_data_dict[motid]['dof_pos'][t,:].clone()
            self._dof_vel[env_id] = torch.zeros_like(self._dof_vel[env_id])

            contact = self._motion_data.hoi_data_dict[motid]['contact1'][t,:]
            root_rot_vel = self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t,:]
            # angle, _ = torch_utils.exp_map_to_angle_axis(root_rot_vel)
            angle = torch.norm(root_rot_vel)
            abnormal = torch.any(torch.abs(angle) > 5.) #Z

            if abnormal == True:
                print("frame:", t, "abnormal:", abnormal, "angle", angle)
                # print(" ", self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t])
                # print(" ", angle)
                self.show_abnorm[env_id] = 10

            handle = self._target_handles[env_id]
            if torch.all(contact == 1., dim=-1):
                # purple means contact
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1., 0, 1.))
            elif torch.all(contact == 0., dim=-1):
                # green means no contact
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(0., 1., 0.))
            elif torch.all(contact == -1., dim=-1):
                # gray means don't consider about the contact
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(0.5, 0.5, 0.5))
            '''
            if abnormal == True or self.show_abnorm[env_id] > 0: #Z
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 1.)) 
                self.show_abnorm[env_id] -= 1
            else:
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.)) 
            '''
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self._set_traj_play_dataset(motid)
        self._refresh_sim_tensors()     
        self.render(t=time)
        self.gym.simulate(self.sim)

        return self.obs_buf
    
    def _draw_task_play(self,t):
        
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32) # color

        self.gym.clear_lines(self.viewer)

        starts = self._motion_data.hoi_data_dict[0]['hoi_data'][t, :3]

        for i, env_ptr in enumerate(self.envs):
            for j in range(len(self._key_body_ids)):
                vec = self._motion_data.hoi_data_dict[0]['key_body_pos'][t, j*3:j*3+3]
                vec = torch.cat([starts, vec], dim=-1).cpu().numpy().reshape([1, 6])
                self.gym.add_lines(self.viewer, env_ptr, 1, vec, cols)

        return
    
    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
            
            if self.save_images:
                env_ids = 0
                os.makedirs("skillmimic/data/images/" + self.save_images_timestamp, exist_ok=True)
                frame_id = t if self.play_dataset else self.progress_buf[env_ids]
                frame_id = len(os.listdir("skillmimic/data/images/" + self.save_images_timestamp))
                rgb_filename = "skillmimic/data/images/" + self.save_images_timestamp + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename)
        return
    
    def get_num_amp_obs(self):
        return self.ref_hoi_obs_size



#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, target_states, hist_obs, progress_buf):

    ## diffvel, set 0 for the first frame
    # hist_dof_pos = hist_obs[:,6:6+156]
    # dof_diffvel = (dof_pos - hist_dof_pos)*fps
    # dof_diffvel = dof_diffvel*(progress_buf!=1).to(float).unsqueeze(dim=-1)

    dof_vel = dof_vel*(progress_buf!=1).unsqueeze(dim=-1)

    contact1 = torch.zeros(key_body_pos.shape[0],1,device=dof_vel.device)
    contact2 = torch.zeros(key_body_pos.shape[0],1,device=dof_vel.device)
    #obs = torch.cat((root_pos, torch_utils.quat_to_exp_map(root_rot), dof_pos, dof_vel, target_states[:,:10], key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), contact), dim=-1)
    obs = torch.cat((root_pos, root_rot, dof_pos, dof_vel, target_states[:,:10], key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), contact1, contact2), dim=-1) #warninggggg changed by me to use quat

    return obs
    
@torch.jit.script
def quaternion_to_rotation_matrix_batch(quaternion):
    # 确保四元数是单位四元数
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
    
    # 分解四元数分量 (B, 4)
    x = quaternion[:, 0]
    y = quaternion[:, 1]
    z = quaternion[:, 2]
    w = quaternion[:, 3]
    
    # 构建旋转矩阵 (B, 3, 3)
    rotation_matrix = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=1)
    ], dim=1)
    
    return rotation_matrix

@torch.jit.script
def transform_keypoints_batch(keypoints, position, quaternion):
    """
    Args:
        keypoints: torch.Tensor of shape (B, N, 3)
        position: torch.Tensor of shape (B, 3)
        quaternion: torch.Tensor of shape (B, 4) (x, y, z, w)
    Returns:
        transformed_keypoints: torch.Tensor of shape (B, N, 3)
    """

    # 获取旋转矩阵 (B, 3, 3)
    R = quaternion_to_rotation_matrix_batch(quaternion)
    
    # 进行旋转 (B, N, 3)
    rotated_points = torch.bmm(keypoints, R.transpose(1, 2))
    
    # 进行平移 (B, N, 3)
    # 将position从(B, 3)扩展为(B, 1, 3)以便广播
    transformed_points = rotated_points + position.unsqueeze(1)
    
    return transformed_points

@torch.jit.script
def compute_local_next_target(wrist_pos: Tensor, wrist_rot: Tensor, 
                              len_keypos: int, current_key_pos: Tensor,
                              current_obj_pos:Tensor, current_obj_quat: Tensor, 
                              next_target_obj_pos: Tensor, next_target_obj_quat: Tensor, 
                              next_target_key_pos: Tensor,
                              next_target_wrist_pos_vel: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    num_envs = wrist_pos.shape[0]
    heading_rot = torch_utils.calc_heading_quat_inv(wrist_rot)
    local_next_target_obj_pos = next_target_obj_pos - wrist_pos
    local_next_target_obj_pos = quat_rotate(heading_rot, local_next_target_obj_pos)
    local_next_target_obj_quat = quat_mul(heading_rot, next_target_obj_quat)

    # heading_rot_expand = heading_rot.unsqueeze(1).repeat((1, 20, 1)) # [num_envs, 20, 4]
    # flat_heading_rot = heading_rot_expand.reshape(-1, 4) # [num_envs*20, 4]
    # local_tar_keypoints = tar_keypoints - wrist_pos.unsqueeze(1)
    # flat_local_tar_keypoints = local_tar_keypoints.reshape(-1, 3) # [num_envs*20, 3]
    # flat_local_tar_keypoints = quat_rotate(flat_heading_rot, flat_local_tar_keypoints)
    # local_tar_keypoints = flat_local_tar_keypoints.view(local_tar_keypoints.shape[0],-1) # [num_envs, 60]
    heading_rot_expand = heading_rot.unsqueeze(1).repeat((1, len_keypos, 1))
    flat_heading_rot = heading_rot_expand.reshape(-1, 4) # [num_envs*16, 4]

    next_target_key_pos = next_target_key_pos.reshape(-1, len_keypos, 3) # [num_envs, 16, 3]
    local_next_target_key_pos = next_target_key_pos - wrist_pos.unsqueeze(1) 
    
    flat_local_next_target_key_pos = local_next_target_key_pos.reshape(-1, 3) # [num_envs*16, 3]
    flat_local_next_target_key_pos = quat_rotate(flat_heading_rot, flat_local_next_target_key_pos)
    local_next_target_key_pos = flat_local_next_target_key_pos.view(num_envs, -1) # [num_envs, 48]

    ####################### Runyi Debug #######################
    local_next_target_key_pos_residual = next_target_key_pos - current_key_pos # [num_envs, 16, 3] 
    flat_local_next_target_key_pos_residual = local_next_target_key_pos_residual.reshape(-1, 3) # [num_envs*16, 3]
    flat_local_next_target_key_pos_residual = quat_rotate(flat_heading_rot, flat_local_next_target_key_pos_residual)
    local_next_target_key_pos_residual = flat_local_next_target_key_pos_residual.view(num_envs, -1) # [num_envs, 48]
    next_target_wrist_pos_vel = quat_rotate(heading_rot, next_target_wrist_pos_vel)

    local_next_target_obj_pos_residual = next_target_obj_pos - current_obj_pos
    local_next_target_obj_pos_residual = quat_rotate(heading_rot, local_next_target_obj_pos_residual)

    obj_heading_rot = torch_utils.calc_heading_quat_inv(current_obj_quat)
    local_next_target_obj_quat_residual = quat_mul(obj_heading_rot, next_target_obj_quat)
    ###########################################################

    return local_next_target_obj_pos, local_next_target_obj_pos_residual, local_next_target_obj_quat, local_next_target_obj_quat_residual, \
        local_next_target_key_pos, local_next_target_key_pos_residual, next_target_wrist_pos_vel

@torch.jit.script
def compute_local_future_target(wrist_pos: Tensor, wrist_rot: Tensor, 
                                len_keypos: int, num_key_frames: int, 
                                seq_target_obj_pos: Tensor, seq_target_key_pos: Tensor, 
                                current_key_pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    num_envs = wrist_pos.shape[0]
    seq_target_key_pos = seq_target_key_pos.reshape(num_envs, num_key_frames, len_keypos, 3)
    heading_rot = torch_utils.calc_heading_quat_inv(wrist_rot)
    
    # to get local_seq_target_obj_pos
    heading_rot_expand = heading_rot.unsqueeze(1).repeat((1, num_key_frames, 1))
    flat_heading_rot = heading_rot_expand.reshape(-1, 4) # [num_envs*5, 4]
    local_seq_target_obj_pos = seq_target_obj_pos - wrist_pos.unsqueeze(1) # [num_envs, 5, 3]
    flat_local_seq_target_obj_pos = local_seq_target_obj_pos.reshape(-1, 3) # [num_envs*5, 3]
    flat_local_seq_target_obj_pos = quat_rotate(flat_heading_rot, flat_local_seq_target_obj_pos)
    local_seq_target_obj_pos = flat_local_seq_target_obj_pos.view(num_envs, num_key_frames, -1) # [num_envs, 5, 3]

    # to get local_seq_target_key_pos
    heading_rot_expand = heading_rot.unsqueeze(1).repeat((1, num_key_frames, len_keypos, 1))
    flat_heading_rot = heading_rot_expand.reshape(-1, 4) # [num_envs*5*16, 4]
    local_seq_target_key_pos = seq_target_key_pos - wrist_pos.unsqueeze(1).unsqueeze(1) # [num_envs, 5, 16, 3]
    flat_local_seq_target_key_pos = local_seq_target_key_pos.reshape(-1, 3) # [num_envs*5*16, 3]
    flat_local_seq_target_key_pos = quat_rotate(flat_heading_rot, flat_local_seq_target_key_pos)
    local_seq_target_key_pos = flat_local_seq_target_key_pos.view(num_envs, num_key_frames, len_keypos, -1) # [num_envs, 5, 16, 3]
    
    ####################### Runyi Debug #######################
    local_seq_target_key_pos_residual = seq_target_key_pos - current_key_pos.unsqueeze(1) # [num_envs, 5, 16, 3] 
    flat_local_seq_target_key_pos_residual = local_seq_target_key_pos_residual.reshape(-1, 3) # [num_envs*5*16, 3]
    flat_local_seq_target_key_pos_residual = quat_rotate(flat_heading_rot, flat_local_seq_target_key_pos_residual)
    local_seq_target_key_pos_residual = flat_local_seq_target_key_pos_residual.view(num_envs, num_key_frames, len_keypos, -1) # [num_envs, 5, 16, 3]
    ###########################################################

    return local_seq_target_obj_pos, local_seq_target_key_pos, local_seq_target_key_pos_residual

# @torch.jit.script
def compute_humanoid_reward(hoi_ref: Tensor, hoi_obs: Tensor, hoi_obs_hist: Tensor,
                            contact_buf: Tensor, tar_contact_forces: Tensor, 
                            tar2_contact_forces: Tensor, len_keypos: int, 
                            w: Dict[str, Tensor], skill_label: Tensor, enable_ig_scale: bool, 
                            ref_obj_keypoints: Tensor,
                            obs_obj_keypoints: Tensor, enable_ig_plus_reward: bool,hand_model) -> Tensor:

    ### data preprocess ###

    #when rot = quat
    if hand_model=="mano":
        root_pos = hoi_obs[:,:3]
        root_rot = hoi_obs[:,3:3+4]
        wrist_pos = hoi_obs[:,7:7+3]
        wrist_rot = hoi_obs[:,10:10+3]
        dof_pos = hoi_obs[:,13:13+15*3]
        wrist_pos_vel = hoi_obs[:,58:58+3]
        dof_pos_vel = hoi_obs[:,58:58+17*3]
        obj_pos = hoi_obs[:,109:109+3]
        obj_rot = hoi_obs[:,112:112+4]
        obj_pos_vel = hoi_obs[:,116:116+3]
        key_pos = hoi_obs[:,119:119+len_keypos*3]
    elif hand_model=="shadow":
        root_pos = hoi_obs[:,:3]
        root_rot = hoi_obs[:,3:3+4]
        wrist_pos = hoi_obs[:,7:7+3]
        wrist_rot = hoi_obs[:,10:10+3]
        dof_pos = hoi_obs[:,13:13+22]
        wrist_pos_vel = hoi_obs[:,35:35+3]
        dof_pos_vel = hoi_obs[:,35:35+28]
        obj_pos = hoi_obs[:,63:63+3]
        obj_rot = hoi_obs[:,66:66+4]
        obj_pos_vel = hoi_obs[:,70:70+3]
        key_pos = hoi_obs[:,73:73+len_keypos*3]
    elif hand_model=="allegro":
        root_pos = hoi_obs[:,:3]
        root_rot = hoi_obs[:,3:3+4]
        wrist_pos = hoi_obs[:,7:7+3]
        wrist_rot = hoi_obs[:,10:10+3]
        dof_pos = hoi_obs[:,13:13+16]
        wrist_pos_vel = hoi_obs[:,29:29+3]
        dof_pos_vel = hoi_obs[:,29:29+22]
        obj_pos = hoi_obs[:,51:51+3]
        obj_rot = hoi_obs[:,54:54+4]
        obj_pos_vel = hoi_obs[:,58:58+3]
        key_pos = hoi_obs[:,61:61+len_keypos*3]
    ###################################################

    ig = key_pos.view(-1,len_keypos,3).transpose(0,1) - obj_pos[:,:3]
    # ig_wrist = ig.transpose(0,1)[:,0:7+1,:].view(-1,(7+1)*3) #ZC
    ig = ig.transpose(0,1).view(-1,(len_keypos)*3)

    #TODO: add relative rot error

    ##############################################################changed by me
    # dof_pos_vel_hist = hoi_obs_hist[:,58:58+17*3] #ZC
    
    
    # reference states
    if hand_model=="mano":
        ref_root_pos = hoi_ref[:,:3]
        ref_root_rot = hoi_ref[:,3:3+4]
        ref_wrist_pos = hoi_ref[:,7:7+3]
        ref_wrist_rot = hoi_ref[:,10:10+3]
        ref_dof_pos = hoi_ref[:,13:13+15*3]
        ref_wrist_pos_vel = hoi_ref[:,58:58+3]
        ref_dof_pos_vel = hoi_ref[:,58:58+17*3]
        ref_obj_pos = hoi_ref[:,109:109+3]
        ref_obj_rot = hoi_ref[:,112:112+4]
        ref_obj_pos_vel = hoi_ref[:,116:116+3]
        ref_key_pos = hoi_ref[:,119:119+len_keypos*3]
        ref_obj_contact = hoi_ref[:,-2:]
        ref_obj2_contact = hoi_ref[:,-1:]
    elif hand_model=="shadow":
        ref_root_pos = hoi_ref[:,:3]
        ref_root_rot = hoi_ref[:,3:3+4]
        ref_wrist_pos = hoi_ref[:,7:7+3]
        ref_wrist_rot = hoi_ref[:,10:10+3]
        ref_dof_pos = hoi_ref[:,13:13+22]
        ref_wrist_pos_vel = hoi_ref[:,35:35+3]
        ref_dof_pos_vel = hoi_ref[:,35:35+28]
        ref_obj_pos = hoi_ref[:,63:63+3]
        ref_obj_rot = hoi_ref[:,66:66+4]
        ref_obj_pos_vel = hoi_ref[:,70:70+3]
        ref_key_pos = hoi_ref[:,73:73+len_keypos*3]
        ref_obj_contact = hoi_ref[:,-2:]
    elif hand_model=="allegro":
        ref_root_pos = hoi_ref[:,:3]
        ref_root_rot = hoi_ref[:,3:3+4]
        ref_wrist_pos = hoi_ref[:,7:7+3]
        ref_wrist_rot = hoi_ref[:,10:10+3]
        ref_dof_pos = hoi_ref[:,13:13+16]
        ref_wrist_pos_vel = hoi_ref[:,29:29+3]
        ref_dof_pos_vel = hoi_ref[:,29:29+22]
        ref_obj_pos = hoi_ref[:,51:51+3]
        ref_obj_rot = hoi_ref[:,54:54+4]
        ref_obj_pos_vel = hoi_ref[:,58:58+3]
        ref_key_pos = hoi_ref[:,61:61+len_keypos*3]
        ref_obj_contact = hoi_ref[:,-2:]
        
        
    ##########################################################################
    ref_ig = ref_key_pos.view(-1,len_keypos,3).transpose(0,1) - ref_obj_pos[:,:3]
    ref_ig = ref_ig.transpose(0,1).view(-1,(len_keypos)*3)

    ####################### Part1: body reward #######################
    # body pos reward
    ep = torch.mean((ref_key_pos - key_pos)**2,dim=-1)
    rp = torch.exp(-ep*w['p'])

    # body rot reward
    er = torch.mean((ref_dof_pos - dof_pos)**2,dim=-1)
    rr = torch.exp(-er*w['r'])
    rb = rp*rr
    ####################### Part1.5: wrist reward #######################
    ewp = torch.mean((ref_wrist_pos - wrist_pos)**2,dim=-1)
    rwp = torch.exp(-ewp*20)
    
    ref_wrist_rot = quat_from_euler_xyz(ref_wrist_rot[..., 0], ref_wrist_rot[..., 1], ref_wrist_rot[..., 2]) # euler
    ref_wrist_rot = ref_wrist_rot / torch.norm(ref_wrist_rot, dim=-1, keepdim=True)
    wrist_rot = quat_from_euler_xyz(wrist_rot[..., 0], wrist_rot[..., 1], wrist_rot[..., 2])
    wrist_rot = wrist_rot / torch.norm(wrist_rot, dim=-1, keepdim=True)
    dot = torch.sum(ref_wrist_rot*wrist_rot,dim=-1)
    dot = torch.clamp(dot, -1.0, 1.0)
    ewr = torch.acos(dot) / torch.pi  # [0,1]
    rwr = torch.exp(-ewr*20) # [exp(-10),1]

    ewpv = torch.mean((ref_wrist_pos_vel - wrist_pos_vel)**2,dim=-1)
    rwpv = torch.exp(-ewpv*20)

    rw = rwp*rwr
    
    # only apply rwpv when the object is in contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(torch.float) # =1 when contact happens to the object
    rw = torch.where((obj_contact == 1) & (skill_label!=8), rw*rwpv, rw)

    ####################### Part2: object reward #######################
    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])

    # object rot reward
    dot = torch.sum(ref_obj_rot*obj_rot,dim=-1)
    obj_rot_adjusted = torch.where(dot.unsqueeze(-1)<0, -obj_rot, obj_rot)
    eor = torch.mean((ref_obj_rot - obj_rot_adjusted)**2,dim=-1)
    ror = torch.exp(-eor*w['or'])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
    ropv = torch.exp(-eopv*w['opv'])

    ro = rop*ror*ropv#*rokp*rorv

    ####################### Part3: interaction graph reward #######################
    eig = torch.mean((ref_ig - ig)**2,dim=-1) #Zw
    # ref_key_pos.shape [n, 48]
    # tar_keypoints.shape [n, 60]
    # 1. 计算参考目标到手的距离（ref_tar2hand_dist）
    #ref_obj_keypoints = transform_keypoints_batch(obj_keypoints, ref_obj_pos, ref_obj_rot) # [n, 20, 3]
    ref_key_pos_reshaped = ref_key_pos.reshape(-1, len_keypos, 3)
    ref_tar2hand_dist = ref_key_pos_reshaped.unsqueeze(2) - ref_obj_keypoints.unsqueeze(1) # [n, 16, 20, 3]
    ref_tar2hand_dist = torch.norm(ref_tar2hand_dist, p=2, dim=-1) # [n, 16, 20]
    # 2. 计算当前目标到手的距离（tar2hand_dist）
    #obj_keypoints = transform_keypoints_batch(obj_keypoints, obj_pos, obj_rot)              # [n, 20, 3]
    key_pos_reshaped = key_pos.reshape(-1, len_keypos, 3)                                    # [n, 16, 3]
    tar2hand_diff = key_pos_reshaped.unsqueeze(2) - obs_obj_keypoints.unsqueeze(1)              # [n, 16, 20, 3]
    tar2hand_dist = torch.norm(tar2hand_diff, p=2, dim=-1)                                  # [n, 16, 20]
    # 3. 计算手到物体的最小距离
    num_obj_keypoiint = ref_obj_keypoints.shape[1]
    min_tar2hand_dist, _ = torch.min(ref_tar2hand_dist.view(-1, len_keypos*num_obj_keypoiint), dim=1) # [n]
    ig_factor = torch.exp(-2 * min_tar2hand_dist) * 2 if enable_ig_scale else 1
    '''
    when min_tar2hand_dist=0, w['ig']*ig_factor = 40
    when min_tar2hand_dist=0.2, w['ig']*ig_factor = 25
    when min_tar2hand_dist=1, w['ig']*ig_factor = 5
    '''
    rig = torch.exp(-eig*w['ig']*ig_factor)
    if enable_ig_plus_reward:
        eigp = torch.mean((tar2hand_dist - ref_tar2hand_dist) ** 2, dim=(1, 2))                # [n]
        rigp = torch.exp(-eigp * 20)
        # no ig plus reward while min_distance > 0.02, or the skill is catch
        rig = torch.where((min_tar2hand_dist < 0.02) & (skill_label!=7), rig * rigp, rig)

    ####################### Part4: simplified contact graph reward #######################
    # object contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(torch.float) # =1 when contact happens to the object
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:,0])
    rcg2 = torch.exp(-ecg2*w['cg2'])
    # if the ref_obj_contact==-1, we don't consider this contact, the reward is 1
    rcg = torch.where(ref_obj_contact[:, 0] == -1, torch.tensor(1.0, device=rcg2.device), rcg2)

    ####################### HOI imitation reward #######################
    # reward = torch.where((skill_label!=7) & (skill_label!=8), 
    #                      rb * rw * ro * rig * rcg,
    #                      rb * rw * rop * rig * rcg) # no object_rotation for throw & catch
    reward = rb * rw * ro * rig * rcg
    # only rb*rw for free_move
    reward = torch.where((skill_label!=9), reward, rb * rw)

    return reward


# @torch.jit.script
def compute_metrics(hoi_ref: Tensor, hoi_obs: Tensor, hoi_obs_hist: Tensor,
                            contact_buf: Tensor, tar_contact_forces: Tensor, 
                            tar2_contact_forces: Tensor, len_keypos: int, 
                            w: Dict[str, Tensor], skill_label: Tensor, enable_ig_scale: bool, ref_obj_keypoints: Tensor,
                            obs_obj_keypoints: Tensor, enable_ig_plus_reward: bool,hand_model="mano") -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if hand_model=="mano":
        obj_pos = hoi_obs[:,109:109+3]
        obj_rot = hoi_obs[:,112:112+4]
        key_pos = hoi_obs[:,119:119+len_keypos*3]
        ref_obj_pos = hoi_ref[:,109:109+3]
        ref_obj_rot = hoi_ref[:,112:112+4]
        ref_key_pos = hoi_ref[:,119:119+len_keypos*3]
    elif hand_model=="shadow":
        obj_pos = hoi_obs[:,63:63+3]
        obj_rot = hoi_obs[:,66:66+4]
        key_pos = hoi_obs[:,73:73+len_keypos*3]
        ref_obj_pos = hoi_ref[:,63:63+3]
        ref_obj_rot = hoi_ref[:,66:66+4]
        ref_key_pos = hoi_ref[:,73:73+len_keypos*3]
    elif hand_model=="allegro":
        obj_pos = hoi_obs[:,51:51+3]
        obj_rot = hoi_obs[:,54:54+4]
        key_pos = hoi_obs[:,61:61+len_keypos*3]
        ref_obj_pos = hoi_ref[:,51:51+3]
        ref_obj_rot = hoi_ref[:,54:54+4]
        ref_key_pos = hoi_ref[:,61:61+len_keypos*3]

    zero_quat_mask = torch.all(ref_obj_rot == 0, dim=-1)
    # Object positional error
    num_obj_keypoiint = ref_obj_keypoints.shape[1]
    ref_obj_keypoints = ref_obj_keypoints.view(-1,num_obj_keypoiint*3)
    curr_obj_keypoints = obs_obj_keypoints.view(-1,num_obj_keypoiint*3)
    E_op = torch.mean(torch.sqrt((ref_obj_keypoints - curr_obj_keypoints)**2),dim=-1) * 100
    E_op = torch.where(zero_quat_mask, torch.tensor(0.0, device=ref_obj_rot.device), E_op)
    # Object orientational error
    # Normalize input quaternions (xyzw format)
    ref_norm = ref_obj_rot / torch.norm(ref_obj_rot, dim=-1, keepdim=True)
    obj_norm = obj_rot / torch.norm(obj_rot, dim=-1, keepdim=True)
    # Compute dot product (cosine of half-angle)
    dot = torch.sum(ref_norm * obj_norm, dim=-1)
    # Handle double coverage and numerical stability
    dot = torch.clamp(dot, min=-1.0, max=1.0)  # Allow negative values for proper angle calculation
    
    # Compute angle difference (radians to degrees)
    # angle_rad = 2 * torch.acos(dot) / torch.pi * 180 # Use abs for angle calculation
    angle_rad = (2 * torch.acos(dot) + torch.pi) % (2 * torch.pi) - torch.pi
    angle_rad = torch.abs(angle_rad) * 180 / torch.pi

    E_or = torch.where(zero_quat_mask, torch.tensor(0.0, device=ref_obj_rot.device), angle_rad)    
    # Mean Per-Joint Position Error for Hands
    ref_key_pos = ref_key_pos.reshape(-1, len_keypos, 3)
    key_pos = key_pos.reshape(-1, len_keypos, 3)
    E_h = torch.norm(ref_key_pos - key_pos, p=2, dim=-1).mean(dim=-1) * 100
    E_h = torch.where(zero_quat_mask, torch.tensor(0.0, device=ref_obj_rot.device), E_h)   
    return E_op, E_or, E_h, zero_quat_mask

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, step, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths,
                           isTest, maxEpisodeLength, E_op, E_or, E_h):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, bool, int, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        # if ObjPosError > 10, or ObjRotError > 90, or KeyPosError > 20, terminate the env
        has_failed_nocontact = ((E_op>10) | (E_or>90) | (E_h>20)) & (progress_buf>5) # (E_or>100) | 
        has_failed_contact = ((E_op>8) | (E_or>90) | (E_h>15)) & (progress_buf>5)
        has_contact = (contact_buf != 0).any(dim=-1).any(dim=-1) # [num_envs]
        terminated = torch.where(has_failed_nocontact & ~has_contact, torch.ones_like(reset_buf), terminated)
        terminated = torch.where(has_failed_contact & has_contact, torch.ones_like(reset_buf), terminated)

    if isTest and maxEpisodeLength > 0 :
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    else:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC
    
    return reset, terminated