from enum import Enum
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

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler

from env.tasks.humanoid_object_task import HumanoidWholeBodyWithObject


class SkillMimicBallPlayInsert(HumanoidWholeBodyWithObject): 
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
        self._enable_text_obs = cfg["env"]["enable_text_obs"]

        self.condition_size = 52

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

        return

    def post_physics_step(self):
        self._update_condition()
        if self.isTest:
            self._set_traj()
        
        # extra calc of self._curr_obs, for imitation reward
        self._compute_hoi_observations()

        super().post_physics_step()

        # self._compute_hoi_observations()
        self._update_hist_hoi_obs()
        return

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
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs
        obj_obs = self._compute_obj_obs(env_ids)
        obj2_obs = self._compute_obj2_obs(env_ids)
        obs = torch.cat([obs, obj_obs, obj2_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)

        if (env_ids is None): #Z
            env_ids = torch.arange(self.num_envs)
        ts = self.progress_buf[env_ids].clone()
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone() #ZC0

        mts = self.motion_times_total[env_ids]
        # next_ts = mts + ts
        # next_target_body_pos= [self._motion_data.hoi_data_dict[mid.item()]['body_pos'][next_ts[idx]].clone()
        #                         for idx, mid in enumerate(self.motion_ids_total[env_ids])]
        # next_target_body_pos = torch.stack(next_target_body_pos, dim=0)

        next_target_obj_pos = self.hoi_data_batch[env_ids,ts][:,109:109+3]
        next_target_obj_quat = self.hoi_data_batch[env_ids,ts][:,112:112+4]
        # next_target_wrist_pos = next_target_body_pos[:,0:3]
        # next_target_wrist_quat = torch_utils.exp_map_to_quat(self.hoi_data_batch[env_ids,ts][:,7:7+3])
        next_target_key_pos = self.hoi_data_batch[env_ids,ts][:,119:119+15*3]

        tracking_obs = torch.cat((next_target_obj_pos, next_target_obj_quat,
                                    # next_target_wrist_pos, next_target_wrist_quat, 
                                    next_target_key_pos), dim=-1)
        
        if self._enable_rand_target_obs:
            # ############################# version 1: random time target #############################
            # # 生成 delta_ts，保证 ts+delta_ts 不超过 motion_lengths
            # delta_ts = torch.randint(20, self.max_episode_length, (len(env_ids),), device=self.device)
            # ml = self._motion_data.motion_lengths[self.motion_ids_total[env_ids]]
            # delta_ts = torch.where(mts + ts + delta_ts >= ml - 1, 
            #                        ml - mts - ts - 1, 
            #                        delta_ts)
            
            # rand_ts = mts + ts + delta_ts
            # rand_target_motion = [self._motion_data.hoi_data_dict[mid.item()]['hoi_data'][rand_ts[idx]].clone()
            #                       for idx, mid in enumerate(self.motion_ids_total[env_ids])]
            # rand_target_body_pos= [self._motion_data.hoi_data_dict[mid.item()]['body_pos'][rand_ts[idx]].clone()
            #                       for idx, mid in enumerate(self.motion_ids_total[env_ids])]
            # rand_target_motion = torch.stack(rand_target_motion, dim=0)
            # rand_target_body_pos = torch.stack(rand_target_body_pos, dim=0)

            # rand_target_obj_pos = rand_target_motion[:,109:109+3]
            # rand_target_obj_quat = rand_target_motion[:,112:112+4]
            # # rand_target_wrist_pos = rand_target_body_pos[:,0:3]
            # # rand_target_wrist_quat = torch_utils.exp_map_to_quat(rand_target_motion[:,7:7+3])
            # rand_target_key_pos = rand_target_motion[:,119:119+15*3]
            
            # tracking_obs = torch.cat((tracking_obs, rand_target_obj_pos, rand_target_obj_quat, 
                                    # rand_target_wrist_pos, rand_target_wrist_quat, 
            #                        rand_target_key_pos, delta_ts.unsqueeze(-1)), dim=-1)
            
            ############################# version 2 hoi_data_batch 6 frames #############################
            # key_frame_ids = torch.tensor([0, 10, 20, 30, 40, 50], device=self.device)
            # seq_target_obj_pos = self.hoi_data_batch[env_ids][:,key_frame_ids,109:109+3] # (num_envs, 6, 3)
            # seq_target_key_pos = self.hoi_data_batch[env_ids][:,key_frame_ids,119:119+15*3] # (num_envs, 6, 45)

            # tracking_obs = torch.cat((tracking_obs, seq_target_obj_pos.view(-1,len(key_frame_ids)*3), 
            #                           seq_target_key_pos.view(-1, len(key_frame_ids)*45), ts.unsqueeze(-1)), dim=-1)
            
            ############################# version 3 future 6 frames #############################
            key_frame_ids = torch.tensor([0, 10, 20, 30, 40, 50], device=self.device).repeat(len(env_ids), 1)
            num_key_frames = key_frame_ids.shape[1]
            key_frame_times = key_frame_ids + mts.unsqueeze(-1) + ts.unsqueeze(-1) # (num_envs, 6)

            # Ensure key_frame_tims smaller than motion_lengths
            ml = self._motion_data.motion_lengths[self.motion_ids_total[env_ids]].unsqueeze(-1)
            key_frame_times = torch.where(key_frame_times >= ml-1, ml-1, key_frame_times)

            ref_motion = [self._motion_data.hoi_data_dict[mid.item()]['hoi_data'][key_frame_times[idx]].clone()
                                  for idx, mid in enumerate(self.motion_ids_total[env_ids])]
            key_ref_motion = torch.stack(ref_motion, dim=0) # (num_envs, 6, dim)
            seq_target_obj_pos = key_ref_motion[:,:,109:109+3].reshape(-1,num_key_frames*3) # (num_envs, 6, 3)
            seq_target_key_pos = key_ref_motion[:,:,119:119+15*3].reshape(-1,num_key_frames*45) # (num_envs, 6, 45)

            tracking_obs = torch.cat((tracking_obs, seq_target_obj_pos, seq_target_key_pos), dim=-1)
        
        obs = torch.cat((obs,tracking_obs),dim=-1)

        if self._enable_text_obs:
            textemb_batch = self.hoi_data_label_batch[env_ids]
            obs = torch.cat((obs, textemb_batch), dim=-1)

        self.obs_buf[env_ids] = obs
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights, 
                                                   self._curr_ref_obs, self._curr_obs, self._motion_data.envid2episode_lengths,
                                                   self.isTest, self.cfg["env"]["episodeLength"]
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
                                                  self._motion_data.reward_weights
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
        self.motion_ids_total[env_ids] = self.motion_ids.to(self.device) # update motion_ids_total
        self.motion_times_total[env_ids] = self.motion_times.to(self.device) # update motion_times_total
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
        motion_lengths = self._motion_data.motion_lengths.to(self.device)
        
        # 计算有效索引
        max_indices = motion_lengths[self.motion_ids] - 1  # 减1防越界
        target_indices = torch.where(
            (self.progress_buf + 1) < max_indices,
            self.progress_buf + 1,
            max_indices
        )
        traj_pos_rot = self.hoi_data_batch[
            torch.arange(self.num_envs, device=self.device), 
            target_indices, 
            109:116
        ]
        self._traj_states[:, :7] = traj_pos_rot
        self._traj_states[:, 7:] = 0.0

        # 15
        for i in range(15):
            key_body_pos = self.hoi_data_batch[
                torch.arange(self.num_envs, device=self.device), 
                target_indices, 
                119+3*i:119+3*i+3
            ]
            keypose_traj_states = getattr(self, f"_keypose_traj_states_{i}")
            keypose_traj_states[:, :3] = key_body_pos

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._traj_actor_ids),
            len(self._traj_actor_ids)
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

        return motion_ids, motion_times
    
    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids], \
        self.init_obj2_pos[env_ids], self.init_obj2_pos_vel[env_ids], self.init_obj2_rot[env_ids], self.init_obj2_rot_vel[env_ids] \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        return motion_ids, motion_times

    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :], 
                                                               self._rigid_body_rot[:, 0, :], 
                                                               self._rigid_body_vel[:, 0, :], 
                                                               self._rigid_body_ang_vel[:, 0, :], 
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                            #    self._target2_states,
                                                               self._hist_obs,
                                                               self.progress_buf)
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :], 
                                                                   self._rigid_body_rot[env_ids][:, 0, :], 
                                                                   self._rigid_body_vel[env_ids][:, 0, :], 
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :], 
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
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
                        self.keybodies[1:].reshape(-1,15*3),
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


    def play_dataset_step(self, time): #Z12

        t = time

        for env_id, env_ptr in enumerate(self.envs):
            # t += self.envid2idt[env_id]

            ### update object ###
            motid = self._motion_data.envid2motid[env_id].item()
            # motid = 0
            self._target_states[env_id, 0:3] = self._motion_data.hoi_data_dict[motid]['obj_pos'][t,:]
            
            # exp_map = torch_utils.angle_axis_to_exp_map(torch.tensor([motid*torch.pi/4]), torch.tensor([0,0,1]))
            # quat = torch_utils.exp_map_to_quat(exp_map)
            # self._target_states[env_id, 3:7] = quat
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

            # target_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) #testing
            # bodies_per_env = target_body_state.shape[0] // self.num_envs
            # target_body_state = gymtorch.wrap_tensor(target_body_state).view(self.num_envs, 55, 13) #testing
            # # target_body_state = gymtorch.wrap_tensor(target_body_state).view(self.num_envs, 55, 13) #testing # changed by me after adding table
            # target_body_pos = target_body_state[..., 52, 0:3] #testing
            self._dof_vel[env_id] = torch.zeros_like(self._dof_vel[env_id])

            contact = self._motion_data.hoi_data_dict[motid]['contact1'][t,:]
            obj_contact = torch.any(contact > 0.1, dim=-1)
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
            if obj_contact == True:
                # print(t, "contact")
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1., 0., 0.))
            else:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(0., 1., 0.))
            '''
            if abnormal == True or self.show_abnorm[env_id] > 0: #Z
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 1.)) 
                self.show_abnorm[env_id] -= 1
            else:
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.)) 
            '''
        # print(time)
        # print("ref_wrist_pose[0]",ref_wrist_pose[0])
        # print("slide_xyz",self._dof_pos[env_id, :3])  
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
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
    

# @torch.jit.script
def compute_humanoid_reward(hoi_ref, hoi_obs, hoi_obs_hist, contact_buf, tar_contact_forces, tar2_contact_forces, len_keypos, w): #ZCr
    ## type: (Tensor, Tensor, Tensor, Tensor, Int, float) -> Tensor

    ### data preprocess ###

    #when rot = quat
    root_pos = hoi_obs[:,:3]
    root_rot = hoi_obs[:,3:3+4]
    dof_pos = hoi_obs[:,7:7+17*3]
    dof_pos_vel = hoi_obs[:,58:58+17*3]
    obj_pos = hoi_obs[:,109:109+3]
    obj_rot = hoi_obs[:,112:112+4]
    obj_pos_vel = hoi_obs[:,116:116+3]
    obj2_start = 116+3
    obj2_pos = hoi_obs[:,119:119+3]
    obj2_rot = hoi_obs[:,122:122+4]
    obj2_pos_vel = hoi_obs[:,126:126+3]
    key_pos = hoi_obs[:,119:119+len_keypos*3]
    # contact = hoi_obs[:,-1:]# fake one
    ###################################################
    # key_pos = torch.cat((root_pos, key_pos),dim=-1)
    # body_rot = torch.cat((root_rot, dof_pos),dim=-1)
    ig = key_pos.view(-1,len_keypos,3).transpose(0,1) - obj_pos[:,:3]
    # ig_wrist = ig.transpose(0,1)[:,0:7+1,:].view(-1,(7+1)*3) #ZC
    ig = ig.transpose(0,1).view(-1,(len_keypos)*3)

    ig2 = obj2_pos[:,:3] - obj_pos[:,:3]

    #TODO: add relative rot error

    ##############################################################changed by me
    dof_pos_vel_hist = hoi_obs_hist[:,58:58+17*3] #ZC
    
    
    # reference states
    ref_root_pos = hoi_ref[:,:3]
    ref_root_rot = hoi_ref[:,3:3+4]
    ref_dof_pos = hoi_ref[:,7:7+17*3]
    ref_dof_pos_vel = hoi_ref[:,58:58+17*3]
    ref_obj_pos = hoi_ref[:,109:109+3]
    ref_obj_rot = hoi_ref[:,112:112+4]
    ref_obj_pos_vel = hoi_ref[:,116:116+3]
    obj2_start = 116+3
    ref_obj2_pos = hoi_ref[:,119:119+3]
    ref_obj2_rot = hoi_ref[:,122:122+4]
    ref_obj2_pos_vel = hoi_ref[:,126:126+3] # total 175 dim
    ref_key_pos = hoi_ref[:,119:119+len_keypos*3]
    ref_obj_contact = hoi_ref[:,-2:]
    ref_obj2_contact = hoi_ref[:,-1:]



    ##########################################################################
    # ref_key_pos = torch.cat((ref_root_pos, ref_key_pos),dim=-1)
    # ref_body_rot = torch.cat((ref_root_rot, ref_dof_pos),dim=-1)
    ref_ig = ref_key_pos.view(-1,len_keypos,3).transpose(0,1) - ref_obj_pos[:,:3]
    # ref_ig_wrist = ref_ig.transpose(0,1)[:,0:7+1,:].view(-1,(7+1)*3) #ZC
    ref_ig = ref_ig.transpose(0,1).view(-1,(len_keypos)*3)

    ref_ig2 = ref_obj2_pos[:,:3] - ref_obj_pos[:,:3]

    ####################### Part1: body reward #######################
    # body pos reward
    ep = torch.mean((ref_key_pos - key_pos)**2,dim=-1)
    # ep = torch.mean((ref_key_pos[:,0:(7+1)*3] - key_pos[:,0:(7+1)*3])**2,dim=-1) #ZC
    rp = torch.exp(-ep*w['p'])

    # body rot reward
    er = torch.mean((ref_dof_pos - dof_pos)**2,dim=-1)
    rr = torch.exp(-er*w['r'])
    rb = rp*rr

    ####################### Part2: object reward #######################
    # object pos reward
    eop = torch.mean((ref_obj_pos - obj_pos)**2,dim=-1)
    rop = torch.exp(-eop*w['op'])

    # object rot reward
    # ref_obj_rot = torch_utils.quat_to_exp_map(ref_obj_rot)
    # obj_rot = torch_utils.quat_to_exp_map(obj_rot)
    dot = torch.sum(ref_obj_rot*obj_rot,dim=-1)
    obj_rot_adjusted = torch.where(dot.unsqueeze(-1)<0, -obj_rot, obj_rot)
    eor = torch.mean((ref_obj_rot - obj_rot_adjusted)**2,dim=-1)
    ror = torch.exp(-eor*w['or'])

    # object pos vel reward
    eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
    ropv = torch.exp(-eopv*w['opv'])

    # # object rot vel reward
    # eorv = torch.zeros_like(ep) #torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
    # rorv = torch.exp(-eorv*w['orv'])

    ro = rop*ror*ropv#*rorv

    ####################### Part3: interaction graph reward #######################
    eig1 = torch.mean((ref_ig - ig)**2,dim=-1) #Zw
    eig2 = torch.mean((ref_ig2 - ig2)**2,dim=-1)
    rig = torch.exp(-eig1*w['ig']) * torch.exp(-eig2*w['ig'])

    ####################### Part4: simplified contact graph reward #######################
    # object contact
    obj_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > 0.1, dim=-1).to(torch.float) # =1 when contact happens to the object
    ecg2 = torch.abs(obj_contact - ref_obj_contact[:,0])
    rcg2 = torch.exp(-ecg2*w['cg2'])
    rcg = rcg2
    
    ####################### HOI imitation reward #######################
    reward = rb * ro * rig * rcg

    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, hoi_ref, hoi_obs, envid2episode_lengths,
                           isTest, maxEpisodeLength):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, Tensor, Tensor, Tensor, Tensor, bool, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights# [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    if isTest and maxEpisodeLength > 0 :
        reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    else:
        reset = torch.where(progress_buf >= envid2episode_lengths-1, torch.ones_like(reset_buf), terminated) #ZC

    # reset = torch.where(progress_buf >= max_episode_length -1, torch.ones_like(reset_buf), terminated)
    # reset = torch.zeros_like(reset_buf) #ZC300

    return reset, terminated