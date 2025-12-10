from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import vmap

from rl_games.common.tr_helpers import unsqueeze_obs
from torch import Tensor
from typing import Tuple, Dict
import glob, os, random, pickle
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime
import time
from collections import defaultdict

from utils import torch_utils
from functorch import make_functional
from utils.obj_distill_motion_data_handler import MultiMotionDataHandler


from rl_games.algos_torch import torch_ext
from env.tasks.skillmimic2_rand import SkillMimicHandRand
from learning import skillmimic_network_builder_denseobj
from learning.distill import skillmimic_models_distill

import yaml
import copy


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


def get_all_paths(dir_path):
    paths = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            paths.append(os.path.join(root, name))
    return paths


class MultiObjDistill(SkillMimicHandRand):
    def print_memory_stats(self, prefix=""):
        if torch.cuda.is_available():
            print(f"{prefix} Memory - Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
                f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB, "
                f"Max Allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        else:
            print("CUDA not available for memory stats")
    
    def _print_timing_stats(self):
        """Print timing statistics for debugging performance bottlenecks"""
        print("\n" + "="*80)
        print(f"TIMING STATISTICS (after {self.step_count} steps)")
        print("="*80)
        
        # Collect all timing data
        timing_data = []
        for key, times in self.timing_stats.items():
            if len(times) > 0:
                avg_time = np.mean(times[-self.print_timing_interval:]) * 1000  # Convert to ms
                max_time = np.max(times[-self.print_timing_interval:]) * 1000
                min_time = np.min(times[-self.print_timing_interval:]) * 1000
                total_time = np.sum(times[-self.print_timing_interval:]) * 1000
                timing_data.append((key, avg_time, max_time, min_time, total_time))
        
        # Sort by average time (descending)
        timing_data.sort(key=lambda x: x[1], reverse=True)
        
        # Print header
        print(f"{'Operation':<30} {'Avg (ms)':<12} {'Max (ms)':<12} {'Min (ms)':<12} {'Total (ms)':<12}")
        print("-"*80)
        
        # Print each operation
        for key, avg_time, max_time, min_time, total_time in timing_data:
            print(f"{key:<30} {avg_time:>10.4f}  {max_time:>10.4f}  {min_time:>10.4f}  {total_time:>10.2f}")
        
        print("="*80)
        
        # Detailed analysis
        print("\n📊 DETAILED ANALYSIS:")
        print("-"*80)
        
        # Exclude outliers (first few steps) for more accurate average
        if self.step_count > 10:
            # Calculate stats excluding first 10 steps
            stats_dict = {}
            for key, times in self.timing_stats.items():
                if len(times) > 10:
                    recent_times = times[10:]  # Exclude first 10 steps
                    stats_dict[key] = np.mean(recent_times) * 1000
            
            if stats_dict:
                print("Average times (excluding first 10 warmup steps):")
                sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1], reverse=True)
                for key, avg_time in sorted_stats[:5]:  # Top 5
                    print(f"  {key:<30} {avg_time:>10.4f} ms")
        
        # Calculate effective step time (excluding resets)
        if 'super_step' in self.timing_stats and 'batched_forward' in self.timing_stats:
            avg_super = np.mean(self.timing_stats['super_step'][-self.print_timing_interval:]) * 1000
            avg_batched = np.mean(self.timing_stats['batched_forward'][-self.print_timing_interval:]) * 1000
            avg_compute_obs = np.mean(self.timing_stats.get('total_compute_obs', [0])[-self.print_timing_interval:]) * 1000
            effective_step = avg_super + avg_batched + avg_compute_obs
            print(f"\nEffective step time (non-reset): ~{effective_step:.2f} ms")
            print(f"  - Physics simulation: {avg_super:.2f} ms ({avg_super/effective_step*100:.1f}%)")
            print(f"  - Policy inference: {avg_batched:.2f} ms ({avg_batched/effective_step*100:.1f}%)")
            print(f"  - Observation compute: {avg_compute_obs:.2f} ms ({avg_compute_obs/effective_step*100:.1f}%)")
        
        print("="*80)
        
        # Identify top 3 bottlenecks (excluding total_* aggregates)
        actionable_bottlenecks = [(k, a, mx, mn, t) for k, a, mx, mn, t in timing_data 
                                   if not k.startswith('total_')]
        print("\n🔥 TOP 3 ACTIONABLE BOTTLENECKS:")
        for i, (key, avg_time, max_time, _, _) in enumerate(actionable_bottlenecks[:3], 1):
            print(f"  {i}. {key}: {avg_time:.4f} ms (max: {max_time:.2f} ms)")
        print("="*80 + "\n")

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.refined_motion_file = cfg['env']['refined_motion_file']
        self.refined_motion_as_obs = cfg['env']['refined_motion_as_obs']
        # Initialize timing statistics
        self.timing_stats = defaultdict(list)
        self.step_count = 0
        self.print_timing_interval = 1000  # Print timing stats every N steps
        
        super().__init__(cfg=cfg,
                        sim_params=sim_params,
                        physics_engine=physics_engine,
                        device_type=device_type,
                        device_id=device_id,
                        headless=headless)

        self.print_memory_stats("After super init")

        network = skillmimic_network_builder_denseobj.SkillMimicBuilderFutureVec()
        self.print_memory_stats("After network creation")

        with open(os.path.join(os.getcwd(), cfg["env"]["teacherPolicyCFG"]), 'r') as f:
            cfg_teacher = yaml.load(f, Loader=yaml.SafeLoader)

        network.load(cfg_teacher['params']['network'])
        network = skillmimic_models_distill.SkillMimicModelContinuous(network)
        teacher_policy = cfg["env"]["teacherPolicy"]
        self.states = None
        self.obs_buf_refined = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        models_path = get_all_paths(teacher_policy)
        self.models = []
        self.functional_models = []
        self.params_list = []
        self.running_means = []
        self.running_vars = []
        self.models_subid = []
        self.models_obj=[]
        config = {
            'actions_num' : 17*3,
            'input_shape' : (1084, ),
            'num_seqs' : cfg["env"]["numEnvs"] * 1,
            'value_size': 1,
        }
        # Use a single functional model for all since architectures are identical. Remove the list of functional models and create one functional model to use with all parameter sets.
        for model_path in models_path:
            print(f"Loading model from {models_path}")
            subid = int(model_path.split('_')[-1].split('.')[0])
            models_obj=model_path.split('/')[-1].split("_")[1]
            ck = torch_ext.load_checkpoint(model_path)
            model = network.build(config)
            model.to(self.device)
            model.load_state_dict(ck['model'])
            model.eval()
            for param in model.parameters():
                param.requires_grad_
            self.models.append(model)
            f_model, params = make_functional(model)

            self.functional_models.append(f_model)
            self.params_list.append(params)
            running_mean, running_var = ck['running_mean_std']['running_mean'], ck['running_mean_std']['running_var']
            self.running_means.append(running_mean)
            self.running_vars.append(running_var)
            self.models_subid.append(subid)
            self.models_obj.append(models_obj)
        self.print_memory_stats("After all models loaded")

        # Transpose list of parameter tuples into tuple of parameter lists
        self.params_zip = list(zip(*self.params_list))
        # Now stack along a new dimension (model dimension = 0)
        self.stacked_params = tuple(torch.stack(p_tensors, dim=0) for p_tensors in self.params_zip)
        self.running_means_all = torch.stack(self.running_means).float()  # shape [num_models, ...]
        self.running_vars_all = torch.stack(self.running_vars).float()   # shape [num_models, ...]
             
        return

    def _load_motion(self, motion_file):
        super()._load_motion(motion_file)
        if self.refined_motion_file:
            self.refined_hoi_data_batch = torch.zeros([self.num_envs, self.max_episode_length, self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
            self._motion_data_refined = MultiMotionDataHandler(self.refined_motion_file, self.device, self._key_body_ids, self.cfg, self.num_envs, 
                                                self.max_episode_length, self.reward_weights_default,  self.play_dataset,
                                                reweight=self.reweight, reweight_alpha=self.reweight_alpha)
            # for key, item in self._motion_data.hoi_data_dict.items():
            #     print(item['hoi_data_path'])            
            #     print(self._motion_data_refined.hoi_data_dict[key]['hoi_data_path'])     
            #     print("-------------------")
        return
    
    def step(self, actions):
        step_start = time.time()
        
        # Time super().step() - This includes physics simulation and post_physics_step
        t0 = time.time()
        super().step(actions=actions)
        super_step_time = time.time() - t0
        self.timing_stats['super_step'].append(super_step_time)
        
        # Log if super_step is unusually slow
        if super_step_time > 0.5:  # More than 500ms
            print(f"⚠️  Warning: super_step took {super_step_time*1000:.2f}ms (unusually slow)")
        
        with torch.no_grad():
            # Time vmap setup
            t0 = time.time()
            batched_forward = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            self.timing_stats['vmap_setup'].append(time.time() - t0)

            # Time observation buffer preparation
            t0 = time.time()
            teacher_obs_buf= self.obs_buf
            obs_buf_time = time.time() - t0
            self.timing_stats['obs_buf_prep'].append(obs_buf_time)
            
            # Log if obs_buf_prep is unusually slow (likely first-time CUDA initialization)
            if obs_buf_time > 1.0:  # More than 1 second
                print(f"⚠️  Warning: obs_buf_prep took {obs_buf_time*1000:.2f}ms (likely CUDA initialization)")
            
            # Time batched forward pass (most critical)
            t0 = time.time()
            selected_action = batched_forward(self.stacked_params, teacher_obs_buf.unsqueeze(0).repeat(self.running_means_all.shape[0], 1, 1), self.running_means_all, self.running_vars_all)
            self.timing_stats['batched_forward'].append(time.time() - t0)
            
            # Time action processing
            t0 = time.time()
            self.sample_indices = torch.arange(self.num_envs, device=self.device)
            teacher_actions_all = torch.clamp(selected_action, min=-1.0, max=1.0)
            self.action_buf = teacher_actions_all[self.model_indices, self.sample_indices]
            self.mu_buf = selected_action[self.model_indices, self.sample_indices]
            self.timing_stats['action_processing'].append(time.time() - t0)
        
        self.timing_stats['total_step'].append(time.time() - step_start)
        
        # Print timing statistics periodically
        self.step_count += 1
        if self.step_count % self.print_timing_interval == 0:
            self._print_timing_stats()


    def single_model_forward(self, params, obs, mean, var):
        self.has_batch_dimension = True
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)

        curr_obs = self._preproc_obs(obs, mean, var)
        # Construct input_dict for the model
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs': curr_obs,
            'rnn_states': self.states
        }
        
        # Run the functional model
        res_dict = self.functional_models[0](params, input_dict)  # We'll rely on vmap to pick correct slice of params
        mu = res_dict['mus']
        self.states = res_dict['rnn_states']
        is_deterministic = True # as teacher are freeze
        if is_deterministic:
            current_action = mu
        #else:
        #    current_action = action

        return current_action   

         
    def _preproc_obs(self, obs_batch, mean, var):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        self.normalize_input = True    
        self.epsilon = 1e-05    

        if self.normalize_input:
            obs_batch = (obs_batch - mean) / torch.sqrt(var + 1e-5)
            obs_batch = torch.clamp(obs_batch, min=-5.0, max=5.0)
            
        return obs_batch
    
    def _reset_envs(self, env_ids=None):
        reset_start = time.time()
        
        # Time super()._reset_envs()
        t0 = time.time()
        super()._reset_envs(env_ids=env_ids)
        self.timing_stats['reset_super'].append(time.time() - t0)
        
        # Time model index mapping
        t0 = time.time()
        id_to_index = {obj: i for i, obj in enumerate(self.models_obj)}
        # if 0 in id_to_index:
        #     id_to_index.update({ 1: id_to_index[0], 2:id_to_index[0], 3 :id_to_index[0] # for grasp, move, place
        #     #id_to_index.update({ 1: id_to_index[0]
        #     })
        self._motion_data_refined._map_env_2_object_tensor
        self.model_indices = torch.tensor([id_to_index[self._motion_data_refined._obj_id_2_name[obj.item()]]
                                             for obj in self._motion_data_refined._map_env_2_object_tensor], 
                             dtype=torch.long, device=self.device)
        self.timing_stats['model_index_mapping'].append(time.time() - t0)

        # Time initial action computation
        t0 = time.time()
        self.sample_indices = torch.arange(self.num_envs, device=self.device)
        with torch.no_grad():
            batched_forward = vmap(self.single_model_forward, in_dims=(0, 0, 0, 0))
            teacher_obs_buf= self.obs_buf
            selected_action = batched_forward(self.stacked_params, teacher_obs_buf.unsqueeze(0).repeat(self.running_means_all.shape[0], 1, 1), self.running_means_all, self.running_vars_all)
            teacher_actions_all = torch.clamp(selected_action, min=-1.0, max=1.0)
            # Select the appropriate action for each env from the batched teacher predictions  
            # model_indices determines which teacher policy to follow per env  
            # sample_indices (usually just arange(num_envs)) picks the right action for each env
            self.action_buf = teacher_actions_all[self.model_indices, self.sample_indices]
            self.mu_buf = selected_action[self.model_indices, self.sample_indices]
        self.timing_stats['reset_action_compute'].append(time.time() - t0)
        
        self.timing_stats['total_reset'].append(time.time() - reset_start)
        return


    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs_start = time.time()
        obs = None

        # Time humanoid and object observation computation
        t0 = time.time()

        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obj_obs = self._compute_obj_obs(env_ids)

        self.timing_stats['compute_basic_obs'].append(time.time() - t0)
        if self.skill_labels[env_ids].shape != torch.Size([1]):
            obj_obs_cond = (self.skill_labels[env_ids]!=9).squeeze(0).unsqueeze(1)
        else:
            obj_obs_cond = self.skill_labels[env_ids]!=9
        obj_obs = torch.where(obj_obs_cond, obj_obs, torch.zeros_like(obj_obs))
        # if self._enable_obj_zero:
        #     obj_obs = torch.zeros_like(obj_obs)

        obs = humanoid_obs
        # obj2_obs = self._compute_obj2_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        # Time reference observation extraction
        t0 = time.time()
        if (env_ids is None): #Z
            env_ids = torch.arange(self.num_envs)
        ts = self.progress_buf[env_ids].clone()
        if self.refined_motion_as_obs:
            self._curr_ref_obs[env_ids] = self.refined_hoi_data_batch[env_ids,ts].clone() #ZC0
        else:
            self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids,ts].clone() #ZC0

        ref_tar_pos = self._curr_ref_obs[:, 109:109+3].clone()
        ref_tar_rot = self._curr_ref_obs[:, 112:112+4].clone()
        self._ref_target_keypoints_per_epoch, self._ref_target_keypoints_vectors_per_epoch = self.extract_keypoints_per_epoch(ref_tar_pos, ref_tar_rot)
        self.timing_stats['ref_obs_extraction'].append(time.time() - t0)

        # Time target object and keypoint extraction
        t0 = time.time()
        mts = self.motion_times_total[env_ids]
        num_key = len(self._key_body_ids)
        if self.refined_motion_as_obs:
            next_target_obj_pos_refined = self.refined_hoi_data_batch[env_ids,ts][:,109:109+3].clone()
            next_target_obj_quat_refined = self.refined_hoi_data_batch[env_ids,ts][:,112:112+4].clone()
            next_target_key_pos_refined = self.refined_hoi_data_batch[env_ids,ts][:,119:119+num_key*3].clone()
            next_target_wrist_pos_vel_refined = self.refined_hoi_data_batch[env_ids,ts][:,58:58+3].clone() # no use
            next_target_contact_refined = self.refined_hoi_data_batch[env_ids,ts][:, -2:-1].clone()
        

        next_target_obj_pos = self.hoi_data_batch[env_ids,ts][:,109:109+3].clone()
        next_target_obj_quat = self.hoi_data_batch[env_ids,ts][:,112:112+4].clone()
        next_target_key_pos = self.hoi_data_batch[env_ids,ts][:,119:119+num_key*3].clone()
        next_target_wrist_pos_vel = self.hoi_data_batch[env_ids,ts][:,58:58+3].clone() # no use
        next_target_contact = self.hoi_data_batch[env_ids,ts][:, -2:-1].clone()
        self.timing_stats['target_extraction'].append(time.time() - t0)

        # Time local coordinate transformation
        t0 = time.time()

        wrist_pos = self._rigid_body_pos[env_ids, self._key_body_ids[-1], :].clone()
        wrist_rot = self._rigid_body_rot[env_ids, self._key_body_ids[-1], :].clone()
        current_obj_pos = self._target_states[env_ids, :3].clone()
        current_obj_quat = self._target_states[env_ids, 3:7].clone()
        current_key_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :].clone()
        next_target_obj_pos, next_target_obj_pos_residual, next_target_obj_quat, next_target_obj_quat_residual, \
        next_target_key_pos, next_target_key_pos_residual, next_target_wrist_pos_vel = \
            compute_local_next_target(wrist_pos, wrist_rot, num_key, current_key_pos, current_obj_pos, current_obj_quat,
                                        next_target_obj_pos, next_target_obj_quat, next_target_key_pos, next_target_wrist_pos_vel)
        if self.refined_motion_as_obs:
            next_target_obj_pos_refined, next_target_obj_pos_residual_refined, next_target_obj_quat_refined, next_target_obj_quat_residual_refined, \
            next_target_key_pos_refined, next_target_key_pos_residual_refined, next_target_wrist_pos_vel_refined = \
            compute_local_next_target(wrist_pos, wrist_rot, num_key, current_key_pos, current_obj_pos, current_obj_quat,
                                        next_target_obj_pos_refined, next_target_obj_quat_refined, next_target_key_pos_refined, next_target_wrist_pos_vel_refined)
    
        self.timing_stats['local_transform'].append(time.time() - t0)
        # if self._enable_obj_zero:
        #     next_target_obj_pos = torch.zeros_like(next_target_obj_pos)
        #     next_target_obj_quat = torch.zeros_like(next_target_obj_quat)
        tracking_obs = torch.cat((next_target_obj_pos, next_target_obj_quat, 
                                  next_target_key_pos, next_target_key_pos_residual, next_target_wrist_pos_vel,
                                  next_target_obj_pos_residual, next_target_obj_quat_residual), dim=-1)
        if self.refined_motion_as_obs:
            tracking_obs_refined = torch.cat((next_target_obj_pos_refined, next_target_obj_quat_refined, 
                                next_target_key_pos_refined, next_target_key_pos_residual_refined, next_target_wrist_pos_vel_refined,
                                next_target_obj_pos_residual_refined, next_target_obj_quat_residual_refined), dim=-1)
          
        ############## Test ######################
        if self._enable_dof_obs:
            if self.refined_motion_file and False:
                next_target_dof_pos = self.refined_hoi_data_batch[env_ids,ts][:,7:7+self.num_joints].clone()
            else:
                next_target_dof_pos = self.hoi_data_batch[env_ids,ts][:,7:7+self.num_joints].clone()
            next_target_dof_pos_residual = next_target_dof_pos - self._dof_pos[env_ids].clone()
            # 将差值调整到[-π, π]范围内
            next_target_dof_pos_residual = (next_target_dof_pos_residual + torch.pi) % (2 * torch.pi) - torch.pi
            tracking_obs = torch.cat((tracking_obs, next_target_contact, next_target_dof_pos, next_target_dof_pos_residual), dim=-1)
        ##########################################
        
        # Time future target observation computation
        t0 = time.time()
        if self._enable_future_target_obs:
            key_frame_ids = torch.tensor([10, 20, 30, 40, 50], device=self.device).repeat(len(env_ids), 1)
            # key_frame_ids = torch.ones_like(key_frame_ids)
            num_key_frames = key_frame_ids.shape[1]
            key_frame_times = key_frame_ids + mts.unsqueeze(-1) + ts.unsqueeze(-1) # (num_envs, 6)

            # Ensure key_frame_tims smaller than motion_lengths
            ml = self._motion_data.motion_lengths[self.motion_ids_total[env_ids]].unsqueeze(-1).clone()
            key_frame_times = torch.where(key_frame_times >= ml-1, ml-1, key_frame_times)
            if self.refined_motion_file and False:
                ref_motion = [self._motion_data_refined.hoi_data_dict[mid.item()]['hoi_data'][key_frame_times[idx]].clone()
                                    for idx, mid in enumerate(self.motion_ids_total[env_ids])]
            else:
                ref_motion = [self._motion_data.hoi_data_dict[mid.item()]['hoi_data'][key_frame_times[idx]].clone()
                                    for idx, mid in enumerate(self.motion_ids_total[env_ids])]
            key_ref_motion = torch.stack(ref_motion, dim=0) # (num_envs, 5, dim)
            seq_target_obj_pos = key_ref_motion[:,:,109:109+3].clone()
            seq_target_key_pos = key_ref_motion[:,:,119:119+num_key*3].clone()

            seq_target_obj_pos, seq_target_key_pos, seq_target_key_pos_residual = \
            compute_local_future_target(wrist_pos, wrist_rot, num_key, num_key_frames,
                                        seq_target_obj_pos, seq_target_key_pos, current_key_pos)

            seq_target_obj_pos = seq_target_obj_pos.reshape(-1,num_key_frames*3) # (num_envs, 5*3)
            seq_target_key_pos = seq_target_key_pos.reshape(-1,num_key_frames*num_key*3) # (num_envs, 5*45/48)
            seq_target_key_pos_residual = seq_target_key_pos_residual.reshape(-1,num_key_frames*num_key*3) # (num_envs, 5*45/48)
            
            if self.refined_motion_as_obs:
                seq_target_obj_pos_refined = seq_target_obj_pos_refined.reshape(-1,num_key_frames*3) # (num_envs, 5*3)
                seq_target_key_pos_refined = seq_target_key_pos_refined.reshape(-1,num_key_frames*num_key*3) # (num_envs, 5*45/48)
                seq_target_key_pos_residual_refined = seq_target_key_pos_residual_refined.reshape(-1,num_key_frames*num_key*3) # (num_envs, 5*45/48)
            
            key_frame_times = key_frame_times - ts.unsqueeze(-1) - mts.unsqueeze(-1)
            key_frame_times = key_frame_times.float() / 50 # normalize to 0-1
            # key_frame_ids = torch.zeros_like(key_frame_ids)
            # if self._enable_obj_zero:
            #     seq_target_obj_pos = torch.zeros_like(seq_target_obj_pos)
            tracking_obs = torch.cat((tracking_obs, seq_target_obj_pos, \
                                      seq_target_key_pos, seq_target_key_pos_residual,
                                      key_frame_times), dim=-1)
            if self.refined_motion_as_obs:
                tracking_obs_refined = torch.cat((tracking_obs_refined, seq_target_obj_pos_refined, \
                                      seq_target_key_pos_refined, seq_target_key_pos_residual_refined,
                                      key_frame_times), dim=-1)
       
        self.timing_stats['future_target_obs'].append(time.time() - t0)
        if self.refined_motion_as_obs:
            obs_refined = torch.cat((obs,tracking_obs_refined),dim=-1)
        obs = torch.cat((obs,tracking_obs),dim=-1)

        self.obs_buf[env_ids] = obs
        self.timing_stats['total_compute_obs'].append(time.time() - obs_start)
        if self.refined_motion_as_obs:
            self.obs_buf_refined[env_ids] = obs_refined 
        else:
            self.obs_buf_refined[env_ids] = obs 
        return


    def _reset_deterministic_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = torch.full(motion_ids.shape, self._state_init, device=self.device, dtype=torch.int)

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids], \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)

        if self.refined_motion_as_obs:
            self.refined_hoi_data_batch[env_ids], \
            _, _,  _, _, \
            _, _, \
            _, _,  _, _, \
                = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)        

            # self.refined_hoi_data_batch[env_ids], \
            # self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
            # self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
            # self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids], \
            # self.init_obj2_pos[env_ids], self.init_obj2_pos_vel[env_ids], self.init_obj2_rot[env_ids], self.init_obj2_rot_vel[env_ids] \
            #     = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)        

        skill_label = self._motion_data.motion_class[motion_ids.cpu().numpy()]
        self.skill_labels[env_ids] = torch.from_numpy(skill_label).to(self.device)

        return motion_ids, motion_times

    def _reset_random_ref_state_init(self, env_ids): #Z11
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_data.sample_motions(num_envs)
        motion_times = self._motion_data.sample_time(motion_ids)

        self.hoi_data_batch[env_ids], \
        self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
        self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
        self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids], \
            = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)
        
        if self.refined_motion_as_obs:
            self.refined_hoi_data_batch[env_ids], \
            _, _,  _, _, \
            _, _, \
            _, _,  _, _, \
            _, _,  _, _ \
                = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)        

            # self.refined_hoi_data_batch[env_ids], \
            # self.init_root_pos[env_ids], self.init_root_rot[env_ids],  self.init_root_pos_vel[env_ids], self.init_root_rot_vel[env_ids], \
            # self.init_dof_pos[env_ids], self.init_dof_pos_vel[env_ids], \
            # self.init_obj_pos[env_ids], self.init_obj_pos_vel[env_ids], self.init_obj_rot[env_ids], self.init_obj_rot_vel[env_ids], \
            # self.init_obj2_pos[env_ids], self.init_obj2_pos_vel[env_ids], self.init_obj2_rot[env_ids], self.init_obj2_rot_vel[env_ids] \
            #     = self._motion_data.get_initial_state(env_ids, motion_ids, motion_times)        

            
        skill_label = self._motion_data.motion_class[motion_ids.cpu().numpy()]
        self.skill_labels[env_ids] = torch.from_numpy(skill_label).to(self.device)

        return motion_ids, motion_times
    
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

def check_nan(tensor: Tensor, name: str) -> Tensor:
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    return tensor