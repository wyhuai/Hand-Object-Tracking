from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
import glob, os, random, pickle
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from utils import torch_utils
from env.tasks.skillmimic2_reweight import SkillMimic2BallPlayReweight


class SkillMimicHandRand(SkillMimic2BallPlayReweight):
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
        if self.skill_name == 'regrasp_kp_hard':
            self.init_dof_pos[env_ids, :3] += self.p_noise(self.init_dof_pos[env_ids, :3].shape, 
                                                       p=self.cfg['env']['state_noise_prob'], scale=0.02)
            self.init_dof_pos[env_ids, 3:6] += self.p_noise(self.init_dof_pos[env_ids, 3:6].shape, 
                                                       p=self.cfg['env']['state_noise_prob'], 
                                                       scale=torch.pi/12)
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
