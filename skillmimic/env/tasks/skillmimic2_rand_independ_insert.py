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
from utils.motion_data_handler import MotionDataHandler

from env.tasks.skillmimic_insert import SkillMimicBallPlayInsert


class SkillMimic2BallPlayRandIndInsert(SkillMimicBallPlayInsert):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                                sim_params=sim_params,
                                physics_engine=physics_engine,
                                device_type=device_type,
                                device_id=device_id,
                                headless=headless)
        
        self.max_sim = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        if 'stateSearchGraph' in cfg['env']: #Z unified
            with open(f"{cfg['env']['stateSearchGraph']}", "rb") as f:
                self.state_search_graph = pickle.load(f)
        self.state_search_to_align_reward = cfg['env']['state_search_to_align_reward']
        self.eval_randskill = cfg['env']['eval_randskill']
        self.play_dataset_switch = [False for _ in range(self.num_envs)]

        self.buffer_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.buffer_steps_init = torch.zeros(self.num_envs, device=self.device, dtype=torch.long) #Z 20250118

    def p_noise(self, shape, p, scale):
        mask = torch.rand(shape[0]).to('cuda') < p
        noise = (torch.rand(shape).to('cuda') * 2 - 1) * scale
        return mask.unsqueeze(1) * noise

    def p_noise_rotate(self, init_obj_rot, max_radians):
        # 生成在 [-max_radians, max_radians] 之间的随机旋转角度 (X, Y, Z)
        rand_angles = self.p_noise(init_obj_rot.shape, p=self.cfg['env']['state_noise_prob'], scale=max_radians)
        # rand_angles = rand_angles.to(init_obj_rot.device)
        roll, pitch, yaw = rand_angles[:,0], rand_angles[:,1], rand_angles[:,2]
        # 获取当前四元数的欧拉角
        current_euler = torch_utils.quat_to_euler(init_obj_rot)  # (N, 3)
        roll_cur, pitch_cur, yaw_cur = current_euler[:,0], current_euler[:,1], current_euler[:,2]
        # 施加随机旋转
        roll_new, pitch_new, yaw_new = roll_cur + roll, pitch_cur + pitch, yaw_cur + yaw
        # 转回四元数
        new_quat = torch_utils.quat_from_euler_xyz(roll_new, pitch_new, yaw_new)
        return new_quat
    
    def _reset_state_init(self, env_ids):
        super()._reset_state_init(env_ids)
        
        self.state_random_flags = [False for _ in env_ids]
        if self.cfg['env']['state_noise_prob'] > 0:
            self.motion_ids, self.motion_times = self._init_with_random_noise(env_ids, self.motion_ids, self.motion_times)
        if self.cfg['env']['state_switch_prob'] > 0:
            self.motion_ids, self.motion_times = self._init_from_random_skill(env_ids, self.motion_ids, self.motion_times)

    def _init_with_random_noise(self, env_ids, motion_ids, motion_times): 
        self.init_dof_pos[env_ids, 6:] += self.p_noise(self.init_dof_pos[env_ids, 6:].shape, 
                                                       p=self.cfg['env']['state_noise_prob'], 
                                                       scale=torch.pi/8)
        self.init_dof_pos_vel[env_ids, 6:] += self.p_noise(self.init_dof_pos_vel[env_ids, 6:].shape, 
                                                          p=self.cfg['env']['state_noise_prob'], 
                                                          scale=0.1)
        self.init_obj_pos[env_ids] += self.p_noise(self.init_obj_pos[env_ids].shape, 
                                                  p=self.cfg['env']['state_noise_prob'], 
                                                  scale=0.02)
        self.init_obj_pos_vel[env_ids] += self.p_noise(self.init_obj_pos_vel[env_ids].shape, 
                                                      p=self.cfg['env']['state_noise_prob'], scale=0.02)
        self.init_obj_rot[env_ids] = self.p_noise_rotate(self.init_obj_rot[env_ids], torch.pi/8)
        self.init_obj_rot_vel[env_ids] += self.p_noise(self.init_obj_rot_vel[env_ids].shape, 
                                                      p=self.cfg['env']['state_noise_prob'], 
                                                      scale=0.02)
        if self.isTest:
            print(f"Random noise added to initial state for env {env_ids}")
        return motion_ids, motion_times
        
    def _init_from_random_skill(self, env_ids, motion_ids, motion_times): 
        # Random init from other skills
        state_switch_flags = [np.random.rand() < self.cfg['env']['state_switch_prob'] for _ in env_ids]
        for ind, env_id in enumerate(env_ids):
            if state_switch_flags[ind] and not self.state_random_flags[ind]: 
                source_motion_class = self._motion_data.motion_class[motion_ids[ind]]
                source_motion_id = motion_ids[ind:ind+1]
                source_motion_time = motion_times[ind:ind+1]

                if self.state_search_to_align_reward:
                    switch_motion_class, switch_motion_id, switch_motion_time, max_sim = \
                        random.choice(self.state_search_graph[source_motion_class][source_motion_id.item()][source_motion_time.item()])
                    if switch_motion_id is None and switch_motion_time is None:
                        continue
                    else:
                        self.max_sim[env_id] = max_sim
                    switch_motion_id = torch.tensor([switch_motion_id], device=self.device)
                    switch_motion_time = torch.tensor([switch_motion_time], device=self.device)

                    # switch_motion_time, new_source_motion_time = self._motion_data.resample_time(source_motion_id, switch_motion_id, weights=self.similarity_weights)
                    # motion_times[ind:ind+1] = new_source_motion_time
                    # resample the hoi_data_batch
                    # self.hoi_data_batch[env_id], init_root_pos_source, init_root_rot_source,  _, _, _, _, init_obj_pos_source , _, _, _ = \
                    #     self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, new_source_motion_time)
                else:
                    switch_motion_id = self._motion_data.sample_switch_motions(source_motion_id)
                    motion_len = self._motion_data.motion_lengths[switch_motion_id].item()
                    switch_motion_time = torch.randint(2, motion_len - 2, (1,), device=self.device)

                    # resample the hoi_data_batch
                    # self.hoi_data_batch[env_id], init_root_pos_source, init_root_rot_source,  _, _, _, _, init_obj_pos_source, _, _, _ = \
                    #     self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, source_motion_time)
                 
                # 从switch中获取待对齐的初始状态
                _, init_root_pos_switch, init_root_rot_switch, init_root_pos_vel_switch, init_root_rot_vel_switch, \
                init_dof_pos_switch, init_dof_pos_vel_switch, \
                init_obj_pos_switch, init_obj_pos_vel_switch, init_obj_rot_switch, init_obj_rot_vel_switch \
                    = self._motion_data.get_initial_state(env_ids[ind:ind+1], switch_motion_id, switch_motion_time)
                # _, _, _, _  \
                
                # 从source中获取初始状态的对齐目标
                self.hoi_data_batch[env_id], init_root_pos_source, init_root_rot_source,  _, _, _, _, init_obj_pos_source , _, _, _ = \
                        self._motion_data.get_initial_state(env_ids[ind:ind+1], source_motion_id, source_motion_time)


                # 计算 yaw 差异: 我们要从switch参考系转到source参考系
                yaw_source = torch_utils.quat_to_euler(init_root_rot_source)[0][2]
                yaw_switch = torch_utils.quat_to_euler(init_root_rot_switch)[0][2]
                yaw_diff = yaw_source - yaw_switch
                yaw_diff = (yaw_diff + torch.pi) % (2*torch.pi) - torch.pi

                # 对齐 root_pos 
                self.init_root_pos[env_id] = self.rotate_xy(init_root_pos_switch[0], init_root_pos_switch[0], yaw_diff, init_root_pos_source[0])
                #从 get_initial_state 返回的 init_root_pos_source、init_root_pos_switch 通常是 (1,3)
                # equal to: init_root_pos_switch[:2] = init_root_pos_source[:2]

                # 对齐 root_rot
                yaw_quat = quat_from_euler_xyz(torch.zeros_like(yaw_diff), torch.zeros_like(yaw_diff), yaw_diff)
                self.init_root_rot[env_id] = torch_utils.quat_multiply(yaw_quat, init_root_rot_switch)

                # 对齐 obj_pos #Z ball
                self.init_obj_pos[env_id] = self.rotate_xy(init_obj_pos_switch[0], init_root_pos_switch[0], yaw_diff, init_root_pos_source[0])

                # 对齐 obj_rot # 因为是ball，所以不需要变换
                # self.init_obj_rot[env_id] # 因为球的旋转不计算奖励，所以不需要更新

                # 速度和dof不需要坐标对齐
                self.init_root_pos_vel[env_id] = init_root_pos_vel_switch
                self.init_root_rot_vel[env_id] = init_root_rot_vel_switch
                self.init_dof_pos[env_id] = init_dof_pos_switch
                self.init_dof_pos_vel[env_id] = init_dof_pos_vel_switch
                self.init_obj_pos_vel[env_id] = init_obj_pos_vel_switch
                self.init_obj_rot_vel[env_id] = init_obj_rot_vel_switch

                if self.isTest:
                    print(f"Switched from skill {switch_motion_class} to {source_motion_class} for env {env_id}")
        
        return motion_ids, motion_times

    def after_reset_actors(self, env_ids):
        super().after_reset_actors(env_ids)
        if self.cfg["env"]["enable_buffernode"] and self.progress_buf_total > 0:
            self.buffer_steps[env_ids] = 0 # 如果上个episode, env_id有buffernode，但是早停或者非常大buffer，导致reset时，没有通过 -1 到0，就会影响本episode
            self._compute_buffer_steps(env_ids)
            self.buffer_steps_init[env_ids] = self.buffer_steps[env_ids] #Z 20250118
            self.max_sim[env_ids] = 0

    def _compute_buffer_steps(self, env_ids):
        for env_id in env_ids:
            if self.max_sim[env_id] > 0.5:
                self.buffer_steps[env_id] = 0
            elif self.max_sim[env_id] != 0:
                self.buffer_steps[env_id] = min(-int(torch.floor(torch.log10(self.max_sim[env_id]))), 5)

    def _hook_post_step(self):
        # extra calc of self._curr_obs, for imitation reward
        self._compute_hoi_observations()
        env_ids = torch.arange(self.num_envs)
        # ts = self.progress_buf[env_ids].clone()
        ts = (self.progress_buf - self.buffer_steps_init).clamp(min=0) #Z 20250118
        self._curr_ref_obs[env_ids] = self.hoi_data_batch[env_ids, ts].clone()


    def _compute_reward(self): #Z 20250118
        """
        1) 调用父类(或上层)的方法, 先计算好每个env的基础奖励 self.rew_buf。
        2) 将当前环境中“缺失帧”或“buffer_steps>0”的环境统一视为 'invalid'，统一替换为同一 motion clip 下的有效平均奖励(若有), 否则用均值。
        """
        # 1) 先用父类算出每个 env 的初步奖励 (self.rew_buf)
        super()._compute_reward()

        # self.rew_buf.shape == [num_envs]
        # self.buffer_steps.shape == [num_envs]
        # self.motion_ids_total.shape == [num_envs]

        # 2) 构造最终 mask: 
        #   -- 缺失帧 (missing_mask)
        #   -- 缓冲期 (buffer_mask)
        env_ids = torch.arange(self.num_envs, device=self.device)
        ts = (self.progress_buf - self.buffer_steps_init).clamp(min=0).long()  # 取当前参考帧索引 t（有 buffer_steps_init 逻辑，可一起考虑）#但是不会用缺失帧初始化
        ref_frame = self.hoi_data_batch[env_ids, ts, :]  # 取参考帧: shape=[num_envs, D]
        # 判断是否全零(=缺失帧)
        missing_mask = torch.all(torch.abs(ref_frame[:,0:162]) < 1e-8, dim=1)  # [num_envs], True表示该env的参考帧缺失
        # 判断是否还在缓冲区
        buffer_mask = (self.buffer_steps > 0)
        # 最终需要替换奖励的 env
        invalid_mask = missing_mask | buffer_mask #   True 表示需要用均值替换

        # assert(buffer_mask.sum() == 0)

        # 若无任何需要替换的环境，直接 buffer_steps -= 1 并 return
        if not invalid_mask.any():
            self.buffer_steps = torch.clamp(self.buffer_steps - 1, min=0)
            return
        
        # 3) 找到剩余“有效环境” valid_mask = ~invalid_mask
        valid_mask = ~invalid_mask
        if not valid_mask.any():
            self.rew_buf[:] = 0.0  # or any fallback
            self.buffer_steps = torch.clamp(self.buffer_steps - 1, min=0)
            return

        # motion_ids_total 是我们跟踪所有 env 的当前 motion clip ID
        motion_ids = self.motion_ids_total  # shape=[num_envs], long
        num_motions = self._motion_data.num_motions

        # 4) 针对 valid_mask 做分组，统计奖励和计数
            #   sums 和 counts 的形状取决于 motion_ids 中的最大值 M:
            #   sums.shape == [M+1], counts.shape == [M+1]
            #   其中 M = motion_ids.max()，因此 sums[i]、counts[i] 对应 motion_id == i
                #   sums[i] = 所有 env 中 motion_id==i 的 reward 求和
                #   counts[i] = motion_id==i 的 env 数量
        sums = torch.bincount(
            motion_ids[valid_mask],
            weights=self.rew_buf[valid_mask].float(),
            minlength=num_motions
        )
        counts = torch.bincount(
            motion_ids[valid_mask],
            minlength=num_motions
        ).float()

        # 4.1) 先算每个 motion clip 的平均
        means = sums / (counts + 1e-8)  # means.shape == [M+1] # means[i] = (所有 motion_id==i 的平均奖励)
        # 4.2) 对 valid env counts=0 的 clip 用 global_mean 填充
        global_mean = self.rew_buf[valid_mask].mean()
        means = torch.where(counts > 0, means, global_mean)

        # 5) 将 invalid_mask 环境的奖励替换为对应 clip 的平均值
        #   - invalid_envs: shape=[X], 其中 X = invalid_mask.sum().item()
        #   - invalid_clip_ids: 这些 X 个 env 的 clip id
        invalid_envs = invalid_mask.nonzero(as_tuple=False).squeeze(-1)
        invalid_clip_ids = motion_ids[invalid_envs]
        # means[final_clip_ids] 就能取到这些 clip 的均值
        self.rew_buf[invalid_envs] = means[invalid_clip_ids]

        # 6) 递减 buffer_steps
        self.buffer_steps = torch.where(
            self.buffer_steps > 0,
            self.buffer_steps - 1,
            self.buffer_steps
        )

        return
    
    def rotate_xy(self, pos, center, angle, target_root_pos):
        # pos, center都是(3,)的Tensor, angle为标量，target_root_pos为(3,)
        x_rel = pos[0] - center[0]
        y_rel = pos[1] - center[1]
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        x_new = x_rel * cos_a - y_rel * sin_a + target_root_pos[0]
        y_new = x_rel * sin_a + y_rel * cos_a + target_root_pos[1]
        # Z保持原样
        z_new = pos[2]
        return torch.tensor([x_new, y_new, z_new], device=pos.device)
    
    def play_dataset_step(self, time): #Z12

        t = time

        for env_id, env_ptr in enumerate(self.envs):
            motid = self._motion_data.envid2motid[env_id].item() # if not self.play_dataset_switch[env_id] else 1
            t = t % self._motion_data.motion_lengths[motid]

            ############# Modified by Runyi #############
            # run -> pickup
            # if motid==1 and not self.play_dataset_switch[env_id]:
            #     continue
            state_switch_flag = np.random.rand()
            if state_switch_flag > 0.95 and not self.play_dataset_switch[env_id] and self.eval_randskill:
                switch_motion_class = self._motion_data.motion_class[motid]
                switch_motion_id = motid
                source_motion_class, source_motion_id, source_motion_time = random.choice(self.state_search_graph[switch_motion_class][switch_motion_id][t.item()])

                if source_motion_id is not None:
                    print(f"Skill {switch_motion_class} id {switch_motion_id} time {t} -> Skill {source_motion_class} id {source_motion_id} time {source_motion_time}")
                    self._motion_data.hoi_data_dict[source_motion_id] = compute_local_hoi_data(self._motion_data.hoi_data_dict[source_motion_id], 
                                                                                self._motion_data.hoi_data_dict[switch_motion_id]['root_pos'][t,:].clone(), 
                                                                                self._motion_data.hoi_data_dict[switch_motion_id]['root_rot'][t,:].clone(), 
                                                                                len(self._key_body_ids),
                                                                                source_motion_time)
                    self.play_dataset_switch[env_id] = True
                    self._motion_data.envid2motid[env_id] = source_motion_id
                    t = torch.tensor(source_motion_time, device=self.device, dtype=torch.long)
                    t = t % self._motion_data.motion_lengths[source_motion_id]
                    motid = source_motion_id
            #############################################


            ### update object ###
            self._target_states[env_id, :3] = self._motion_data.hoi_data_dict[motid]['obj_pos'][t,:]
            self._target_states[env_id, 3:7] = self._motion_data.hoi_data_dict[motid]['obj_rot'][t,:]
            self._target_states[env_id, 7:10] = torch.zeros_like(self._target_states[env_id, 7:10])
            self._target_states[env_id, 10:13] = torch.zeros_like(self._target_states[env_id, 10:13])

            ### update subject ###
            _humanoid_root_pos = self._motion_data.hoi_data_dict[motid]['root_pos'][t,:].clone()
            _humanoid_root_rot = self._motion_data.hoi_data_dict[motid]['root_rot'][t,:].clone()
            self._humanoid_root_states[env_id, 0:3] = _humanoid_root_pos
            self._humanoid_root_states[env_id, 3:7] = _humanoid_root_rot
            self._humanoid_root_states[env_id, 7:10] = torch.zeros_like(self._humanoid_root_states[env_id, 7:10])
            self._humanoid_root_states[env_id, 10:13] = torch.zeros_like(self._humanoid_root_states[env_id, 10:13])
            
            self._dof_pos[env_id] = self._motion_data.hoi_data_dict[motid]['dof_pos'][t,:].clone()
            self._dof_vel[env_id] = torch.zeros_like(self._dof_vel[env_id])

            # env_id_int32 = self._humanoid_actor_ids[env_id].unsqueeze(0)

            contact = self._motion_data.hoi_data_dict[motid]['contact'][t,:]
            obj_contact = torch.any(contact > 0.1, dim=-1)
            root_rot_vel = self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t,:]
            angle = torch.norm(root_rot_vel)
            abnormal = torch.any(torch.abs(angle) > 5.) #Z

            if abnormal == True:
                # print("frame:", t, "abnormal:", abnormal, "angle", angle)
                # print(" ", self._motion_data.hoi_data_dict[motid]['root_rot_vel'][t])
                # print(" ", angle)
                self.show_abnorm[env_id] = 10

            handle = self._target_handles[env_id]
            if obj_contact == True:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1., 0., 0.))
            else:
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(0., 1., 0.))
            
            if abnormal == True or self.show_abnorm[env_id] > 0: #Z
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 0., 1.)) 
                self.show_abnorm[env_id] -= 1
            else:
                for j in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, j, gymapi.MESH_VISUAL, gymapi.Vec3(0., 1., 0.)) 
            
            ############# Modified by Runyi #############
            if self.play_dataset_switch[env_id]:
                for i in range(self.num_bodies): #Z humanoid_handle == 0
                    self.gym.set_rigid_body_color(env_ptr, 0, i, gymapi.MESH_VISUAL, gymapi.Vec3(1., 0., 0.)) 
            #############################################

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))
        self._refresh_sim_tensors()     

        self.render(t=time)
        self.gym.simulate(self.sim)

        self._compute_observations()

        return self.obs_buf
    
    


#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
def compute_local_hoi_data(hoi_data_dict, switch_root_pos, switch_root_rot, len_keypos, new_source_motion_time):
    # switch_root_rot (3)
    # switch_root_rot (1, 4)
    local_hoi_data_dict = {}
    init_root_pos = hoi_data_dict['root_pos'][new_source_motion_time]
    init_root_rot_quat = hoi_data_dict['root_rot'][new_source_motion_time]

    root_pos = hoi_data_dict['root_pos']
    root_rot = hoi_data_dict['root_rot']
    root_rot_3d = hoi_data_dict['root_rot_3d']
    root_rot_vel = hoi_data_dict['root_rot_vel']
    dof_pos = hoi_data_dict['dof_pos']
    dof_pos_vel = hoi_data_dict['dof_pos_vel']
    obj_pos = hoi_data_dict['obj_pos']
    obj_rot = hoi_data_dict['obj_rot']
    obj_pos_vel = hoi_data_dict['obj_pos_vel']
    contact = hoi_data_dict['contact']
    nframes = root_pos.shape[0]
    
    switch_root_rot_euler_z = torch_utils.quat_to_euler(switch_root_rot)[2] # (1)
    init_root_rot_euler_z = torch_utils.quat_to_euler(init_root_rot_quat)[2] # (3)
    source_to_switch_euler_z = switch_root_rot_euler_z - init_root_rot_euler_z
    source_to_switch_euler_z = (source_to_switch_euler_z + torch.pi) % (2 * torch.pi) - torch.pi  # 归一化到 [-pi, pi]
    source_to_switch_euler_z = source_to_switch_euler_z.squeeze()
    zeros = torch.zeros_like(source_to_switch_euler_z)
    source_to_switch = quat_from_euler_xyz(zeros, zeros, source_to_switch_euler_z)
    source_to_switch = source_to_switch.repeat(nframes, 1) # (nframes, 4)

    # referece to the new root
    # local_root_pos
    relative_root_pos = root_pos - init_root_pos
    local_relative_root_pos = torch_utils.quat_rotate(source_to_switch, relative_root_pos)
    local_root_pos = local_relative_root_pos + switch_root_pos
    local_root_pos[:, 2] = root_pos[:, 2]
    # local_root_rot
    root_rot_quat = root_rot
    local_root_rot = torch_utils.quat_multiply(source_to_switch, root_rot_quat)
    # local_root_rot_3d
    local_root_rot_3d = torch_utils.exp_map_to_quat(root_rot_3d)
    local_root_rot_3d = torch_utils.quat_multiply(source_to_switch, local_root_rot_3d)
    local_root_rot_3d = torch_utils.quat_to_exp_map(local_root_rot_3d)
    # local_obj_pos
    relative_obj_pos = obj_pos - init_root_pos
    local_relative_obj_pos = torch_utils.quat_rotate(source_to_switch, relative_obj_pos)
    local_obj_pos = local_relative_obj_pos + switch_root_pos
    local_obj_pos[:, 2] = obj_pos[:, 2]
    # local_obj_pos_vel
    local_obj_pos_vel = torch_utils.quat_rotate(source_to_switch, obj_pos_vel)
    

    local_hoi_data_dict['root_pos'] = local_root_pos
    local_hoi_data_dict['root_rot'] = local_root_rot
    local_hoi_data_dict['root_rot_3d'] = local_root_rot_3d
    local_hoi_data_dict['root_rot_vel'] = root_rot_vel
    local_hoi_data_dict['dof_pos'] = dof_pos
    local_hoi_data_dict['dof_pos_vel'] = dof_pos_vel
    local_hoi_data_dict['obj_pos'] = local_obj_pos
    local_hoi_data_dict['obj_rot'] = obj_rot
    local_hoi_data_dict['obj_pos_vel'] = local_obj_pos_vel
    local_hoi_data_dict['contact'] = contact

    return local_hoi_data_dict