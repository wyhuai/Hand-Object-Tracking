import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
import re
import copy
from pprint import pformat

from utils import torch_utils
from isaacgym.torch_utils import *


class MotionDataHandler:
    def __init__(self, motion_file, device, key_body_ids, cfg, num_envs, max_episode_length, reward_weights_default, play_dataset=False, reweight=False, reweight_alpha=0):
        self.device = device
        self._key_body_ids = key_body_ids
        self.cfg = cfg
        self.play_dataset = play_dataset
        self.hoi_data_dict = {}
        self.hoi_data_label_batch = None
        self.motion_lengths = None
        self.load_motion(motion_file)

        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.envid2motid = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.envid2sframe = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reweight = reweight
        self.reweight_alpha = reweight_alpha

        self.envid2episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.reward_weights_default = reward_weights_default
        self.reward_weights = {}
        self.reward_weights["p"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["p"])
        self.reward_weights["r"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["r"])
        self.reward_weights["pv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["pv"])
        self.reward_weights["rv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["rv"])
        self.reward_weights["wp"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["wp"])
        self.reward_weights["wr"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["wr"])
        self.reward_weights["wpv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["wpv"])
        self.reward_weights["wrv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["wrv"])
        self.reward_weights["op"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["op"])
        self.reward_weights["or"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["or"])
        self.reward_weights["opv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["opv"])
        self.reward_weights["orv"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["orv"])
        self.reward_weights["ig"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["ig"])
        self.reward_weights["cg1"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg1"])
        self.reward_weights["cg2"] = (torch.ones((self.num_envs), device=self.device, dtype=torch.float)*self.reward_weights_default["cg2"])
        self._init_vectorized_buffers() #ZQH
    
    def _init_vectorized_buffers(self):
        # # 1) clip-level采样权重，先全部置为1，稍后由 _compute_motion_weights(...) 或 reweight函数修正
        # self._motion_weights_tensor = torch.ones(
        #     self.num_motions, device=self.device, dtype=torch.float32
        # ) 
        # #已在 _compute_motion_weights 中求得
        
        # 2) time-level采样权重
        max_len_minus3 = int(self.motion_lengths.max().item() - 3)
        if max_len_minus3 <= 0:
            max_len_minus3 = 1  # 以防万一
        
        self.time_sample_rate_tensor = torch.zeros(
            (self.num_motions, max_len_minus3),
            device=self.device, dtype=torch.float32
        )
        
        # 将每条 motion clip 的 [2, length-1) 段设为均匀分布，但排除缺失帧
        for m_id in range(self.num_motions):
            cur_len = self.motion_lengths[m_id].item()
            cur_len_minus3 = int(cur_len - 3)
            if cur_len_minus3 <= 0:
                continue
            self.time_sample_rate_tensor[m_id, :cur_len_minus3] = 1.0 / cur_len_minus3
        return
    
    def load_motion(self, motion_file):
        self.skill_name = motion_file.split('/')[-1]
        all_seqs = [motion_file] if os.path.isfile(motion_file) \
            else glob.glob(os.path.join(motion_file, '**', '*.pt'), recursive=True)
        self.num_motions = len(all_seqs)
        self.motion_lengths = torch.zeros(len(all_seqs), device=self.device, dtype=torch.long)
        self.motion_class = np.zeros(len(all_seqs), dtype=int)        
        self.root_target = torch.zeros((len(all_seqs), 3), device=self.device, dtype=torch.float)
        all_seqs.sort(key=self._sort_key)
        for i, seq_path in enumerate(all_seqs):
            loaded_dict = self._process_sequence(seq_path)
            self.hoi_data_dict[i] = loaded_dict
            self.motion_lengths[i] = loaded_dict['hoi_data'].shape[0]
            self.motion_class[i] = int(loaded_dict['hoi_data_text'])

        self._compute_motion_weights(self.motion_class)
        self.motion_class_tensor = torch.tensor(self.motion_class, dtype=torch.long, device=self.device) #ZQH
        print(f"--------Having loaded {len(all_seqs)} motions--------")

    
    def _sort_key(self, filename):
        match = re.search(r'\d+.pt$', filename)
        return int(match.group().replace('.pt', '')) if match else -1

    def get_object(self, filename):
        match = re.search(r'\d+.pt$', filename)
        return int(match.group().replace('.pt', '')) if match else -1

    def _process_sequence(self, seq_path):
        loaded_dict = {}
        hoi_data = torch.load(seq_path)
        loaded_dict['hoi_data_path'] = seq_path
        loaded_dict['hoi_data_text'] = os.path.basename(seq_path)[0:3]
        data_frames_scale = self.cfg["env"]["dataFramesScale"]
        fps_data = self.cfg["env"]["dataFPS"] * data_frames_scale
        
        loaded_dict['hoi_data'] = hoi_data
        loaded_dict['root_pos'] = loaded_dict['hoi_data']['root_pos'].clone().to(self.device)
        loaded_dict['root_pos_vel'] = self._compute_velocity(loaded_dict['root_pos'], fps_data)
        loaded_dict['root_rot'] = loaded_dict['root_rot'] = loaded_dict['hoi_data']['root_rot'].clone().to(self.device) 
        self.smooth_quat_seq(loaded_dict['root_rot'])

        q_diff = torch_utils.quat_multiply(
            torch_utils.quat_conjugate(loaded_dict['root_rot'][:-1, :].clone()), 
            loaded_dict['root_rot'][1:, :].clone()
        )
        angle, axis = torch_utils.quat_to_angle_axis(q_diff)
        exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
        loaded_dict['root_rot_vel'] = exp_map*fps_data
        loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, 3)).to(self.device),loaded_dict['root_rot_vel']),dim=0)
        
        loaded_dict['dof_pos'] = torch.cat((loaded_dict['hoi_data']['wrist_dof'], loaded_dict['hoi_data']['fingers_dof']), dim=1).clone().float().to(self.device) #changeged by me
        loaded_dict['dof_pos_vel'] = self._compute_velocity(loaded_dict['dof_pos'], fps_data)
        
        loaded_dict['body_pos'] = loaded_dict['hoi_data']['body_pos'].clone().float().to(self.device)
        loaded_dict['key_body_pos'] = loaded_dict['body_pos'].clone()
        loaded_dict['key_body_pos_vel'] = self._compute_velocity(loaded_dict['key_body_pos'], fps_data)

        loaded_dict['obj_pos'] = loaded_dict['hoi_data']['obj_pos'].clone().float().to(self.device)
        loaded_dict['obj_pos_vel'] = loaded_dict['hoi_data']['obj_pos_vel']
        loaded_dict['obj_rot'] = loaded_dict['hoi_data']['obj_rot'].clone().float().to(self.device)
        self.smooth_quat_seq(loaded_dict['obj_rot'])

        q_diff = torch_utils.quat_multiply(
            torch_utils.quat_conjugate(loaded_dict['obj_rot'][:-1, :].clone()), 
            loaded_dict['obj_rot'][1:, :].clone()
        )
        angle, axis = torch_utils.quat_to_angle_axis(q_diff)
        exp_map = torch_utils.angle_axis_to_exp_map(angle, axis)
        loaded_dict['obj_rot_vel'] = exp_map*fps_data
        loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).float().to(self.device), loaded_dict['obj_rot_vel']),dim=0)
        
        loaded_dict['contact1'] = torch.round(loaded_dict['hoi_data']['contact1'].clone().float().to(self.device))
        
        loaded_dict['hoi_data'] = torch.cat((
            loaded_dict['root_pos'],
            loaded_dict['root_rot'],
            loaded_dict['dof_pos'], 
            loaded_dict['dof_pos_vel'],
            loaded_dict['obj_pos'],
            loaded_dict['obj_rot'],
            loaded_dict['obj_pos_vel'],
            loaded_dict['key_body_pos'],
            loaded_dict['contact1'].to(self.device),
            torch.zeros_like(loaded_dict['contact1']),
        ), dim=-1)

        return loaded_dict

    def _compute_velocity(self, positions, fps):
        velocity = (positions[1:, :].clone() - positions[:-1, :].clone()) * fps
        velocity = torch.cat((torch.zeros((1, positions.shape[-1])).to(self.device), velocity), dim=0)
        return velocity

    def smooth_quat_seq(self, quat_seq):
        n = quat_seq.size(0)
        for i in range(1, n):
            dot_product = torch.dot(quat_seq[i-1], quat_seq[i])
            if dot_product < 0:
                quat_seq[i] *=-1
        return quat_seq

    def _compute_motion_weights(self, motion_class):
        unique_classes, counts = np.unique(motion_class, return_counts=True)
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        class_weights = 1.0 / counts
        if 1 in class_to_index: # raise sampling probs of skill pick
            class_weights[class_to_index[1]]*=2
        self.indexed_classes = np.array([class_to_index[int(cls)] for cls in motion_class], dtype=int)
        w_array = class_weights[self.indexed_classes]  # shape=[num_motions]
        self._motion_weights_tensor = torch.tensor(w_array, device=self.device, dtype=torch.float32)

    def get_motions(self, n):
        motion_available = n if n < len(self.motion_class_id) else len(self.motion_class_id)
        motion_ids_list = [self.motion_class_id.pop() for _ in range(motion_available)]
        motion_ids_list.extend([0] * (n - motion_available))
        motion_ids = torch.tensor(motion_ids_list)
        print(motion_ids)

        return motion_ids
    
    def _reweight_clip_sampling_rate_vectorized(self, average_rewards_tensor: torch.Tensor):
        """
         功能：
           1) 先对同一类别内所有 clip 的 reward 做平均，得到 class_avg_rewards；
           2) 对 class_avg_rewards 做 exp(-5 * class_avg) 后 softmax，得到类别级别的 weight；
           3) 类别内部，再根据各 clip 的 reward 做 exp(-5 * reward_clip) 归一化，得到 clip 在类别内的相对权重；
           4) 最终 clip i 的 weight = baseline + alpha * [class_weight(class_i) * clip_intra_class_weight(i)]，
              其中 baseline = (1 - alpha) / num_motions。

         参数:
            average_rewards_tensor: shape=[num_motions], 每条 motion clip 的平均奖励 (GPU 上)

         输出:
            无显式返回值，但更新 self._motion_weights_tensor：shape=[num_motions]
        
         Tip:
           - 避免了手动 for i in range(self.num_motions)。
           - 使用 PyTorch 的 index_add_、unique + return_inverse=True 等操作完成向量化。
        """
        if self.num_motions < 1:
            return

        alpha = self.reweight_alpha  # reweight 系数
        device = self.device

        # -- 1) 计算每个类别的平均奖励 (class-level) ---------------------------------

        # 取得所有 clip 所属的类别（整型），并找到 unique class
        # cls_idx 的形状与 motion_class_tensor 相同，表示每个 clip 属于 unique_classes 中的哪个下标
        unique_classes, cls_idx = torch.unique(self.motion_class_tensor, return_inverse=True)
        # unique_classes.shape = [C], cls_idx.shape = [num_motions]

        # 用 index_add_ 做聚合：将 average_rewards_tensor 根据 cls_idx 加到 class_sum_rewards
        class_sum_rewards = torch.zeros_like(unique_classes, dtype=average_rewards_tensor.dtype, device=device)
        class_counts = torch.zeros_like(unique_classes, dtype=average_rewards_tensor.dtype, device=device)

        # 例如：对属于同一类别 c 的 clip，其奖励会加到 class_sum_rewards[c]
        class_sum_rewards.index_add_(0, cls_idx, average_rewards_tensor)
        class_counts.index_add_(0, cls_idx, torch.ones_like(average_rewards_tensor))

        # 避免除零
        class_avg_rewards = class_sum_rewards / (class_counts + 1e-8)  # shape=[C]

        # -- 2) 计算类别级别的 exp(-5*avg_reward)，再 softmax -------------------------
        negative_exp_class = torch.exp(-5.0 * class_avg_rewards)  # shape=[C]
        sum_neg_class = negative_exp_class.sum() + 1e-8
        class_weights = negative_exp_class / sum_neg_class        # shape=[C]
        print('Class Labels:', unique_classes)
        print('Class Average Weights:', class_weights)

        # -- 3) 对各 clip 的 reward 同样做 exp(-5*reward)，并在类别内部进行归一化 ----
        negative_exp_motion = torch.exp(-5.0 * average_rewards_tensor)  # shape=[num_motions]
        sum_neg_motion_per_class = torch.zeros_like(unique_classes, dtype=average_rewards_tensor.dtype, device=device)

        # 将每个 clip 的 negative_exp_motion 累加到所属类别
        sum_neg_motion_per_class.index_add_(0, cls_idx, negative_exp_motion)
        # sum_neg_motion_per_class[k] 表示第 k 个类别内部所有 clip 的 exp(-5*reward) 之和

        # 对单条 clip 而言，其在类别内的相对权重 = negative_exp_motion[i] / sum_neg_motion_per_class[ cls_idx[i] ]
        clip_intra_class = negative_exp_motion / (sum_neg_motion_per_class[cls_idx] + 1e-8)

        # -- 4) 最终 clip 权重 = baseline + alpha * [class_weights[cls_idx] * clip_intra_class]
        baseline = (1.0 - alpha) / float(self.num_motions)
        motion_weights = class_weights[cls_idx] * clip_intra_class  # shape=[num_motions]
        motion_weights = baseline + alpha * motion_weights

        # 存储到 self._motion_weights_tensor 以便后续采样时使用
        self._motion_weights_tensor = motion_weights
        print('##### Reweight clip sampling rate #####')
        print(self._motion_weights_tensor)


    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights_tensor, num_samples=n, replacement=True)
        return motion_ids
    
    def _reweight_time_sampling_rate_vectorized(self, motion_time_seqreward_tensor: torch.Tensor):
        #ZQH 新增：用向量化方式更新 time-level 采样权重 motion_time_seqreward_tensor: shape=[num_motions, max_len_minus3], 在GPU
        alpha = self.reweight_alpha
        # 若没有可用列, 直接返回
        if motion_time_seqreward_tensor.size(1) == 0:
            return
        #----------------1) 计算 baseline + alpha * exp(-10 * R)----------------#
        baseline = (1.0 - alpha) / float(motion_time_seqreward_tensor.size(1))
        negative_exp = torch.exp(-10.0 * motion_time_seqreward_tensor)  # same shape
        sum_per_clip = negative_exp.sum(dim=1, keepdim=True) + 1e-8
        self.time_sample_rate_tensor = baseline + alpha * (negative_exp / sum_per_clip)
        # print('motion_time_seqreward:', motion_time_seqreward_tensor)
        # print('Reweighted time sampling rate:', self.time_sample_rate_tensor)

    def sample_time(self, motion_ids, truncate_time=None):
        """#ZQH
        改动后版本：使用 GPU 上的 self.time_sample_rate_tensor 做采样
        （假设 self.time_sample_rate_tensor[m_id, :L] 存储该 motion_id
        在 [2, L+2) 范围内的 time 概率分布）
        """
        # motion_times shape与 motion_ids 一致
        motion_times = torch.zeros_like(motion_ids, dtype=torch.int32, device=self.device)

        if not self.reweight: # 若没有 reweight，可直接走均匀分布(可以在外部把 time_sample_rate_tensor 初始化成均匀即可)
            for i in range(len(motion_ids)):
                m_id = motion_ids[i].item()
                L = self.motion_lengths[m_id] - 3
                dist = torch.ones(L, device=self.device)/(L+1e-6)
                idx = torch.multinomial(dist, 1)
                motion_times[i] = idx.item() + 2
        else:
            for i in range(len(motion_ids)):
                m_id = motion_ids[i].item()
                length_minus3 = int(self.motion_lengths[m_id].item() - 3)
                if length_minus3 <= 0:
                    # motion太短，强行置0或别的逻辑
                    motion_times[i] = 0
                    continue
                
                # dist shape=[length_minus3], GPU
                dist = self.time_sample_rate_tensor[m_id, :length_minus3]

                # 在 dist 上多项式采样1个time index, 结果 shape=[1]
                t_idx = torch.multinomial(dist, 1)
                # +2 抵消因为我们只存了 [2, length-1) 这段的概率
                motion_times[i] = t_idx.item() + 2

        # 截断逻辑
        if truncate_time is not None:
            assert truncate_time >= 0
            # clamp
            # (self.motion_lengths[motion_ids] - truncate_time) shape=[len(motion_ids)]
            max_allowed = self.motion_lengths[motion_ids] - truncate_time
            # 需要确保 motion_times和 max_allowed在同一device
            motion_times = torch.min(motion_times, max_allowed.to(self.device).long())

        if self.play_dataset:
            motion_times = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)

        return motion_times


    def get_initial_state(self, env_ids, motion_ids, start_frames):
        """
        Get the initial state for given motion_ids and start_frames.
        
        Parameters:
        motion_ids (Tensor): A tensor containing the motion id for each environment.
        start_frames (Tensor): A tensor containing the starting frame number for each environment.
        
        Returns:
        Tuple: A tuple containing the initial state
        """
        assert len(motion_ids) == len(env_ids)
        valid_lengths = self.motion_lengths[motion_ids] - start_frames
        self.envid2episode_lengths[env_ids] = torch.where(valid_lengths < self.max_episode_length,
                                    valid_lengths, self.max_episode_length)
        # reward_weights_list = []
        hoi_data_list = []
        root_pos_list = []
        root_rot_list = []
        root_vel_list = []
        root_ang_vel_list = []
        dof_pos_list = []
        dof_vel_list = []
        obj_pos_list = []
        obj_pos_vel_list = []
        obj_rot_list = []
        obj_rot_vel_list = []

        for i, env_id in enumerate(env_ids):
            motion_id = motion_ids[i].item()
            if self.play_dataset:
                start_frame = 0 
            else:
                start_frame = start_frames[i].item() 
            
            self.envid2motid[env_id] = motion_id
            self.envid2sframe[env_id] = start_frame 
            episode_length = self.envid2episode_lengths[env_id].item()

            state = self._get_initial_state(motion_id, start_frame, episode_length)
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

            for k in self.reward_weights_default:
                self.reward_weights[k][env_id] =  torch.as_tensor(state['reward_weights'][k], device=self.device).float()
            assert env_id.max() < self.reward_weights[k].shape[0], \
                f"env_id {env_id.max()} 超出 reward_weights 的维度 {self.reward_weights[k].shape[0]}"
            hoi_data_list.append(state["hoi_data"])
            root_pos_list.append(state['init_root_pos'])
            root_rot_list.append(state['init_root_rot'])
            root_vel_list.append(state['init_root_pos_vel'])
            root_ang_vel_list.append(state['init_root_rot_vel'])
            dof_pos_list.append(state['init_dof_pos'])
            dof_vel_list.append(state['init_dof_pos_vel'])
            obj_pos_list.append(state["init_obj_pos"])
            obj_pos_vel_list.append(state["init_obj_pos_vel"])
            obj_rot_list.append(state["init_obj_rot"])
            obj_rot_vel_list.append(state["init_obj_rot_vel"])

        hoi_data = torch.stack(hoi_data_list, dim=0)
        root_pos = torch.stack(root_pos_list, dim=0)
        root_rot = torch.stack(root_rot_list, dim=0)
        root_vel = torch.stack(root_vel_list, dim=0)
        root_ang_vel = torch.stack(root_ang_vel_list, dim=0)
        dof_pos = torch.stack(dof_pos_list, dim=0)
        dof_vel = torch.stack(dof_vel_list, dim=0)
        obj_pos = torch.stack(obj_pos_list, dim =0)
        obj_pos_vel = torch.stack(obj_pos_vel_list, dim =0)
        obj_rot = torch.stack(obj_rot_list, dim =0)
        obj_rot_vel = torch.stack(obj_rot_vel_list, dim =0)

        return hoi_data, \
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, \
                obj_pos, obj_pos_vel, obj_rot, obj_rot_vel
                

    def _get_initial_state(self, motion_id, start_frame, episode_length):
        hoi_data = F.pad(
            self.hoi_data_dict[motion_id]['hoi_data'][start_frame:start_frame + episode_length],
            (0, 0, 0, self.max_episode_length - episode_length)
        )
        return {
            "reward_weights": self._get_reward_weights(),
            "hoi_data": hoi_data,
            "init_root_pos": self.hoi_data_dict[motion_id]['root_pos'][start_frame, :],
            "init_root_rot": self.hoi_data_dict[motion_id]['root_rot'][start_frame, :],
            "init_root_pos_vel": self.hoi_data_dict[motion_id]['root_pos_vel'][start_frame, :],
            "init_root_rot_vel": self.hoi_data_dict[motion_id]['root_rot_vel'][start_frame, :],
            "init_dof_pos": self.hoi_data_dict[motion_id]['dof_pos'][start_frame, :],
            "init_dof_pos_vel": self.hoi_data_dict[motion_id]['dof_pos_vel'][start_frame, :],
            "init_obj_pos": self.hoi_data_dict[motion_id]['obj_pos'][start_frame, :],
            "init_obj_pos_vel": self.hoi_data_dict[motion_id]['obj_pos_vel'][start_frame, :],
            "init_obj_rot": self.hoi_data_dict[motion_id]['obj_rot'][start_frame, :],
            "init_obj_rot_vel": self.hoi_data_dict[motion_id]['obj_rot_vel'][start_frame, :],
        }

    def _get_reward_weights(self):
        reward_weights = self.reward_weights_default
        return {
            "p": reward_weights["p"],
            "r": reward_weights["r"],
            "pv": reward_weights["pv"],
            "rv": reward_weights["rv"],
            "wp": reward_weights["wp"],
            "wr": reward_weights["wr"],
            "wpv": reward_weights["wpv"],
            "wrv": reward_weights["wrv"],
            "op": reward_weights["op"],
            "or": reward_weights["or"],
            "opv": reward_weights["opv"],
            "orv": reward_weights["orv"],
            "ig": reward_weights["ig"],
            "cg1": reward_weights["cg1"],
            "cg2": reward_weights["cg2"],
        }
