# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class RegraspMetric(BaseMetric):
    def __init__(self, num_envs, device, h_offset, max_episode_length):
        super().__init__(num_envs, device, h_offset)
        self.at_target = torch.ones(num_envs, device=device, dtype=torch.bool)  # Assuming max progress_buf = 500
        self.reached_target = torch.ones(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.at_target = True
        self.reached_target = True

    def update(self, state):
        obj_pos = state['obj_pos']
        wrist_pos = state['wrist_pos']
        key_pos_error = state['key_pos_error']
        obj_pos_error = state['obj_pos_error']
        obj_rot_error = state['obj_rot_error']
        distance_obj2wrist = torch.norm(obj_pos - wrist_pos, dim=-1)
        
        current_step = state['progress']
        if current_step not in [30, 116]:
            at_target = True
        else:
            at_target = (distance_obj2wrist < 0.3)  & (key_pos_error < 5) & (obj_pos_error < 5) & (obj_rot_error < 20)
            # print('ObjPosError:', obj_rot_error, 'ObjRotError:', obj_rot_error, 'at_targe:', at_target)

        self.reached_target = self.reached_target & at_target 

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
