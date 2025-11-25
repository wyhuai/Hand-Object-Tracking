# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class ThrowMetric(BaseMetric):
    def __init__(self, num_envs, device, h_offset, max_episode_length):
        super().__init__(num_envs, device, h_offset)
        self.reached_target = torch.ones(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.reached_target = True

    def update(self, state):
        obj_pos = state['obj_pos']
        wrist_pos = state['wrist_pos']
        key_pos_error = state['key_pos_error']
        obj_pos_error = state['obj_pos_error']
        obj_rot_error = state['obj_rot_error']
        distance_obj2wrist = torch.norm(obj_pos - wrist_pos, dim=-1)
        
        current_step = state['progress']
        if current_step in range(30, 35):
            at_target = (key_pos_error < 40) & (obj_pos_error < 40) & (obj_rot_error < 90)
        else:
            at_target = True

        self.reached_target = self.reached_target & at_target

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
