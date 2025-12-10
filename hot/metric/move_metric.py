# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class MoveMetric(BaseMetric):
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
        distance_obj2wrist = torch.norm(obj_pos - wrist_pos, dim=-1)
        
        current_step = state['progress']
        if current_step <= 10:
            at_target = True
        else:
            threshold = 0.15 - self.h_offset
            at_target = (distance_obj2wrist < 0.3) & (obj_pos[:, 2] > threshold) 

        self.at_target = self.at_target & at_target
        self.reached_target = self.at_target  # 直接覆盖，不累积

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
