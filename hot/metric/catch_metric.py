# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class CatchMetric(BaseMetric):
    def __init__(self, num_envs, device, h_offset, max_episode_length):
        super().__init__(num_envs, device, h_offset)
        self.at_target = torch.ones(num_envs, device=device, dtype=torch.bool)  # Assuming max progress_buf = 500
        self.reached_target = torch.ones(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.at_target = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.reached_target = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

    def update(self, state):
        obj_pos = state['obj_pos']
        wrist_pos = state['wrist_pos']
        distance_obj2wrist = torch.norm(obj_pos - wrist_pos, dim=-1)
        
        current_step = state['progress']
        if current_step <= 50:
            at_target = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            threshold = 0.3 + self.h_offset
            at_target = (distance_obj2wrist < 0.3) & (obj_pos[:, 2] > threshold) & (wrist_pos[:, 2] > threshold) 

        self.at_target = self.at_target & at_target
        if current_step == 88:
            self.reached_target = self.at_target

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
