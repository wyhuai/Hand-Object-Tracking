# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class PlaceMetric(BaseMetric):
    def __init__(self, num_envs, device, h_offset, max_episode_length):
        super().__init__(num_envs, device, h_offset)
        self.at_target = torch.zeros((num_envs, 220), device=device, dtype=torch.bool)  # Assuming max progress_buf = 500
        self.reached_target = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def reset(self, env_ids):
        self.at_target = torch.zeros((self.num_envs, 220), device=self.device, dtype=torch.bool)
        self.reached_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def update(self, state):
        obj_pos = state['obj_pos']
        obj_pos_vel = state['obj_pos_vel']
        wrist_pos = state['wrist_pos']
        distance_obj2wrist = torch.norm(obj_pos - wrist_pos, dim=-1)
        
        current_step = state['progress']
        if current_step <= 160:
            at_target = (distance_obj2wrist < 0.3) & (obj_pos_vel[:, 0] < 0.1) & (obj_pos_vel[:, 1] < 0.1) & (obj_pos_vel[:, 2] <= 0.1)
        else:
            at_target = torch.norm(obj_pos_vel, dim=-1) <= 0.4
        self.at_target[:, current_step] = at_target
        self.reached_target = torch.sum(self.at_target, dim=-1) > 190

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
