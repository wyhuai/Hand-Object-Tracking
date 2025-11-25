# metric/shot_metric.py
from .base_metric import BaseMetric
import torch

class TrackingMetric(BaseMetric):
    def __init__(self, num_envs, device, h_offset, max_episode_length):
        super().__init__(num_envs, device, h_offset)
        self.at_target = torch.ones(num_envs, device=device, dtype=torch.bool)  # Assuming max progress_buf = 500
        self.reached_target = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.max_episode_length = max_episode_length

    def reset(self, env_ids):
        self.at_target = True
        self.reached_target = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

    def update(self, state):
        obj_pos = state['obj_pos']
        wrist_pos = state['wrist_pos']
        key_pos_error = state['key_pos_error']
        obj_pos_error = state['obj_pos_error']
        obj_rot_error = state['obj_rot_error']
        distance_obj2wrist = torch.norm(obj_pos - wrist_pos, dim=-1)
        
        current_step = state['progress']
        if current_step > 20 and current_step < self.max_episode_length - 5:
            at_target = (obj_pos_error < 12)  & (obj_rot_error < 90) & (key_pos_error < 25)
        else:
            at_target = True

        self.reached_target = self.reached_target & at_target 

    def compute(self):
        success_rate = torch.sum(self.reached_target) / self.num_envs
        return success_rate
