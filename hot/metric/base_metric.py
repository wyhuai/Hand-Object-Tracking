from abc import ABC, abstractmethod
import torch

class BaseMetric(ABC):
    def __init__(self, num_envs, device, h_offset):
        self.num_envs = num_envs
        self.device = device
        self.h_offset = h_offset

    @abstractmethod
    def reset(self, env_ids):
        pass

    @abstractmethod
    def update(self, state):
        pass

    @abstractmethod
    def compute(self):
        pass
    
    def get_succ_envs(self):
        succ_envs_ids = torch.nonzero(self.reached_target, as_tuple=True)[0]
        return succ_envs_ids