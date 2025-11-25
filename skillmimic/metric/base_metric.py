from abc import ABC, abstractmethod

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