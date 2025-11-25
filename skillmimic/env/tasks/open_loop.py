from enum import Enum
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import glob, os, random
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import time as pyton_time# changed by me
from pathlib import Path
from datetime import datetime
import math 

from utils import torch_utils
from utils.motion_data_handler import MotionDataHandler

from env.tasks.skillmimic import SkillMimicBallPlay


class OpenLoop(SkillMimicBallPlay): 
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):



        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless= headless)



    def open_loop_step(self):

        # apply actions
        self.pre_physics_step()

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self):
        ts = self.progress_buf.clone()
        self.actions = self.hoi_data_batch[torch.arange(self.num_envs),ts+1][:, 7:7+self.num_joints].clone() # [num_envs, 56]
        #self.actions = torch.clamp(self.actions, -1.0, 1.0)
        if self._pd_control:
            pd_tar = self.actions
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        self.evts = list(self.gym.query_viewer_action_events(self.viewer))

        return


    
