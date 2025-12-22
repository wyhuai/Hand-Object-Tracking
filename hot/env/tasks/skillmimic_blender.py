from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
import glob, os, random, pickle
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from datetime import datetime

from env.tasks.skillmimic import SkillMimicBallPlay


class SkillMimicBallPlayBlender(SkillMimicBallPlay):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                                sim_params=sim_params,
                                physics_engine=physics_engine,
                                device_type=device_type,
                                device_id=device_id,
                                headless=headless)
        self.build_blender_motion = cfg['env']['build_blender_motion']
        self.blender_motion_length = cfg['env']['blender_motion_length'] if cfg['env']['blender_motion_length'] > 0 else self.max_episode_length
        self.blender_motion_name = cfg['env']['blender_motion_name'] if len(cfg['env']['blender_motion_name']) > 0 else f"{self.object_names[0]}_motion.pt"
        self.motion_dict = {'wristpos':[], 'wristrot':[], 'dofpos':[], 'ballpos':[], 'ballrot':[]} # 'dofrot':[], 
        

    def _build_frame_for_blender(self, motion_dict, rootpos, rootrot, dofpos, ballpos, ballrot): # dofrot, 
        motion_dict['wristpos'].append(rootpos.clone())
        motion_dict['wristrot'].append(rootrot.clone())
        motion_dict['dofpos'].append(dofpos.clone())
        # motion_dict['dofrot'].append(dofrot.clone())
        motion_dict['ballpos'].append(ballpos.clone())
        motion_dict['ballrot'].append(ballrot.clone())
        
    def _save_motion_dict(self, motion_dict, filename='motion.pickle'):
        for key in motion_dict:
            if len(motion_dict[key]) > 0:
                motion_dict[key] = torch.stack(motion_dict[key])

        # torch.save(motion_dict, filename)
        motion_data = {_: motion_dict[_].to('cpu').numpy() for _ in motion_dict}
        with open(filename, 'wb') as file:
            pickle.dump(motion_data, file)
        print(f'Successfully save the motion_dict to {filename}!')
        exit()
        
    def _build_blender_motion(self):
        body_ids = list(range(51))
        self._build_frame_for_blender(self.motion_dict,
                        self._rigid_body_pos[0, body_ids, :],
                        self._rigid_body_rot[0, body_ids, :],
                        self._dof_pos,
                        self._target_states[0, :3],
                        self._target_states[0, 3:7])
        if self.progress_buf_total == self.blender_motion_length:
            self._save_motion_dict(self.motion_dict, f'blender_motion/{self.blender_motion_name}')

    def play_dataset_step(self, time, motid=None, length=None):
        super().play_dataset_step(time, motid, length)
        self.progress_buf_total += 1
        if self.build_blender_motion:
            self._build_blender_motion()
        return self.obs_buf
        
    def post_physics_step(self):
        super().post_physics_step()
        if self.build_blender_motion:
            self._build_blender_motion()