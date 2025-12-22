
from isaacgym.torch_utils import *
from hop_skill import *

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import torch_utils

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.spatial.transform import Rotation as R
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import glob
import os



class GraspRootProcessor(SkillGenerator):
    def __init__(self, obj_name="box01",
                 basic_grasps_path="../../data/motions/dexgrasp_train/box/grasp_rot",
                 output_path="../../data/motions/dexgrasp_train/box/grasp_rot_transformed_root",
                 root_pos=torch.tensor([0.0, 0.0, 0.5]),
                 root_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
                 hand_model = "mano",
                 device='cpu'):
        self.skill_code = "000"
        super().__init__(obj_name=obj_name,
                    basic_grasps_path=basic_grasps_path,
                    output_path=output_path,
                    operation_space_ranges=None,
                    device=device,
                    hand_model=hand_model,
                    num_data_samples=0)
        self.desired_root_pos = root_pos
        self.desired_root_quat = root_quat    
        self.load_basic_trajectories()
        self.process_all_trajectories()

    def _load_desired_grasp(self):
        """Load the reference grasp trajectory"""
        return load_grasp_traj(self.paths['desired_grasp'], self.device)


    def process_all_trajectories(self):
       
        os.makedirs(self.paths['output'], exist_ok=True)
        num_trajs = len(self.basic_trajectories)

        for i in range(num_trajs):
            self._process_single_traj(i)

    def _process_single_traj(self, traj_idx):
        """Process individual frame in a grasp trajectory"""
        num_frame = self.basic_trajectories[traj_idx]['root_pos'].shape[0]
        large_error = False
        wrist_pos_traj_global, wrist_quat_traj_global = self.get_wrist_global_poses(self.basic_trajectories[traj_idx])
        # Compute new wrist DOF
        wrist_dof, large_error, deleted_frame_id = self.calculate_wrist_dofs(
            wrist_pos_traj_global, wrist_quat_traj_global,
            self.desired_root_pos, self.desired_root_quat,
            per_frame=True
        )
        if self.hand_model == "mano":
                key_body_dim = 48
        elif self.hand_model == "shadow":
                key_body_dim = 23*3
        elif self.hand_model == "allegro":
                key_body_dim = 17*3

        if len(deleted_frame_id) > 0:
            deleted_tensor = torch.tensor(deleted_frame_id)
            # Create a mask for frames that are not deleted
            mask = torch.ones(num_frame, dtype=torch.bool)
            mask[deleted_tensor] = False
            # Construct HOI data frame

            generated_data = {
                'root_pos': self.desired_root_pos.repeat(num_frame-len(deleted_frame_id), 1),
                'root_rot': self.desired_root_quat.repeat(num_frame-len(deleted_frame_id), 1),
                'wrist_dof':  wrist_dof.squeeze(0),
                'fingers_dof': self.basic_trajectories[traj_idx]['fingers_dof'][mask] ,
                'body_pos': torch.zeros(num_frame-len(deleted_frame_id), key_body_dim), #key_body_pos
                'obj_pos': self.basic_trajectories[traj_idx]['obj_pos'][mask] ,
                'obj_pos_vel': None,
                'obj_rot':self.basic_trajectories[traj_idx]['obj_rot'][mask] ,
                'obj2_pos': self.basic_trajectories[traj_idx]['obj2_pos'][mask] ,
                'obj2_rot': self.basic_trajectories[traj_idx]['obj2_rot'][mask] ,
                'contact1':  self.basic_trajectories[traj_idx]['contact1'][mask] ,
                'contact2':  self.basic_trajectories[traj_idx]['contact2'][mask] 
                }
        else:
            generated_data = {
                'root_pos': self.desired_root_pos.repeat(num_frame, 1),
                'root_rot': self.desired_root_quat.repeat(num_frame, 1),
                'wrist_dof':  wrist_dof.squeeze(0),
                'fingers_dof': self.basic_trajectories[traj_idx]['fingers_dof'],
                'body_pos': torch.zeros(num_frame, key_body_dim), #key_body_pos
                'obj_pos': self.basic_trajectories[traj_idx]['obj_pos'],
                'obj_pos_vel': None,
                'obj_rot': self.basic_trajectories[traj_idx]['obj_rot'],
                'obj2_pos': self.basic_trajectories[traj_idx]['obj2_pos'],
                'obj2_rot': self.basic_trajectories[traj_idx]['obj2_rot'],
                'contact1':  self.basic_trajectories[traj_idx]['contact1'],
                'contact2':  self.basic_trajectories[traj_idx]['contact2']
                }
        if len(deleted_frame_id) == 0:
            self._save_transformed_data(generated_data, traj_idx, num_frame)
        else :
            print(f"\nSkip frame {deleted_frame_id} as it contain large error, please check if both of the objects are moving within the robot operation space！\n")
            self._save_transformed_data(generated_data, traj_idx, num_frame)

    def get_wrist_global_poses(self, grasp_traj):
        """Convert wrist DOFs to global poses"""
        wrist_pos_traj_global_list = []
        wrist_quat_traj_global_list = []
        
        for i, dof in enumerate(grasp_traj['wrist_dof']):
            wrist_pos = dof[:3]
            wrist_rot = quat_from_euler_xyz_extrinsic(dof[3], dof[4], dof[5])
            global_wrist_pos, global_wrist_rot = AinB_local_to_global(
                wrist_pos.unsqueeze(0), 
                wrist_rot.unsqueeze(0),
                grasp_traj['root_pos'][i],
                grasp_traj['root_rot'][i]
            )
            wrist_pos_traj_global_list.append(global_wrist_pos.squeeze())
            wrist_quat_traj_global_list.append(global_wrist_rot.squeeze())

        return torch.stack(wrist_pos_traj_global_list), torch.stack(wrist_quat_traj_global_list)



    def _save_transformed_data(self, data_dict, idx, grasp_idx):
        """Save transformed data to file"""
        save_path = os.path.join(
            self.paths['output'],
            f'{self.output_name}_{idx}_{grasp_idx}.pt'
        )
        torch.save(data_dict, save_path)
        print(f"\nSaved {self.output_name}_{idx}_{grasp_idx}.pt!")


if __name__ == "__main__":
    processor = GraspRootProcessor(
        #basic_grasps_path="../../data/motions/graspmimic/sword/grasp_rot",
        #output_path="../../data/motions/graspmimic/sword/grasp_rot_transformed_root",
        basic_grasps_path="../../data/motions/graspskill/shoe_merged",
        output_path="../../data/motions/graspskill/shoe_change_root",
        hand_model= "allegro",
        #basic_grasps_path="../../data/motions/graspmimic/gun/grasp_rot",
        #output_path="../../data/motions/graspmimic/gun/grasp_rot_transformed_root",
        #basic_grasps_path="../../data/motions/graspmimic/screwdriver/grasp_rot",
        #output_path="../../data/motions/graspmimic/screwdriver/grasp_rot_transformed_root",
        device='cpu'
    )
    print("\nAll processing completed!\n")