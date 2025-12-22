
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

#############################################################
# Set the random seed
# seed = 42  # You can choose any integer you like
# np.random.seed(seed)            # Set NumPy random seed
# torch.manual_seed(seed)         # Set PyTorch random seed
# torch.cuda.manual_seed(seed)    # Set CUDA random seed (if using GPU)
# torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs (if using multiple GPUs
################################################################

class GraspMoveGenerator(SkillGenerator):
    def __init__(self, obj_name="box", 
                 basic_grasps_path="../../data/motions/graspmimic/sword/grasp_rot_transformed_root",
                 output_path="../../data/motions/graspmimic/sword/move",
                 frame_rate = 0.5/100,
                 num_data_samples = 5,
                 operation_space_ranges=None,
                 device='cpu',
                 hand_model = "mano"
                 ):
        self.skill_code = "002"
        super().__init__(obj_name=obj_name,
                    basic_grasps_path=basic_grasps_path,
                    output_path=output_path,
                    operation_space_ranges=None,
                    device=device,
                    hand_model=hand_model,
                    num_data_samples=num_data_samples)
        self.frame_rate = frame_rate
        self.operation_space = operation_space_ranges or {
            'x_offset': (-.5, .5),
            'y_offset': (-.5, .5),
            'z_range': (0.5, 1)
        }

        self.load_basic_trajectories()
        self.process_all_grasps()


    def process_all_grasps(self):
        os.makedirs(self.paths['output'], exist_ok=True)
        num_grasps = self.basic_trajectories[0]['root_pos'].shape[0]
        for i in range(num_grasps):
            self._process_single_grasp(i)

    def _process_single_grasp(self, grasp_idx):
        """Process a single grasp trajectory"""
        idx = 0
        while True: 
            obj_pos_traj, obj_rot_traj, num_steps = self._generate_object_trajectory(
               self.basic_trajectories[0]['root_pos'][grasp_idx] 
            )

            # Calculate wrist trajectory
            wrist_dof_traj, large_error = self._calculate_wrist_trajectory(self.basic_trajectories[0], grasp_idx, obj_pos_traj, obj_rot_traj)
            
            obj_pos_traj = torch.stack(obj_pos_traj).squeeze(1)
            obj_rot_traj = torch.stack(obj_rot_traj).squeeze(1)
            obj_rot_traj = smooth_quat_seq(obj_rot_traj)

            if not large_error:
                if self.hand_model == "mano":
                    key_body_dim = 48
                elif self.hand_model == "shadow":
                    key_body_dim = 23*3
                elif self.hand_model == "allegro":
                    key_body_dim = 17*3
                generated_data = {
                'root_pos':  self.basic_trajectories[0]['root_pos'][grasp_idx].repeat(num_steps, 1),
                'root_rot':  self.basic_trajectories[0]['root_rot'][grasp_idx].repeat(num_steps, 1),           
                'wrist_dof':  wrist_dof_traj,     
                'fingers_dof': self.basic_trajectories[0]['fingers_dof'][grasp_idx].repeat(num_steps, 1),
                'body_pos': torch.zeros(num_steps, key_body_dim),
                'obj_pos': obj_pos_traj,
                'obj_pos_vel': None,
                'obj_rot': obj_rot_traj,
                'obj2_pos': torch.zeros_like(obj_pos_traj),
                'obj2_rot': torch.zeros_like(obj_rot_traj),
                'contact1':  torch.ones(num_steps, 1),
                'contact2':  torch.zeros(num_steps, 1),
                }
                # Randomly select one of the three options
                choice = torch.randint(0, 3, (1,)).item()
                if choice == 1:
                    generated_data = self.duplicate_last_frame(generated_data)
                elif choice == 2:
                    generated_data = self.duplicate_first_frame(generated_data)
                self._save_transformed_data(generated_data, idx, grasp_idx)
                idx += 1
                if idx == self.num_data_samples:
                    return
            else :              
                print(f"\nSkip {self.output_name}_{grasp_idx}_{idx}.pt as it contain large error, please check if both of the objects are moving within the robot operation space！\n")
            

    def _generate_object_trajectory(self, base_pos, fixed_rotation_magnitude_deg=180.0):
        """Generate complete object trajectory"""
        # Define operation space
        x_min = base_pos[0] + self.operation_space['x_offset'][0]
        x_max = base_pos[0] + self.operation_space['x_offset'][1]
        y_min = base_pos[1] + self.operation_space['y_offset'][0]
        y_max = base_pos[1] + self.operation_space['y_offset'][1]
        z_min, z_max =  self.operation_space['z_range']
        # Generate random target poses
        target_obj_pos1 = random_position(x_min, x_max, y_min, y_max, z_min, z_max)
        target_obj_pos2 = random_position(x_min, x_max, y_min, y_max, z_min, z_max)
        target_obj_quat1 = random_quaternion()
        # Compute fixed rotation magnitude
        fixed_angle_rad = torch.deg2rad(torch.tensor(fixed_rotation_magnitude_deg))  # Convert to radians

        # Generate a random rotation axis (unit vector)
        random_axis = torch.randn(3)
        random_axis = random_axis / torch.norm(random_axis)  # Normalize to unit length

        # Create target_obj_quat2 by rotating target_obj_quat1 by fixed_angle_rad around random_axis
        target_obj_quat1_tensor = torch.tensor(target_obj_quat1, dtype=torch.float32)

        # Create rotation vector and quaternion
        rotation_vec = random_axis * fixed_angle_rad
        delta_rotation = R.from_rotvec(rotation_vec.numpy())

        # Quaternion multiplication 
        target_obj_quat2 = (R.from_quat(target_obj_quat1_tensor.numpy()) * delta_rotation).as_quat()
        target_obj_quat2 = torch.tensor(target_obj_quat2, dtype=torch.float32)

        distance = torch.norm(target_obj_pos2 - target_obj_pos1).item()  # Compute Euclidean distance
        num_frame = int(round(distance / self.frame_rate))

        obj_pos_traj, obj_rot_traj = generate_trajectory_between_poses(
            target_obj_pos1, target_obj_quat1,
            target_obj_pos2, target_obj_quat2,
            num_frame
        )
        return obj_pos_traj, obj_rot_traj, num_frame

    def duplicate_last_frame(self, generated_data):
        num_copies = torch.randint(low=1, high=31, size=(1,)).item()  # Random number between 1 and 30
        extended_data = {}
        for key, value in generated_data.items():
            if key != 'obj_pos_vel' :
                # Copy the last frame for the specified number of times
                last_frame = value[-1:]  # Get the last frame
                extended_data[key] = torch.cat((value, last_frame.repeat(num_copies, 1)), dim=0)
            else: 
                extended_data[key] = None
        return extended_data
    
    def duplicate_first_frame(self, generated_data):
            num_copies = torch.randint(low=1, high=31, size=(1,)).item()  # Random number between 1 and 30
            extended_data = {}
            for key, value in generated_data.items():
                if key != 'obj_pos_vel':
                    # Copy the first frame for the random number of times
                    first_frame = value[:1]  # Get the first frame
                    extended_data[key] = torch.cat((first_frame.repeat(num_copies, 1), value), dim=0)
                else:
                    extended_data[key] = None
            return extended_data


if __name__ == "__main__":
    generator = GraspMoveGenerator(
        obj_name="hammer", 
        basic_grasps_path="../../data/motions/dexgrasp_train_mano/hammer/grasp_rot_transformed_root",
        output_path="../../data/motions/dexgrasp_train_mano/hammer/move_test",
        hand_model= "mano"

        #basic_grasps_path="../../data/motions/graspmimic/stick/grasp_rot_transformed_root",
        #output_path="../../data/motions/graspmimic/stick/move",

        #basic_grasps_path="../../data/motions/graspmimic/gun/grasp_rot_transformed_root",
        #output_path="../../data/motions/graspmimic/gun/move",
        
        #basic_grasps_path="../../data/motions/graspmimic/screwdriver/grasp_rot_transformed_root",
        #output_path="../../data/motions/graspmimic/screwdriver/move",
    )
    print("\nAll steps completed!\n")