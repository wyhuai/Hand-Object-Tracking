
from isaacgym.torch_utils import *
from hop_skill import *

import os
import sys
import random
import math
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
#Set the random seed
#seed = 42  # You can choose any integer you like
#np.random.seed(seed)            # Set NumPy random seed
#torch.manual_seed(seed)         # Set PyTorch random seed
#torch.cuda.manual_seed(seed)    # Set CUDA random seed (if using GPU)
#torch.cuda.manual_seed_all(seed) # Set the seed for all GPUs (if using multiple GPUs
################################################################
class GraspAndPlaceGenerator(SkillGenerator):
    def __init__(self,
                 obj_name="mug",
                 basic_grasps_path="../../data/motions/graspmimic/stick/grasp_rot_transformed_root",
                 stable_pose_path ="../../data/free_drop_test_image/Stick/unique_quats.txt",
                 output_path="../../data/motions/graspmimic/stick/grasp",
                 output_place_path="../../data/motions/graspmimic/stick/place",
                 device='cpu',
                 frame_rate = 0.5/100,
                 num_data_samples=3, #21
                 operation_space_ranges=None,
                 hand_model = "mano",
                 span_angle=45):
        self.skill_code = "001"
        super().__init__(obj_name=obj_name,
                    basic_grasps_path=basic_grasps_path,
                    output_path=output_path,
                    operation_space_ranges=operation_space_ranges,
                    device=device,
                    hand_model=hand_model,
                    num_data_samples=num_data_samples)
        self.output_name_1 = f"003_{obj_name}"
        self.paths['stable_pose_path'] = stable_pose_path
        self.paths['output_place'] = output_place_path
        self.unreachable_pos = 0
        self.ik = 0
        self.stable_poses = {}
        self.frame_rate = frame_rate
        self.num_duplicated_grasp_frame = 20
        self.span_angle = span_angle
        self.valid_wrist_z = 0.05
        self.load_stable_pose()
        self.load_basic_trajectories()
        self.process_stable_pose()

    def load_stable_pose(self):
        self.stable_poses = {}  # Initialize stable_poses as a dictionary
        with open(self.paths['stable_pose_path'], 'r') as file:
            for line in file:
                parts = line.split(':')
                if len(parts) > 1:
                    group_id = int(parts[0].split()[1])  # Extract group ID (e.g., Group 0)
                    values = parts[1].strip().split(';')
                    
                    # Parse position and quaternion
                    pos_values = list(map(float, values[0].strip().split(',')))  # Position
                    quat_values = list(map(float, values[1].strip().split(',')))  # Quaternion
                    
                    # Store in the stable_poses dictionary
                    self.stable_poses[group_id] = {
                        'pos': torch.tensor(pos_values),
                        'quat': torch.tensor(quat_values)
                    }
        print(f"Loaded stable poses: {self.stable_poses}")


    def process_stable_pose(self):        
        os.makedirs(self.paths['output'], exist_ok=True)
        os.makedirs(self.paths['output_place'], exist_ok=True)
        for idx, stable_pose in self.stable_poses.items():
            self.process_all_grasps(idx, stable_pose)
  
    def process_all_grasps(self, stable_idx, stable_pose):
        num_grasps = self.basic_trajectories[0]['root_pos'].shape[0]        
        for i in range(num_grasps):
            self._process_single_grasp(i, stable_idx, stable_pose)
            
    def universal_obj_pos(self, init_root_pos, stable_pose):
        base_pos = init_root_pos
        # Define the ranges for x, y, and z
        ranges = torch.tensor([
            [base_pos[0] + self.operation_space['x_offset'][0],
            base_pos[0] + self.operation_space['x_offset'][1]],
            [base_pos[1] + self.operation_space['y_offset'][0],
            base_pos[1] + self.operation_space['y_offset'][1]],
        ], device=self.device)
        # Calculate the middle values for x and y
        middle_x = (ranges[0][0] + ranges[0][1]) / 2
        middle_y = (ranges[1][0] + ranges[1][1]) / 2
        # Create a tensor with the middle values for x, y, and z
        return torch.tensor([middle_x, middle_y, stable_pose["pos"][2]], device=self.device)

    def _process_single_grasp(self, grasp_idx, stable_idx, stable_pos):
        idx = 0
        fail = 0
        while True:
            """Augment a single trajectory with random transformation"""

            init_data = {key: self.basic_trajectories[0][key][grasp_idx].clone() for key in ['obj_pos', 'obj_rot', 'wrist_dof', 'root_pos', 'root_rot']}
            init_wrist_rot_quat_local = quat_from_euler_xyz_extrinsic(
                                        init_data['wrist_dof'][3],
                                        init_data['wrist_dof'][4],
                                        init_data['wrist_dof'][5]).unsqueeze(0)  
            stable_obj_pos = self.universal_obj_pos(init_data['root_pos'], stable_pos).clone()
            stable_obj_quat = stable_pos["quat"].clone()
            # Check if the hand pose is within valid range
            if_vaild_wrist = self.valid_wrist(init_data, stable_obj_pos, stable_obj_quat)
            if not if_vaild_wrist:
                print(f"Invalid hand grasp {grasp_idx} for stable pose {stable_idx}")
                self.unreachable_pos += self.num_data_samples
                return
            
            # Generate self.num_data_samples of sample
            target_obj_pos, target_obj_quat = self.generate_random_target_pose(init_data['root_pos'], stable_obj_pos, stable_obj_quat)

            #get wrist global 
            init_wrist_pos_global, init_wrist_rot_quat_global=\
                AinB_local_to_global(init_data['wrist_dof'][:3].unsqueeze(0), init_wrist_rot_quat_local, init_data['root_pos'], init_data['root_rot'])

            # get rel pos between wrist and obj for grasp frame to be last frame
            rel_trans, rel_rot=\
                AinB_global_to_local(init_wrist_pos_global, 
                                        init_wrist_rot_quat_global, 
                                        init_data['obj_pos'],
                                        init_data['obj_rot']
                                )
            # get end wrist pose from target object pose
            desired_wrist_pos_global, desired_wrist_rot_quat_global=\
                AinB_local_to_global(rel_trans, rel_rot, target_obj_pos, target_obj_quat)

            desired_init_wrist_pos_global = self.sample_init_wrist_pos_from_cone(init_data['root_pos'], desired_wrist_pos_global.squeeze(0), target_obj_pos, angle=45, distance_upper_bound=init_data['root_pos'][2]+self.operation_space['z_range'][1])
            if desired_init_wrist_pos_global is not None:
                for single_desired_init_wrist_pos_global in desired_init_wrist_pos_global:
                    print("num of pos",len(desired_init_wrist_pos_global))
                    large_error, generated_data = self.generate_single_grasp_traj(grasp_idx, init_data, desired_wrist_pos_global, desired_wrist_rot_quat_global, single_desired_init_wrist_pos_global, target_obj_pos, target_obj_quat)
                    if not large_error:
                        self._save_transformed_data(generated_data, idx, grasp_idx, stable_idx)
                        idx += 1
                        if idx == self.num_data_samples:
                            return
                    else :
                        fail += 1
                        self.ik +=1
                        print(f"\nSkip {self.output_name}_{stable_idx}_{grasp_idx}-{idx}.pt as it contain large error, please check if both of the objects are moving within the robot operation space！\n")
                    if fail > 100:
                        return
            else:
                return
            
    def generate_single_grasp_traj(self, grasp_idx, init_data, desired_wrist_pos_global, desired_wrist_rot_quat_global, single_desired_init_wrist_pos_global, target_obj_pos, target_obj_quat):
            # such that the hand will move in random speed
            pre_grasp_distance = torch.norm(single_desired_init_wrist_pos_global - desired_wrist_pos_global).item()  # Compute Euclidean distance
            num_pre_grasp_frame = int(round(pre_grasp_distance / self.frame_rate))

            wrist_pos_pre_grasp_traj_global, _ = self. _generate_trajectory_between_poses(
                desired_wrist_pos_global.squeeze(0), desired_wrist_rot_quat_global.squeeze(0), 
                single_desired_init_wrist_pos_global, 
                num_pre_grasp_frame
            )
            # wrist rot remain unchange
            wrist_quat_pre_grasp_traj_global = desired_wrist_rot_quat_global.repeat(num_pre_grasp_frame, 1)

            # Move the object upward 
            upward_desired_wrist_pos_global = desired_wrist_pos_global.clone()
            upward_desired_wrist_pos_global[:, 2] += 0.5  

            post_grasp_distance = torch.norm(upward_desired_wrist_pos_global - desired_wrist_pos_global).item()  # Compute Euclidean distance
            num_post_grasp_frame = int(round(post_grasp_distance / self.frame_rate))

            wrist_pos_post_grasp_traj_global, _ = self. _generate_trajectory_between_poses(
                upward_desired_wrist_pos_global.squeeze(0), desired_wrist_rot_quat_global.squeeze(0), 
                desired_wrist_pos_global.squeeze(0), 
                num_post_grasp_frame
            )
            
            wrist_quat_post_grasp_traj_global = desired_wrist_rot_quat_global.repeat(num_post_grasp_frame, 1)

            wrist_pos_full_traj_global, wrist_quat_full_traj_global = \
                torch.cat((wrist_pos_pre_grasp_traj_global, desired_wrist_pos_global.repeat(self.num_duplicated_grasp_frame, 1), wrist_pos_post_grasp_traj_global),dim=0), \
                torch.cat((wrist_quat_pre_grasp_traj_global, desired_wrist_rot_quat_global.repeat(self.num_duplicated_grasp_frame, 1), wrist_quat_post_grasp_traj_global),dim=0)
                        
            wrist_quat_full_traj_global = smooth_quat_seq(wrist_quat_full_traj_global)
            wrist_dof_traj, large_error = self.calculate_wrist_dofs(
                wrist_pos_full_traj_global, wrist_quat_full_traj_global,
                init_data['root_pos'], init_data['root_rot']
            )

            # Generate post grasp object trajectory
            obj_pos_post_grasp_traj_global = []
            obj_rot_post_grasp_traj_global = []
            for wrist_pos_global, wrist_quat_global \
                in zip(wrist_pos_post_grasp_traj_global, wrist_quat_post_grasp_traj_global):
                rel_trans, rel_rot = AinB_global_to_local(
                    target_obj_pos.unsqueeze(0), target_obj_quat.unsqueeze(0),
                    desired_wrist_pos_global.squeeze(0), desired_wrist_rot_quat_global.squeeze(0)
                )

                obj_pos_global, obj_rot_global = AinB_local_to_global(
                    rel_trans, rel_rot, 
                     wrist_pos_global, wrist_quat_global
                )
                obj_pos_post_grasp_traj_global.append(obj_pos_global.squeeze())
                obj_rot_post_grasp_traj_global.append(obj_rot_global.squeeze())
            obj_pos_post_grasp_traj_global,  obj_rot_post_grasp_traj_global= torch.stack(obj_pos_post_grasp_traj_global), torch.stack(obj_rot_post_grasp_traj_global)

            finger_pos_pre_grasp_traj_global, _ = self. _generate_trajectory_between_poses(
                self.basic_trajectories[0]['fingers_dof'][grasp_idx], desired_wrist_rot_quat_global.squeeze(0), 
                torch.zeros_like(self.basic_trajectories[0]['fingers_dof'][grasp_idx]), 
                num_pre_grasp_frame+1
            )
            if self.hand_model == "mano":
                key_body_dim = 48
            elif self.hand_model == "shadow":
                key_body_dim = 23*3
            elif self.hand_model == "allegro":
                key_body_dim = 17*3
            generated_data = {
                'root_pos':  init_data['root_pos'].repeat(num_pre_grasp_frame+self.num_duplicated_grasp_frame+num_post_grasp_frame, 1),
                'root_rot':  init_data['root_rot'].repeat(num_pre_grasp_frame+self.num_duplicated_grasp_frame+num_post_grasp_frame, 1),           
                'wrist_dof':  wrist_dof_traj,     
                # zeros like for hand to open
                'fingers_dof':  torch.cat((finger_pos_pre_grasp_traj_global,finger_pos_pre_grasp_traj_global[-1].repeat(self.num_duplicated_grasp_frame-1+num_post_grasp_frame, 1)),dim=0),
                'body_pos': torch.zeros(num_pre_grasp_frame+self.num_duplicated_grasp_frame+num_post_grasp_frame, key_body_dim),
                'obj_pos': torch.cat((target_obj_pos.repeat(num_pre_grasp_frame+self.num_duplicated_grasp_frame, 1), obj_pos_post_grasp_traj_global),dim=0),
                'obj_pos_vel': None,
                'obj_rot':  torch.cat((target_obj_quat.repeat(num_pre_grasp_frame+self.num_duplicated_grasp_frame, 1),obj_rot_post_grasp_traj_global),dim=0),
                'obj2_pos':  self.basic_trajectories[0]['obj2_pos'][grasp_idx].repeat(num_pre_grasp_frame+self.num_duplicated_grasp_frame+num_post_grasp_frame, 1),
                'obj2_rot':  self.basic_trajectories[0]['obj2_rot'][grasp_idx].repeat(num_pre_grasp_frame+self.num_duplicated_grasp_frame+num_post_grasp_frame, 1),
                'contact1':  torch.cat((torch.zeros_like(self.basic_trajectories[0]['contact1'][grasp_idx].repeat(num_pre_grasp_frame-4, 1)),-torch.ones_like(self.basic_trajectories[0]['contact1'][grasp_idx].repeat(4, 1)),torch.ones_like(self.basic_trajectories[0]['contact1'][grasp_idx].repeat(+self.num_duplicated_grasp_frame+num_post_grasp_frame, 1))),dim=0),
                'contact2':  torch.zeros_like(self.basic_trajectories[0]['contact2'][grasp_idx].repeat(num_pre_grasp_frame+self.num_duplicated_grasp_frame+num_post_grasp_frame, 1))
                }
            duplicated_generated_data = self.duplicate_last_frame(generated_data)
            return large_error, duplicated_generated_data
    
    def valid_wrist(self, init_data, stable_obj_pos, stable_obj_quat):
        init_wrist_rot = quat_from_euler_xyz_extrinsic(
                                    init_data['wrist_dof'][3],
                                    init_data['wrist_dof'][4],
                                    init_data['wrist_dof'][5]).unsqueeze(0)  
        
        init_root_pos = init_data['root_pos']
        init_root_rot = init_data['root_rot']
        init_obj_pos = init_data['obj_pos']
        init_obj_rot = init_data['obj_rot']
        
        # Get global wrist pose
        wrist_pos_global, wrist_rot_global = AinB_local_to_global(
            init_data['wrist_dof'][:3].unsqueeze(0),
            init_wrist_rot,
            init_root_pos,
            init_root_rot
        )
        # Calculate relative transform between wrist and object
        rel_trans, rel_rot = AinB_global_to_local(
            wrist_pos_global,
            wrist_rot_global,
            init_obj_pos,
            init_obj_rot
        )
        
        # get wrist pose from target object pose
        desired_wrist_pos_global, desired_wrist_rot_quat_global=\
            AinB_local_to_global(rel_trans, rel_rot, stable_obj_pos, stable_obj_quat)
        if desired_wrist_pos_global.squeeze(0)[2] > self.valid_wrist_z:
            return True
        else:            
            print(f"Invalid ", desired_wrist_pos_global.squeeze(0)[2])
            return False
  
  
    
    def generate_random_target_pose(self, init_root_pos, init_obj_pos, init_obj_rot):
        """Generate random target position and rotation"""
        ranges = torch.tensor([ # as default root is at [0,0,0.5]
            [-0.5,
             0.5],
            [-0.5,
             0.5],
            [0.0,
             0.0]
        ], device=self.device)

        coords = torch.rand(3, device=self.device)
        target_pos = coords * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        target_pos[2] = init_obj_pos[2]
        #  generates a random quaternion that represents a rotation around the Z-axis by a random angle between 0 and 2pi, mainly to augment the wristy
        rand_rz = torch.rand(1, device=self.device) * 2 * torch.pi
        random_quat = torch.zeros(1,4).to(self.device) # axis quat
        random_quat[:, 2] = torch.sin(rand_rz/2)
        random_quat[:, 3] = torch.cos(rand_rz/2)
        target_quat = quat_multiply(random_quat,init_obj_rot).squeeze(0)
        
        return target_pos, target_quat

    

    def sample_init_wrist_pos_from_cone(self, init_root_pos, desired_wrist_pos_global, target_obj_pos, angle=45, distance_lower_bound=0.2, distance_upper_bound=3.0):
        """
        Samples a random initial wrist position within a conical volume that spans 45 degrees from a direction vector, 
        with the tip of the cone at the target object position 
        extending outward to a specified distance
        """
        # Tip of cone is object pose
        tip = desired_wrist_pos_global
        # Normalize the direction vector
        direction = (
            desired_wrist_pos_global[0] - target_obj_pos[0],
            desired_wrist_pos_global[1] - target_obj_pos[1],
            desired_wrist_pos_global[2] - target_obj_pos[2]
        )
        magnitude = math.sqrt(sum(component ** 2 for component in direction))
        normalized_direction = (
            direction[0] / magnitude,
            direction[1] / magnitude,
            direction[2] / magnitude
        )
        normalized_direction =torch.tensor([
                normalized_direction[0],
                normalized_direction[1],
                normalized_direction[2]  # Keep the z-component based on the original direction
            ], dtype=torch.float32)
        # An additional range is the operation base
        base_pos = init_root_pos
        ranges = torch.tensor([
            [base_pos[0] + self.operation_space['x_offset'][0],
             base_pos[0] + self.operation_space['x_offset'][1]],
            [base_pos[1] + self.operation_space['y_offset'][0],
             base_pos[1] + self.operation_space['y_offset'][1]],
            self.operation_space['z_range']
        ], device=self.device)


        def sample_cone(p, v, num_samples=1000000, r_min=0.2, r_max=0.2):
            device = p.device
            dtype = p.dtype
            
            # 标准化方向向量
            v_norm = v / torch.norm(v)
            
            # 生成球坐标系参数
            u = torch.rand(num_samples, 3, device=device, dtype=dtype)
            
            # 采样半径（体积均匀分布）
            r = (u[:,0] * (r_max**3 - r_min**3) + r_min**3).pow(1/3)
            
            # 采样角度
            theta_max = torch.tensor([torch.pi/4])#math.radians(R_deg)
            cos_theta = torch.cos(theta_max) + (1 - torch.cos(theta_max)) * u[:,1]
            theta = torch.acos(cos_theta)
            phi = 2 * torch.pi * u[:,2]
            
            # 转换为笛卡尔坐标（局部坐标系）
            x_local = r * torch.sin(theta) * torch.cos(phi)
            y_local = r * torch.sin(theta) * torch.sin(phi)
            z_local = r * torch.cos(theta)
            points_local = torch.stack([x_local, y_local, z_local], dim=1)
            
            # 构建旋转矩阵
            if torch.abs(v_norm[0]) < 0.9:
                aux = torch.tensor([1,0,0], device=device, dtype=dtype)
            else:
                aux = torch.tensor([0,1,0], device=device, dtype=dtype)
            y_axis = torch.cross(v_norm, aux)
            if torch.norm(y_axis) < 1e-6:
                aux = torch.tensor([0,0,1], device=device, dtype=dtype)
                y_axis = torch.cross(v_norm, aux)
            y_axis = y_axis / torch.norm(y_axis)
            x_axis = torch.cross(y_axis, v_norm)
            x_axis = x_axis / torch.norm(x_axis)
            rotation = torch.stack([x_axis, y_axis, v_norm], dim=1)
            
            # 转换到世界坐标系
            points = p + torch.mm(points_local, rotation.T)
    
            # 筛选有效点
            valid = (r >= r_min) & (r <= r_max)
                
            return points[valid]
        # Calculate final position in 3D space
        sampled_positions =sample_cone(
            tip, normalized_direction
        )
        ranges = torch.tensor([
                [base_pos[0] + self.operation_space['x_offset'][0],
                base_pos[0] + self.operation_space['x_offset'][1]],
                [base_pos[1] + self.operation_space['y_offset'][0],
                base_pos[1] + self.operation_space['y_offset'][1]],
                self.operation_space['z_range']
            ])
        # 筛选有效点
        x_min, x_max = ranges[0][0].item(), ranges[0][1].item()
        y_min, y_max = ranges[1][0].item(), ranges[1][1].item()
        z_min, z_max = desired_wrist_pos_global[2], ranges[2][1].item()


        # Check if sampled positions are within the defined ranges
        valid_mask = (
            (sampled_positions[:, 0] >= x_min) & (sampled_positions[:, 0] <= x_max) &
            (sampled_positions[:, 1] >= y_min) & (sampled_positions[:, 1] <= y_max) &
            (sampled_positions[:, 2] >= z_min) & (sampled_positions[:, 2] <= z_max)
        )
        # Filter valid samples
        valid_samples = sampled_positions[valid_mask]
        cone_valid_candidates = []
        angle_threshold = math.radians(45)  # 45 degrees in radians
        z_axis = torch.tensor([0, 0, 1], device=valid_samples.device, dtype=valid_samples.dtype)

        for candidate in valid_samples:
            approach_vector = candidate - desired_wrist_pos_global 
            approach_magnitude = torch.norm(approach_vector)
            if approach_magnitude < 1e-6:  # avoid division by zero
                continue
            normalized_approach = approach_vector / approach_magnitude
            
            # Calculate angle between approach vector and z-axis
            dot_product = torch.dot(normalized_approach, z_axis)
            angle_between = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            
            if angle_between <= angle_threshold:
                cone_valid_candidates.append(candidate)
        if cone_valid_candidates:
            return torch.stack(cone_valid_candidates)
            
    def _save_transformed_data(self, data_dict, idx, grasp_idx, stable_idx=None):
        """Save augmented trajectory in the required format"""

        ######################################
        # print(data_dict['wrist_dof'])
        # import sys
        # print("Ending the program...")
        # sys.exit()
        #####################################  
        save_path = os.path.join(
            self.paths['output'],
            f'{self.output_name}_{stable_idx}_{grasp_idx}-{idx}.pt'
        )
        torch.save(data_dict, save_path)
        print(f"\nSaved {self.output_name}_{stable_idx}_{grasp_idx}-{idx}.pt!")

        save_path = os.path.join(
            self.paths['output_place'],
            f'{self.output_name_1}_{stable_idx}_{grasp_idx}-{idx}.pt'
        )
        flipped_data_dict = {
            key: torch.flip(value.clone().float(), [0]) if key != 'obj_pos_vel' else None
            for key, value in data_dict.items()
        }        
        torch.save(flipped_data_dict, save_path)
        print(f"\nSaved {self.output_name_1}_{stable_idx}_{grasp_idx}-{idx}.pt!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Select a method for processing.")
    
    parser.add_argument('--obj_name', 
                        type=str, default=None, 
                        help="File name for the grasp trajectory to be saved.")


    parser.add_argument('--asset_name', 
                        type=str, default=None, 
                        help="Folder name for the grasp trajectory to be saved.")
    args = parser.parse_args()

    augmenter = GraspAndPlaceGenerator(
        obj_name=args.obj_name,
        basic_grasps_path=f"../../data/motions/dexgrasp_train_mano/{args.obj_name}/grasp_rot_transformed_root",
        stable_pose_path =f"../../data/free_drop_test_image/{args.asset_name}/unique_quats.txt",
        output_path=f"../../data/motions/dexgrasp_train_mano/{args.obj_name}/grasp_test",
        output_place_path=f"../../data/motions/dexgrasp_train_mano/{args.obj_name}/place_test",
        hand_model= "mano"

    )
    print(f"Trajectory augmentation completed! There are {augmenter.unreachable_pos} unreachable wrist poses. {augmenter.ik}")