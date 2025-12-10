from isaacgym.torch_utils import *

import sys
import os
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
import math


def quat_multiply(q1, q2):
    # Compute the product of two quaternions.
    # The input quaternion format is [x, y, z, w], where w is the real part.
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack((x, y, z, w), dim=-1)

def quat2rotmat(quat):

    if not isinstance(quat, torch.Tensor):
        quat = torch.tensor(quat)

    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    x, y, z, w = torch.unbind(quat, dim=-1)

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotmat = torch.stack([
    w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (xz + wy),
    2 * (xy + wz), w2 - x2 + y2 - z2, 2 * (yz - wx),
    2 * (xz - wy), 2 * (yz + wx), w2 - x2 - y2 + z2
    ], dim=-1)

    output_shape = quat.shape[:-1] + (3, 3)
    rotmat = rotmat.reshape(output_shape)

    return rotmat


def validate_traj_coherence(
        last_dof_of_prev_traj,
        init_dof_of_curr_traj
    ):
    print("Start validating coherence between last traj and current traj")
    error_dof = np.linalg.norm(last_dof_of_prev_traj - init_dof_of_curr_traj)
    if error_dof > 1e-4:
            print("error_dof > 1e-4:",error_dof)
    print("Finish validating coherence between last traj and current traj")
    return

def quat_from_euler_xyz_extrinsic(x, y, z):
    quat = R.from_euler('XYZ',[x,y,z]).as_quat()
    quat = torch.tensor(quat).float()
    return quat

def rotmat_to_quat(rotmat):
    """
    quat format: xyzw
    """
    B = rotmat.shape[0]
    quat = torch.zeros((B, 4), device=rotmat.device, dtype=rotmat.dtype)

    trace = rotmat[:, 0, 0] + rotmat[:, 1, 1] + rotmat[:, 2, 2]
    mask = trace > 0
    S = torch.zeros(B, device=rotmat.device, dtype=rotmat.dtype)
    S[mask] = torch.sqrt(trace[mask] + 1.0) * 2 # S=4*qw
    quat[mask, 0] = 0.25 * S[mask]
    quat[mask, 1] = (rotmat[mask, 2, 1] - rotmat[mask, 1, 2]) / S[mask]
    quat[mask, 2] = (rotmat[mask, 0, 2] - rotmat[mask, 2, 0]) / S[mask]
    quat[mask, 3] = (rotmat[mask, 1, 0] - rotmat[mask, 0, 1]) / S[mask]

    mask = ~mask
    A = rotmat[mask, 0, 0] - rotmat[mask, 1, 1] - rotmat[mask, 2, 2] + 1.0
    S = torch.sqrt(A) * 2 # S=4*qx
    quat[mask, 1] = 0.25 * S
    quat[mask, 0] = (rotmat[mask, 2, 1] - rotmat[mask, 1, 2]) / S
    quat[mask, 2] = (rotmat[mask, 0, 1] + rotmat[mask, 1, 0]) / S
    quat[mask, 3] = (rotmat[mask, 0, 2] + rotmat[mask, 2, 0]) / S

    quat = F.normalize(quat, p=2, dim=1)

    quat_xyzw = quat.clone()
    quat_xyzw[:,0] = quat[:,1]
    quat_xyzw[:,1] = quat[:,2]
    quat_xyzw[:,2] = quat[:,3]
    quat_xyzw[:,3] = quat[:,0]

    return quat_xyzw


def transform_traj(obj_quat_traj, obj_pos_traj, wrist_quat_traj, wrist_pos_traj,
                      target_obj_quat, target_obj_pos):
    # q_diff
    q_initial = obj_quat_traj[-1].clone()
    q_initial_inv = quat_conjugate(q_initial)
    q_diff = torch_utils.quat_multiply(target_obj_quat, q_initial_inv)

    # pos_diff
    pos_initial = obj_pos_traj[-1].clone() 
    pos_target = target_obj_pos.clone()
    pos_diff = pos_target - quat_rotate(q_diff, pos_initial.unsqueeze(0)).squeeze(0)

    # traj transform
    obj_quat_traj_new = torch_utils.quat_multiply(q_diff.expand_as(obj_quat_traj), obj_quat_traj)
    obj_pos_traj_new = quat_rotate(q_diff.expand_as(obj_quat_traj), obj_pos_traj) + pos_diff

    wrist_quat_traj_new = torch_utils.quat_multiply(q_diff.expand_as(wrist_quat_traj), wrist_quat_traj)
    wrist_pos_traj_new = quat_rotate(q_diff.expand_as(wrist_quat_traj), wrist_pos_traj) + pos_diff

    return obj_quat_traj_new, obj_pos_traj_new, wrist_quat_traj_new, wrist_pos_traj_new


def create_robot_chain(root_rot, root_pos):

    robot_chain = Chain(
        name='wrist',
        links=[
            OriginLink(),

            URDFLink(
                name="prismatic_x",
                origin_translation=root_pos,
                origin_orientation=root_rot,
                translation=[1, 0, 0],  # x-axis
                joint_type='prismatic',
                bounds=(-2, 2)
            ),

            URDFLink(
                name="prismatic_y",
                origin_translation=[0.0, 0.0, 0.0],  
                origin_orientation=[0, 0, 0],
                translation=[0, 1, 0],  # y-axis
                joint_type='prismatic',
                bounds=(-2, 2)
            ),

            URDFLink(
                name="prismatic_z",
                origin_translation=[0.0, 0.0, 0.0],
                origin_orientation=[0, 0, 0],
                translation=[0, 0, 1],  # z-axis
                joint_type='prismatic',
                bounds=(-2, 2)
            ),

            URDFLink(
                name="rx",
                origin_translation=[0.0, 0.0, 0.0],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],  # Rotation around x-axis
                joint_type='revolute',
                bounds=(-2*np.pi, 2*np.pi)
            ),

            URDFLink(
                name="ry",
                origin_translation=[0.0, 0.0, 0.0],  # No translation; same position as elbow_x
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],  # Rotation around y-axis
                joint_type='revolute',
                bounds=(-2*np.pi, 2*np.pi)
            ),

            URDFLink(
                name="rz",
                origin_translation=[0.0, 0.0, 0.0],  # No translation; same position as elbow_y
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],  # Rotation around z-axis
                joint_type='revolute',
                bounds=(-2*np.pi, 2*np.pi)
            ),
        ],active_links_mask=[False, True, True, True, True, True, True]
    )
    return robot_chain



def inverse_kinematics(wrist_pos_traj,
                wrist_rot_traj, root_rot, root_pos,
                init_dof_xyz,
                init_dof_rxryrz, per_frame = False):

    large_error = False
    deleted_frame_idx = []
    robot_chain = create_robot_chain(root_rot, root_pos)

    arm_dof_trajectory = []
    initial_joint_angles = [0,
                            init_dof_xyz[0],
                            init_dof_xyz[1],
                            init_dof_xyz[2],
                            init_dof_rxryrz[0],
                            init_dof_rxryrz[1],
                            init_dof_rxryrz[2]
                            ]
    
    for idx, (pos, rotation_matrix) in enumerate(zip(wrist_pos_traj, 
                                          wrist_rot_traj)):
        if per_frame:
            large_error = False
        ik_result = robot_chain.inverse_kinematics(
            target_position=pos,
            target_orientation=rotation_matrix,
            orientation_mode="all",
            initial_position=initial_joint_angles
        )

        # check ik with fk
        joint_angles = ik_result
        result_pos, result_rotmat = verify_fk(robot_chain, joint_angles)
        result_quat = R.from_matrix(result_rotmat).as_quat()

        target_quat = R.from_matrix(rotation_matrix).as_quat()
        r_target = R.from_quat(target_quat)
        r_computed = R.from_quat(result_quat)
        relative_rotation = r_target.inv() * r_computed
        error_angle = relative_rotation.magnitude()

        if error_angle > 1e-4:
            large_error  = True
        error_pos = np.linalg.norm(result_pos-pos)
        if error_pos > 1e-4:
            print("ik_error_pos > 1e-4:",error_pos)
            large_error = True

        # update init dof for smooth
        #initial_joint_angles = joint_angles.copy()
        # remove the OriginLink
        arm_dof = ik_result[1:]
        if large_error and per_frame:
            deleted_frame_idx.append(idx)
        else:
            arm_dof_trajectory.append(arm_dof)
        
        #if idx % 10 == 0:
        #    print(f"Calculating IK for step {idx}/{len(wrist_pos_traj)}")

    arm_dof_trajectory = np.array(arm_dof_trajectory)
    arm_dof_trajectory = torch.tensor(arm_dof_trajectory)
    if per_frame:
        return arm_dof_trajectory, large_error, deleted_frame_idx
    else:
        return arm_dof_trajectory, large_error


def verify_fk(robot_chain, joint_angles):
    frames = robot_chain.forward_kinematics(joint_angles, full_kinematics=True)
    end_effector_frame = frames[-1]
    position_computed = end_effector_frame[:3, 3]
    rotation_computed = end_effector_frame[:3, :3]

    return position_computed, rotation_computed

def smooth_quat_seq(quat_seq):
    """
    ensure that a sequence of quaternions (representing rotations) 
    is smoothly interpolated, avoiding any sudden flips in orientation. 
    """
    n = quat_seq.size(0)

    for i in range(1, n):
        dot_product = torch.dot(quat_seq[i-1], quat_seq[i])
        if dot_product < 0: #If the dot product is negative, quaternions is greater than 90 degrees, meaning they are pointing in opposite directions
            quat_seq[i] *=-1

    return quat_seq


def global_to_robot_local(pos1, quat1, pos2, quat2):
    rel_pos = pos1-pos2
    quat2_conj = quat2.clone()
    quat2_conj[:3] = -quat2_conj[:3]
    local_pos = quat_rotate(quat2_conj.unsqueeze(0), rel_pos)
    local_quat = torch_utils.quat_multiply(quat2_conj, quat1)
    return local_pos, local_quat


def load_grasp_traj(seq_path, device='cpu'):

    graspxl_data = torch.load(seq_path)#[:traj_length]
    if isinstance(graspxl_data, torch.Tensor):
        graspxl_data = torch.load(seq_path, weights_only=True).to(device)#[:traj_length]
        grasp_traj = {
        'root_pos': graspxl_data[:, 0:3].clone(), # torch.Size([1, 3])
        'root_rot': smooth_quat_seq(graspxl_data[:, 3:7].clone()), # torch.Size([1, 4])
        'wrist_dof': graspxl_data[:, 7:13].clone(),
        'fingers_dof': graspxl_data[:, 13:58].clone(),
        'keybody_pos': graspxl_data[:, 58:58+15*3].clone(),
        'obj_pos': graspxl_data[:, 103:106].clone(),
        'obj_pos_vel': graspxl_data[:, 103:106].clone(),
        'obj_rot': smooth_quat_seq(graspxl_data[:, 106:110].clone()),
        'obj2_pos': graspxl_data[:, 103:106].clone()*0+10., #graspxl_data[:, 110:113].clone(),
        'obj2_rot': smooth_quat_seq(graspxl_data[:, 106:110].clone())*0, #smooth_quat_seq(graspxl_data[:, 113:117].clone()),
        'contact1': graspxl_data[:, 117:118].clone(),
        'contact2':  torch.zeros((graspxl_data.shape[0], 1)) #graspxl_data[:, 118:119].clone()
        }
    elif isinstance(graspxl_data, dict):

        graspxl_data = torch.load(seq_path)#[:traj_length]
        if graspxl_data['obj_pos_vel'] is None:
            obj_pos_vel = graspxl_data['obj_pos_vel']
        else:
            obj_pos_vel = graspxl_data['obj_pos_vel'].clone().float()
        grasp_traj = {
        'root_pos': graspxl_data['root_pos'].clone().float(), # torch.Size([1, 3])
        'root_rot': smooth_quat_seq(graspxl_data['root_rot'].clone().float()), # torch.Size([1, 4])
        'wrist_dof': graspxl_data['wrist_dof'].clone().float(),
        'fingers_dof': graspxl_data['fingers_dof'].clone().float(),
        'keybody_pos': graspxl_data['body_pos'].clone().float(),
        'obj_pos': graspxl_data['obj_pos'].clone().float(),
        'obj_pos_vel': obj_pos_vel,
        'obj_rot': smooth_quat_seq(graspxl_data['obj_rot'].clone().float()),
        'obj2_pos': graspxl_data['obj2_pos'].clone().float()*0+1., #graspxl_data[:, 110:113].clone(),
        'obj2_rot': smooth_quat_seq(graspxl_data['obj2_rot'].clone().float())*0, #smooth_quat_seq(graspxl_data[:, 113:117].clone()),
        'contact1': graspxl_data['contact1'].clone().float(),
        'contact2':  torch.zeros((graspxl_data['contact2'].shape[0], 1)), #graspxl_data[:, 118:119].clone()
        }
    return grasp_traj



def random_quaternion(batch_size=1):
    #torch.manual_seed(42)         # Set PyTorch random seed

    # 生成随机单位向量作为轴
    axis = torch.randn(batch_size, 3)
    axis = axis / torch.norm(axis, dim=1, keepdim=True)

    # 生成随机角度 [0, 2π]
    angle = torch.rand(batch_size, 1) * 2 * torch.pi

    # 计算四元数
    sin_half = torch.sin(angle / 2)
    cos_half = torch.cos(angle / 2)

    quat = torch.cat([
    axis[:, 0:1] * sin_half, # x
    axis[:, 1:2] * sin_half, # y
    axis[:, 2:3] * sin_half, # z
    cos_half # w
    ], dim=1)

    return quat

def random_position(x_min, x_max, y_min, y_max, z_min, z_max):
    #torch.manual_seed(42)         # Set PyTorch random seed

    coords = torch.rand(1,3)
    ranges = torch.tensor([
        [x_min,x_max],
        [y_min,y_max],
        [z_min,z_max]]
    )
    return coords * (ranges[:,1]-ranges[:,0]) + ranges[:,0]

def interpolate_pose(
        trans1,
        q1,
        trans2,
        q2,
        t):
    trans = (1 - t) * trans1 + t * trans2 # lerp = weighted average between the two 3D point or can view as strting_point + distance_traveled = strting_point + protion * distance = trans1 + t(trans2 - trans1)
    q_interp = torch_utils.slerp(q1, q2, torch.tensor(t)) #slerp function in PyTorch is used for spherical linear interpolation between two quaternions or vectors on a sphere

    return trans, q_interp

def generate_trajectory_between_poses(
        init_obj_pos,
        init_obj_rot,
        target_obj_pos,
        target_obj_rot,
        num_steps):
    """
    Generate a list of poses interpolated between pose_start and pose_end.
    """
    obj_pos_traj = []
    obj_rot_traj = []
    for i in range(num_steps):
        t = i / (num_steps - 1)
        obj_pos, obj_rot = interpolate_pose(
        init_obj_pos,
        init_obj_rot,
        target_obj_pos,
        target_obj_rot,
        t)
        obj_pos_traj.append(obj_pos)
        obj_rot_traj.append(obj_rot)
    return obj_pos_traj, obj_rot_traj

def AinB_local_to_global(posA_local, quatA_local, posB_global, quatB_global):
    global_quat = torch_utils.quat_multiply(quatB_global, quatA_local)
    rotated_pos = quat_rotate(quatB_global.unsqueeze(0).expand_as(quatA_local), posA_local)
    global_pos = rotated_pos+posB_global
    return global_pos, global_quat

def AinB_global_to_local(posA_global, quatA_global, posB_global, quatB_global):
    rel_pos = posA_global-posB_global
    quatB_global_conj = quatB_global.clone()
    quatB_global_conj[:3] = -quatB_global_conj[:3]
    posA_local = quat_rotate(quatB_global_conj.unsqueeze(0), rel_pos)
    quatA_local = torch_utils.quat_multiply(quatB_global_conj, quatA_global)
    return posA_local, quatA_local

class SkillGenerator:
    def __init__(self, obj_name=None,
                    basic_grasps_path=None,
                    output_path=None,
                    operation_space_ranges=None,
                    device='cpu',
                    hand_model=None,
                    num_data_samples=-1
                    ):
        self.obj_name = obj_name
        self.paths = {
            'basic_grasps': basic_grasps_path,
            'output': output_path,
        }
        self.output_name = f"{self.skill_code}_{self.obj_name}"
        self.device = torch.device(device)
        self.basic_trajectories = []
        self.operation_space = operation_space_ranges or {
            'x_offset': (-0.5, 0.5),
            'y_offset': (-0.5, 0.5),
            'z_range': (0.2, 1.0)
        }
        self.hand_model = hand_model
        self.traj_name = []
        self.num_data_samples = num_data_samples

    def load_basic_trajectories(self):
        """Load all basic grasp trajectories from .pt files"""
        all_seqs = glob.glob(os.path.join(self.paths['basic_grasps'], '*.pt'))
        for seq_path in all_seqs:
            self.basic_trajectories.append(load_grasp_traj(seq_path))
            
    def calculate_wrist_dofs(self, wrist_pos_traj_new, global_rot, root_pos, root_rot, per_frame=False):
        """Convert global wrist poses back to DOFs using inverse kinematics"""

        wrist_mat_traj_new = quat2rotmat(global_rot)
        root_rot_euler = get_euler_xyz(root_rot.unsqueeze(0))
        # find the dof

        if per_frame:
            wrist_dof_traj_new, large_error, deleted_frame_id = inverse_kinematics(
                    wrist_pos_traj_new.clone().numpy(),
                    wrist_mat_traj_new.clone().numpy(),
                    [root_rot_euler[0].clone().numpy(),
                    root_rot_euler[1].clone().numpy(),
                    root_rot_euler[2].clone().numpy()],
                    root_pos.clone().numpy(),
                    [0,0,0],
                    [0,0,0],
                    per_frame=per_frame
            )
            return wrist_dof_traj_new, large_error, deleted_frame_id
        else:
            wrist_dof_traj_new = inverse_kinematics(
                    wrist_pos_traj_new.clone().numpy(),
                    wrist_mat_traj_new.clone().numpy(),
                    [root_rot_euler[0].clone().numpy(),
                    root_rot_euler[1].clone().numpy(),
                    root_rot_euler[2].clone().numpy()],
                    root_pos.clone().numpy(),
                    [0,0,0],
                    [0,0,0],
                    per_frame=per_frame
            )
            return wrist_dof_traj_new
    
    def _generate_random_pos(self, base_pos):
        """Generate random position in operation space"""
        # Define operation space
        x_min = base_pos[0] + self.operation_space['x_offset'][0]
        x_max = base_pos[0] + self.operation_space['x_offset'][1]
        y_min = base_pos[1] + self.operation_space['y_offset'][0]
        y_max = base_pos[1] + self.operation_space['y_offset'][1]
        z_min, z_max =  self.operation_space['z_range']

        target_pos = random_position(x_min, x_max, y_min, y_max, z_min, z_max) #wanmingggg
        target_quat = random_quaternion()
        return target_pos, target_quat       
    
    def _calculate_wrist_trajectory(self, grasp_traj, grasp_idx, obj_pos_traj, obj_rot_traj):
        """Calculate wrist trajectory based on object trajectory"""
        # Initialization and coordinate transformations
        init_data = {key: grasp_traj[key][grasp_idx].clone() for key in ['obj_pos', 'obj_rot', 'wrist_dof', 'root_pos', 'root_rot']}
        
        # Convert wrist rotation to quaternion
        init_wrist_rot_quat_local = quat_from_euler_xyz_extrinsic(
                                    init_data['wrist_dof'][3],
                                    init_data['wrist_dof'][4],
                                    init_data['wrist_dof'][5]).unsqueeze(0)   
        # Get global wrist pose
        wrist_pos_global, wrist_rot_global = AinB_local_to_global(
            init_data['wrist_dof'][:3].unsqueeze(0),
            init_wrist_rot_quat_local,
            init_data['root_pos'],
            init_data['root_rot']
        )

        # Calculate relative transform between wrist and object
        rel_trans, rel_rot = AinB_global_to_local(
            wrist_pos_global,
            wrist_rot_global,
            init_data['obj_pos'],
            init_data['obj_rot']
        )

        # Generate wrist trajectory
        wrist_pos_traj_global = []
        wrist_rot_traj_global = []
        for obj_pos, obj_rot in zip(obj_pos_traj, obj_rot_traj):
            pos, rot = AinB_local_to_global(rel_trans, rel_rot, obj_pos.squeeze(0), obj_rot.squeeze(0))
            wrist_pos_traj_global.append(pos.squeeze())
            wrist_rot_traj_global.append(rot)

        wrist_pos_traj_global,  wrist_rot_traj_global= torch.stack(wrist_pos_traj_global), torch.stack(wrist_rot_traj_global)

        return self.calculate_wrist_dofs(wrist_pos_traj_global, wrist_rot_traj_global, 
                                          init_data['root_pos'], init_data['root_rot'])

    def duplicate_each_frame(self, data_dict, num_frames=None, duplicate_time = 60):
        """
        Duplicates each frame 60 times in all tensors of the input dictionary.
        
        Args:
            data_dict: Dictionary containing motion data with tensors
            num_frames: Original number of frames (optional, inferred if None)
        
        Returns:
            New dictionary with each frame repeated 60 times consecutively
        """
        if num_frames is None:
            # Find the first tensor to infer the number of frames
            for val in data_dict.values():
                if isinstance(val, torch.Tensor):
                    num_frames = val.shape[0]
                    break
        
        new_data = {}
        for key, val in data_dict.items():
            if val is None:
                new_data[key] = None
            elif isinstance(val, torch.Tensor):
                # Repeat each frame 60 times along dim=0 (time axis)
                new_data[key] = val.repeat_interleave(duplicate_time, dim=0)
            else:
                # Copy non-tensor values as-is
                new_data[key] = val
        
        return new_data
        
    def duplicate_last_frame(self, generated_data, num_copies=60):
        num_copies = 60
        # Create a new dictionary to hold the modified data
        extended_data = {}

        # Iterate over each key in the original generated_data
        for key, value in generated_data.items():
            # Copy the last frame for the specified number of times
            if key != 'obj_pos_vel' or  generated_data['obj_pos_vel'] is not None:
                # Copy the last frame for the specified number of times
                last_frame = value[-1:]  # Get the last frame
                extended_data[key] = torch.cat((value, last_frame.repeat(num_copies, 1)), dim=0)
            else: 
                extended_data[key] = None
        return extended_data
    
    def _save_transformed_data(self, data_dict, idx, grasp_idx):
        """Save transformed data to file"""
        ######################################
        # print(data_dict['wrist_dof'])
        # import sys
        # print("Ending the program...")
        # sys.exit()
        #####################################  
        save_path = os.path.join(
            self.paths['output'],
            f'{self.output_name}_{grasp_idx}_{idx}.pt'
        )
        torch.save(data_dict, save_path)
        print(f"\nSaved {self.output_name}_{grasp_idx}_{idx}.pt!")
    


    def _generate_trajectory_between_poses(self, target_wrist_pos, target_wrist_rot, target_init_wrist_pos, num_steps): # similar to _generate_object_trajectory
        """Generate pre_grasp trajectory"""
        # Generate trajectory between poses
        wrist_pos_traj, wrist_rot_traj = generate_trajectory_between_poses(
            target_init_wrist_pos, target_wrist_rot,
            target_wrist_pos.unsqueeze(0), target_wrist_rot.unsqueeze(0),
            num_steps
        )
        wrist_pos_traj = torch.stack(wrist_pos_traj).squeeze(1)
        wrist_rot_traj = torch.stack(wrist_rot_traj).squeeze(1)
        return wrist_pos_traj, wrist_rot_traj
    