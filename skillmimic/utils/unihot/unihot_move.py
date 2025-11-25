
from isaacgym.torch_utils import *
import torch_utils
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
                init_dof_rxryrz):

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
            print("ik_error_angle > 1e-4:",error_angle)
        error_pos = np.linalg.norm(result_pos-pos)
        if error_pos > 1e-4:
            print("ik_error_pos > 1e-4:",error_pos)

        # update init dof for smooth
        initial_joint_angles = joint_angles.copy()

        # remove the OriginLink
        arm_dof = ik_result[1:]
        arm_dof_trajectory.append(arm_dof)

        if idx % 10 == 0:
            print(f"Calculating IK for step {idx}/{len(wrist_pos_traj)}")

    arm_dof_trajectory = np.array(arm_dof_trajectory)
    arm_dof_trajectory = torch.tensor(arm_dof_trajectory)
    return arm_dof_trajectory

def verify_fk(robot_chain, joint_angles):
    frames = robot_chain.forward_kinematics(joint_angles, full_kinematics=True)
    end_effector_frame = frames[-1]
    position_computed = end_effector_frame[:3, 3]
    rotation_computed = end_effector_frame[:3, :3]

    return position_computed, rotation_computed

def smooth_quat_seq(quat_seq):
    n = quat_seq.size(0)

    for i in range(1, n):
        dot_product = torch.dot(quat_seq[i-1], quat_seq[i])
        if dot_product < 0:
            quat_seq[i] *=-1

    return quat_seq


def global_to_robot_local(pos1, quat1, pos2, quat2):
    rel_pos = pos1-pos2
    quat2_conj = quat2.clone()
    quat2_conj[:3] = -quat2_conj[:3]
    local_pos = quat_rotate(quat2_conj.unsqueeze(0), rel_pos)
    local_quat = torch_utils.quat_multiply(quat2_conj, quat1)
    return local_pos, local_quat

def robot_local_to_global(pos1, quat1, pos2, quat2):
    global_quat = torch_utils.quat_multiply(quat2, quat1)
    rotated_pos = quat_rotate(quat2.unsqueeze(0).expand_as(quat1), pos1)
    global_pos = rotated_pos+pos2
    return global_pos, global_quat

def load_grasp_traj(seq_path, device='cpu'):
    """
    加载抓取轨迹并进行预处理。

    参数:
    seq_path (str): 抓取轨迹文件路径。
    device (str): 设备类型，默认为'cpu'。
    traj_length (int): 需要加载的轨迹长度，默认为20。

    返回:
    dict: 包含各个轨迹分量的字典。
    """
    print("开始加载抓取轨迹...")
    graspxl_data = torch.load(seq_path, weights_only=True).to(device)#[:traj_length]

    grasp_traj = {
    'root_pos': graspxl_data[:, 0:3].clone(),
    'root_rot': smooth_quat_seq(graspxl_data[:, 3:7].clone()),
    'wrist_dof': graspxl_data[:, 7:13].clone(),
    'fingers_dof': graspxl_data[:, 13:58].clone(),
    'keybody_pos': graspxl_data[:, 58:58+15*3].clone(),
    'obj_pos': graspxl_data[:, 103:106].clone(),
    'obj_rot': smooth_quat_seq(graspxl_data[:, 106:110].clone()),
    'obj2_pos': graspxl_data[:, 110:113].clone(),
    'obj2_rot': smooth_quat_seq(graspxl_data[:, 113:117].clone()),
    'contact1': graspxl_data[:, 117:118].clone(),
    'contact2': graspxl_data[:, 118:119].clone()
    }

    print("抓取轨迹加载完成。")
    return grasp_traj

def random_quaternion(batch_size=1):
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
    trans = (1 - t) * trans1 + t * trans2
    q_interp = torch_utils.slerp(q1, q2, torch.tensor(t))
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


if __name__ == "__main__":

    device='cpu'
    #define the range of operation space
    x_min = -1.6
    x_max = +1.6
    y_min = -1.6
    y_max = +1.6
    z_min = 0.2
    z_max = +1.6

    basic_grasp_frames = "/home/hkust/yinhuai/unihot_dataset/basic_grasp_frames"
    all_grasps = glob.glob(basic_grasp_frames + '/*.pt')
    for idx, seq_path in enumerate(all_grasps):
        grasp_traj = load_grasp_traj(seq_path, device=device)
        num_grasps = grasp_traj['root_pos'].shape[0]

        x_min += grasp_traj['root_pos'][-1][0]
        x_max += grasp_traj['root_pos'][-1][0]
        y_min += grasp_traj['root_pos'][-1][1]
        y_max += grasp_traj['root_pos'][-1][1]

        #generate a move traj for each unique grasp
        for i in range(num_grasps):

            target_obj_pos1 = random_position(x_min, x_max, y_min, y_max, z_min, z_max)
            target_obj_pos2 = random_position(x_min, x_max, y_min, y_max, z_min, z_max)
            target_obj_quat1 = random_quaternion()
            target_obj_quat2 = random_quaternion()
            num_steps = 100

            obj_pos_traj, obj_rot_traj = generate_trajectory_between_poses(
                        target_obj_pos1.to(device),
                        target_obj_quat1.to(device),
                        target_obj_pos2.to(device),
                        target_obj_quat2.to(device),
                        num_steps)
            
            # get relative pose between wrist and obj
            init_obj_pos = grasp_traj['obj_pos'][i].clone()
            init_obj_rot = grasp_traj['obj_rot'][i].clone()
            init_wrist_pos = grasp_traj['wrist_dof'][i][:3].clone()
            init_wrist_rot = grasp_traj['wrist_dof'][i][3:6].clone()
            init_root_pos = grasp_traj['root_pos'][i].clone()
            init_root_rot = grasp_traj['root_rot'][i].clone()

            init_obj_rot_mat = quat2rotmat(init_obj_rot)
            init_wrist_rot_quat_local = quat_from_euler_xyz_extrinsic(
                                        init_wrist_rot[0],
                                        init_wrist_rot[1],
                                        init_wrist_rot[2])
            init_wrist_rot_mat_local = quat2rotmat(init_wrist_rot_quat_local)
            init_root_rot_mat = quat2rotmat(init_root_rot)
            init_wrist_rot_mat_global = init_root_rot_mat @ init_wrist_rot_mat_local
            init_wrist_pos_global = init_root_pos + init_root_rot_mat @ init_wrist_pos

            rel_rot = init_obj_rot_mat.T @ init_wrist_rot_mat_global
            rel_trans = init_obj_rot_mat.T @ (init_wrist_pos_global - init_obj_pos)

            wrist_pos_traj_global = []
            wrist_rot_traj_global = []

            # get wrist traj from object traj
            for j in range(len(obj_pos_traj)):
                obj_pos = obj_pos_traj[j]
                obj_rot = obj_rot_traj[j]

                obj_rot_mat = quat2rotmat(obj_rot)

                wrist_rot_global = obj_rot_mat @ rel_rot
                wrist_pos_global = obj_rot_mat @ rel_trans + obj_pos

                wrist_rot_traj_global.append(wrist_rot_global)
                wrist_pos_traj_global.append(wrist_pos_global)

            wrist_pos_traj_global = torch.stack(wrist_pos_traj_global)
            wrist_rot_traj_global = torch.stack(wrist_rot_traj_global)

            obj_pos_traj = torch.stack(obj_pos_traj).squeeze(1)
            obj_rot_traj = torch.stack(obj_rot_traj).squeeze(1)
            obj_rot_traj = smooth_quat_seq(obj_rot_traj)

            # for each wrist pose, compute the ik to get wirst dof
            '''wrist dof: x, y, z, rx, ry, rz'''

            root_rot_euler = get_euler_xyz(init_root_rot.unsqueeze(0))
            wrist_dof_traj = inverse_kinematics(
                    wrist_pos_traj_global.clone().numpy(),
                    wrist_rot_traj_global.clone().numpy(),
                    [root_rot_euler[0].clone().numpy(),
                    root_rot_euler[1].clone().numpy(),
                    root_rot_euler[2].clone().numpy()],
                    init_root_pos.clone().numpy(),
                    [0,0,0],
                    [0,0,0])
            wrist_dof_traj = wrist_dof_traj.float()
    
            root_pos = grasp_traj['root_pos'][i].repeat(num_steps, 1)
            root_rot = grasp_traj['root_rot'][i].repeat(num_steps, 1)
            fingers_dof = grasp_traj['fingers_dof'][i].repeat(num_steps, 1)
            obj2_pos_traj = torch.zeros_like(obj_pos_traj).to(device)
            obj2_pos_traj[:,2] += 1.
            obj2_rot_traj = torch.zeros_like(obj_rot_traj).to(device)
            obj2_rot_traj[:,3] = 1.
            contact1 = torch.ones_like(obj_pos_traj[:,:1]).to(device)
            contact2 = torch.zeros_like(obj_pos_traj[:,:1]).to(device)

            hoi_data = torch.cat((
            root_pos,
            root_rot,
            wrist_dof_traj,
            fingers_dof,
            torch.zeros(num_steps, 15*3),
            obj_pos_traj,
            obj_rot_traj,
            obj2_pos_traj,
            obj2_rot_traj,
            contact1,
            contact2
            ), dim=-1)

            # 保存最终轨迹数据
            os.makedirs('/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/move',exist_ok=True)
            save_path = f'/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/move/002_box01_{idx}_{i}.pt'
            torch.save(hoi_data.clone().float(), save_path)
            print(f"\nsave 002_box01_{idx}_{i}.pt！\n")

    print("\n所有步骤完成！\n")