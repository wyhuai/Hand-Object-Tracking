
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

def generate_manipulation_trajectory(
        init_obj_pos, 
        init_obj_rot, 
        init_wrist_pos, 
        init_wrist_rot, 
        init_root_pos,
        init_root_rot,
        target_obj_pos,
        target_obj_rot,
        num_steps=100):

    # Generate object trajectory: init -> target
    obj_pos_traj, obj_rot_traj = generate_trajectory_between_poses(
        init_obj_pos,
        init_obj_rot,
        target_obj_pos,
        target_obj_rot,
        num_steps)
    # print('obj_pos_traj',obj_pos_traj)
    # print('obj_rot_traj',obj_rot_traj)

    # 初始旋转矩阵
    init_obj_rot_mat = quat2rotmat(init_obj_rot)

    # 手腕初始局部旋转矩阵（相对于根部）
    init_wrist_rot_quat_local = quat_from_euler_xyz_extrinsic(
    init_wrist_rot[0],
    init_wrist_rot[1],
    init_wrist_rot[2]
    )
    init_wrist_rot_mat_local = quat2rotmat(init_wrist_rot_quat_local)

    # 根部初始旋转矩阵
    init_root_rot_mat = quat2rotmat(init_root_rot)

    # 手腕的全局初始旋转矩阵
    init_wrist_rot_mat_global = init_root_rot_mat @ init_wrist_rot_mat_local

    # 手腕的全局初始位置（修正使用 init_root_rot_mat 而非其转置）
    init_wrist_pos_global = init_root_pos + init_root_rot_mat @ init_wrist_pos

    # 计算物体相对于手腕的初始旋转和平移
    rel_rot = init_obj_rot_mat.T @ init_wrist_rot_mat_global
    rel_trans = init_obj_rot_mat.T @ (init_wrist_pos_global - init_obj_pos)

    # 初始化轨迹列表
    wrist_pos_traj_global = []
    wrist_rot_traj_global = []
    wrist_pos_traj_local = []
    wrist_rot_traj_local = []

    for i in range(len(obj_pos_traj)):
        obj_pos = obj_pos_traj[i]
        obj_rot = obj_rot_traj[i]

        # 当前物体的旋转矩阵
        obj_rot_mat = quat2rotmat(obj_rot)

        # 计算手腕的全局旋转和位置，保持相对姿态不变
        wrist_rot_global = obj_rot_mat @ rel_rot
        wrist_pos_global = obj_rot_mat @ rel_trans + obj_pos

        wrist_rot_traj_global.append(wrist_rot_global)
        wrist_pos_traj_global.append(wrist_pos_global)

        # 计算手腕的局部旋转和位置（相对于根部）
        wrist_rot_local = init_root_rot_mat.T @ wrist_rot_global
        wrist_pos_local = init_root_rot_mat.T @ (wrist_pos_global - init_root_pos)

        wrist_rot_traj_local.append(wrist_rot_local)
        wrist_pos_traj_local.append(wrist_pos_local)

    # wrist_rot_traj = wrist_rot_traj_global
    # wrist_pos_traj = wrist_pos_traj_global
    # wrist_rot_traj = wrist_rot_traj_local
    # wrist_pos_traj = wrist_pos_traj_local

    # wrist_pos_traj = torch.stack(wrist_pos_traj)
    # wrist_rot_traj = torch.stack(wrist_rot_traj)
    wrist_pos_traj_local = torch.stack(wrist_pos_traj_local)
    wrist_rot_traj_local = torch.stack(wrist_rot_traj_local)
    wrist_pos_traj_global = torch.stack(wrist_pos_traj_global)
    wrist_rot_traj_global = torch.stack(wrist_rot_traj_global)
    # print('\n')
    # print('manipulation final wrist pos',wrist_pos_traj[-1])
    # print('manipulation final wrist rotmat',wrist_rot_traj[-1])
    # print('\n')


    ### debug ik and transformation
    robot_chain = create_robot_chain([0,0,0], [0,0,0])
    initial_joint_angles = [0,
                        init_wrist_pos[0],
                        init_wrist_pos[1],
                        init_wrist_pos[2],
                        init_wrist_rot[0],
                        init_wrist_rot[1],
                        init_wrist_rot[2]
                        ]
    test_pos, test_mat = verify_fk(robot_chain, initial_joint_angles)
    test_quat_ik = R.from_matrix(test_mat).as_quat()
    test_quat_given = quat_from_euler_xyz_extrinsic(*init_wrist_rot)
    # test_quat_given = R.from_matrix(wrist_rot_traj_local[0]).as_quat()
    test_quat_error = test_quat_ik - test_quat_given.numpy()
    print("test_quat_error:",test_quat_error)





    # plot_trajectories(wrist_pos_traj, obj_pos_traj)
    obj_pos_traj = torch.stack(obj_pos_traj)
    obj_rot_traj = torch.stack(obj_rot_traj)

    #check grasp pose consistency
    validate_grasp_pose_consistency(
        wrist_pos_traj_global.clone().numpy(),
        wrist_rot_traj_global.clone().numpy(),
        obj_pos_traj.clone().numpy(),
        obj_rot_traj.clone().numpy(),
        rel_rot.clone().numpy(),
        rel_trans.clone().numpy(),
        num_steps
    )

    # For each wrist pose, compute the ik
    '''wrist dof: x, y, z, rx, ry, rz'''

    root_rot_euler = get_euler_xyz(init_root_rot.unsqueeze(0))
    wrist_dof_traj = inverse_kinematics(
            wrist_pos_traj_local.clone().numpy(),
            wrist_rot_traj_local.clone().numpy(),
            # [root_rot_euler[0].clone().numpy(),
            #  root_rot_euler[1].clone().numpy(),
            #  root_rot_euler[2].clone().numpy()],
            # init_root_pos.clone().numpy(),
            [0,0,0],
            [0,0,0],
            init_wrist_pos.clone().numpy(),
            init_wrist_rot.clone().numpy())

    # check traj coherence
    last_dof_of_prev_traj = torch.cat((
        init_wrist_pos,
        init_wrist_rot),dim=-1)
    validate_traj_coherence(
        last_dof_of_prev_traj,
        # last_wrist_pos_of_prev_traj,
        # last_wrist_rot_of_prev_traj,
        init_dof_of_curr_traj=wrist_dof_traj[0],
        # init_wrist_pos_of_curr_traj,
        # init_wrist_rot_of_curr_traj,
    )

    return obj_pos_traj, obj_rot_traj, wrist_dof_traj


def validate_traj_coherence(
        last_dof_of_prev_traj,
        # last_wrist_pos_of_prev_traj,
        # last_wrist_rot_of_prev_traj,
        init_dof_of_curr_traj
        # init_wrist_pos_of_curr_traj,
        # init_wrist_rot_of_curr_traj,
    ):
    print("Start validating coherence between last traj and current traj")
    error_dof = np.linalg.norm(last_dof_of_prev_traj - init_dof_of_curr_traj)
    # # global pos
    # error_wrist_pos = np.linalg.norm(last_wrist_pos_of_prev_traj - init_wrist_pos_of_curr_traj)
    # # global angle
    # error_wrist_rot = R.from_quat(last_wrist_rot_of_prev_traj).inv() * R.from_quat(init_wrist_rot_of_curr_traj)
    # error_wrist_angle = error_wrist_rot.magnitude()
    if error_dof > 1e-4:
            print("error_dof > 1e-4:",error_dof)
    # if error_wrist_pos > 1e-4:
    #         print("error_wrist_pos > 1e-4:",error_wrist_pos)
    # if error_wrist_angle > 1e-4:
    #         print("error_wrist_angle > 1e-4:",error_wrist_angle)
    print("Finish validating coherence between last traj and current traj")
    return


def validate_grasp_pose_consistency(
        wrist_pos_traj,
        wrist_rot_traj,
        obj_pos_traj,
        obj_rot_traj,
        rel_rot,
        rel_trans,
        num_steps,
        position_threshold=1e-3,
        rotation_threshold=1e-3):
    
    print("开始manipulation轨迹验证...")

    len_traj = wrist_pos_traj.shape[0]

    # # 验证腕部轨迹连续性
    # for i in range(1, len_traj):
    #     wrist_prev, wrist_prev_quat = wrist_traj[i-1]
    #     wrist_curr, wrist_curr_quat = wrist_traj[i]
    #     pos_diff = np.linalg.norm(wrist_curr - wrist_prev)
    #     rot_diff = R.from_quat(wrist_prev_quat).inv() * R.from_quat(wrist_curr_quat)
    #     angle_diff = rot_diff.magnitude()
    #     if pos_diff > position_threshold:
    #         print(f"Wrist位置跳变在步数 {i}：位移差={pos_diff}")
    #     if angle_diff > rotation_threshold:
    #         print(f"Wrist旋转跳变在步数 {i}：角度差={angle_diff} 弧度")
    
    # # 验证物体轨迹连续性
    # for i in range(1, len_traj):
    #     obj_prev, obj_prev_quat = object_traj[i-1]
    #     obj_curr, obj_curr_quat = object_traj[i]
    #     pos_diff = np.linalg.norm(obj_curr - obj_prev)
    #     rot_diff = R.from_quat(obj_prev_quat).inv() * R.from_quat(obj_curr_quat)
    #     angle_diff = rot_diff.magnitude()
    #     if pos_diff > position_threshold:
    #         print(f"Object位置跳变在步数 {i}：位移差={pos_diff}")
    #     if angle_diff > rotation_threshold:
    #         print(f"Object旋转跳变在步数 {i}：角度差={angle_diff} 弧度")
    
    # 验证抓取相对位姿
    rel_quat = R.from_matrix(rel_rot).as_quat()
    rel_rot = R.from_quat(rel_quat)
    for i in range(len_traj):
        obj_pos = obj_pos_traj[i]
        # obj_quat = R.from_matrix(obj_rot_traj[i]).as_quat()
        obj_rot = R.from_quat(obj_rot_traj[i])
        expected_wrist_rot = obj_rot * rel_rot
        expected_wrist_pos = obj_rot.apply(rel_trans) + obj_pos
        actual_wrist_pos = wrist_pos_traj[i]
        actual_wrist_quat = R.from_matrix(wrist_rot_traj[i]).as_quat()
        pos_diff = np.linalg.norm(expected_wrist_pos - actual_wrist_pos)
        rot_diff = expected_wrist_rot.inv() * R.from_quat(actual_wrist_quat)
        angle_diff = rot_diff.magnitude()
        if pos_diff > position_threshold:
            print(f"Wrist位置不匹配在步数 {i}：期望位置={expected_wrist_pos}, 实际位置={actual_wrist_pos}, 位移差={pos_diff}")
        if angle_diff > rotation_threshold:
            print(f"Wrist旋转不匹配在步数 {i}：角度差={angle_diff} 弧度")
    
    print("manipulation轨迹验证完成。")


# def generate_release_wrist_trajectory(
#     target_wrist_pos, # 目标手腕xyz dof
#     target_wrist_euler, # 目标手腕rxryrz dof
#     init_root_pos,
#     init_root_quat, # 初始根旋转四元数 (4,)
#     wrist_dof_traj, # 手腕自由度轨迹，形状为 (T, 6) [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
#     ):
#     wrist_pos_traj = wrist_dof_traj[:,:3].clone()
#     wrist_euler_traj = wrist_dof_traj[:,3:6].clone()
#     wrist_quat_traj = quat_from_euler_xyz_extrinsic(
#                         wrist_euler_traj[:,0],
#                         wrist_euler_traj[:,1],
#                         wrist_euler_traj[:,2],)

#     wrist_quat_traj = torch_utils.quat_multiply(
#         init_root_quat.unsqueeze(0).expand_as(wrist_quat_traj),
#         wrist_quat_traj)
#     wrist_pos_traj = quat_rotate(
#         init_root_quat.unsqueeze(0).expand_as(wrist_quat_traj),
#         wrist_pos_traj) + init_root_pos

#     target_wrist_quat = quat_from_euler_xyz_extrinsic(
#                         target_wrist_euler[0],
#                         target_wrist_euler[1],
#                         target_wrist_euler[2],)
#     target_wrist_quat_global = torch_utils.quat_multiply(
#         init_root_quat.unsqueeze(0),
#         target_wrist_quat.unsqueeze(0))
#     target_wrist_pos_global = quat_rotate(
#         init_root_quat.unsqueeze(0),
#         target_wrist_pos.unsqueeze(0)) + init_root_pos

#     # === 1. 计算需要应用的旋转和位移 ===
#     q_initial = wrist_quat_traj[0].clone() # 初始对象旋转四元数 [qx, qy, qz, qw]
#     q_initial_inv = quat_conjugate(q_initial) # 四元数的共轭

#     q_diff = torch_utils.quat_multiply(target_wrist_quat_global, q_initial_inv) # 旋转差异

#     pos_initial = target_wrist_pos_global[0].clone() 
#     pos_target = target_wrist_pos_global.clone()
#     pos_diff = pos_target - quat_rotate(q_diff, pos_initial.unsqueeze(0)).squeeze(0)

#     # === 3. 应用变换到手腕轨迹 ===
#     wrist_quat_traj_new_global = torch_utils.quat_multiply(q_diff.expand_as(wrist_quat_traj), wrist_quat_traj)
#     wrist_pos_traj_new_global = quat_rotate(q_diff.expand_as(wrist_quat_traj), wrist_pos_traj) + pos_diff
#     wrist_mat_traj_new_global = quat2rotmat(wrist_quat_traj_new_global)

#     print("target_wrist_pos_global[0]",target_wrist_pos_global[0])
#     print("wrist_pos_traj_new_global[0]",wrist_pos_traj_new_global[0])
    
#     print("target_wrist_quat_global[0]",target_wrist_quat_global[0])
#     print("wrist_quat_traj_new_global[0]",wrist_quat_traj_new_global[0])

#     # # ####################global transform
#     # wrist_quat_traj_new_global = torch_utils.quat_multiply(
#     #     init_root_quat.unsqueeze(0).expand_as(wrist_quat_traj_new),
#     #     wrist_quat_traj_new)
#     # wrist_pos_traj_new_global = quat_rotate(
#     #     init_root_quat.unsqueeze(0).expand_as(wrist_quat_traj),
#     #     wrist_pos_traj_new) + init_root_pos
#     # wrist_mat_traj_new_global = quat2rotmat(wrist_quat_traj_new_global)

#     # # === 4. 计算新的手腕自由度轨迹 ===
#     init_root_euler = get_euler_xyz(init_root_quat.unsqueeze(0))
#     wrist_dof_traj_new = inverse_kinematics(
#     wrist_pos_traj_new_global,
#     wrist_mat_traj_new_global,
#     init_root_euler,
#     init_root_pos,
#     target_wrist_pos,
#     target_wrist_euler
#     )

#     return wrist_dof_traj_new

def quat_from_euler_xyz_extrinsic(x, y, z):
    quat = R.from_euler('XYZ',[x,y,z]).as_quat()
    quat = torch.tensor(quat).float()
    return quat
    
def generate_release_wrist_trajectory(
    target_wrist_pos, # 目标手腕xyz dof
    target_wrist_euler, # 目标手腕rxryrz dof
    init_root_pos,
    init_root_quat, # 初始根旋转四元数 (4,)
    wrist_dof_traj, # 手腕自由度轨迹，形状为 (T, 6) [rot_x, rot_y, rot_z, pos_x, pos_y, pos_z]
    ):

    wrist_pos_traj = wrist_dof_traj[:,:3].clone()
    wrist_euler_traj = wrist_dof_traj[:,3:6].clone()
    wrist_quat_traj = torch.ones(wrist_euler_traj.shape[0],4)
    for i in range(wrist_euler_traj.shape[0]):
        wrist_quat_traj[i] = quat_from_euler_xyz_extrinsic(
                            wrist_euler_traj[i,0],
                            wrist_euler_traj[i,1],
                            wrist_euler_traj[i,2],)

    wrist_quat_traj_new, wrist_pos_traj_new = transform_wrist_pose_with_rotmat(
    wrist_quat_traj,
    wrist_pos_traj,
    target_wrist_euler,
    target_wrist_pos
    )

    target_wrist_quat = quat_from_euler_xyz_extrinsic(*target_wrist_euler).squeeze()

    # ### debug ik and transformation
    # robot_chain = create_robot_chain([0,0,0], [0,0,0])
    # initial_joint_angles = [0,
    #                     target_wrist_pos[0],
    #                     target_wrist_pos[1],
    #                     target_wrist_pos[2],
    #                     target_wrist_euler[0],
    #                     target_wrist_euler[1],
    #                     target_wrist_euler[2]
    #                     ]
    # test_pos, test_mat = verify_fk(robot_chain, initial_joint_angles)
    # test_quat = R.from_matrix(test_mat).as_quat()
    # test_quat_error = test_quat - wrist_quat_traj_new[0]


    print('\n')
    print('target quat:',target_wrist_quat)
    print('quat before transform',wrist_quat_traj[0])
    print('quat after transform',wrist_quat_traj_new[0])
    print('\n')
    print('target pos:',target_wrist_pos)
    print('pos before transform',wrist_pos_traj[0])
    print('pos after transform',wrist_pos_traj_new[0])
    print('\n')

    # # 验证第一帧是否对齐
    # print('\n验证结果:')
    # print('目标四元数:', quat_from_euler_xyz_extrinsic(*target_wrist_euler).squeeze())
    # print('第一帧变换前四元数:', wrist_quat_traj[0])
    # print('第一帧变换后四元数:', wrist_quat_traj_new[0])
    # print('目标位置:', target_wrist_pos)
    # print('第一帧变换前位置:', wrist_pos_traj[0])
    # print('第一帧变换后位置:', wrist_pos_traj_new[0])

    # ####################global 

    # wrist_quat_traj_new_global = torch_utils.quat_multiply(
    #     init_root_quat.unsqueeze(0).expand_as(wrist_quat_traj_new),
    #     wrist_quat_traj_new)
    # wrist_pos_traj_new_global = quat_rotate(
    #     init_root_quat.unsqueeze(0).expand_as(wrist_quat_traj),
    #     wrist_pos_traj_new) + init_root_pos
    # wrist_mat_traj_new_global = quat2rotmat(wrist_quat_traj_new_global)

    # print('\n')
    # print('release start wrist pos global',wrist_pos_traj_new_global[0])
    # print('release start wrist rotmat global',wrist_mat_traj_new_global[0])
    # print('\n')

    # print('\n')
    # print('release start wrist pos local',wrist_pos_traj_new[0])
    # print('release start wrist rotmat local',quat2rotmat(wrist_quat_traj_new)[0])
    # print('\n')

    wrist_mat_traj_new = quat2rotmat(wrist_quat_traj_new)

    print("check quat2mat:", rotmat_to_quat(wrist_mat_traj_new[:1]) - wrist_quat_traj_new[:1])

    # init_root_euler = get_euler_xyz(init_root_rot.unsqueeze(0))
    wrist_dof_traj_new = inverse_kinematics(
    wrist_pos_traj_new.clone().numpy(),
    wrist_mat_traj_new.clone().numpy(),
    [0,0,0],
    [0,0,0],
    target_wrist_pos.clone().numpy(),
    target_wrist_euler.clone().numpy()
    # torch.zeros_like(target_wrist_euler.clone()).numpy()
    )

    # check traj coherence
    last_dof_of_prev_traj = torch.cat((
        target_wrist_pos,
        target_wrist_euler),dim=-1)
    validate_traj_coherence(
        last_dof_of_prev_traj,
        # last_wrist_pos_of_prev_traj,
        # last_wrist_rot_of_prev_traj,
        init_dof_of_curr_traj=wrist_dof_traj_new[0],
        # init_wrist_pos_of_curr_traj,
        # init_wrist_rot_of_curr_traj,
    )

    test_q1 = quat_from_euler_xyz_extrinsic(*wrist_dof_traj_new[0,3:6])
    test_q2 = quat_from_euler_xyz_extrinsic(*last_dof_of_prev_traj[3:6])
    # # print("target_wrist_pos",target_wrist_pos)
    # # print("gen_release_pos[0]",wrist_dof_traj_new[0,:3])
    # print("target_wrist_quat",target_wrist_quat)
    # print("wrist_quat_traj_new[0]",wrist_quat_traj_new[0])
    # # print("gen_release_rot[0]",wrist_dof_traj_new[0,3:])
    # wrist_dof_traj_new_2quat = quat_from_euler_xyz_extrinsic(
    #                     wrist_dof_traj_new[0,3],
    #                     wrist_dof_traj_new[0,4],
    #                     wrist_dof_traj_new[0,5],)
    # print("wrist_dof_traj_new_2quat",wrist_dof_traj_new_2quat)
    return wrist_dof_traj_new


def rotmat_to_quat(rotmat):
    """
    将旋转矩阵转换为四元数。
    输出的四元数格式为 (x, y, z, w)。
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


def transform_wrist_pose_with_rotmat(wrist_quat_traj, wrist_pos_traj, target_wrist_euler, target_wrist_pos):
    # q_diff
    q_initial = wrist_quat_traj[0].clone()
    q_initial_inv = quat_conjugate(q_initial)
    target_wrist_quat = quat_from_euler_xyz_extrinsic(
                        target_wrist_euler[0],
                        target_wrist_euler[1],
                        target_wrist_euler[2],)
    q_diff = torch_utils.quat_multiply(target_wrist_quat, q_initial_inv) # 旋转差异

    # pos_diff
    pos_initial = wrist_pos_traj[0].clone() 
    pos_target = target_wrist_pos.clone()
    pos_diff = pos_target - quat_rotate(q_diff.unsqueeze(0), pos_initial.unsqueeze(0)).squeeze(0)

    # traj transform
    wrist_quat_traj_new = torch_utils.quat_multiply(q_diff.unsqueeze(0).expand_as(wrist_quat_traj), wrist_quat_traj)
    wrist_pos_traj_new = quat_rotate(q_diff.unsqueeze(0).expand_as(wrist_quat_traj), wrist_pos_traj) + pos_diff

    return wrist_quat_traj_new, wrist_pos_traj_new

def create_robot_chain(root_rot, root_pos):

    robot_chain = Chain(
        name='hand',
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
    """
    """
    # 创建机械臂的运动学链
    # root_pos must be euler
    robot_chain = create_robot_chain(root_rot, root_pos)

    # 初始化关节角度列表
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
        # # 将四元数转换为旋转矩阵
        # rot = R.from_quat(quat)
        # rotation_matrix = rot.as_matrix()

        ik_result = robot_chain.inverse_kinematics(
            target_position=pos,
            target_orientation=rotation_matrix,
            orientation_mode="all",
            initial_position=initial_joint_angles
        )

        # 验证正运动学结果
        joint_angles = ik_result
        wrist_pose = (pos, rotation_matrix)
        result_pos, result_rotmat = verify_fk(robot_chain, joint_angles)
        result_quat = R.from_matrix(result_rotmat).as_quat()
        target_quat = R.from_matrix(rotation_matrix).as_quat()
        # error_rot = torch.norm(
        #     torch.tensor(result_quat)-torch.tensor(target_quat))
        r_target = R.from_quat(target_quat)
        r_computed = R.from_quat(result_quat)
        relative_rotation = r_target.inv() * r_computed
        error_angle = relative_rotation.magnitude() * (180.0 / np.pi)  # 转换为度
        if error_angle > 1e-4:
            print("ik_error_angle > 1e-4:",error_angle)
        error_pos = np.linalg.norm(result_pos-pos)
        if error_pos > 1e-4:
            print("ik_error_pos > 1e-4:",error_pos)

        # debug
        # robot_chain = create_robot_chain(root_rot, root_pos)
        test_pos, test_mat = verify_fk(robot_chain, initial_joint_angles)
        test_quat = R.from_matrix(test_mat).as_quat()
        # result_quat = R.from_matrix(result_rotmat).as_quat()
        error_rot = torch.mean(
            torch.tensor(test_quat)-torch.tensor(target_quat))
        # print('\n')
        # print('release start wrist pos (from fk)',test_pos)
        # print('release start wrist rotmat (from fk)',test_mat)
        # print('\n')


        # 更新初始关节角度为当前解，以便下一次求解使用
        initial_joint_angles = joint_angles.copy()

        # 移除基座关节角度（如果有）
        arm_dof = ik_result[1:]  # 假设第一个关节是基座，不需要
        arm_dof_trajectory.append(arm_dof)

        # 可选：打印进度
        if idx % 10 == 0:
            print(f"Calculating IK for step {idx}/{len(wrist_pos_traj)}")

    # 将关节角度列表转换为numpy数组
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
    

def plot_trajectories(wrist_traj, object_traj):
    # Extract positions
    obj_positions = torch.stack([pose for pose in object_traj]).cpu().numpy()
    wrist_positions = torch.stack([pose for pose in wrist_traj]).cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot object trajectory
    ax.plot(obj_positions[:, 0], obj_positions[:, 1], obj_positions[:, 2], label='Object Trajectory', color='blue')
    ax.scatter(obj_positions[0, 0], obj_positions[0, 1], obj_positions[0, 2], color='blue', marker='o', label='Start Object')
    ax.scatter(obj_positions[-1, 0], obj_positions[-1, 1], obj_positions[-1, 2], color='blue', marker='^', label='End Object')

    # Plot wrist trajectory
    ax.plot(wrist_positions[:, 0], wrist_positions[:, 1], wrist_positions[:, 2], label='Wrist Trajectory', color='red')
    ax.scatter(wrist_positions[0, 0], wrist_positions[0, 1], wrist_positions[0, 2], color='red', marker='o', label='Start Wrist')
    ax.scatter(wrist_positions[-1, 0], wrist_positions[-1, 1], wrist_positions[-1, 2], color='red', marker='^', label='End Wrist')

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Manipulation Trajectories')
    ax.legend()
    ax.grid(True)

    # Equal aspect ratio
    max_range = max(obj_positions.max(axis=0) - obj_positions.min(axis=0))
    mid_x = (obj_positions[:,0].max() + obj_positions[:,0].min()) * 0.5
    mid_y = (obj_positions[:,1].max() + obj_positions[:,1].min()) * 0.5
    mid_z = (obj_positions[:,2].max() + obj_positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    plt.show()

if __name__ == "__main__":


    # load grasp traj
    """
    The grasp traj is generated by off-the-shelf model.
    """
    seq_path = "/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/graspmimic/Camera_kp/camera1.pt"
    graspxl_data = torch.load(seq_path).to('cpu')[:20]
    len_grasp_traj = graspxl_data.shape[0]
    grasp_traj = {}
    grasp_traj['root_pos'] = graspxl_data[:, 0:3].clone()
    grasp_traj['root_rot'] = graspxl_data[:, 3:7].clone()
    grasp_traj['root_rot'] = smooth_quat_seq(grasp_traj['root_rot'])
    # grasp_traj['dof_pos'] = graspxl_data[:, 7:7+51].clone()
    grasp_traj['wrist_dof'] = graspxl_data[:, 7:7+6].clone()
    grasp_traj['fingers_dof'] = graspxl_data[:, 13:13+45].clone()
    grasp_traj['obj_pos'] = graspxl_data[:, 103:103+3].clone()
    grasp_traj['obj_rot'] = graspxl_data[:, 106:106+4].clone()
    grasp_traj['obj_rot'] = smooth_quat_seq(grasp_traj['obj_rot'])
    grasp_traj['contact'] = graspxl_data[:, 110:110+1].clone()

    # generate manipulation traj
    """
    Generate manipulation traj based on object pose interpolation.
    The relative pose between object and wrist is unchanged.
        1. interpolate the object pose
        2. generate wrist pose based on object pose
        3. use IK to generate wrist dof based on wrist pose,
           make sure the dof traj is smooth.
    """
    manipulation_traj = {}
    init_obj_pos = grasp_traj['obj_pos'][-1].clone() # global
    init_obj_rot = grasp_traj['obj_rot'][-1].clone() # global quat (x,y,z,w)
    init_wrist_pos = grasp_traj['wrist_dof'][-1,:3].clone() # local
    init_wrist_rot = grasp_traj['wrist_dof'][-1,3:].clone() # local euler xyz
    init_root_pos = grasp_traj['root_pos'][-1].clone() # global
    init_root_rot = grasp_traj['root_rot'][-1].clone() # global quat
    # print('init_wrist_pos',init_wrist_pos)
    # print('init_wrist_rot',init_wrist_rot)
    # print('init_obj_pos',init_obj_pos)
    # print('init_obj_rot',init_obj_rot)

    target_obj_pos = torch.rand(3)
    target_obj_rot = torch_utils.exp_map_to_quat(torch.rand(3)*2)
    # target_obj_rot = init_obj_rot

    manipulation_traj['obj_pos'],\
    manipulation_traj['obj_rot'],\
    manipulation_traj['wrist_dof']\
    = generate_manipulation_trajectory(
        init_obj_pos, 
        init_obj_rot, 
        init_wrist_pos, 
        init_wrist_rot, 
        init_root_pos,
        init_root_rot,
        target_obj_pos,
        target_obj_rot,
        num_steps=50
    )

    len_manipulation_traj = manipulation_traj['obj_pos'].shape[0]
    manipulation_traj['root_pos'] = grasp_traj['root_pos'][-1,:].repeat(len_manipulation_traj,1)
    manipulation_traj['root_rot'] = grasp_traj['root_rot'][-1,:].repeat(len_manipulation_traj,1)
    manipulation_traj['fingers_dof'] = grasp_traj['fingers_dof'][-1,:].repeat(len_manipulation_traj,1)
    manipulation_traj['contact'] = torch.ones(len_manipulation_traj,1)
    # print('\n')
    # print('grasp_traj[wrist_dof][-1]',grasp_traj['wrist_dof'][-1])
    # print('manipulation_traj[wrist_dof]',manipulation_traj['wrist_dof'])


    # generate release traj
    """
    The release traj is the inverse of the grasp traj
    with transformation based on current object pose
    """
    release_traj = {}
    release_traj['wrist_dof'] = torch.flip(grasp_traj['wrist_dof'],dims=[0])
    release_traj['root_pos'] = torch.flip(grasp_traj['root_pos'],dims=[0])
    release_traj['root_rot'] = torch.flip(grasp_traj['root_rot'],dims=[0])
    release_traj['fingers_dof'] = torch.flip(grasp_traj['fingers_dof'],dims=[0])
    release_traj['contact'] = torch.flip(grasp_traj['contact'],dims=[0])

    target_obj_pos = manipulation_traj['obj_pos'][-1].clone() # global
    target_obj_rot = manipulation_traj['obj_rot'][-1].clone() # global quat (x,y,z,w)
    target_wrist_pos = manipulation_traj['wrist_dof'][-1,:3].clone() # local
    target_wrist_rot = manipulation_traj['wrist_dof'][-1,3:].clone() # local euler xyz
    init_root_pos = manipulation_traj['root_pos'][-1].clone() # global
    init_root_rot = manipulation_traj['root_rot'][-1].clone() # global quat

    len_release_traj = release_traj['fingers_dof'].shape[0]
    release_traj['obj_rot'] = target_obj_rot.repeat(len_release_traj,1)
    release_traj['obj_pos'] = target_obj_pos.repeat(len_release_traj,1)

    release_traj['wrist_dof']\
    = generate_release_wrist_trajectory(
        target_wrist_pos.float(), 
        target_wrist_rot.float(), 
        init_root_pos,
        init_root_rot,
        release_traj['wrist_dof'],
    )

    # traj fusion
    """
    Fuse the grasp, manipulation, and release traj,
    as the full traj.
    """
    full_traj = {}
    full_traj['obj_pos'] = torch.cat((grasp_traj['obj_pos'],
                                      manipulation_traj['obj_pos'],
                                      release_traj['obj_pos']),dim=0)
    full_traj['obj_rot'] = torch.cat((grasp_traj['obj_rot'],
                                      manipulation_traj['obj_rot'],
                                      release_traj['obj_rot']),dim=0)
    full_traj['wrist_dof'] = torch.cat((grasp_traj['wrist_dof'],
                                      manipulation_traj['wrist_dof'],
                                      release_traj['wrist_dof']),dim=0)
    full_traj['root_pos'] = torch.cat((grasp_traj['root_pos'],
                                      manipulation_traj['root_pos'],
                                      release_traj['root_pos']),dim=0)
    full_traj['root_rot'] = torch.cat((grasp_traj['root_rot'],
                                      manipulation_traj['root_rot'],
                                      release_traj['root_rot']),dim=0)
    full_traj['fingers_dof'] = torch.cat((grasp_traj['fingers_dof'],
                                      manipulation_traj['fingers_dof'],
                                      release_traj['fingers_dof']),dim=0)
    full_traj['contact'] = torch.cat((grasp_traj['contact'],
                                      manipulation_traj['contact'],
                                      release_traj['contact']),dim=0)
    len_full_traj = full_traj['obj_pos'].shape[0]

    hoi_data = torch.cat((
        full_traj['root_pos'],
        full_traj['root_rot'],
        full_traj['wrist_dof'],
        full_traj['fingers_dof'],

        torch.zeros(len_full_traj,15*3),

        full_traj['obj_pos'],
        full_traj['obj_rot'],
        full_traj['contact'],
        ),dim=-1)

    save_hoi_data = hoi_data.clone().float()
    torch.save(save_hoi_data, '/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/graspmimic/Camera_gen_release2/camera.pt')


    ## TODO: 
    '''
    1. implement the release traj generation code
        - change the obj rot into quat transform, rather than mat.
    1.1 check the dof consistency
    2. optimize the code structure
      to support multiple manipulation traj, i.e., more waypoints
    3. implement auto generation of many traj files.
    4. update the contact.
    '''