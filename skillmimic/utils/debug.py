from isaacgym.torch_utils import *
import torch_utils
import torch
import torch.nn.functional as F

# 假设以下变量已经定义并且是torch.Tensor类型
# wrist_quat_traj: Tensor of shape (N, 4) with format (w, x, y, z)
# wrist_pos_traj: Tensor of shape (N, 3)
# target_wrist_euler: Tensor or list of 3 elements representing Euler angles (x, y, z)
# target_wrist_pos: Tensor of shape (3, )

# def quat_from_euler_xyz(euler_x, euler_y, euler_z):
#     """
#     根据给定的欧拉角按 XYZ 顺序生成四元数。
#     角度单位为弧度。
#     """
#     cy = torch.cos(euler_z * 0.5)
#     sy = torch.sin(euler_z * 0.5)
#     cp = torch.cos(euler_y * 0.5)
#     sp = torch.sin(euler_y * 0.5)
#     cr = torch.cos(euler_x * 0.5)
#     sr = torch.sin(euler_x * 0.5)

#     w = cr * cp * cy + sr * sp * sy
#     x = sr * cp * cy - cr * sp * sy
#     y = cr * sp * cy + sr * cp * sy
#     z = cr * cp * sy - sr * sp * cy

#     quat = torch.stack([x, y, z, w], dim=-1)
#     quat = F.normalize(quat, p=2, dim=-1)
#     return quat

# def transform_wrist_pose_with_rotmat(wrist_quat_traj, wrist_pos_traj, target_wrist_euler, target_wrist_pos):
#     """
#     将腕部姿态轨迹整体变换，使得第一帧与目标姿态一致。
#     使用旋转矩阵实现。
#     """
#     # 1. 确保四元数归一化
#     wrist_quat_traj = F.normalize(wrist_quat_traj, p=2, dim=1)

#     # 2. 转换四元数到旋转矩阵
#     R_traj = quat_to_rotmat(wrist_quat_traj) # (N, 3, 3)

#     # 3. 计算目标四元数并转换为旋转矩阵
#     target_wrist_quat = quat_from_euler_xyz(
#     target_wrist_euler[0],
#     target_wrist_euler[1],
#     target_wrist_euler[2],
#     ) # (1, 4) assuming target_wrist_euler is a list or tensor of 3 elements
#     R_target = quat_to_rotmat(target_wrist_quat.unsqueeze(0)) # (1, 3, 3)

#     # 4. 计算旋转差异 R_diff = R_target * R_initial^T
#     R_initial = R_traj[0].unsqueeze(0) # (1, 3, 3)
#     R_diff = torch.bmm(R_target, R_initial.transpose(1, 2)) # (1, 3, 3)

#     # 5. 计算位置差异 t_diff = t_target - R_diff * t_initial
#     t_initial = wrist_pos_traj[0].unsqueeze(0).unsqueeze(2) # (1, 3, 1)
#     t_target = target_wrist_pos.unsqueeze(0).unsqueeze(2) # (1, 3, 1)
#     t_diff = t_target - torch.bmm(R_diff, t_initial) # (1, 3, 1)

#     # 6. 应用整体旋转和平移到整个轨迹
#     R_traj_new = torch.bmm(R_diff.repeat(wrist_quat_traj.shape[0], 1, 1), R_traj) # (N, 3, 3)
#     wrist_quat_traj_new = rotmat_to_quat(R_traj_new) # (N, 4)

#     a = R_diff.repeat(wrist_quat_traj.shape[0], 1, 1)
#     b = wrist_pos_traj.unsqueeze(2)
#     wrist_pos_traj_new = torch.bmm(a, b) + t_diff
#     wrist_pos_traj_new = wrist_pos_traj_new.squeeze(2) # (N, 3)

#     return wrist_quat_traj_new, wrist_pos_traj_new


import torch
import torch.nn.functional as F

def quat_to_rotmat(quat):
    """
    将四元数转换为旋转矩阵。
    四元数格式假设为 (w, x, y, z)。
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    B = quat.shape[0]
    rotmat = torch.zeros((B, 3, 3), device=quat.device, dtype=quat.dtype)

    rotmat[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    rotmat[:, 0, 1] = 2*x*y - 2*z*w
    rotmat[:, 0, 2] = 2*x*z + 2*y*w

    rotmat[:, 1, 0] = 2*x*y + 2*z*w
    rotmat[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    rotmat[:, 1, 2] = 2*y*z - 2*x*w

    rotmat[:, 2, 0] = 2*x*z - 2*y*w
    rotmat[:, 2, 1] = 2*y*z + 2*x*w
    rotmat[:, 2, 2] = 1 - 2*x**2 - 2*y**2

    return rotmat

def rotmat_to_quat(rotmat):
    """
    将旋转矩阵转换为四元数。
    输出的四元数格式为 (w, x, y, z)。
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
    target_wrist_quat = quat_from_euler_xyz(
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




# # 使用示例
# # 请根据你的实际数据来填充 wrist_quat_traj, wrist_pos_traj, target_wrist_euler, target_wrist_pos

# # 示例数据（需要替换为实际数据）
# N = 10
# wrist_quat_traj = F.normalize(torch.randn(N, 4), p=2, dim=1)
# wrist_pos_traj = torch.randn(N, 3)
# target_wrist_euler = torch.tensor([0.1, 0.5, 0.8]) # 目标欧拉角
# target_wrist_pos = torch.tensor([1.0, 2.0, 3.0]) # 目标位置

# # 应用变换
# wrist_quat_traj_new, wrist_pos_traj_new = transform_wrist_pose_with_rotmat(
# wrist_quat_traj,
# wrist_pos_traj,
# target_wrist_euler,
# target_wrist_pos
# )

# # 验证第一帧是否对齐
# print('\n验证结果:')
# print('目标四元数:', quat_from_euler_xyz(*target_wrist_euler).squeeze())
# print('第一帧变换前四元数:', wrist_quat_traj[0])
# print('第一帧变换后四元数:', wrist_quat_traj_new[0])
# print('目标位置:', target_wrist_pos)
# print('第一帧变换前位置:', wrist_pos_traj[0])
# print('第一帧变换后位置:', wrist_pos_traj_new[0])

# # 验证最后一帧（或其他帧）是否按预期变换
# print('\n最后一帧验证:')
# print('最后一帧变换前四元数:', wrist_quat_traj[-1])
# print('最后一帧变换后四元数:', wrist_quat_traj_new[-1])
# print('最后一帧变换前位置:', wrist_pos_traj[-1])
# print('最后一帧变换后位置:', wrist_pos_traj_new[-1])

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from scipy.spatial.transform import Rotation as R

def create_robot_chain():

    robot_chain = Chain(
        name='wrist_rot',
        links=[
            OriginLink(),
            URDFLink(
                name="rx",
                origin_translation=[0.0, 0.0, 0.0],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],  # Rotation around x-axis
                bounds=(-2*np.pi, 2*np.pi)
            ),

            URDFLink(
                name="ry",
                origin_translation=[0.0, 0.0, 0.0],  # No translation; same position as elbow_x
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],  # Rotation around y-axis
                bounds=(-2*np.pi, 2*np.pi)
            ),

            URDFLink(
                name="rz",
                origin_translation=[0.0, 0.0, 0.0],  # No translation; same position as elbow_y
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],  # Rotation around z-axis
                bounds=(-2*np.pi, 2*np.pi)
            ),
        ],active_links_mask=[False, True, True, True]
    )
    return robot_chain

def verify_fk(robot_chain, joint_angles):
    frames = robot_chain.forward_kinematics(joint_angles, full_kinematics=True)
    end_effector_frame = frames[-1]
    print("frames",frames)
    position_computed = end_effector_frame[:3, 3]
    rotation_computed = end_effector_frame[:3, :3]

    return position_computed, rotation_computed

def quat_from_euler_xyz_extrinsic(x, y, z):
    quat = R.from_euler('XYZ',[x,y,z]).as_quat()
    quat = torch.tensor(quat)
    return quat


### debug ik

init_wrist_euler = torch.tensor([torch.pi/2,0.22,torch.pi/2])

robot_chain = create_robot_chain()
initial_joint_angles = [0,
                    init_wrist_euler[0],
                    init_wrist_euler[1],
                    init_wrist_euler[2]
                    ]
test_pos, test_mat = verify_fk(robot_chain, initial_joint_angles)
test_quat_ik = R.from_matrix(test_mat).as_quat()
# test_quat_given = quat_from_euler_xyz(*init_wrist_euler)
test_quat_given_111 = quat_from_euler_xyz(*init_wrist_euler)
test_quat_given = quat_from_euler_xyz_extrinsic(*init_wrist_euler)
# test_quat_given = R.from_euler('XYZ',[
#                     init_wrist_euler[0],
#                     init_wrist_euler[1],
#                     init_wrist_euler[2]
#                     ]).as_quat()

print("test_quat_given.shape():",test_quat_given.shape)
print("test_quat_given_111.shape():",test_quat_given_111.shape)
print("test_quat_given:",test_quat_given)
print("test_quat_ik:",test_quat_ik)

# test_quat_error = test_quat_ik - test_quat_given.numpy()
# print("test_quat_error:",test_quat_error)
# test_eulerxyz_ik = get_euler_xyz(torch.from_numpy(test_quat_ik).unsqueeze(0))
# print("\n")
# print("test_eulerxyz_given:",init_wrist_euler)
# print("test_eulerxyz_ik:",test_eulerxyz_ik)

