
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

def quat_from_euler_xyz_extrinsic(x, y, z):
    quat = R.from_euler('XYZ',[x,y,z]).as_quat()
    quat = torch.tensor(quat).float()
    return quat

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


if __name__ == "__main__":

    device='cpu'

    basic_grasp_frames_path = "/home/hkust/yinhuai/unihot_dataset/basic_grasp_frames/000_traj_0.pt"
    basic_grasp_frames = load_grasp_traj(basic_grasp_frames_path, device=device)
    move_trajs_path = "/home/hkust/yinhuai/unihot_dataset/move"
    all_move_trajs = glob.glob(move_trajs_path + '/*.pt')
    for idx, seq_path in enumerate(all_move_trajs):
        move_traj = load_grasp_traj(seq_path, device=device)
        num_grasps = basic_grasp_frames['root_pos'].shape[0]

        #for each basic_grasp_frame, align the
        #               obj pose 
        #to the first frame of current move_traj
        for i in range(num_grasps):

            target_obj_pos = move_traj['obj_pos'][0].clone()
            target_obj_rot = move_traj['obj_rot'][0].clone()
            
            # get relative pose between wrist and obj
            init_obj_pos = basic_grasp_frames['obj_pos'][i].clone()
            init_obj_rot = basic_grasp_frames['obj_rot'][i].clone()
            init_wrist_pos = basic_grasp_frames['wrist_dof'][i][:3].clone()
            init_wrist_rot = basic_grasp_frames['wrist_dof'][i][3:6].clone()
            init_root_pos = basic_grasp_frames['root_pos'][i].clone()
            init_root_rot = basic_grasp_frames['root_rot'][i].clone()

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

            # get wrist pose from target object pose
            obj_pos = target_obj_pos
            obj_rot = target_obj_rot

            obj_rot_mat = quat2rotmat(obj_rot)

            wrist_rot_global = obj_rot_mat @ rel_rot
            wrist_pos_global = obj_rot_mat @ rel_trans + obj_pos

            wrist_rot_traj_global.append(wrist_rot_global)
            wrist_pos_traj_global.append(wrist_pos_global)

            wrist_pos_traj_global = torch.stack(wrist_pos_traj_global)
            wrist_rot_traj_global = torch.stack(wrist_rot_traj_global)

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
                    move_traj['wrist_dof'][0,:3].clone().numpy(),
                    move_traj['wrist_dof'][0,3:6].clone().numpy())
                    # [0,0,0],
                    # [0,0,0])
            wrist_dof_traj = wrist_dof_traj.float()
    
            # cat the transformed grasp frame with current move_traj
            root_pos = torch.cat((move_traj['root_pos'][:1],move_traj['root_pos']),dim=0)
            root_rot = torch.cat((move_traj['root_rot'][:1],move_traj['root_rot']),dim=0)
            wrist_dof = torch.cat((wrist_dof_traj,move_traj['wrist_dof']),dim=0)
            fingers_dof = torch.cat((basic_grasp_frames['fingers_dof'][i:i+1],move_traj['fingers_dof']),dim=0)
            obj_pos_traj = torch.cat((move_traj['obj_pos'][:1],move_traj['obj_pos']),dim=0)
            obj_rot_traj = torch.cat((move_traj['obj_rot'][:1],move_traj['obj_rot']),dim=0)
            obj2_pos_traj = torch.cat((move_traj['obj2_pos'][:1],move_traj['obj2_pos']),dim=0)
            obj2_rot_traj = torch.cat((move_traj['obj2_rot'][:1],move_traj['obj2_rot']),dim=0)
            contact1 = torch.cat((move_traj['contact1'][:1],move_traj['contact1']),dim=0)
            contact2 = torch.cat((move_traj['contact2'][:1],move_traj['contact2']),dim=0)

            hoi_data = torch.cat((
            root_pos,
            root_rot,
            wrist_dof,
            fingers_dof,
            torch.zeros(root_pos.shape[0], 15*3),
            obj_pos_traj,
            obj_rot_traj,
            obj2_pos_traj,
            obj2_rot_traj,
            contact1,
            contact2
            ), dim=-1)

            # 保存最终轨迹数据
            os.makedirs('/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/graspswitch',exist_ok=True)
            save_path = f'/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/graspswitch/005_box01_{idx}_{i}.pt'
            torch.save(hoi_data.clone().float(), save_path)
            print(f"\nsave 005_box01_{idx}_{i}.pt！\n")

    print("\n所有步骤完成！\n")