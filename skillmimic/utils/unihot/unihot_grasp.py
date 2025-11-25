
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


def transform_hoitraj(grasp_traj, target_obj_quat, target_obj_pos):

    new_grasp_traj = {
    'wrist_dof': grasp_traj['wrist_dof'].clone(),
    'root_pos': grasp_traj['root_pos'].clone(),
    'root_rot': grasp_traj['root_rot'].clone(),
    'fingers_dof': grasp_traj['fingers_dof'].clone(),
    'obj_pos': grasp_traj['obj_pos'].clone(),
    'obj_rot': grasp_traj['obj_rot'].clone(),
    'obj2_pos': grasp_traj['obj2_pos'].clone(),
    'obj2_rot': grasp_traj['obj2_rot'].clone(),
    'contact1': grasp_traj['contact1'].clone(),
    'contact2': grasp_traj['contact2'].clone(),
    }

    # get wrist local coordinate
    len_wrist_dof_traj = grasp_traj['wrist_dof'].shape[0]
    wrist_quat_traj = torch.zeros(len_wrist_dof_traj,4)
    wrist_pos_traj = torch.zeros(len_wrist_dof_traj,3)
    for i in range(len_wrist_dof_traj):
        wrist_quat_traj[i] = quat_from_euler_xyz_extrinsic(
                            grasp_traj['wrist_dof'][i,3],
                            grasp_traj['wrist_dof'][i,4],
                            grasp_traj['wrist_dof'][i,5],)
        wrist_pos_traj[i] = grasp_traj['wrist_dof'][i,:3].clone()
        
    # global
    wrist_pos_traj_global, wrist_quat_traj_global=\
        robot_local_to_global(wrist_pos_traj, wrist_quat_traj, grasp_traj['root_pos'][0], grasp_traj['root_rot'][0])
    
    new_grasp_traj['obj_rot'], new_grasp_traj['obj_pos'],\
    wrist_quat_traj_new, wrist_pos_traj_new = \
        transform_traj(grasp_traj['obj_rot'], grasp_traj['obj_pos'],
                    wrist_quat_traj_global, wrist_pos_traj_global,
                    target_obj_quat, target_obj_pos)

    new_grasp_traj['obj_rot'] = smooth_quat_seq(new_grasp_traj['obj_rot'])

    wrist_mat_traj_new = quat2rotmat(wrist_quat_traj_new)

    root_rot_euler = get_euler_xyz(grasp_traj['root_rot'][0].unsqueeze(0))
    wrist_dof_traj_new = inverse_kinematics(
    wrist_pos_traj_new.clone().numpy(),
    wrist_mat_traj_new.clone().numpy(),
    [root_rot_euler[0].clone().numpy(),
    root_rot_euler[1].clone().numpy(),
    root_rot_euler[2].clone().numpy()],
    grasp_traj['root_pos'][0].clone().numpy(),
    [0,0,0],
    [0,0,0]
    )
    new_grasp_traj['wrist_dof'] = wrist_dof_traj_new

    return new_grasp_traj


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

    # function: augment basic grasp trajs to cover the operation space.
    # input: 16K basic grasp traj
    # output: 16K*M augmented grasp traj

    device='cpu'
    M = 10

    #define the range of operation space
    x_min = -1.6
    x_max = +1.6
    y_min = -1.6
    y_max = +1.6
    z_min = 0.
    z_max = +1.6

    basic_grasp_trajs = "/home/hkust/yinhuai/unihot_dataset/basic_grasp_trajs"
    all_seqs = glob.glob(basic_grasp_trajs + '/*.pt')
    for idx, seq_path in enumerate(all_seqs):
        grasp_traj = load_grasp_traj(seq_path, device=device)

        # adjust the range by the root location
        z_min = grasp_traj['obj_pos'][-1][2].clone()
        z_max = z_min
        x_min += grasp_traj['root_pos'][-1][0]
        x_max += grasp_traj['root_pos'][-1][0]
        y_min += grasp_traj['root_pos'][-1][1]
        y_max += grasp_traj['root_pos'][-1][1]

        for i in range(M):
            # M augmented traj. rand obj xy and rz,  
            coords = torch.rand(1,3,device=device)
            ranges = torch.tensor([
                [x_min,x_max],
                [y_min,y_max],
                [z_min,z_max]],
                device=device
            )
            target_obj_pos = coords * (ranges[:,1]-ranges[:,0]) + ranges[:,0]

            rand_rz = torch.rand(1,device=device)*2*torch.pi
            target_obj_quat = torch.zeros(1,4).to(device) # axis quat
            target_obj_quat[:,2] = torch.sin(rand_rz/2) # z
            target_obj_quat[:,3] = torch.cos(rand_rz/2) # w

            new_grasp_traj = \
                transform_hoitraj(grasp_traj, target_obj_quat, target_obj_pos)

            len_traj = new_grasp_traj['root_pos'].shape[0]
            contact1 = new_grasp_traj['contact1']
            contact2 = torch.zeros_like(new_grasp_traj['contact1']).to(device)
            hoi_data = torch.cat((
            new_grasp_traj['root_pos'],
            new_grasp_traj['root_rot'],
            new_grasp_traj['wrist_dof'],
            new_grasp_traj['fingers_dof'],
            torch.zeros(len_traj, 15*3),
            new_grasp_traj['obj_pos'],
            new_grasp_traj['obj_rot'],
            new_grasp_traj['obj2_pos'],
            new_grasp_traj['obj2_rot'],
            contact1,
            contact2
            ), dim=-1)

            # 保存最终grasp轨迹数据
            os.makedirs('/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/grasp',exist_ok=True)
            save_path = f'/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/grasp/001_box01_{idx}_{i}.pt'
            torch.save(hoi_data.clone().float(), save_path)
            print(f"\nsave 001_box01_{idx}_{i}.pt！\n")

            # 保存最终place轨迹数据(reverse of grasp)
            os.makedirs('/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/place',exist_ok=True)
            save_path = f'/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/unihot/place/003_box01_{idx}_{i}.pt'
            torch.save(hoi_data.clone().float(), save_path)
            print(f"\nsave 003_box01_{idx}_{i}.pt！\n")

    print("\n所有步骤完成！\n")