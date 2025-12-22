"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath('.'))

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
#from utils.hand_model import HandModel
#from utils.object_model import ObjectModel
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_batch', default=128, type=int)
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    parser.add_argument('--grasp_path', default="../data/experiments_hammer_02/exp_32/results", type=str)
    parser.add_argument('--result_path', default="../data/dataset_new", type=str)
    parser.add_argument('--object_code',
                        default="sem-Hammer-5d4da30b8c0eaae46d7014c7d6ce68fc",
                        #default="ddg-kit_CokePlasticLarge",
                        type=str)
    # if index is received, then the debug mode is on
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)

    args = parser.parse_args()

    #translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    #rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    '''
    joint_keys = [
        'right_index1_x', 'right_index1_y', 'right_index1_z',
        'right_index2_x', 'right_index2_y', 'right_index2_z',
        'right_index3_x', 'right_index3_y', 'right_index3_z',
        'right_middle1_x', 'right_middle1_y', 'right_middle1_z',
        'right_middle2_x', 'right_middle2_y', 'right_middle2_z',
        'right_middle3_x', 'right_middle3_y', 'right_middle3_z',
        'right_pinky1_x', 'right_pinky1_y', 'right_pinky1_z',
        'right_pinky2_x', 'right_pinky2_y', 'right_pinky2_z',
        'right_pinky3_x', 'right_pinky3_y', 'right_pinky3_z',
        'right_ring1_x', 'right_ring1_y', 'right_ring1_z',
        'right_ring2_x', 'right_ring2_y', 'right_ring2_z',
        'right_ring3_x', 'right_ring3_y', 'right_ring3_z',
        'right_thumb1_x', 'right_thumb1_y', 'right_thumb1_z',
        'right_thumb2_x', 'right_thumb2_y', 'right_thumb2_z',
        'right_thumb3_x', 'right_thumb3_y', 'right_thumb3_z',
        'right_wrist_0rx', 'right_wrist_0ry', 'right_wrist_0rz',
        'right_wrist_0x', 'right_wrist_0y', 'right_wrist_0z'
    ]
    '''
    #os.environ.pop("CUDA_VISIBLE_DEVICES")
    os.makedirs(args.result_path, exist_ok=True)
    
    if not args.no_force:
        device = torch.device(
            f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        data_dict = np.load(os.path.join(
            args.grasp_path, args.object_code + '_for_isaacgym.npy'), allow_pickle=True)
        batch_size = data_dict.shape[0]
        hand_state = []
        scale_tensor = []
        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            #rot = np.array(transforms3d.euler.euler2mat(
            #    *[qpos[name] for name in rot_names]))
            rot = np.array(transforms3d.quaternions.quat2mat(
                np.array([qpos['rot'][3], qpos['rot'][0], qpos['rot'][1], qpos['rot'][2]])))
            rot = rot[:, :2].T.ravel().tolist()
            #hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [
            #    qpos[name] for name in joint_names], dtype=torch.float, device=device)
            hand_pose = torch.tensor(qpos['trans'].tolist()  + rot + 
                qpos['thetas'], dtype=torch.float, device=device)
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)
        '''
        # print(scale_tensor.dtype)
        hand_model = HandModel(
            mano_root='mano', 
            contact_indices_path='mano/contact_indices.json', 
            pose_distrib_path='mano/pose_distrib.pt', 
            device=device
        )

        hand_model.set_parameters(hand_state)
        # object model
        object_model = ObjectModel(
            data_root_path=args.mesh_path,
            batch_size_each=batch_size,
            num_samples=0,
            device=device
        )
        object_model.initialize(args.object_code)
        object_model.object_scale_tensor = scale_tensor
        # calculate contact points and contact normals
        contact_points_hand = torch.zeros((batch_size, 19, 3)).to(device)
        contact_normals = torch.zeros((batch_size, 19, 3)).to(device)

        for i, link_name in enumerate(hand_model.mesh):
            if len(hand_model.mesh[link_name]['surface_points']) == 0:
                continue
            surface_points = hand_model.current_status[link_name].transform_points(
                hand_model.mesh[link_name]['surface_points']).expand(batch_size, -1, 3)
            surface_points = surface_points @ hand_model.global_rotation.transpose(
                1, 2) + hand_model.global_translation.unsqueeze(1)
            distances, normals = object_model.cal_distance(
                surface_points)
            nearest_point_index = distances.argmax(dim=1)
            nearest_distances = torch.gather(
                distances, 1, nearest_point_index.unsqueeze(1))
            nearest_points_hand = torch.gather(
                surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            nearest_normals = torch.gather(
                normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3))
            admited = -nearest_distances < args.thres_cont
            admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
            contact_points_hand[:, i:i+1, :] = torch.where(
                admited, nearest_points_hand, contact_points_hand[:, i:i+1, :])
            contact_normals[:, i:i+1, :] = torch.where(
                admited, nearest_normals, contact_normals[:, i:i+1, :])

        target_points = contact_points_hand + contact_normals * args.dis_move
        loss = (target_points.detach().clone() -
                contact_points_hand).square().sum()
        loss.backward()
        with torch.no_grad():
            hand_state[:, 9:] += hand_state.grad[:, 9:] * args.grad_move
            hand_state.grad.zero_()
        '''
    sim = IsaacValidator(gpu=args.gpu, mode="gui")
    #if (args.index is not None):
    #    sim = IsaacValidator(gpu=args.gpu, mode="gui")

    data_dict = np.load(os.path.join(
        args.grasp_path, args.object_code + '_for_isaacgym.npy'), allow_pickle=True)
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']
        #rot = [qpos[name] for name in rot_names]
        rot = np.array([qpos['rot'][3], qpos['rot'][0], qpos['rot'][1], qpos['rot'][2]])
        #rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        #translations.append(np.array([qpos[name]
        #                    for name in translation_names]))        
        translations.append(np.array(qpos['trans'].tolist()))
        #hand_poses.append(np.array([qpos[name] for name in joint_names]))
        hand_poses.append(np.array(qpos['thetas']))
        scale_array.append(scale)
        E_pen_array.append(data_dict[i]["E_pen"])
    E_pen_array = np.array(E_pen_array)
    if not args.no_force:
        hand_poses = hand_state[:, 9:]

    if (args.index is not None):
        sim.set_asset("mjcf", "rhand_mano_low_mass.xml",
                       os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")  
        index = args.index
        sim.add_env_single(rotations[index], translations[index], hand_poses[index],
                           scale_array[index]) #0 here is by rotating the hand and object simultaneously, there are in rotation 6 rotation to validate, here 0 mean only the first one is chosen
        result = sim.run_sim()
        print(result)
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            sim.set_asset("mjcf", "rhand_mano_low_mass.xml",
                        os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")  
            for index in range(offset, offset_):
                sim.add_env(rotations[index], translations[index], hand_poses[index],
                            scale_array[index])
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            offset = offset_
        for i in range(batch_size):
            simulated[i] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)

        estimated = E_pen_array < args.penetration_threshold
        true_indices = [index for index, value in enumerate(estimated) if value]
        print(true_indices)
        valid = simulated * estimated
        print(valid)
        true_indices = [index for index, value in enumerate(valid) if value]
        print(true_indices)
        with open(os.path.join(os.path.dirname(args.grasp_path), "saved_ids.txt"), 'w') as f:
            for index in true_indices:
                f.write(f"{index}\n")
        print(
            f'estimated: {estimated.sum().item()}/{batch_size}, '
            f'simulated: {simulated.sum().item()}/{batch_size}, '
            f'valid: {valid.sum().item()}/{batch_size}')
        result_list = []
        for i in range(batch_size):
            if (valid[i]):
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                result_list.append(new_data_dict)
        np.save(os.path.join(args.result_path, args.object_code +
                '.npy'), result_list, allow_pickle=True)
    sim.destroy()
