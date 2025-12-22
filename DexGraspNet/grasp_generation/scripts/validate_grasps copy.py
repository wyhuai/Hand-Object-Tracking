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
from utils.hand_model import HandModel
from utils.object_model import ObjectModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--val_batch', default=128, type=int)
    parser.add_argument('--mesh_path', default="../data/meshdata", type=str)
    parser.add_argument('--grasp_path', default="../data/experiments_ball_006/exp_33/results", type=str)
    parser.add_argument('--result_path', default="../data/1223", type=str)
    parser.add_argument('--object_code',
                        default="ddg-gd_soccer_ball_poisson_000",
                        type=str)
    # if index is received, then the debug mode is on
    parser.add_argument('--index', type=int)
    parser.add_argument('--no_force', action='store_true')
    parser.add_argument('--thres_cont', default=0.001, type=float)
    parser.add_argument('--dis_move', default=0.001, type=float)
    parser.add_argument('--grad_move', default=500, type=float)
    parser.add_argument('--penetration_threshold', default=0.001, type=float)

    args = parser.parse_args()

    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
        'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
        'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
        'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
        'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
    ]

    #os.environ.pop("CUDA_VISIBLE_DEVICES")
    os.makedirs(args.result_path, exist_ok=True)

    if not args.no_force:
        device = torch.device(
            f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        data_dict = np.load(os.path.join(
            args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
        batch_size = data_dict.shape[0]
        hand_state = []
        scale_tensor = []
        for i in range(batch_size):
            qpos = data_dict[i]['qpos']
            scale = data_dict[i]['scale']
            rot = np.array(transforms3d.euler.euler2mat(
                *[qpos[name] for name in rot_names]))
            rot = rot[:, :2].T.ravel().tolist()
            hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [
                qpos[name] for name in joint_names], dtype=torch.float, device=device)
            hand_state.append(hand_pose)
            scale_tensor.append(scale)
        hand_state = torch.stack(hand_state).to(device).requires_grad_()
        scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)


    #sim = IsaacValidator(gpu=args.gpu)
    #if (args.index is not None):
    sim = IsaacValidator(gpu=args.gpu, mode="gui")
    data_dict = np.load(os.path.join(
        args.grasp_path, args.object_code + '.npy'), allow_pickle=True)
    batch_size = data_dict.shape[0]
    scale_array = []
    hand_poses = []
    rotations = []
    translations = []
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]['qpos']
        scale = data_dict[i]['scale']
        rot = [qpos[name] for name in rot_names]
        rot = transforms3d.euler.euler2quat(*rot)
        rotations.append(rot)
        translations.append(np.array([qpos[name]
                            for name in translation_names]))
        hand_poses.append(np.array([qpos[name] for name in joint_names]))
        scale_array.append(scale)
        E_pen_array.append(data_dict[i]["E_pen"])
    E_pen_array = np.array(E_pen_array)
    if not args.no_force:
        hand_poses = hand_state[:, 9:]

    if (args.index is not None):
        sim.set_asset("open_ai_assets", "hand/allegro_hand.xml",
                       os.path.join(args.mesh_path, args.object_code, "coacd"), "coacd.urdf")
        index = args.index
        sim.add_env_single(rotations[index], translations[index], hand_poses[index],
                           scale_array[index], 0)
        result = sim.run_sim()
        print(result)
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            sim.set_asset("open_ai_assets", "hand/allegro_hand.xml",
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
        #with open(os.path.join(os.path.dirname(args.grasp_path), "saved_ids.txt"), 'w') as f:
        #    for index in true_indices:
        #        f.write(f"{index}\n")
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
