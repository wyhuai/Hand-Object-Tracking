"""
Last modified date: 2023.02.23
Author: Jialiang Zhang, Ruicheng Wang
Description: Entry of the program, generate small-scale experiments
"""

import os
print(__file__)
os.chdir(os.path.dirname("./main.py"))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
from manopth import rodrigues_layer
import torch.nn.functional as F

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger

def rotmat_to_extrinsic_euler_xyz(rotmat):
    """
    Convert rotation matrix to extrinsic XYZ Euler angles.
    Input: rotmat (B, 3, 3)
    Output: euler_angles (B, 3) in radians
    """
    sy = torch.sqrt(rotmat[:, 0, 0] ** 2 + rotmat[:, 1, 0] ** 2)
    singular = sy < 1e-6

    x = torch.atan2(rotmat[:, 2, 1], rotmat[:, 2, 2])
    y = torch.atan2(-rotmat[:, 2, 0], sy)
    z = torch.atan2(rotmat[:, 1, 0], rotmat[:, 0, 0])

    # Handle singularities (gimbal lock)
    x[singular] = torch.atan2(-rotmat[singular, 1, 2], rotmat[singular, 1, 1])
    y[singular] = torch.atan2(-rotmat[singular, 2, 0], sy[singular])
    z[singular] = 0.0

    return torch.stack([x, y, z], dim=1)
    
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

# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--object_code_list', default=
    [
        #'sem-Car-2f28e2bd754977da8cfac9da0ff28f62',
        #'sem-Car-27e267f0570f121869a949ac99a843c4',
        'sem-Hammer-5d4da30b8c0eaae46d7014c7d6ce68fc',
        #'core-mug-1a1c0a8d4bad82169f0594e65f756cf5',
        #'sem-Gun-898424eaa40e821c2bf47341dbd96eb',
        #'ddg-gd_champagne_glass_poisson_001',
        #'sem-Pencil-8bf01717c0d166fd271f8bae2df0a074',
        #'sem-USBStick-1baa93373407c8924315bea999e66ce3',
        #'ddg-gd_pliers_poisson_015',
        #'ddg-gd_soccer_ball_poisson_000',
        #'mujoco-Footed_Bowl_Sand',
        #'core-knife-bcc178786ae893835f7f383d5cbb672d',
        #'sem-Book-1fd8d3cdf525532b5b5a685c28abd3e',
        #'mujoco-ASICS_GEL1140V_WhiteBlackSilver',
        #'ddg-kit_CokePlasticLarge',
        #'ddg-gd_rubik_cube_poisson_004',
        #'ddg-ycb_044_flat_screwdriver'
        #'sem-Pan-436e1ae9112384fbf4cc5d95933e54b',
    ], type=list)
parser.add_argument('--name', default='exp_32', type=str)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
# hyper parameters (** Magic, don't touch! **)
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--step_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_prior', default=0.5, type=float)
parser.add_argument('--w_spen', default=10.0, type=float)
# initialization settings
parser.add_argument('--jitter_strength', default=0., type=float)
parser.add_argument('--distance_lower', default=0.1, type=float)
parser.add_argument('--distance_upper', default=0.1, type=float)
parser.add_argument('--theta_lower', default=0, type=float)
parser.add_argument('--theta_upper', default=0, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

args = parser.parse_args()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# prepare models

total_batch_size = len(args.object_code_list) * args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

hand_model = HandModel(
    mano_root='mano', 
    contact_indices_path='mano/contact_indices.json', 
    pose_distrib_path='mano/pose_distrib.pt', 
    device=device
)

object_model = ObjectModel(
    data_root_path='../data/meshdata',
    batch_size_each=args.batch_size,
    num_samples=2000, 
    device=device
)
object_model.initialize(args.object_code_list)

initialize_convex_hull(hand_model, object_model, args)

print('total batch size', total_batch_size)
hand_pose_st = hand_model.hand_pose.detach()

optim_config = {
    'switch_possibility': args.switch_possibility,
    'starting_temperature': args.starting_temperature,
    'temperature_decay': args.temperature_decay,
    'annealing_period': args.annealing_period,
    'step_size': args.step_size,
    'stepsize_period': args.stepsize_period,
    'mu': args.mu,
    'device': device
}
optimizer = Annealing(hand_model, **optim_config)

try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'logs'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'logs'), exist_ok=True)
logger_config = {
    'thres_fc': args.thres_fc,
    'thres_dis': args.thres_dis,
    'thres_pen': args.thres_pen
}
logger = Logger(log_dir=os.path.join('../data/experiments', args.name, 'logs'), **logger_config)


# log settings

with open(os.path.join('../data/experiments', args.name, 'output.txt'), 'w') as f:
    f.write(str(args) + '\n')


# optimize

weight_dict = dict(
    w_dis=args.w_dis,
    w_pen=args.w_pen,
    w_prior=args.w_prior,
    w_spen=args.w_spen
)
energy, E_fc, E_dis, E_pen, E_prior, E_spen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

energy.sum().backward(retain_graph=True)
logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, 0, show=False)

for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
    s = optimizer.try_step()

    optimizer.zero_grad()
    new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_prior, new_E_spen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    new_energy.sum().backward(retain_graph=True)

    with torch.no_grad():
        accept, t = optimizer.accept_step(energy, new_energy)

        energy[accept] = new_energy[accept]
        E_dis[accept] = new_E_dis[accept]
        E_fc[accept] = new_E_fc[accept]
        E_pen[accept] = new_E_pen[accept]
        E_prior[accept] = new_E_prior[accept]
        E_spen[accept] = new_E_spen[accept]

        logger.log(energy, E_fc, E_dis, E_pen, E_prior, E_spen, step, show=False)


# save results
try:
    shutil.rmtree(os.path.join('../data/experiments', args.name, 'results'))
except FileNotFoundError:
    pass
os.makedirs(os.path.join('../data/experiments', args.name, 'results'), exist_ok=True)
result_path = os.path.join('../data/experiments', args.name, 'results')
os.makedirs(result_path, exist_ok=True)
for i in range(len(args.object_code_list)):
    data_list = []
    data__isaacgym_list = []
    for j in range(args.batch_size):
        idx = i * args.batch_size + j
        scale = object_model.object_scale_tensor[i][j].item()
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        rotation_matrix = rodrigues_layer.batch_rodrigues(hand_pose[3:6].unsqueeze(0))
        rot = rotmat_to_quat(rotation_matrix.unsqueeze(0).view(1,3,3)).squeeze(0)
        qpos = dict(
            trans=hand_pose[:3].tolist(),
            rot=hand_pose[3:6].tolist(),
            thetas=hand_pose[6:].tolist(),
        )
        hand_pose = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(
            trans=hand_pose[:3].tolist(),
            rot=hand_pose[3:6].tolist(),
            thetas=hand_pose[6:].tolist(),
        )
        data_list.append(dict(
            scale=scale,
            qpos=qpos,
            contact_point_indices=hand_model.contact_point_indices[idx].detach().cpu().tolist(), 
            qpos_st=qpos_st,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_prior=E_prior[idx].item(),
            E_spen=E_spen[idx].item(),
        ))
        #print(data_list)
        np.save(os.path.join(result_path, args.object_code_list[i] + '.npy'), data_list, allow_pickle=True)
        hand_pose = hand_model.hand_pose[idx].detach().cpu()
        
        
        
        
        results = []  # Initialize an empty list to store results
        for j in range(6,  hand_pose.shape[0], 3):
            axis = hand_pose[j:j + 3]
            result = rotmat_to_extrinsic_euler_xyz(rodrigues_layer.batch_rodrigues(torch.tensor(axis).unsqueeze(0)).unsqueeze(0).view(1,3,3))
            results.append(result)
        final_output = torch.cat(results, dim=0)  # Dim 0 for batch dimension
        final_output = final_output.view(-1) 







        rotation_matrix = rodrigues_layer.batch_rodrigues(hand_pose[3:6].unsqueeze(0))
        rot = rotmat_to_quat(rotation_matrix.unsqueeze(0).view(1,3,3)).squeeze(0)
        qpos = dict(
            #trans=hand_pose[:3].tolist(),
            trans=hand_model.keypoints[idx,0],
            #rot=hand_pose[3:6].tolist(),
            rot=rot.tolist(),
            thetas=hand_pose[6:].tolist(),
        )
        hand_pose = hand_pose_st[idx].detach().cpu()
        qpos_st = dict(
            trans=hand_pose[:3].tolist(),
            rot=hand_pose[3:6].tolist(),
            #thetas=hand_pose[6:].tolist(), ################
            thetas=final_output.tolist(),
        )
        data__isaacgym_list.append(dict(
            scale=scale,
            qpos=qpos,
            contact_point_indices=hand_model.contact_point_indices[idx].detach().cpu().tolist(), 
            qpos_st=qpos_st,
            energy=energy[idx].item(),
            E_fc=E_fc[idx].item(),
            E_dis=E_dis[idx].item(),
            E_pen=E_pen[idx].item(),
            E_prior=E_prior[idx].item(),
            E_spen=E_spen[idx].item(),
        ))
        np.save(os.path.join(result_path, args.object_code_list[i] + '_for_isaacgym.npy'), data__isaacgym_list, allow_pickle=True)
