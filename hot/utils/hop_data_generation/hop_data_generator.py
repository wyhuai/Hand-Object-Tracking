
from hop_grasp_change_root import GraspRootProcessor
from hop_move import GraspMoveGenerator
from hop_grasp_place import GraspAndPlaceGenerator
# from hop_insert import InsertGraspGenerator
# from hop_regrasp_hard import RegraspGenerator
# from hop_inhand_rotation_easy import InhandRotation
# from hop_catch_action_sample import CatchOrientationGenerator
# from hop_catch_obj import CatchObjTrajGen
# from hop_catch import CatchGenerator
# from hop_throw_obj import ThrowObjTrajGen
# from hop_throw_action_sample_new import ThrowDirectionGenerator
# from hop_throw import ThrowGenerator

from free_drop import FindStablePose
from free_drop_test import FindStablePoseTest

import os
import subprocess
import argparse
import torch
import numpy as np
import json

# Define the directory and script paths
original_dir = os.getcwd()

# Create the argument parser
parser = argparse.ArgumentParser(description="Select a method for processing.")
    
# Add the 'method' argument

parser.add_argument('--root_pos',
                    type=str,
                    help="Comma-separated values for the list, e.g., '1.0,2.0,3.0'")

parser.add_argument('--root_quat',
                    type=str,
                    help="Comma-separated values for the list, e.g., '1.0,2.0,3.0,4.0'")


parser.add_argument('--obj_name', 
                    type=str, default=None, 
                    help="File name for the grasp trajectory to be saved.")

#unused
parser.add_argument('--folder_name', 
                    type=str, default=None, 
                    help="Folder name for the grasp trajectory to be saved.")

parser.add_argument('--hand_model', 
                    type=str, default="mano", 
                    help="allegro or shadow or mano")

parser.add_argument('--headless',                        
                    action='store_true', 
                    help="Rebuild the environments")

parser.add_argument('--rotation_path', 
                    type=str, default=None, 
                    help="File path of starting unique rotation in txt format")

parser.add_argument('--asset_path', 
                        type=str,  
                        help="The urdf file of the asset.")
args = parser.parse_args()

if args.root_pos:
    root_pos = torch.tensor([float(x) for x in args.list_arg.split(',')])
else:
    root_pos = torch.tensor([0.0, 0.0, 0.5]) 

if args.root_quat:
    root_quat = torch.tensor([float(x) for x in args.list_arg.split(',')])
else:
    root_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]) 

if args.hand_model is not None:
    args.hand_model = args.hand_model.lower()

if args.hand_model == "mano":
    grasp_dir = "dexgrasp_mano"
elif args.hand_model == "allegro":
    grasp_dir = "dexgrasp_allegro"
elif args.hand_model == "shadow":
    grasp_dir = "dexgrasp_shadow"

grasp_directory = f'../../data/motions/graspmimic/{grasp_dir}/{args.obj_name}'
sorted_filenames = sorted(os.listdir(grasp_directory), key=lambda x: int(x.split('_')[-1].split('.pt')[0]))


# Get the current working directory
os.chdir(original_dir)

if args.hand_model == "mano":
    output_dir = "dexgrasp_train_mano"
elif args.hand_model == "allegro":
    output_dir = "dexgrasp_train_allegro"
elif args.hand_model == "shadow":
    output_dir = "dexgrasp_train_shadow"

motion_dir = os.path.join(f'../../data/motions/{output_dir}', args.obj_name)
os.makedirs(motion_dir, exist_ok=True)
os.makedirs(os.path.join(motion_dir, "full_data"), exist_ok=True)
os.makedirs(os.path.join(motion_dir, "grasp_rot"), exist_ok=True)


extracted_grasp_tensors = []

for i, filename in enumerate(sorted_filenames):
        if filename.endswith('.pt'):
            file_path = os.path.join(grasp_directory, filename)
            # Load the .pt file
            data = torch.load(file_path)
            extracted_grasp_tensors.append(data)
            torch.save(data, f'{motion_dir}/full_data/000_{args.obj_name}_{0}_{i}.pt') 

merged_tensor = {}
# Stack all extracted tensors into a single tensor
if extracted_grasp_tensors:
    # Concatenate along the first dimension for each key
    for key in extracted_grasp_tensors[0].keys():  # Assuming all dictionaries have the same keys
        if key == 'obj_pos_vel':
            merged_tensor[key] = None  # Set to None if the key is 'obj_pos_vel'
        else:
            merged_tensor[key] = torch.cat([d[key] for d in extracted_grasp_tensors], dim=0)   
    # Save the merged tensor to a new .pt file
    torch.save(merged_tensor, f'{motion_dir}/grasp_rot/000_{args.obj_name}_{0}_{0}.pt') 

    
GraspRootProcessor(obj_name = args.obj_name,
                   basic_grasps_path = f"{motion_dir}/grasp_rot",
                   output_path = f"{motion_dir}/grasp_rot_transformed_root",
                   root_pos = root_pos,
                   root_quat = root_quat,
                    hand_model= args.hand_model
                 )


GraspMoveGenerator(obj_name = args.obj_name,
                   basic_grasps_path = f"{motion_dir}/grasp_rot_transformed_root",
                   output_path = f"{motion_dir}/move",
                   num_data_samples = 10,
                   hand_model= args.hand_model
                 )


if not args.rotation_path:
    if os.path.isfile(f"../../data/free_drop_test_image/{(args.obj_name).capitalize()}/unique_quats.txt"):
        args.rotation_path = os.path.abspath(f"../../data/free_drop_test_image/{(args.obj_name).capitalize()}/unique_quats.txt")
        print("The stable pos directory exists.")
    elif os.path.isfile(f"../../data/free_drop_test_image/{(args.obj_name).upper()}/unique_quats.txt"):
        args.rotation_path = os.path.abspath(f"../../data/free_drop_test_image/{(args.obj_name).upper()}/unique_quats.txt")
        print("The stable pos directory exists.")
    else:
        pose_finder = FindStablePose(asset_path = args.asset_path, save_images = True, headless = args.headless)
        stable_pos_path = pose_finder.compute_pose()
        pose_finder.release_GPU()
        pose_confirmer = FindStablePoseTest(asset_path = args.asset_path, save_images = True, headless = args.headless, stable_pos_path=stable_pos_path)
        args.rotation_path = pose_confirmer.compute_pose()
        pose_confirmer.release_GPU()

GraspAndPlaceGenerator(obj_name = args.obj_name,
                   basic_grasps_path = f"{motion_dir}/grasp_rot_transformed_root",
                   stable_pose_path =args.rotation_path,
                   output_path = f"{motion_dir}/grasp",
                   output_place_path = f"{motion_dir}/place",
                   num_data_samples = 5,    
                   operation_space_ranges=None,
                   span_angle=45,
                   hand_model= args.hand_model
                 )



# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['DRI_PRIME'] = '1'
# command = [
#     'python',
#     'hot/run.py',
#     '--test',
#     '--task', 'SkillMimicBallPlay',
#     '--num_envs', '1',
#     '--cfg_env', 'hot/data/cfg/skillmimic.yaml',
#     '--cfg_train', 'hot/data/cfg/train/rlg/skillmimic.yaml',
#     '--motion_file', absolute_motion_dir,
#     '--object_asset', args.asset_path,
#     '--postproc_hopdata',
#     '--headless'
# ]
# subprocess.run(command, check=True)

