import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# Define the base path to your experiments folder
base_path = '../DexGraspNet_allegro/data/experiments_allegro'  # Change this to your actual path
output_path = './data/motions/graspmimic/'  # Change this to your actual path

# Initialize a list to store the first 20 indices
def duplicate_last_frame(generated_data):
        num_copies = 1 # this is number of repeated frame not number of frame
        # Create a new dictionary to hold the modified data
        extended_data = {}

        # Iterate over each key in the original generated_data
        for key, value in generated_data.items():
            if key != 'obj_pos_vel' :
                # Copy the last frame for the specified number of times
                last_frame = value[-1:]  # Get the last frame
                extended_data[key] = torch.cat((value, last_frame.repeat(num_copies, 1)), dim=0)
            else: 
                extended_data[key] = None
        return extended_data


# Initialize a list to store the first 20 indices
def quat_from_euler_xyz_intrinsic(x, y, z):
    # Intrinsic rotations: applied to the moving coordinate frame
    quat = R.from_euler('ZYX', [z, y, x]).as_quat()
    return torch.tensor(quat).float()


theta_dict = {
    'joint_0.0': 6,
    'joint_1.0': 7,
    'joint_2.0': 8,
    'joint_3.0': 9,
    'joint_4.0': 10,
    'joint_5.0': 11,
    'joint_6.0': 12,
    'joint_7.0': 13,
    'joint_8.0': 14,
    'joint_9.0': 15,
    'joint_10.0': 16,
    'joint_11.0': 17,
    'joint_12.0': 18,
    'joint_13.0': 19,
    'joint_14.0': 20,
    'joint_15.0': 21,

}
sorted_theta_keys = sorted(theta_dict.keys(), key=lambda x: theta_dict.get(x, float('inf')))


# Process the selected dictionaries
for experiment_folder in os.listdir(base_path):
    experiment_path = os.path.join(base_path, experiment_folder)
    first_20_indices = []

    if os.path.isdir(experiment_path):
        exp_33_path = os.path.join(experiment_path, 'exp_33')
        
        # Check if 'exp_33' exists
        if os.path.exists(exp_33_path):
            # Read saved_ids.txt
            saved_ids_path = os.path.join(exp_33_path, 'saved_ids.txt')
            with open(saved_ids_path, 'r') as file:
                lines = file.readlines()
                first_20_indices = [int(line.strip()) for line in lines[:20]]
            # Load the .npy file from the results subfolder
            results_path = os.path.join(exp_33_path, 'results')
            npy_files = [f for f in os.listdir(results_path) if f.endswith('.npy')]
            if npy_files:
                npy_file_path = os.path.join(results_path, npy_files[0])
                qpos_data_list = np.load(npy_file_path, allow_pickle=True)

                # Extract the object name from the experiment folder
                obj_name = experiment_folder.split('_')[1]  # Assuming the format is 'experiments_obj_XX'

                # Create output directory for saved .pt files
                output_dir = os.path.join(output_path, 'dexgrasp_allegro', obj_name)
                os.makedirs(output_dir, exist_ok=True)

                for idx in first_20_indices:
                    qpos_data = qpos_data_list[idx]
                    qpos_data_dict = {
                        'scale': qpos_data['scale'],
                        'trans': torch.tensor([qpos_data['qpos']['WRJTx'], qpos_data['qpos']['WRJTy'], qpos_data['qpos']['WRJTz']]),
                        'rot': quat_from_euler_xyz_intrinsic(qpos_data['qpos']['WRJRx'], qpos_data['qpos']['WRJRy'], qpos_data['qpos']['WRJRz']),
                        'thetas': torch.tensor([qpos_data['qpos'][key] for key in sorted_theta_keys if key in qpos_data['qpos']])
                    }
                    print([qpos_data['qpos'][key] for key in sorted_theta_keys if key in qpos_data['qpos']])
                    qpos_data_dict['trans'] = qpos_data_dict['trans'].to('cpu')
                    qpos_data_dict['rot'] = qpos_data_dict['rot'].to('cpu')
                    qpos_data_dict['thetas'] = qpos_data_dict['thetas'].to('cpu')
                    print(idx, obj_name, qpos_data_dict['scale'] )
                    # Process right hand data
                    right_hand_trans = qpos_data_dict['trans'] + torch.tensor([[0, 0, 1]], dtype=torch.float32).to('cpu')
                    right_hand_rot_quat = qpos_data_dict['rot']
                    #right_hand_dof_pos = torch.cat((torch.zeros(6, dtype=torch.float32).to('cpu'), qpos_data_dict['thetas'])).view(1, 28)
                    right_hand_dof_pos = qpos_data_dict['thetas'].view(1, 16)
                    print(right_hand_trans)

                    right_hand_pose = torch.zeros((1, 17*3))  # Adjust shape as needed

                    hoi_data = {}
                    hoi_data['root_pos'] = right_hand_trans
                    hoi_data['root_rot'] = right_hand_rot_quat.unsqueeze(0)
                    hoi_data['wrist_dof'] = torch.zeros((1,6), dtype=torch.float32).to('cpu')
                    hoi_data['fingers_dof'] = qpos_data_dict['thetas'].unsqueeze(0)
                    hoi_data['body_pos'] = right_hand_pose

                    hoi_data['obj_pos'] = torch.zeros(1, 3) + torch.tensor([[0, 0, 1]])
                    hoi_data['obj_pos_vel'] = None
                    hoi_data['obj_rot'] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])


                    hoi_data['obj2_pos'] = hoi_data['obj_pos'].clone() * 0 + 2
                    hoi_data['obj2_rot'] = hoi_data['obj_rot'].clone() * 0

                    hoi_data['contact1'] = torch.ones(1, 1)
                    hoi_data['contact2'] = torch.zeros(1, 1)
                    # Save each hoi_data to a separate .pt file
                    for key, value in hoi_data.items():
                        if value is not None:
                            print(f"{key}: {value.shape}")
                        else:
                            print(f"{key}: None")

                    repeated_hoi_data = duplicate_last_frame(hoi_data)
                    file_path = os.path.join(output_dir, f'000_{obj_name}_{idx}.pt')
                    torch.save(hoi_data, file_path)
                    #torch.save(repeated_hoi_data, file_path)

        print("Processing and saving completed.")
