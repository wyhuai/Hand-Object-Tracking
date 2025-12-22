import torch
import numpy as np
import os

# Define the base path to your experiments folder
base_path = '../DexGraspNet/data/experiments_mano'  # Change this to your actual path
output_path = './data/motions/graspmimic/'  # Change this to your actual path

# Initialize a list to store the first 20 indices
def duplicate_last_frame(generated_data):
        num_copies = 100
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

# Traverse through each top-level experiment folder
for experiment_folder in os.listdir(base_path):
    experiment_path = os.path.join(base_path, experiment_folder)
    first_20_indices = []

    if os.path.isdir(experiment_path):
        exp_32_path = os.path.join(experiment_path, 'exp_32')
        
        # Check if 'exp_32' exists
        if os.path.exists(exp_32_path):
            # Read saved_ids.txt
            saved_ids_path = os.path.join(exp_32_path, 'saved_ids.txt')
            with open(saved_ids_path, 'r') as file:
                lines = file.readlines()
                first_20_indices = [int(line.strip()) for line in lines[:20]]

            # Load the .npy file from the results subfolder
            results_path = os.path.join(exp_32_path, 'results')
            npy_files = [f for f in os.listdir(results_path) if f.endswith('_isaacgym.npy')]
            if npy_files:
                npy_file_path = os.path.join(results_path, npy_files[0])
                qpos_data_list = np.load(npy_file_path, allow_pickle=True)

                # Extract the object name from the experiment folder
                obj_name = experiment_folder.split('_')[1]  # Assuming the format is 'experiments_obj_XX'

                # Create output directory for saved .pt files
                output_dir = os.path.join(output_path, 'dexgrasp_mano', obj_name)
                os.makedirs(output_dir, exist_ok=True)

                # Process the selected dictionaries
                for idx in first_20_indices:
                    if idx < len(qpos_data_list):
                        qpos_data = qpos_data_list[idx]

                        qpos_data_dict = {
                            'scale': qpos_data['scale'],
                            'trans': torch.tensor(qpos_data['qpos']['trans']),
                            'rot': torch.tensor(qpos_data['qpos']['rot']),
                            'thetas': torch.tensor(qpos_data['qpos']['thetas'])
                        }
                        qpos_data_dict['trans'] = qpos_data_dict['trans'].to('cpu')
                        qpos_data_dict['rot'] = qpos_data_dict['rot'].to('cpu')
                        qpos_data_dict['thetas'] = qpos_data_dict['thetas'].to('cpu')
                        print(idx, obj_name, qpos_data_dict['scale'] )
                        # Process right hand data
                        right_hand_trans = qpos_data_dict['trans'] + torch.tensor([[0, 0, 1]], dtype=torch.float32).to('cpu')
                        right_hand_rot_quat = qpos_data_dict['rot']
                        right_hand_dof_pos = torch.cat((torch.zeros(6, dtype=torch.float32).to('cpu'), qpos_data_dict['thetas'])).view(1, 51)

                        right_hand_pose = torch.zeros((1, 48))  # Adjust shape as needed

                        # Prepare hoi_data
                        '''
                        hoi_data = torch.zeros((right_hand_trans.shape[0], 119), dtype=torch.float)
                        hoi_data[:right_hand_trans.shape[0], 0:3] = right_hand_trans
                        hoi_data[:right_hand_rot_quat.shape[0], 3:7] = right_hand_rot_quat
                        hoi_data[:right_hand_dof_pos.shape[0], 7:7 + 51] = right_hand_dof_pos
                        hoi_data[:right_hand_pose.shape[0], 58:58 + 45] = right_hand_pose

                        obj_trans = torch.zeros(1, 3) + torch.tensor([[0, 0, 1]])
                        obj_rot = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
                        hoi_data[:obj_trans.shape[0], 103:103 + 3] = obj_trans
                        hoi_data[:obj_rot.shape[0], 106:106 + 4] = obj_rot

                        obj_trans = obj_trans.clone() * 0 + 2
                        obj_rot = obj_rot.clone() * 0
                        hoi_data[:obj_trans.shape[0], 110:110 + 3] = obj_trans
                        hoi_data[:obj_rot.shape[0], 113:113 + 4] = obj_rot

                        hoi_data[:obj_rot.shape[0], 117:117 + 1] = torch.ones(1, 1)
                        hoi_data[:obj_rot.shape[0], 118:118 + 1] = torch.zeros(1, 1)
                        #repeated_hoi_data = hoi_data[:right_hand_pose.shape[0]].repeat(10, 1)
                        '''
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
                        file_path = os.path.join(output_dir, f'000_{obj_name}_{idx}.pt')
                        print(file_path)
                        repeated_hoi_data = duplicate_last_frame(hoi_data)
                        torch.save(hoi_data, file_path)
                        #torch.save(repeated_hoi_data, file_path)

print("Processing and saving completed.")