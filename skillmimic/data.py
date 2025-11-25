import torch
import numpy as np
import os

# Load the .pt file
# pt_file_path = '/home/super/Downloads/skill/graspskill/skillmimic/data/motions/BallPlay-M/layup/031_015pickle_layup_001_001.pt'
# old_hoi_data = torch.load(pt_file_path)
# Load the .npy file
npy_file_path = '/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/diverse_seq_npy'
for npy_file_name in os.listdir(npy_file_path):
    # Check if the file has a .npy extension
    if npy_file_name.endswith('.npy'):
        # Construct the full file path
        file_path = os.path.join(npy_file_path, npy_file_name)    
        data_npy = np.load(file_path, allow_pickle=True).item()  # Load as dictionary
        obj_item = npy_file_name.split('.', 1)[0]
        right_hand_trans = torch.tensor(data_npy['right_hand']['trans'])  # Shape: (140, 3)
        right_hand_rot = torch.tensor(data_npy['right_hand']['rot'])      # Shape: (140, 3)
        right_hand_dof_pos = torch.tensor(data_npy['right_hand']['dof_pos'])    # Shape: (140, 51)
        right_hand_pose = torch.tensor(data_npy['right_hand']['pose'])    # Shape: (140, 45)
        #print(data_npy['right_hand']['trans'])
        hoi_data = torch.zeros((right_hand_trans.shape[0], 111), dtype=torch.float)

        hoi_data[:right_hand_trans.shape[0], 0:3] = right_hand_trans
        hoi_data[:right_hand_rot.shape[0], 3:7] = right_hand_rot
        hoi_data[:right_hand_dof_pos.shape[0], 7:7+51] = right_hand_dof_pos
        hoi_data[:right_hand_pose.shape[0], 58:58+45] = right_hand_pose


        obj_trans = torch.tensor(data_npy[obj_item]['trans'])  # Shape: (140, 3)
        obj_rot = torch.tensor(data_npy[obj_item]['rot'])      # Shape: (140, 3)        
        #obj_angle = torch.tensor(data_npy[obj_item]['angle'])   # Shape: (140, 1)
        hoi_data[:obj_trans.shape[0], 103:103+3] = obj_trans
        hoi_data[:obj_rot.shape[0], 106:106+4] = obj_rot
        hoi_data[:obj_rot.shape[0], 110:110+1] = torch.zeros_like(hoi_data[:obj_rot.shape[0], 110:110+1])#torch.round(old_hoi_data[:140, 330+6:331+6].clone())
        folder_name =  obj_item.split('_', 1)[0]
        os.makedirs(os.path.join('/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/graspmimic', folder_name), exist_ok=True)
        print(folder_name)
        # Optionally save the updated tensor back to a new .pt file
        torch.save(hoi_data[:right_hand_pose.shape[0]], '/home/hkust/yinhuai/skillmimic_hand/skillmimic/data/motions/graspmimic/'+folder_name+'/'+obj_item+'.pt') #only save to number frame for npy as it is shorter
        print(hoi_data.shape)
        #print(right_hand_trans)        
        #print(obj_trans)
        #print(right_hand_trans)     
        if folder_name == "Camera":
            print(right_hand_trans)