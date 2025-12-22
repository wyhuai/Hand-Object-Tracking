import pickle
import os
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
pkl_file_path = '../mano/MANO_RIGHT.pkl'

smpl_data = ready_arguments(pkl_file_path)

#hands_components = smpl_data['th_hands_mean_rotmat']
hands_components = smpl_data['kintree_table']
print(smpl_data)
# Define the path to your .pkl file
pkl_file_path = '../mano/MANO_RIGHT.pkl'

# Check if the file exists
if os.path.isfile(pkl_file_path):
    # Open the file and load the data
    with open(pkl_file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Print or process the loaded data
    print(data)
else:
    print("File not found.")