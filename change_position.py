import torch
import glob
import os
main_path =  'skillmimic/data/motions/dexgrasp_train_shadow/bottle/grasp_kp/*.pt'
paths = glob.glob(main_path)

if 'allegro' in main_path:
    len_body_pos = 17
elif 'shadow' in main_path:
    len_body_pos = 23
else:
    len_body_pos = 16

dir_folder = os.path.dirname(paths[0])
dir_folder_name = os.path.basename(dir_folder)
new_dir_folder_name = dir_folder_name.replace('kp', 'higher_kp')
os.makedirs(os.path.dirname(paths[0].replace(dir_folder_name, new_dir_folder_name)), exist_ok=True)

for path in paths:
    x = torch.load(path)
    for key in x:
        if key == 'body_pos':
            x[key] = x[key].reshape(-1,len_body_pos,3)
            x[key][..., 2] += 0.1
            x[key] = x[key].reshape(-1,len_body_pos*3)
        if key == 'wrist_dof':
            x[key][..., 2] += 0.1
        if key == 'obj_pos':
            x[key][..., 2] += 0.1

    new_path = path.replace(dir_folder_name, new_dir_folder_name)
    print(new_path)
    torch.save(x, new_path)