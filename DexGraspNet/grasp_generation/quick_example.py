# %%

# %%
import os
import random
from utils.hand_model_lite import HandModelMJCFLite
import numpy as np
import transforms3d
import torch
import trimesh


# %%
mesh_path = "../data/meshdata"
data_path = "../data/dataset"

use_visual_mesh = False

hand_file = "mjcf/shadow_hand_vis.xml" if use_visual_mesh else "mjcf/shadow_hand_wrist_free.xml"

joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']


# %%
hand_model = HandModelMJCFLite(
    hand_file,
    "mjcf/meshes")


# %%
grasp_code_list = []
for code in os.listdir(data_path):
    grasp_code_list.append(code[:-4])


# %%
grasp_code = random.choice(grasp_code_list)
grasp_data = np.load(
    os.path.join(data_path, grasp_code+".npy"), allow_pickle=True)
object_mesh_origin = trimesh.load(os.path.join(
    mesh_path, grasp_code, "coacd/decomposed.obj"))
print(grasp_code)


# %%
index = random.randint(0, len(grasp_data) - 1)


qpos = grasp_data[index]['qpos']
rot = np.array(transforms3d.euler.euler2mat(
    *[qpos[name] for name in rot_names]))
rot = rot[:, :2].T.ravel().tolist()
hand_pose = torch.tensor([qpos[name] for name in translation_names] + rot + [qpos[name]
                         for name in joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
hand_model.set_parameters(hand_pose)
hand_mesh = hand_model.get_trimesh_data(0)
object_mesh = object_mesh_origin.copy().apply_scale(grasp_data[index]["scale"])


# %%
(hand_mesh+object_mesh).show()



