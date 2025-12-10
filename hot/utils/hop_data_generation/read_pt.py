import time
import torch
import pytorch_kinematics as pk
import viser
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

VISUALIZE = False
FPS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROBOT_SPHERE_RADIUS = 0.003
ROBOT_SPHERE_COLOR = (1.0, 0.0, 0.0)
BODY_SPHERE_RADIUS = 0.0035
BODY_SPHERE_COLOR = (0.0, 1.0, 0.0)

HAND_CONFIGS = {
    'mano': {
        'xml_path': "hot/data/assets/mjcf/rhand_mano_pk.xml",
        'key_bodies': [
            "right_index1_z", "right_index2_z", "right_index3_z",
            "right_middle1_z", "right_middle2_z", "right_middle3_z",
            "right_pinky1_z", "right_pinky2_z", "right_pinky3_z",
            "right_ring1_z", "right_ring2_z", "right_ring3_z",
            "right_thumb1_z", "right_thumb2_z", "right_thumb3_z",
            "right_wrist"
        ]
    },
    'shadow': {
        'xml_path': "hot/data/assets/mjcf/shadow/shadow_hand_kp.xml", 
        'key_bodies': [
            "robot0:ffknuckle", "robot0:ffproximal", "robot0:ffmiddle", "robot0:ffdistal",
            "robot0:mfknuckle", "robot0:mfproximal", "robot0:mfmiddle", "robot0:mfdistal", 
            "robot0:rfknuckle", "robot0:rfproximal", "robot0:rfmiddle", "robot0:rfdistal", 
            "robot0:lfmetacarpal", "robot0:lfknuckle", "robot0:lfproximal", "robot0:lfmiddle", "robot0:lfdistal", 
            "robot0:thbase", "robot0:thproximal", "robot0:thhub", "robot0:thmiddle", "robot0:thdistal",
            "robot0:palm"
        ]
    },
    'allegro': {
        'xml_path': "hot/data/assets/mjcf/allegro/allegro_hand_pk.xml",
        'key_bodies': [
            "link_0.0","link_1.0", "link_2.0", "link_3.0", "link_4.0", 
            "link_5.0","link_6.0", "link_7.0", "link_8.0", "link_9.0", 
            "link_10.0", "link_11.0", "link_12.0", "link_13.0", "link_14.0", 
            "link_15.0",'base_link'
        ]
    }
}

def load_robot_chain(xml_path, device):
    with open(xml_path, "r") as f: xml_data = f.read()
    try: return pk.build_chain_from_mjcf(xml_data).to(dtype=torch.float32, device=device)
    except: return pk.build_chain_from_urdf(xml_data).to(dtype=torch.float32, device=device)

def load_pt_data(path):
    try:
        raw_data = torch.load(path, map_location=DEVICE)
    except Exception as e:
        print(f"Error reading: {path.name} - {e}")
        return None, None, None, None

    qpos, root_pos, root_rot = None, None, None
    
    if isinstance(raw_data, dict):
        if 'qpos' in raw_data: qpos = raw_data['qpos']
        elif 'q' in raw_data: qpos = raw_data['q']
        elif 'wrist_dof' in raw_data: qpos = torch.cat([raw_data['wrist_dof'], raw_data['fingers_dof']], dim=1)
            
        if 'root_pos' in raw_data: root_pos = raw_data['root_pos']
        if 'root_rot' in raw_data: root_rot = raw_data['root_rot']
    else:
        qpos = raw_data 

    if qpos is not None:
        if qpos.dim() == 1: qpos = qpos.unsqueeze(0)
        qpos = qpos.to(dtype=torch.float32, device=DEVICE)
        
    if root_pos is not None:
        if root_pos.dim() == 1: root_pos = root_pos.unsqueeze(0)
        root_pos = root_pos.to(dtype=torch.float32, device=DEVICE)

    if root_rot is not None:
        if root_rot.dim() == 1: root_rot = root_rot.unsqueeze(0)
        root_rot = root_rot.to(dtype=torch.float32, device=DEVICE)
    
    return raw_data, qpos, root_pos, root_rot

def process_single_file(file_path, chain, target_key_bodies, server=None):
    file_path = Path(file_path)
    
    raw_data_dict, qpos_seq, root_pos_seq, root_rot_seq = load_pt_data(file_path)
    if qpos_seq is None: return

    n_joints = len(chain.get_joint_parameter_names())
    if qpos_seq.shape[1] > n_joints: qpos_seq = qpos_seq[:, :n_joints]
    num_frames = qpos_seq.shape[0]

    with torch.no_grad():
        tg_batch = chain.forward_kinematics(qpos_seq)

        if root_pos_seq is not None and root_rot_seq is not None:
            root_rot_wxyz = root_rot_seq[:, [3, 0, 1, 2]]
            root_tf = pk.Transform3d(pos=root_pos_seq, rot=root_rot_wxyz, device=DEVICE)
            
            new_tg_batch = {}
            for link_name, local_tf in tg_batch.items():
                new_tg_batch[link_name] = root_tf.compose(local_tf)
            tg_batch = new_tg_batch

        all_link_names = list(tg_batch.keys())
        matrix_list = [t.get_matrix() for t in tg_batch.values()]
        all_matrices = torch.stack(matrix_list).permute(1, 0, 2, 3)
        pos_3d = all_matrices[..., :3, 3]

        key_body_ids = []
        for name in target_key_bodies:
            try: key_body_ids.append(all_link_names.index(name))
            except ValueError: pass
        
        if len(key_body_ids) == 0:
            print(f"Warning: No key bodies found for {file_path.name}")
            return

        key_body_pose = pos_3d[:, key_body_ids, :]
        key_body_pose_flat = key_body_pose.reshape(num_frames, -1)

    if not isinstance(raw_data_dict, dict):
        output_dict = {'qpos': qpos_seq.cpu()}
        if root_pos_seq is not None: output_dict['root_pos'] = root_pos_seq.cpu()
        if root_rot_seq is not None: output_dict['root_rot'] = root_rot_seq.cpu()
    else:
        output_dict = raw_data_dict

    output_dict['body_pos'] = key_body_pose_flat.cpu()

    parent_dir = file_path.parent
    new_dir_name = parent_dir.name + "_kp" 
    new_save_dir = parent_dir.parent / new_dir_name
    new_save_dir.mkdir(parents=True, exist_ok=True)
    save_path = new_save_dir / file_path.name
    
    torch.save(output_dict, save_path)
    
    if server is not None:
        pos_3d_np = pos_3d.cpu().numpy()
        key_body_pose_np = key_body_pose.cpu().numpy().reshape(num_frames, -1, 3)
        
        for i in range(num_frames):
            step_start = time.time()
            server.scene.add_point_cloud(
                name="/robot/full_joints",
                points=pos_3d_np[i],
                colors=np.tile(ROBOT_SPHERE_COLOR, (pos_3d_np[i].shape[0], 1)),
                point_size=ROBOT_SPHERE_RADIUS,
            )
            server.scene.add_point_cloud(
                name="/robot/saved_body_pos",
                points=key_body_pose_np[i],
                colors=np.tile(BODY_SPHERE_COLOR, (key_body_pose_np[i].shape[0], 1)),
                point_size=BODY_SPHERE_RADIUS,
            )
            time.sleep(max(0, (1.0 / FPS) - (time.time() - step_start)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Input folder path")
    parser.add_argument("--hand", type=str, required=True, choices=['shadow', 'allegro', 'mano'], help="Hand type")
    args = parser.parse_args()

    hand_config = HAND_CONFIGS[args.hand]
    xml_path = hand_config['xml_path']
    target_key_bodies = hand_config['key_bodies']

    print(f"Selected Hand: {args.hand}")
    print(f"Model Path: {xml_path}")

    server = None
    if VISUALIZE:
        print("\nViser started at http://localhost:8080")
        server = viser.ViserServer()

    chain = load_robot_chain(xml_path, DEVICE)
    
    root_path = Path(args.path)
    if not root_path.exists():
        print(f"Path does not exist: {root_path}")
        return

    all_files = [p for p in root_path.rglob("*.pt")]
    print(f"Found {len(all_files)} files.")

    iterator = tqdm(all_files)
    for file_path in iterator:
        try:
            process_single_file(file_path, chain, target_key_bodies, server)
        except Exception as e:
            iterator.write(f"Error processing {file_path.name}: {e}")

    print("\nDone!")
    if VISUALIZE:
        while True: time.sleep(1)

if __name__ == "__main__":
    main()