"""
Last modified date: 2023.02.23
Author: Jialiang Zhang
Description: visualize grasp result using plotly.graph_objects
"""

import os
import sys
import math
#os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from manopth import rodrigues_layer
from manopth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                               subtract_flat_id, make_list)

import argparse
import torch
import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F

from utils.hand_model import HandModel
from utils.object_model import ObjectModel

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

def quat_to_euler_xyzw(quat):
    # Ensure that the input is a tensor
    if isinstance(quat, torch.Tensor):
        # Unpack the tensor assuming it's of shape (1, 4)
        x, y, z, w = quat[0, 0], quat[0, 1], quat[0, 2], quat[0, 3]
    else:
        x, y, z, w = quat  # Unpack regular list/array

    # Calculate roll, pitch, yaw
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return torch.tensor([roll, pitch, yaw])  # Returns a tensor
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

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

def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_from_euler_xyz_extrinsic(x, y, z):
    quat = R.from_euler('XYZ',[x,y,z]).as_quat()
    quat = torch.tensor(quat).float()
    return quat

def quat_multiply(q1, q2):
    # Compute the product of two quaternions.
    # The input quaternion format is [x, y, z, w], where w is the real part.
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack((x, y, z, w), dim=-1)

def AinB_local_to_global(posA_local, posB_global):
    #global_quat = quat_multiply(quatB_global, quatA_local)
    # rotated_pos = quat_rotate(quatB_global.unsqueeze(0).expand_as(quatA_local), posA_local)
    global_pos = posA_local+posB_global
    return global_pos

def get_wrist_global_poses(wrist_dof, root_pos):
    """Convert wrist DOFs to global poses"""

    #wrist_rot = quat_from_euler_xyz_extrinsic(dof[3], dof[4], dof[5])
    global_wrist_pos = AinB_local_to_global(
        wrist_dof,
        root_pos
    )

    return global_wrist_pos

def axis_angle_to_euler(axis, angle):
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    theta = angle

    # Compute the rotation matrix
    K = np.array([[0, -z, y],
                  [z, 0, -x],
                  [-y, x, 0]])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # Extract Euler angles from the rotation matrix
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = -np.arcsin(R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    return roll, pitch, yaw  # In radians



def axis_angle_to_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)  # Ensure unit vector
    ux, uy, uz = axis
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    I = np.eye(3)
    R = np.cos(theta) * I + (1 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * K
    return R

def matrix_to_extrinsic_euler_xyz(R):
    # Extract angles for extrinsic XYZ order
    beta = np.arcsin(-R[2, 0])
    cos_beta = np.cos(beta)
    
    alpha = np.arctan2(R[1, 0] / cos_beta, R[0, 0] / cos_beta)
    gamma = np.arctan2(R[2, 1] / cos_beta, R[2, 2] / cos_beta)
    
    return np.array([alpha, beta, gamma]) 


def convert_hand_dof(hand_axis_angles, euler_order='xyz'):
    euler_angles = []
    for axis, theta in hand_axis_angles:
        R = axis_angle_to_matrix(axis, theta)
        # Replace with your decomposition function
        angles = matrix_to_extrinsic_euler_xyz(R)
        euler_angles.append(angles)
    return np.array(euler_angles)

if __name__ == '__main__':
    angle = math.pi / 3
    parser = argparse.ArgumentParser()
    #parser.add_argument('--object_code', type=str, default='ddg-ycb_044_flat_screwdriver')
    #parser.add_argument('--object_code', type=str, default='ddg-gd_champagne_glass_poisson_001')
    parser.add_argument('--object_code', type=str, default='sem-Hammer-5d4da30b8c0eaae46d7014c7d6ce68fc')
    #parser.add_argument('--object_code', type=str, default='ddg-gd_rubik_cube_poisson_004')
    #parser.add_argument('--object_code', type=str, default='sem-USBStick-1baa93373407c8924315bea999e66ce3')
    #parser.add_argument('--object_code', type=str, default='core-knife-bcc178786ae893835f7f383d5cbb672d')
    #parser.add_argument('--object_code', type=str, default='sem-Gun-898424eaa40e821c2bf47341dbd96eb')
    #parser.add_argument('--object_code', type=str, default='sem-Gun-898424eaa40e821c2bf47341dbd96eb')
    #parser.add_argument('--object_code', type=str, default='ddg-gd_rubik_cube_poisson_004')
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--result_path', type=str, default='../data/experiments_hammer_02/exp_32/results')
    parser.add_argument('--image_path', type=str, default='../data/experiments_hammer_02/exp_32/images')
    args = parser.parse_args()

    device = 'cpu'
    for id in range(args.num):
        # load results
        #data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[id]
        data_dict = np.load(os.path.join(args.result_path, args.object_code + '.npy'), allow_pickle=True)[id]
        qpos = data_dict['qpos']
        hand_pose = torch.concat([torch.tensor(qpos[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])
        if 'contact_point_indices' in data_dict:
            contact_point_indices = torch.tensor(data_dict['contact_point_indices'], dtype=torch.long, device=device)
        if 'qpos_st' in data_dict:
            qpos_st = data_dict['qpos_st']
            hand_pose_st = torch.concat([torch.tensor(qpos_st[key], dtype=torch.float, device=device) for key in ['trans', 'rot', 'thetas']])

        # hand model
        hand_model = HandModel(
            mano_root='mano', 
            contact_indices_path='mano/contact_indices.json', 
            pose_distrib_path='mano/pose_distrib.pt', 
            device=device
        )

        # object model
        object_model = ObjectModel(
            data_root_path='../data/meshdata',
            batch_size_each=1,
            num_samples=2000, 
            device=device
        )
        object_model.initialize(args.object_code)
        object_model.object_scale_tensor = torch.tensor(data_dict['scale'], dtype=torch.float, device=device).reshape(1, 1)
        print(data_dict)
        # visualize
        if 'qpos_st' in data_dict:
            hand_model.set_parameters(hand_pose_st.unsqueeze(0))
            hand_st_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue')
        else:
            hand_st_plotly = []
        if 'contact_point_indices' in data_dict:
            #hand_pose[:]=0
            #hand_pose[43]=angle
            #hand_pose[46]=angle
            #hand_pose[49]=angle
            hand_model.set_parameters(hand_pose.unsqueeze(0), contact_point_indices.unsqueeze(0))
            hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=True)
        else:
            hand_model.set_parameters(hand_pose.unsqueeze(0))
            hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue')
        print("ppp",hand_en_plotly)
        object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
        print(object_plotly)

        fig = go.Figure(hand_st_plotly + hand_en_plotly + object_plotly)
        #fig = go.Figure(object_plotly)
        if 'energy' in data_dict:
            scale = round(data_dict['scale'], 2)
            energy = data_dict['energy']
            E_fc = round(data_dict['E_fc'], 3)
            E_dis = round(data_dict['E_dis'], 5)
            E_pen = round(data_dict['E_pen'], 5)
            E_prior = round(data_dict['E_prior'], 3)
            E_spen = round(data_dict['E_spen'], 4)
            result = f'Index {args.num}  scale {scale}  E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}  E_prior {E_prior}  E_spen {E_spen}'
            fig.add_annotation(text=result, x=0.5, y=0.1, xref='paper', yref='paper')
            wrist_position = hand_model.keypoints[0, 0] # Shape: [3]
            mano_index = hand_model.keypoints[0, 5] # +np.array([0,  0+0.0075+0.001,  0])# Shape: [3]
            mano_index_tip = hand_model.keypoints[0, 7]  #+np.array([0,  0+0.01+0.001,  0])# Shape: [3]
            mano_thumb = hand_model.keypoints[0, 1]  #+np.array([0,  0+0.009+0.001,  0])# Shape: [3]
            mano_thumb_tip = hand_model.keypoints[0, 3]  #+np.array([0,  0+0.01+0.001,  0])# Shape: [3]
            print(f"Wrist Global Position: {wrist_position}")
            point = np.array(wrist_position)
            print("ddd",hand_model.keypoints)
            # point2 = get_wrist_global_poses(qpos_st['thetas'][:3],qpos_st['trans']) #wrongggggg
            #point2 = np.array(wrist_position)
        #index =np.array([-0.0828,  0.0269,  0.9456-1])
        #thumb =np.array([-0.0906,  0.0090,  0.8835-1])
        #index_tip  =np.array([-0.0393,  0.0156,  0.9753-1])
        #thumb_tip  =np.array([-0.0433,  0.0019,  0.9133-1])

        #index =np.array([-0.0828,  0.0270,  0.9456-1])
        #thumb =np.array([-0.0902,  0.0088,  0.8836-1])
        #index_tip  =np.array([-0.0694,  0.0468,  0.9950-1])
        #thumb_tip  =np.array([-0.0666,  0.0553,  0.9062-1])

        #index =np.array([-0.0655,  0.0254,  0.9273-1])
        #thumb =np.array([-0.0953, -0.0057,  0.8785-1])
        #index_tip  =np.array([-0.0173,  0.0197,  0.9458-1])
        #thumb_tip  =np.array([-0.0556, -0.0190,  0.8510-1])

        #index =np.array([-0.0463, -0.0090,  0.9218-1])
        #thumb =np.array([-0.0727, -0.0676,  0.9320-1])
        #index_tip  =np.array([-0.0170,  0.0108,  0.9611-1])
        #thumb_tip  =np.array([-0.0377, -0.0401,  0.9672-1])

        #index =np.array([-0.0461, -0.0092,  0.9218-1])
        #thumb =np.array([-0.0726, -0.0676,  0.9323-1])
        #index_tip  =np.array([-0.0156,  0.0094,  0.9608-1])
        #thumb_tip  =np.array([-0.0413, -0.0408,  0.9716-1])
        #index =np.array([-0.0462, -0.0091,  0.9216-1])
        #thumb =np.array([-0.0726, -0.0675,  0.9325-1])
        #index_tip  =np.array([-0.0163,  0.0364,  0.9141-1])
        #thumb_tip  =np.array([-0.0291, -0.0936,  0.9061-1])

        #index =np.array([-0.0462, -0.0091,  0.9216-1])
        #thumb =np.array([-0.0726, -0.0675,  0.9326-1])
        #index_tip  =np.array([-0.0162,  0.0363,  0.9141-1])
        #thumb_tip  =np.array([-0.0608, -0.1176,  0.9173-1])
        
        '''
        rotation_matrix = rodrigues_layer.batch_rodrigues(hand_pose[3:6].unsqueeze(0)).view(3,3)
        axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # x, y, z axes

        origin = np.array([[-0.1165, -0.0179, -0.1238]])  # Origin point

        # Apply the rotation matrix to the axes
        rotated_axes = np.dot(axes, rotation_matrix.T)
        for i, color in enumerate(['darkred', 'darkgreen', 'darkblue']):
            fig.add_trace(go.Scatter3d(
                x=[origin[0, 0], axes[i, 0]],
                y=[origin[0, 1], axes[i, 1]],
                z=[origin[0, 2], rotated_axes[i, 2]],
                mode='lines',
                line=dict(color=color, width=4, dash='dash'),
                name=f'Rotated {["X", "Y", "Z"][i]}'
            ))
        '''
        # Add the point to the figure
        fig.add_trace(go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode='markers',
            marker=dict(size=5, color='yellow'),  # Adjust size and color as needed
            name='Point'
        ))
        # # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[index[0]],
        #     y=[index[1]],
        #     z=[index[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='green'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        # # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[thumb[0]],
        #     y=[thumb[1]],
        #     z=[thumb[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='green'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        #         # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[index_tip[0]],
        #     y=[index_tip[1]],
        #     z=[index_tip[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='green'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        #         # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[thumb_tip[0]],
        #     y=[thumb_tip[1]],
        #     z=[thumb_tip[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='green'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        #         # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[mano_index[0]],
        #     y=[mano_index[1]],
        #     z=[mano_index[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='blue'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        # # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[mano_index_tip[0]],
        #     y=[mano_index_tip[1]],
        #     z=[mano_index_tip[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='blue'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        #         # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[mano_thumb[0]],
        #     y=[mano_thumb[1]],
        #     z=[mano_thumb[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='blue'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        #         # Add the point to the figure
        # fig.add_trace(go.Scatter3d(
        #     x=[mano_thumb_tip[0]],
        #     y=[mano_thumb_tip[1]],
        #     z=[mano_thumb_tip[2]],
        #     mode='markers',
        #     marker=dict(size=5, color='blue'),  # Adjust size and color as needed
        #     name='Point'
        # ))
        fig.update_layout(title=f'{id}')
        fig.update_layout(scene_aspectmode='data')
        fig.write_image(os.path.join(args.result_path, f"{args.object_code}_index_{id}.png"))
        #fig.show()
        html_file_path = os.path.join(args.result_path, f"{args.object_code}_index_{id}.html")
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Create full HTML with button
        complete_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{args.object_code} - Index {id}</title>
            <script>
                function saveId(id) {{
                    fetch('/save_id', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ id: id }}),
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        alert('ID saved: ' + id);
                        window.close();

                    }})
                    .catch((error) => {{
                        console.error('Error:', error);
                    }});
                }}
            </script>
        </head>
        <body>
            <h1>{args.object_code} - Index {id}</h1>
            {fig_html}
            <button onclick="saveId({id})">Save ID</button>
        </body>
        </html>
        """

        # Save the generated complete HTML content
        with open(html_file_path, 'w') as f:
            f.write(complete_html)
            
        results = []  # Initialize an empty list to store results

        for i in range(0, len(data_dict['qpos']["thetas"]), 3):
            axis = data_dict['qpos']["thetas"][i:i + 3]
            result = quat_to_euler_xyzw(rotmat_to_quat(rodrigues_layer.batch_rodrigues(torch.tensor(axis).unsqueeze(0)).unsqueeze(0).view(1,3,3)))
            print(torch.tensor(axis).unsqueeze(0).shape)

            print("lll",result)
            
            results.append(result)  # Append the result
        '''
        for i in range(6,  hand_pose.shape[0], 3):
            axis = hand_pose[i:i + 3]
            print(torch.tensor(axis).unsqueeze(0).shape)
            result = rotmat_to_extrinsic_euler_xyz(rodrigues_layer.batch_rodrigues(torch.tensor(axis).unsqueeze(0)).unsqueeze(0).view(1,3,3))

            print("bbb",result)
        '''
        # Concatenate all results into a single tensor
        final_output = torch.cat(results, dim=0)  # Dim 0 for batch dimension

        # Ensure the final output is of the desired shape
        final_output = final_output.view(-1) 
        print("fff",final_output)  # Should be (45,)

        rotation_matrix = rodrigues_layer.batch_rodrigues(hand_pose[3:6].unsqueeze(0))
        #print(rotmat_to_quat(rotation_matrix.unsqueeze(0).view(1,3,3)))            

        rotation_matrix = rodrigues_layer.batch_rodrigues(torch.tensor(data_dict['qpos']["rot"]).unsqueeze(0))

        #print(rotation_matrix.unsqueeze(0).view(1,3,3).shape)                                     
        #print("hhhh",hand_pose[0:3],torch.tensor(data_dict['qpos']["rot"]).unsqueeze(0))                                      
        #print(rotmat_to_quat(rotation_matrix.unsqueeze(0).view(1,3,3)))            
        