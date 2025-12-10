import numpy as np
import torch
from torch import Tensor
import glob, os, random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.humanoid_task import HumanoidWholeBody
from utils.metrics import Metrics, compute_evaluation_metrics
import xml.etree.ElementTree as ET

import trimesh
from pathlib import Path

NUM_KEY_POS = {"mano": 16, "shadow": 23, "allegro": 17}    

class HumanoidWholeBodyWithObject(HumanoidWholeBody): #metric
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Object Properties
        self.obj_size = cfg['env']['objSize']
        self.obj_restitution = cfg['env']['objRestitution']
        self.obj_density = cfg['env']['objDensity']

        self.object_names = cfg['env']['objectNames']
        self._obj_name_id = {name: idx for idx, name in enumerate(self.object_names)}
        self._obj_id_name = {idx: name for name, idx in self._obj_name_id.items()}
        num_envs = cfg["env"]["numEnvs"]
        num_objects = len(self.object_names)
        env_object_array = np.tile(np.arange(num_objects), (num_envs // num_objects) + 1)[:num_envs]
        self._envid_to_objid = torch.tensor(env_object_array, dtype=torch.long)

        self._num_sampled_obj_surface_pt = self.get_num_obj_keypoints()
        self._num_target_keypoints = 20
        self._obj_scale_range = [0.75, 1.5]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._build_target_tensors()
        if self.cfg['args'].test:
            self._build_traj_tensors()
        self._target_keypoints = torch.zeros([self.num_envs, self._num_target_keypoints, 3], device=self.device, dtype=torch.float)
    
    def get_num_obj_keypoints(self):
        if self._enable_dense_obj:
            num_keypoints = 200
        else:
            num_keypoints = 20
        return num_keypoints

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        self.obj_obs_size = 15
        obs_size += self.obj_obs_size
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._load_target_asset()
        if self.cfg['args'].test:
            self._traj_handles = []
            self._load_vis_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def extract_objpoints(self, mesh_path, num_keypoints=200, num_vertics=20):
        mesh = trimesh.load(mesh_path)
        # Get the closest points for each finger tip
        if self._enable_dense_obj:
            keypoints = np.zeros((num_keypoints, 3))
            assert num_vertics < num_keypoints, "num_vertics should be less than num_keypoints"
            keypoints, aff_face_id = trimesh.sample.sample_surface(mesh, num_keypoints-num_vertics)
            keypoints_tensor = torch.from_numpy(keypoints).float()
            vertices = mesh.vertices
            key_vertices = np.zeros((num_vertics, 3))
            for i in range(1, num_vertics):
                distances = np.min(np.linalg.norm(vertices[:, None] - key_vertices[:i], axis=2), axis=1)
                key_vertices[i] = vertices[np.argmax(distances)]
            vertices_tensor = torch.from_numpy(key_vertices).float()
            keypoints_tensor = torch.cat((keypoints_tensor, vertices_tensor), dim=0)
        # Use FPS(Farthest Point Sampling) to sample the object keypoints
        else:
            vertices = mesh.vertices
            keypoints = np.zeros((num_keypoints, 3))
            keypoints[0] = vertices[0]
            for i in range(1, num_keypoints):
                distances = np.min(np.linalg.norm(vertices[:, None] - keypoints[:i], axis=2), axis=1)
                keypoints[i] = vertices[np.argmax(distances)]
            keypoints_tensor = torch.from_numpy(keypoints).float()
        return keypoints_tensor

    def extract_keypoints(self, env_ids=None, motid=None):
        keypoints_tensor = self._obj_surface_pt.clone()
        return keypoints_tensor

    def extract_keypoints_per_epoch(self, tar_pos, tar_rot):
        target_keypoints_per_epoch = transform_keypoints_batch(self._target_keypoints, tar_pos, tar_rot)
        return target_keypoints_per_epoch, target_keypoints_per_epoch
            
    
    def _load_target_asset(self):
        self._obj_surface_pt = torch.zeros([self.num_envs, self._num_sampled_obj_surface_pt, 3], device=self.device, dtype=torch.float)
        asset_root = "hot/data/assets/urdf"

        self.loaded_assets = {}
        self.traj_sphere_asset = {}
        for obj_name in self.object_names:
            obj_asset_path = obj_name + "/" + obj_name + ".urdf"
            obj_asset_path_full = Path(asset_root) / obj_asset_path
            if obj_asset_path_full.exists():
                asset_options = gymapi.AssetOptions()
                asset_options.angular_damping = 0.01
                asset_options.linear_damping = 0.01
                asset_options.max_angular_velocity = 100.0
                asset_options.density = self.obj_density
                asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                asset_options.vhacd_enabled = True
                asset_options.vhacd_params.max_convex_hulls = 20
                asset_options.vhacd_params.max_num_vertices_per_ch = 64
                asset_options.vhacd_params.resolution = 300000

                self.loaded_assets[obj_name] = self.gym.load_asset(self.sim, asset_root, obj_asset_path, asset_options)
                ################## load obj keypoints while building the env ##################
                # # 加载mesh并提取关键点
                # mesh_path = os.path.join(asset_root, obj_name, 'top_watertight_tiny.obj')
                # obj_keypoints = self.extract_keypoints(mesh_path, num_keypoints=20).to(self.device)
                # self.loaded_assets_keypoints[obj_name] = obj_keypoints

                # object tracking asset, no_collision
                asset_options_tk = gymapi.AssetOptions()
                asset_options_tk.disable_gravity = True
                self.traj_sphere_asset[obj_name] = self.gym.load_asset(self.sim, asset_root, obj_asset_path, asset_options_tk)
            else:
                print(f"Warning: Asset not found for object {obj_name}")

        self.num_objects = len(self.loaded_assets)

        return
    
    def _load_vis_asset(self):
        asset_root = "hot/data/assets/urdf"
        num_key_pos = NUM_KEY_POS[self.hand_model]
        for i in range(num_key_pos):
            asset_file = "ball_no_collision.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = True
            asset_options.density = 0.0
            setattr(self, f"keypose_traj_asset_{i}", self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        if self.cfg['args'].test:
            self._build_traj(env_id, env_ptr)
        return
    
    def _build_target(self, env_id, env_ptr):
        
        obj_id = self._envid_to_objid[env_id]
        obj_name = self._obj_id_name[obj_id.item()]

        _target_asset = self.loaded_assets[obj_name]
        mesh_type = "stl"
        mesh_path = os.path.join('hot/data/assets/urdf', obj_name, f'top_watertight_tiny.{mesh_type}')
        self._obj_surface_pt[env_id] = self.extract_objpoints(mesh_path, num_keypoints=self._num_sampled_obj_surface_pt).to(self.device)
        col_group = env_id 
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        target_handle = self.gym.create_actor(env_ptr, _target_asset, default_pose, f"target_{env_id}", col_group, col_filter, segmentation_id)
        ball_props =  self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        # Modify the properties
        for b in ball_props:
            b.restitution = self.obj_restitution
            b.friction = 2
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, ball_props)  
        
        # set object color
        if self.cfg["headless"] == False:
            self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.5, 1.5, 1.5))

        self._target_handles.append(target_handle)
        
        # 20% probability to scale the object while self._obj_rand_scale=True
        if self._obj_rand_scale and np.random.random() < 0.2:
            self.obj_size = random.uniform(self._obj_scale_range[0], self._obj_scale_range[1])
            self._obj_surface_pt[env_id] *= self.obj_size 
        self.gym.set_actor_scale(env_ptr, target_handle, self.obj_size)

        return
    
    def _build_traj(self, env_id, env_ptr):
        obj_id = self._envid_to_objid[env_id]
        obj_name = self._obj_id_name[obj_id.item()]
        traj_sphere_asset = self.traj_sphere_asset[obj_name]

        # build object traj
        traj_col_group = self.num_envs + env_id
        traj_pose = gymapi.Transform()
        traj_pose.p.x = 0
        traj_pose.p.y = 0
        traj_pose.p.z = 1
        traj_handle = self.gym.create_actor(env_ptr, traj_sphere_asset, traj_pose, f"traj_point_{env_id}", traj_col_group, 0)
        if self.cfg["headless"] == False:
            self.gym.set_rigid_body_color(env_ptr, traj_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0)) # red color
        self._traj_handles.append(traj_handle)
        
        # build keypoints traj
        num_key_pos = NUM_KEY_POS[self.hand_model]
        for i in range(num_key_pos):
            # ensure no collision among the trajectory points
            traj_col_group = self.num_envs * 2 + env_id * num_key_pos + i 
            traj_pose = gymapi.Transform()
            traj_pose.p.x = 0
            traj_pose.p.y = 0
            traj_pose.p.z = 1
            traj_handle = self.gym.create_actor(env_ptr, getattr(self, f"keypose_traj_asset_{i}"), traj_pose, f"keypose_traj_point_{i}", traj_col_group, 0)
            if self.cfg["headless"] == False:
                self.gym.set_rigid_body_color(env_ptr, traj_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0)) # red color
            self._traj_handles.append(traj_handle)
            self.gym.set_actor_scale(env_ptr, traj_handle, 1.0)


    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_rigid_bodies, :]
        
        self.init_obj_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_pos_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_rot = torch.tensor([1., 0., 0., 0.], device=self.device, dtype=torch.float).repeat(self.num_envs, 1) 
        self.init_obj_rot_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        return

    def _build_traj_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._traj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        num_key_pos = NUM_KEY_POS[self.hand_model]
        self._keypose_traj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 3:3+num_key_pos, :]        
        self._traj_actor_ids = num_actors * np.arange(self.num_envs)
        self._traj_actor_ids = np.expand_dims(self._traj_actor_ids, axis=-1)
        self._traj_actor_ids = self._traj_actor_ids + np.reshape(np.array(self._traj_handles), [self.num_envs, num_key_pos+1])
        self._traj_actor_ids = self._traj_actor_ids.flatten()
        self._traj_actor_ids = to_torch(self._traj_actor_ids, device=self.device, dtype=torch.int32)
        
        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        return

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.init_obj_pos[env_ids]
        self._target_states[env_ids, 3:7] = self.init_obj_rot[env_ids] 
        self._target_states[env_ids, 7:10] = self.init_obj_pos_vel[env_ids]
        self._target_states[env_ids, 10:13] = self.init_obj_rot_vel[env_ids]
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.evts = list(self.gym.query_viewer_action_events(self.viewer))
        return 

    def _compute_observations(self, env_ids=None): # called @ reset & post step #not used
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs
        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if (env_ids is None): 
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return        

    def _compute_obj_obs(self, env_ids=None):
        tar_pos = self._target_states[:, 0:3].clone()
        tar_rot = self._target_states[:, 3:7].clone()
        self._obs_target_keypoints_per_epoch, self._obs_target_keypoints_vectors_per_epoch = self.extract_keypoints_per_epoch(tar_pos, tar_rot)

        # Unified index processing
        indexer = slice(None) if env_ids is None else env_ids
        # Unified value getting
        tar_states = self._target_states[indexer].clone()
        body_pos = self._rigid_body_pos[indexer][:, self._key_body_ids, :].clone()
        body_rot = self._rigid_body_rot[indexer][:, self._key_body_ids, :].clone()
        tar_keypoints_feature = self._obs_target_keypoints_per_epoch[indexer].clone()
        
        obs = compute_obj_observations(tar_states, tar_keypoints_feature, body_pos, body_rot, self._enable_obj_keypoints)
        return obs


class HumanoidWholeBodyWithObjectPlane(HumanoidWholeBodyWithObject):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_plane = True
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
    
    def _load_target_asset(self):
        super()._load_target_asset()
        if self._enable_plane:
            plane_asset_options = gymapi.AssetOptions()
            plane_asset_options.fix_base_link = True
            self._plane_proj_asset = self.gym.create_box(self.sim, 3, 3, 0.01, plane_asset_options)
            

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        if self._enable_plane:
            self._build_plane(env_id, env_ptr)
        return
    
    def _build_plane(self, env_id, env_ptr):
        col_group = env_id 
        col_filter = 1
        segmentation_id = 0

        plane_pose = gymapi.Transform()
        plane_pose.p.x = 0
        plane_pose.p.y = 0
        plane_pose.p.z = 0.1
        plane_handle = self.gym.create_actor(env_ptr, self._plane_proj_asset, plane_pose, "plane", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, plane_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.5, 0.5))
        

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def quaternion_to_rotation_matrix_batch(quaternion: Tensor) -> Tensor:
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
    
    x = quaternion[:, 0]
    y = quaternion[:, 1]
    z = quaternion[:, 2]
    w = quaternion[:, 3]
    
    rotation_matrix = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=1)
    ], dim=1)
    
    return rotation_matrix

@torch.jit.script
def transform_keypoints_batch(keypoints: Tensor, position: Tensor, quaternion: Tensor) -> Tensor:
    """
    Args:
        keypoints: torch.Tensor of shape (B, N, 3)
        position: torch.Tensor of shape (B, 3)
        quaternion: torch.Tensor of shape (B, 4) (x, y, z, w)
    Returns:
        transformed_keypoints: torch.Tensor of shape (B, N, 3)
    """
    R = quaternion_to_rotation_matrix_batch(quaternion)
    rotated_points = torch.bmm(keypoints, R.transpose(1, 2))
    transformed_points = rotated_points + position.unsqueeze(1)
    return transformed_points


# @torch.jit.script
def compute_obj_observations(tar_states, tar_keypoints_orig, body_pos, body_rot, enable_obj_keypoints=False):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    wrist_pos = body_pos[:, -1, :] # right_wrist global xyz
    wrist_rot = body_rot[:, -1, :] # right_wrist global quat rotation

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(wrist_rot)
    local_tar_pos = tar_pos - wrist_pos
    norm = torch.norm(local_tar_pos, dim=-1, keepdim=True)
    local_tar_pos = torch.where(norm > 0.5, (local_tar_pos / norm) * 0.5, local_tar_pos)
    
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)

    # transform obj keypoints to current obj pose
    tar_keypoints = tar_keypoints_orig
    if enable_obj_keypoints:
        num_pbj_keypoints = tar_keypoints_orig.shape[1]
        heading_rot_expand = heading_rot.unsqueeze(1).repeat((1, num_pbj_keypoints, 1)) # [num_envs, 20, 4]
        flat_heading_rot = heading_rot_expand.reshape(-1, 4) # [num_envs*20, 4]
        local_tar_keypoints = tar_keypoints - wrist_pos.unsqueeze(1)
        norm = torch.norm(local_tar_keypoints, dim=-1, keepdim=True)
        local_tar_keypoints = torch.where(norm > 0.5, (local_tar_keypoints / norm) * 0.5, local_tar_keypoints)
        flat_local_tar_keypoints = local_tar_keypoints.reshape(-1, 3) # [num_envs*20, 3]
        flat_local_tar_keypoints = quat_rotate(flat_heading_rot, flat_local_tar_keypoints)
        local_tar_keypoints = flat_local_tar_keypoints.view(local_tar_keypoints.shape[0],-1) # [num_envs, 60]

        obs = torch.cat([obs, local_tar_keypoints], dim=-1)
    return obs