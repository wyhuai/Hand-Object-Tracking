from enum import Enum
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


PERTURB_PROJECTORS = [
    ["small", 60],
    # ["large", 60],
]

class HumanoidWholeBodyWithObject(HumanoidWholeBody): #metric
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.projtype = cfg['env']['projtype']
        self.apply_disturbance = cfg['env']['applyDisturbance']
        self.disturbance_force_scale = cfg['env']['disturbanceForceScale']       
        self.body_to_obj_keypoint = cfg['env']['bodyObjKeypoint']       
        # Ball Properties
        self.ball_size = cfg['env']['ballSize']
        self.ball_restitution = cfg['env']['ballRestitution']
        self.ball_density = cfg['env']['ballDensity']

        ############## objects related code ##############
        self.object_names = cfg['env']['objectNames']
        self._obj_name_2_id = {name: idx for idx, name in enumerate(self.object_names)}
        self._obj_id_2_name = {idx: name for name, idx in self._obj_name_2_id.items()}
        # 让对象在 num_envs 个环境中均匀分布
        num_envs = cfg["env"]["numEnvs"]
        num_objects = len(self.object_names)
        env_object_array = np.tile(np.arange(num_objects), (num_envs // num_objects) + 1)[:num_envs]
        self._map_env_2_object_tensor = torch.tensor(env_object_array, dtype=torch.long)
        self._num_sampled_obj_surface_pt = self.get_num_obj_keypoints()
        self._num_target_keypoints = len(cfg['env']["keyBodies"]) if self.body_to_obj_keypoint else 20
        # object scale randomization
        self.obj_scale_dict = {'Ball': [0.6, 1.0], 'Book': [0.9, 1.1], 'Bowl': [0.6, 1.6], 'Bottle': [0.8, 1.4], 'Box':[0.8, 1.4], 
                            'Gun': [0.9, 2.0], 'Hammer': [0.9, 2.0], 'Mug': [0.6, 1.6], 'Screwdriver': [0.9, 2.0], 'Shoe': [0.8, 1.4], 
                            'Stick': [0.6, 1.6], 'Sword': [0.6, 1.6], 'Wineglass': [0.9, 2.0]}
        # self.obj_scale_dict = {'Ball': [0.5, 0.8], 'Book': [0.8, 1.0], 'Bowl': [0.6, 1.0], 'Bottle': [0.6, 1.0], 'Box':[0.6, 0.8], 
        #                     'Gun': [0.8, 1.1], 'Hammer': [0.8, 1.0], 'Mug': [0.6, 1.0], 'Screwdriver': [0.9, 1.0], 'Shoe': [0.8, 1.0], 
        #                     'Stick': [0.6, 1.0], 'Sword': [0.5, 0.8], 'Wineglass': [0.8, 0.9]}
        ##################################################

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._build_target_tensors()
        self._build_target2_tensors()
        if self.cfg['args'].test:
            self._build_traj_tensors()
            if self.cfg['env']['showObjectKeyPoints']:
                self._build_objkeypos_tensors()

        self.init_obj2_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj2_pos_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj2_rot = torch.tensor([1., 0., 0., 0.], device=self.device, dtype=torch.float).repeat(self.num_envs, 1) 
        self.init_obj2_rot_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._target_keypoints = torch.zeros([self.num_envs, self._num_target_keypoints, 3], device=self.device, dtype=torch.float)
        

        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj_tensors()
    
    def get_num_obj_keypoints(self):
        if self.body_to_obj_keypoint:
            num_keypoints = 1000
        elif self._enable_dense_obj:
            num_keypoints = 200
        else:
            num_keypoints = 20
        return num_keypoints

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        self.obj_obs_size = 15
        obs_size += self.obj_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._target2_handles = []
        self._load_target_asset()
        self._load_target2_asset()
        if self.cfg['args'].test:
            self._traj_handles = []
            self._load_vis_asset()
            if self.cfg['env']['showObjectKeyPoints']:
                self._objkp_handles = []
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._proj_handles = []
            self._load_proj_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def extract_objpoints(self, mesh_path, num_keypoints=200, num_vertics=20):
        mesh = trimesh.load(mesh_path)
        if self.body_to_obj_keypoint or self._enable_dense_obj:
            keypoints = np.zeros((num_keypoints, 3))
            ################# Modified by Runyi #################
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
            #####################################################
        else:
            vertices = mesh.vertices
            # 使用FPS(Farthest Point Sampling)算法选择关键点
            keypoints = np.zeros((num_keypoints, 3))
            ################ fix the keypoints ################
            # keypoints[0] = vertices[np.random.randint(len(vertices))]
            keypoints[0] = vertices[0]
            ###################################################
            
            for i in range(1, num_keypoints):
                distances = np.min(np.linalg.norm(vertices[:, None] - keypoints[:i], axis=2), axis=1)
                keypoints[i] = vertices[np.argmax(distances)]
            
            # 转换为tensor
            keypoints_tensor = torch.from_numpy(keypoints).float()
            
        return keypoints_tensor

    def extract_keypoints(self, env_ids=None, motid=None):
        keypoints_tensor = self._obj_surface_pt.clone()
        
        if self.body_to_obj_keypoint:
            '''
                use the first frame to get the contact info for place task
                use the last frame to get the contact info for other tasks
            '''
            num_key = len(self._key_body_ids)
            
            if motid is not None:
                # For play_dataset mode (single motion)
                # if self.hoi_data_text[motid] == '003':
                #     contact_frame_id = 0
                # else:
                #     contact_frame_id = self._motion_data.motion_lengths[motid]-1
                #     # contact_frame_id = 100
                motion_data = self._motion_data.hoi_data_dict[motid]['hoi_data']
                motion_data_contact = motion_data[:,-2]
                contact_frame_id = torch.where(motion_data_contact == 1)[0][10] # reference motion里的第10个接触帧
                ref_obs = self._motion_data.hoi_data_dict[motid] 
                ref_tar_pos = ref_obs['obj_pos'][contact_frame_id,:].clone().unsqueeze(0)
                ref_tar_rot = ref_obs['obj_rot'][contact_frame_id,:].clone().unsqueeze(0)
                key_body_pos = ref_obs['key_body_pos'][contact_frame_id,:].clone().view(-1, num_key, 3)
            else:
                keypoints_tensor = keypoints_tensor[env_ids]
                # contact_frame_ids = torch.min(
                #     self._motion_data.envid2episode_lengths[env_ids] - 1,
                #     torch.tensor(self.max_episode_length - 1, device=self.device)
                # )
                # contact_frame_ids = torch.where(self.skill_labels[env_ids] == 3, torch.zeros_like(contact_frame_ids), contact_frame_ids)
                contact_frame_ids = torch.zeros_like(env_ids)
                last_frame_ref_obs = torch.zeros([len(env_ids), self.ref_hoi_obs_size], device=self.device, dtype=torch.float)
                for ind, env_id in enumerate(env_ids):
                    mid = self.motion_ids_total[env_id]
                    motion_data = self._motion_data.hoi_data_dict[mid.item()]['hoi_data']
                    motion_data_contact = motion_data[:,-2]
                    contact_frame_ids[ind] = torch.where(motion_data_contact == 1)[0][10] # reference motion里的第10个接触帧
                    # Get reference observations in batch
                    last_frame_ref_obs[ind] = motion_data[contact_frame_ids[ind]]
                if self.hand_model == "mano":
                    # Extract target positions and rotations
                    ref_tar_pos = last_frame_ref_obs[:, 109:109+3]  # shape: [B, 3]
                    ref_tar_rot = last_frame_ref_obs[:, 112:112+4]  # shape: [B, 4]
                    
                    # Extract key body positions
                    key_body_pos = last_frame_ref_obs[:, 119:119+num_key*3].view(-1, num_key, 3)  # shape: [B, num_key, 3]
                elif self.hand_model == "shadow":
                    # Extract target positions and rotations
                    ref_tar_pos = last_frame_ref_obs[:, 63:66]  # shape: [B, 3]
                    ref_tar_rot = last_frame_ref_obs[:, 66:66+4]  # shape: [B, 4]
                    
                    # Extract key body positions
                    key_body_pos = last_frame_ref_obs[:, 73:73+num_key*3].view(-1, num_key, 3)  # shape: [B, num_key, 3]
                elif self.hand_model == "allegro":
                    # Extract target positions and rotations
                    ref_tar_pos = last_frame_ref_obs[:, 51:51+3]  # shape: [B, 3]
                    ref_tar_rot = last_frame_ref_obs[:, 54:54+4]  # shape: [B, 4]
                    
                    # Extract key body positions
                    key_body_pos = last_frame_ref_obs[:, 61:61+num_key*3].view(-1, num_key, 3)  # shape: [B, num_key, 3]

            # Transform keypoints to target frame
            target_keypoints_per_epoch = transform_keypoints_batch(keypoints_tensor, ref_tar_pos, ref_tar_rot)
            # calculate the Euclidean distances between two sets of points.
            body_obj_keypoints_dis = torch.cdist(key_body_pos, target_keypoints_per_epoch)
            min_body_obj_keypoints_dis, min_min_body_obj_keypoints_dis_idx = torch.min(body_obj_keypoints_dis, dim=2)
            keypoints_tensor = torch.gather(keypoints_tensor, 1,  # keypoints_tensor - use the keypoints in the origin space
                                                min_min_body_obj_keypoints_dis_idx.unsqueeze(2).expand(-1, -1, 3))
        return keypoints_tensor

    def extract_keypoints_per_epoch(self, tar_pos, tar_rot):
        target_keypoints_per_epoch = transform_keypoints_batch(self._target_keypoints, tar_pos, tar_rot)
        if self.body_to_obj_keypoint:
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :].clone()
            target_keypoints_vectors_per_epoch = target_keypoints_per_epoch - key_body_pos
            return target_keypoints_per_epoch, target_keypoints_vectors_per_epoch
        return target_keypoints_per_epoch, target_keypoints_per_epoch
            
    
        

    def _load_target_asset(self):
        self._obj_surface_pt = torch.zeros([self.num_envs, self._num_sampled_obj_surface_pt, 3], device=self.device, dtype=torch.float)
        #self._obs_target_keypoints_per_epoch = torch.zeros([self.num_envs, self._num_target_keypoints, 3], device=self.device, dtype=torch.float)
        #self._ref_target_keypoints_per_epoch = torch.zeros([self.num_envs, self._num_target_keypoints, 3], device=self.device, dtype=torch.float)
        #self._obs_target_keypoints_vectors_per_epoch = torch.zeros([self.num_envs, self._num_target_keypoints, 3], device=self.device, dtype=torch.float)
        #self._ref_target_keypoints_vectors_per_epoch = torch.zeros([self.num_envs, self._num_target_keypoints, 3], device=self.device, dtype=torch.float)

        asset_root = "skillmimic/data/assets/urdf"

        # get object names
        self.loaded_assets = {}
        # self.loaded_assets_keypoints = {}
        self.traj_sphere_asset = {}

        # Load assets for each object
        for obj_name in self.object_names:
            # try:
            obj_asset_path = obj_name + "/" + obj_name + ".urdf"
            obj_asset_path_full = Path(asset_root) / obj_asset_path
            if obj_asset_path_full.exists():
                asset_options = gymapi.AssetOptions()
                asset_options.angular_damping = 0.01
                asset_options.linear_damping = 0.01
                asset_options.max_angular_velocity = 100.0
                if obj_asset_path in ['Sword/Sword.urdf']:
                    asset_options.density = 10 #0.2 * self.ball_density
                else:
                    asset_options.density = 500 #self.ball_density
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
    
    def _load_target2_asset(self):
        asset_root = "skillmimic/data/assets/urdf" #projectname
        asset_file = "box02_container.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = self.ball_density #85.0#*6
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.max_convex_hulls = 1
        asset_options.vhacd_params.max_num_vertices_per_ch = 64
        asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.disable_gravity = True
        asset_options.fix_base_link = True

        self._target2_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        return
    
    def _load_vis_asset(self):
        asset_root = "skillmimic/data/assets/urdf"
        # keypose traj asset
        if self.hand_model == "mano":
            num_key_pos = 16
        elif self.hand_model == "shadow":
            num_key_pos = 23   
        elif self.hand_model == "allegro":
            num_key_pos = 17
        for i in range(num_key_pos):
            asset_file = "ball_no_collision.urdf"
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = True
            asset_options.density = 0.0
            setattr(self, f"keypose_traj_asset_{i}", self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))
                    
        return

    def _load_proj_asset(self):
        asset_root = "skillmimic/data/assets/urdf/" #projectname
        small_asset_file = "block_projectile.urdf"
        
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 1000.0
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)
        self._build_target2(env_id, env_ptr)
        if self.cfg['args'].test:
            self._build_traj(env_id, env_ptr)
            if self.cfg['env']['showObjectKeyPoints']:
                self._build_objkeypos(env_id, env_ptr)
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._build_proj(env_id, env_ptr)

        return
    
    def _build_target(self, env_id, env_ptr):
        
        ############## objects related code ##############
        obj_id = self._map_env_2_object_tensor[env_id]
        obj_name = self._obj_id_2_name[obj_id.item()]
        ##################################################

        _target_asset = self.loaded_assets[obj_name]
        # _target_keypoints = self.loaded_assets_keypoints[obj_name].clone()
        # self._target_keypoints[env_id] = _target_keypoints
        
        ################## load obj keypoints while building the env ##################
        # 加载mesh并提取关键点
        mesh_type = "obj"
        mesh_path = os.path.join('skillmimic/data/assets/urdf', obj_name, f'top_watertight_tiny.{mesh_type}')
        self._obj_surface_pt[env_id] = self.extract_objpoints(mesh_path, num_keypoints=self._num_sampled_obj_surface_pt).to(self.device)
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(env_ptr, _target_asset, default_pose, f"target_{env_id}", col_group, col_filter, segmentation_id)

        ball_props =  self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        # Modify the properties
        for b in ball_props:
            b.restitution = self.ball_restitution #0.66 #1.6
            # b.friction = 10
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, ball_props)  
        
        # set ball color
        if self.cfg["headless"] == False:
            self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(1.5, 1.5, 1.5))

        self._target_handles.append(target_handle)
        
        # 20% of the time, randomize the scale
        if self._obj_rand_scale and np.random.random() < 0.2:
            scale_factor = random.uniform(self.obj_scale_dict[obj_name][0], self.obj_scale_dict[obj_name][1])
            self._target_keypoints[env_id] *= scale_factor
            self.gym.set_actor_scale(env_ptr, target_handle, scale_factor)
        else:
            self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

        if obj_name == 'Apple':
            self.gym.set_actor_scale(env_ptr, target_handle, 0.8)
        return
    
    def _build_target2(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()

        target_handle = self.gym.create_actor(env_ptr, self._target2_asset, default_pose, "target2", col_group, col_filter, segmentation_id)

        ball_props =  self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        # Modify the properties
        for b in ball_props:
            b.restitution = self.ball_restitution #0.66 #1.6
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, ball_props)  
        
        # set ball color
        if self.cfg["headless"] == False:
            self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(1.5, 1.5, 1.5))
                                        # gymapi.Vec3(0., 1.0, 1.5))
            # h = self.gym.create_texture_from_file(self.sim, 'skillmimic/data/assets/urdf/basketball.png') #projectname
            # self.gym.set_rigid_body_texture(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, h)


        self._target2_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)
        return
    
    def _build_traj(self, env_id, env_ptr):

        ############## objects related code ##############
        obj_id = self._map_env_2_object_tensor[env_id]
        obj_name = self._obj_id_2_name[obj_id.item()]
        ##################################################
        traj_sphere_asset = self.traj_sphere_asset[obj_name]

        # build object traj
        traj_col_group = self.num_envs + env_id
        traj_pose = gymapi.Transform()
        traj_pose.p.x = 0
        traj_pose.p.y = 0
        traj_pose.p.z = 1
        traj_handle = self.gym.create_actor(env_ptr, traj_sphere_asset, traj_pose, 
                                              f"traj_point_{env_id}", traj_col_group, 0)
        # set ball color
        if self.cfg["headless"] == False:
            self.gym.set_rigid_body_color(env_ptr, traj_handle, 0, gymapi.MESH_VISUAL,
                                        gymapi.Vec3(1, 0, 0))
        
        self._traj_handles.append(traj_handle)
        if self.cfg['env']['showObjectKeyPoints']:
            self.gym.set_actor_scale(env_ptr, traj_handle, 0.0001)
        shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, traj_handle)

        if obj_name == 'Apple':
            self.gym.set_actor_scale(env_ptr, traj_handle, 0.8)
        for prop in shape_props:
            prop.filter = 0  # 让物体不与任何物体发生碰撞
        self.gym.set_actor_rigid_shape_properties(env_ptr, traj_handle, shape_props)
        
        # build key pose traj
        if self.hand_model == "mano":
            num_key_pos = 16
        elif self.hand_model == "shadow":
            num_key_pos = 23
        elif self.hand_model == "allegro":
            num_key_pos = 17

        for i in range(num_key_pos):
            # build object traj
            tmp_col_group = self.num_envs*2 + env_id*num_key_pos + i # 互相不碰撞
            traj_pose = gymapi.Transform()
            traj_pose.p.x = 0
            traj_pose.p.y = 0
            traj_pose.p.z = 1
            traj_handle = self.gym.create_actor(env_ptr, getattr(self, f"keypose_traj_asset_{i}"), traj_pose, 
                                                f"keypose_traj_point_{i}", tmp_col_group, 0)
            # set ball color
            if self.cfg["headless"] == False:
                self.gym.set_rigid_body_color(env_ptr, traj_handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1, 0, 0))
            self._traj_handles.append(traj_handle)
            
            self.gym.set_actor_scale(env_ptr, traj_handle, 1.0)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, traj_handle)
            for prop in shape_props:
                prop.filter = 0  # 让物体不与任何物体发生碰撞
            self.gym.set_actor_rigid_shape_properties(env_ptr, traj_handle, shape_props)

    def _build_objkeypos(self, env_id, env_ptr):
        for i in range(self._num_target_keypoints):
            # create asset
            keypoint_asset_options = gymapi.AssetOptions()
            keypoint_asset_options.disable_gravity = True
            keypoint_asset_options.fix_base_link = False
            keypoint_asset_options.density = 0
            keypoint_asset = self.gym.create_sphere(self.sim, 0.01, keypoint_asset_options)  # 5mm radius

            # build asset
            kp_col_group = self.num_envs*2+env_id*num_key_pos +env_id*num_key_pos+ i
            kp_pose = gymapi.Transform()
            kp_pose.p.x = 0
            kp_pose.p.y = 0
            kp_pose.p.z = 0
            kp_handle = self.gym.create_actor(env_ptr, keypoint_asset, kp_pose, f"kp_{env_id}_{i}", kp_col_group, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, kp_handle)
            for prop in shape_props:
                prop.filter = 0  # Disable all collisions
            # Set keypoint color (yellow for visualization)
            self.gym.set_actor_rigid_shape_properties(env_ptr, kp_handle, shape_props)  
            self._objkp_handles.append(kp_handle)
        return

    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_PROJECTORS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)
            self.gym.set_actor_scale(env_ptr, proj_handle, 0.5)

        return
    
    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_rigid_bodies, :]
        # #debug
        # self.full_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)

        self.init_obj_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_pos_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.init_obj_rot = torch.tensor([1., 0., 0., 0.], device=self.device, dtype=torch.float).repeat(self.num_envs, 1) 
        self.init_obj_rot_vel = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        return

    def _build_target2_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target2_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 2, :]
        
        self._tar2_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 2
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar2_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies+1, :]

        return
    
    def _build_traj_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._traj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 3, :]
        if self.hand_model == "mano":
            num_key_pos = 16
        elif self.hand_model == "shadow":
            num_key_pos = 23   
        elif self.hand_model == "allegro":
            num_key_pos = 17

        self._keypose_traj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 4:4+num_key_pos, :]        
        self._traj_actor_ids = num_actors * np.arange(self.num_envs)
        self._traj_actor_ids = np.expand_dims(self._traj_actor_ids, axis=-1)
        self._traj_actor_ids = self._traj_actor_ids + np.reshape(np.array(self._traj_handles), [self.num_envs, 1+num_key_pos])
        self._traj_actor_ids = self._traj_actor_ids.flatten()
        self._traj_actor_ids = to_torch(self._traj_actor_ids, device=self.device, dtype=torch.int32)
        
        return

    def _build_objkeypos_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._obj_keypoint_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 20:20+self._num_target_keypoints, :]
        
        self._objkeypos_actor_ids = num_actors * np.arange(self.num_envs)
        self._objkeypos_actor_ids = np.expand_dims(self._objkeypos_actor_ids, axis=-1)
        self._objkeypos_actor_ids = self._objkeypos_actor_ids + np.reshape(np.array(self._objkp_handles), [self.num_envs, self._num_target_keypoints])
        self._objkeypos_actor_ids = self._objkeypos_actor_ids.flatten()
        self._objkeypos_actor_ids = to_torch(self._objkeypos_actor_ids, device=self.device, dtype=torch.int32)
        self._traj_actor_ids = torch.cat([self._objkeypos_actor_ids, self._traj_actor_ids], dim=0)
        

    def _build_proj_tensors(self):
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40

        num_actors = self.get_num_actors_per_env()
        num_objs = len(PERTURB_PROJECTORS)
        self._proj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., (num_actors - num_objs):, :]
        
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]

        self._calc_perturb_times()

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot") #ZC0
        self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")
        
        return

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_PROJECTORS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

        return


    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_target(env_ids)
        self._reset_target2(env_ids)
        return

    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.init_obj_pos[env_ids]#.clone()+0.5
        self._target_states[env_ids, 3:7] = self.init_obj_rot[env_ids] 
        self._target_states[env_ids, 7:10] = self.init_obj_pos_vel[env_ids]#.clone()
        self._target_states[env_ids, 10:13] = self.init_obj_rot_vel[env_ids]
        return


    def _reset_target2(self, env_ids):
        # self.init_obj_pos[env_ids, 2] += 8
        self._target2_states[env_ids, :3] = self.init_obj2_pos[env_ids]#.clone()+0.5
        # self._target2_states[env_ids, :1] += torch.rand_like(self._target_states[env_ids, :1]).to(self.device)*0.2
        
        self._target2_states[env_ids, 3:7] = self.init_obj2_rot[env_ids]#.clone() #rand_rot
        self._target2_states[env_ids, 7:10] = self.init_obj2_pos_vel[env_ids]#.clone()
        self._target2_states[env_ids, 10:13] = self.init_obj2_rot_vel[env_ids]
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        # env_ids_int32 = torch.cat(
        #     (
        #         self._tar_actor_ids[env_ids],
        #         self._tar2_actor_ids[env_ids]
        #     ), dim=-1
        # )
        env_ids_int32 = self._tar_actor_ids[env_ids]
        # env_ids_int32 = torch.cat([self._tar_actor_ids[env_ids], self._tar2_actor_ids[env_ids]], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # env_ids_int32 = self._tar2_actor_ids[env_ids]
        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
        #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.evts = list(self.gym.query_viewer_action_events(self.viewer))        
        return 
    
    def post_physics_step(self):
        if self.projtype == "Mouse" or self.projtype == "Auto":
            self._update_proj()
        # if self.apply_disturbance:
        #     self._apply_disturbance_forces()
        super().post_physics_step()

    def _apply_disturbance_forces(self):
        # Generate random forces
        force_offset = 0.0
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        ## Apply forces at the target's current position
        positions = self._rigid_body_state[..., 0:3].clone().reshape(self.num_envs, bodies_per_env, 3)
        positions[:, self.num_bodies, 1] += force_offset
        positions = positions.reshape(1, -1, 3)
        forces = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float).reshape(self.num_envs, bodies_per_env, 3)
        torques = torch.zeros((1, self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        random_forces = torch.randn((self.num_envs, 3), device=self.device) * self.disturbance_force_scale
        forces[:, self.num_bodies, :] = random_forces.reshape(-1, 3)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

        self.gym.clear_lines(self.viewer)

        # Draw force vectors as red lines
        for env_idx in range(self.num_envs):
            # Get the force and position for the object in this environment
            force = forces[env_idx, self.num_bodies, :]
            pos_idx = env_idx * bodies_per_env + self.num_bodies
            pos = positions.reshape(-1, 3)[pos_idx]
            
            # Convert to numpy arrays for visualization
            pos_np = pos.cpu().numpy()
            force_np = force.cpu().numpy()
            
            # Scale the force for better visualization
            scale = 0.1  # Adjust this value to change line length
            end_pos = pos_np + force_np * scale
            
            # Draw line from object position to force direction
            color = gymapi.Vec3(1.0, 0.0, 0.0)  # Red color
            self.gym.add_lines(
                self.viewer,
                self.envs[env_idx],
                1,  # Number of lines
                [
                    pos_np[0], pos_np[1], pos_np[2],
                end_pos[0], end_pos[1], end_pos[2]
                ]
                , [0.85, 0.1, 0.1])
 
    def _update_proj(self):

        if self.projtype == 'Auto':
            curr_timestep = self.progress_buf.cpu().numpy()[0]
            curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
            perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]
            
            if (len(perturb_step) > 0):
                perturb_id = perturb_step[0]
                n = self.num_envs
                humanoid_root_pos = self._humanoid_root_states[..., 0:3]

                rand_theta = torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device)
                rand_theta *= 2 * np.pi
                rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_dist_min
                pos_x = rand_dist * torch.cos(rand_theta)
                pos_y = -rand_dist * torch.sin(rand_theta)
                pos_z = (self._proj_h_max - self._proj_h_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_h_min
                
                self._proj_states[..., perturb_id, 0] = humanoid_root_pos[..., 0] + pos_x
                self._proj_states[..., perturb_id, 1] = humanoid_root_pos[..., 1] + pos_y
                self._proj_states[..., perturb_id, 2] = pos_z
                self._proj_states[..., perturb_id, 3:6] = 0.0
                self._proj_states[..., perturb_id, 6] = 1.0
                
                tar_body_idx = np.random.randint(self.num_bodies)
                tar_body_idx = 1

                launch_tar_pos = self._rigid_body_pos[..., tar_body_idx, :]
                launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
                launch_dir += 0.1 * torch.randn_like(launch_dir)
                launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
                launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
                launch_vel = launch_speed * launch_dir
                launch_vel[..., 0:2] += self._rigid_body_vel[..., tar_body_idx, 0:2]
                self._proj_states[..., perturb_id, 7:10] = launch_vel
                self._proj_states[..., perturb_id, 10:13] = 0.0

                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                             gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                             len(self._proj_actor_ids))
            
        elif self.projtype == 'Mouse':
            # mouse control
            for evt in self.evts:
                if evt.action == "reset" and evt.value > 0:
                    self.gym.set_sim_rigid_body_states(self.sim, self._proj_states, gymapi.STATE_ALL)
                elif (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0:
                    if evt.action == "mouse_shoot":
                        pos = self.gym.get_viewer_mouse_position(self.viewer)
                        window_size = self.gym.get_viewer_size(self.viewer)
                        xcoord = round(pos.x * window_size.x)
                        ycoord = round(pos.y * window_size.y)
                        print(f"Fired projectile with mouse at coords: {xcoord} {ycoord}")

                    cam_pose = self.gym.get_viewer_camera_transform(self.viewer, None)
                    cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))

                    spawn = cam_pose.p
                    speed = 25
                    vel = cam_fwd * speed

                    angvel = 1.57 - 3.14 * np.random.random(3)

                    self._proj_states[..., 0] = spawn.x
                    self._proj_states[..., 1] = spawn.y
                    self._proj_states[..., 2] = spawn.z
                    self._proj_states[..., 7] = vel.x
                    self._proj_states[..., 8] = vel.y
                    self._proj_states[..., 9] = vel.z

                    self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                            gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                            len(self._proj_actor_ids))

        return


    def _set_traj(self):
        if self.cfg['env']['showObjectKeyPoints']:
            self._obj_keypoint_states[:, :, :3] = self._obs_target_keypoints_per_epoch
        if self.body_to_obj_keypoint:
            self.visualize_keypoint_vectors()
    
    def _set_traj_current(self):
        if self.cfg['env']['showObjectKeyPoints']:
            self._obj_keypoint_states[:, :, :3] = self._obs_target_keypoints_per_epoch
        if self.body_to_obj_keypoint:
            self.visualize_keypoint_vectors()
    
    def _set_traj_play_dataset(self, motids):
        if self.cfg['env']['showObjectKeyPoints']:
            tar_pos = self._target_states[:, 0:3].clone()
            tar_rot = self._target_states[:, 3:7].clone()
            obj_keypoint_pos, obj_keypoint_vector = self.extract_keypoints_per_epoch(tar_pos, tar_rot)
            self._obj_keypoint_states[:, :, :3] = obj_keypoint_pos
            self._obs_target_keypoints_per_epoch = obj_keypoint_pos
        if self.body_to_obj_keypoint:
            self.visualize_keypoint_vectors(obj_keypoint_vector)
    
    def visualize_keypoint_vectors(self, obj_keypoint_vector=None):
        """
        Visualize vectors between key_body_pos and target_keypoints_per_epoch
        using Isaac Gym's line rendering capabilities.
        """
        # Ensure we have the latest data
        self.gym.clear_lines(self.viewer)
        # Convert tensors to numpy for visualization
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :].cpu().numpy()
        target_keypoints = self._obs_target_keypoints_per_epoch.cpu().numpy()
        
        # Clear previous visualizations
        if hasattr(self, '_vector_visual_handles'):
            for handle in self._vector_visual_handles:
                self.gym.clear_lines(self.viewer)
            self._vector_visual_handles = []
        
        # Create line segments for each environment
        for env_idx in range(self.num_envs):
            # Prepare line vertices (start and end points)
            line_verts = []
            for i in range(target_keypoints.shape[1]):
                if obj_keypoint_vector is not None:
                    start = target_keypoints[env_idx, i] - obj_keypoint_vector[env_idx, i].cpu().numpy()
                else:
                    start = key_body_pos [env_idx, i]
                end = target_keypoints[env_idx, i]
                
                # Add both points (start and end) to create a line segment
                line_verts.append(start)
                line_verts.append(end)
            
            # Convert to numpy array
            line_verts = np.array(line_verts, dtype=np.float32)
            
            # Create line colors (red for all vectors)
            line_colors = np.array([[1, 0, 0]] * len(line_verts), dtype=np.float32)
            
            # Draw the lines
            handle = self.gym.add_lines(self.viewer, self.envs[env_idx], len(line_verts)//2+2, 
                                    line_verts, line_colors)

    def _compute_observations(self, env_ids=None): # called @ reset & post step #not used
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs
        obj_obs = self._compute_obj_obs(env_ids)
        obs = torch.cat([obs, obj_obs], dim=-1)

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)

        if (env_ids is None): 
            self.obs_buf[:] = obs

        else:
            self.obs_buf[env_ids] = obs

        return        

    def _compute_obj_obs(self, env_ids=None):
        tar_pos = self._target_states[:, 0:3].clone()
        tar_rot = self._target_states[:, 3:7].clone()
        self._obs_target_keypoints_per_epoch, self._obs_target_keypoints_vectors_per_epoch = self.extract_keypoints_per_epoch(tar_pos, tar_rot)
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
            body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            if self.body_to_obj_keypoint:
                tar_keypoints_feature = self._obs_target_keypoints_vectors_per_epoch.clone()
            else:
                tar_keypoints_feature = self._obs_target_keypoints_per_epoch.clone()
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
            body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            if self.body_to_obj_keypoint:
                tar_keypoints_feature = self._obs_target_keypoints_vectors_per_epoch[env_ids].clone()
            else:
                tar_keypoints_feature = self._obs_target_keypoints_per_epoch[env_ids].clone()       
        obs = compute_obj_observations(root_states, tar_states, tar_keypoints_feature, body_pos,
                                                                      self._enable_nearest_vector, self._enable_obj_keypoints)

        return obs

    def _compute_obj_local_obs(self, env_ids=None):
        tar_pos = self._target_states[:, 0:3].clone()
        tar_rot = self._target_states[:, 3:7].clone()
        self._obs_target_keypoints_per_epoch, self._obs_target_keypoints_vectors_per_epoch = self.extract_keypoints_per_epoch(tar_pos, tar_rot)
        if (env_ids is None):
            tar_states = self._target_states.clone()
            body_pos = self._rigid_body_pos[:, self._key_body_ids, :].clone()
            body_rot = self._rigid_body_rot[:, self._key_body_ids, :].clone()
            if self.body_to_obj_keypoint:
                tar_keypoints_feature = self._obs_target_keypoints_vectors_per_epoch.clone()
            else:
                tar_keypoints_feature = self._obs_target_keypoints_per_epoch.clone()        
        else:
            tar_states = self._target_states[env_ids].clone()
            body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :].clone()
            body_rot = self._rigid_body_rot[env_ids][:, self._key_body_ids, :].clone()
            if self.body_to_obj_keypoint:
                tar_keypoints_feature = self._obs_target_keypoints_vectors_per_epoch[env_ids].clone()
            else:
                tar_keypoints_feature = self._obs_target_keypoints_per_epoch[env_ids].clone()          
        # x y rz are local, z is global
        obs = compute_obj_local_observations(tar_states, tar_keypoints_feature, body_pos, body_rot, self._enable_nearest_vector, self._enable_obj_keypoints, self.body_to_obj_keypoint)
        return obs

    def _compute_obj2_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target2_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target2_states[env_ids]

        obs = compute_obj_observations(root_states, tar_states)
        return obs

class HumanoidWholeBodyWithObjectPlane(HumanoidWholeBodyWithObject): #metric
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # cfg['env'].get('enable_plane', False)
        self._enable_plane = True if 'higher' in cfg['args'].motion_file else False
        self._grab_plane = False
        if 'grab' in cfg['args'].motion_file:
            self._enable_plane = True
            self._grab_plane = True
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
            plane_asset_options.fix_base_link = True  # 固定桌子，使其不会移动
            if self._grab_plane:
                self._plane_proj_asset = self.gym.create_box(self.sim, 0.3, 0.3, 0.01, plane_asset_options)
            else:
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
        plane_pose.p.x = 0 # 0
        plane_pose.p.y = 0 # 0
        plane_pose.p.z = 0.1 # 0.1
        plane_handle = self.gym.create_actor(env_ptr, self._plane_proj_asset, plane_pose, "plane", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, plane_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.5, 0.5))
        

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def quaternion_to_rotation_matrix_batch(quaternion: Tensor) -> Tensor:
    # 确保四元数是单位四元数
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
    
    # 分解四元数分量 (B, 4)
    x = quaternion[:, 0]
    y = quaternion[:, 1]
    z = quaternion[:, 2]
    w = quaternion[:, 3]
    
    # 构建旋转矩阵 (B, 3, 3)
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
    # 获取旋转矩阵 (B, 3, 3)
    R = quaternion_to_rotation_matrix_batch(quaternion)
    # 进行旋转 (B, N, 3)
    rotated_points = torch.bmm(keypoints, R.transpose(1, 2))
    
    # 进行平移 (B, N, 3)
    # 将position从(B, 3)扩展为(B, 1, 3)以便广播
    transformed_points = rotated_points + position.unsqueeze(1)
    
    return transformed_points

@torch.jit.script
def compute_obj_observations(root_states: Tensor, tar_states: Tensor, tar_keypoints_orig: Tensor, body_pos,
                             enable_nearest_vector: bool, enable_obj_keypoints: bool) -> Tensor:
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    ## for disturbance test
    # local_tar_pos += torch.rand_like(local_tar_pos).to(self.device)*0.05
    # local_tar_rot_obs += torch.rand_like(local_tar_rot_obs).to(self.device)*0.5
    # local_tar_vel += torch.rand_like(local_tar_vel).to(self.device)*0.5
    # local_tar_ang_vel += torch.rand_like(local_tar_ang_vel).to(self.device)*0.5

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)


    # transform obj keypoints to current obj pose
    tar_keypoints = tar_keypoints_orig
    # Do not need transform to local for the time being, because root is fixed.
    # TODO: transform to wrist local coordinates.
    if enable_obj_keypoints:
        tar_keypoints = tar_keypoints.view(tar_keypoints.shape[0],-1)
        obs = torch.cat([obs, tar_keypoints], dim=-1)
    return obs

@torch.jit.script
def compute_obj_local_observations(tar_states, tar_keypoints_orig, body_pos, body_rot,
                             enable_nearest_vector=False, enable_obj_keypoints=False, enable_body_to_obj_keypoint=False):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, bool) -> Tensor

    wrist_pos = body_pos[:, -1, :] # right_wrist global xyz
    wrist_rot = body_rot[:, -1, :] # right_wrist global quat rotation

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(wrist_rot)
    local_tar_pos = tar_pos - wrist_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)

    # transform obj keypoints to current obj pose
    tar_keypoints = tar_keypoints_orig
    if enable_obj_keypoints:
        num_obj_keypoints = tar_keypoints_orig.shape[1]
        heading_rot_expand = heading_rot.unsqueeze(1).repeat((1, num_obj_keypoints, 1)) # [num_envs, 20, 4]
        flat_heading_rot = heading_rot_expand.reshape(-1, 4) # [num_envs*20, 4]
        if enable_body_to_obj_keypoint:
            local_tar_keypoints = tar_keypoints
            flat_local_tar_keypoints = local_tar_keypoints.reshape(-1, 3) # [num_envs*20, 3]
            local_tar_keypoints = flat_local_tar_keypoints.view(local_tar_keypoints.shape[0],-1) # [num_envs, 60]
        else:
            local_tar_keypoints = tar_keypoints - wrist_pos.unsqueeze(1)
            local_tar_keypoints[..., -1] = tar_keypoints[..., -1]
            flat_local_tar_keypoints = local_tar_keypoints.reshape(-1, 3) # [num_envs*20, 3]
            flat_local_tar_keypoints = quat_rotate(flat_heading_rot, flat_local_tar_keypoints)
            local_tar_keypoints = flat_local_tar_keypoints.view(local_tar_keypoints.shape[0],-1) # [num_envs, 60]

        obs = torch.cat([obs, local_tar_keypoints], dim=-1)
    return obs