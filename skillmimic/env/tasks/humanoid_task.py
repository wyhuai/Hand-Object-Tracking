from enum import Enum
import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import glob, os, random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils

from env.tasks.base_task import BaseTask


PERTURB_OBJS = [
    ["small", 60],
    # ["large", 60],
]

class HumanoidWholeBody(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.projtype = cfg['env']['projtype']

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"] #V1
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"] #warning by me not setting termination condition
        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.use_action_scale = False
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim) #V1
        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim) # Although the performance impact is usually quite small, it is best to only enable the sensors when needed.
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self._root_states = gymtorch.wrap_tensor(actor_root_state) # shape = [16, 13] (num_env, actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        num_actors = self.get_num_actors_per_env()
        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :] 
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0
        self.init_root_pos = self._initial_humanoid_root_states[:, 0:3]
        self.init_root_rot = torch.zeros_like(self._initial_humanoid_root_states[:,3:7], device=self.device, dtype=torch.float)
        self.init_root_pos_vel = self._initial_humanoid_root_states[:, 7:10]
        self.init_root_rot_vel = torch.zeros_like(self._initial_humanoid_root_states[:, 10:13], device=self.device, dtype=torch.float)

        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
        self.init_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self.init_dof_pos_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        #########################changed by me
        self.num_rigid_bodies = self.gym.get_actor_rigid_body_count(self.envs[0], self.humanoid_handles[0])  #warning as isaacgym may not be able to read xml properly it treat all link as rigid bodies even xyz =63 for urdf =52 xml for xml
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_rigid_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_rigid_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_rigid_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_rigid_bodies, 10:13]
        ##############################################################

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_rigid_bodies, :] #changed by me warning useless
                
        self._build_termination_heights()
        
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._key_body_wrist_ids = self._build_key_body_ids_tensor(self.cfg["env"]["keyBodiesWrist"])
        self._contact_body_ids = self._build_contact_body_ids_tensor(self.cfg["env"]["contactBodies"])
        
        if self.viewer != None:
            self._init_camera()

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long) # in  extras/info
        return

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)
        asset_file = "mjcf/rhand_mano_low_mass.xml"
        self._dof_obs_size = 17*3 #changed by me 17 as included wris_r
        self._num_actions = 17*3  #changed by me
        obj_obs_size = 15
        # 5 *3 is contact force
        self._num_obs = 1 + (num_key_bodies + 1) * (3 + 6 + 3 + 3) - 3 + 5*3 + (num_key_bodies - 15) * 3 + 48 + 3 + 7
        if self._enable_text_obs:
            self._num_obs += 52
        if self._enable_future_target_obs:
            if "futureVectorDim" in self.cfg['env']:
                self._num_obs += self.future_vecotr_dim
            else:
                self._num_obs += (num_key_bodies + 1) * 5 * 3 + 240 + 5
        if self._enable_dense_obj:
            self._num_obs += 180 * 3
        if self._enable_nearest_vector:
            self._num_obs += (num_key_bodies + 1) * 3
        if self._enable_obj_keypoints:
            self._num_obs += self._num_target_keypoints*3
        if self._enable_dof_obs:
            self._num_obs += 51 * 2 + 1
        if "Insert" in self.cfg['name']:
            self._num_obs += obj_obs_size
        if self._enable_wrist_local_obs:
            self._num_obs -= 15
        if "DexGenBallPlay" in self.cfg['name']:
            self._num_obs = 387
        return
    
    
    def get_obs_size(self):
        obs_size = 0
        humanoid_obs_size = self._num_obs
        obs_size += humanoid_obs_size
        if self._enable_task_obs:
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors
    
    def get_task_obs_size(self):
        return 0

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["assetFileName"]

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.fix_base_link = True
        # asset_options.disable_gravity = True
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        #right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        #left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        #self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        #self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts) #changed by me warning gym cannot support actuator in urdf https://forums.developer.nvidia.com/t/dofs-vs-actuators/177537 #aaaaaaaaaaaaaaaaa
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        ################################################################################# changed by me
        # self.num_bodies = len(self.cfg["env"]["keyBodies"]) + 1 # as root is not included in key body
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.num_dof

        ######################################################
        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        max_agg_bodies = self.num_humanoid_bodies + 40
        max_agg_shapes = self.num_humanoid_shapes + 40
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self._build_env(i, env_ptr, humanoid_asset)
            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        #print("yyyyyyyyyyyy",self.num_dof)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        # dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        # print(dof_prop["driveMode"])
        # print(gymapi.DOF_MODE_POS)
        # print(dof_prop["stiffness"])
        # import sys
        # sys.exit()
        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id 
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        char_h = 0.89

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
        self.humanoid_handles.append(humanoid_handle)
        return


    ###########################by me
    def _create_table(self, env_id, env_ptr):
        table_size = gymapi.Vec3(2.0, 1.0, 0.5)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True 
        table_asset = self.gym.create_box(self.sim, table_size.x, table_size.y, table_size.z, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(1.25, 0, 0.25)
        table_pose.r = gymapi.Quat(1, 0, 0, 0)
        table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", 0, 1, 0)
        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.9, 0.9))
        table_props = self.gym.get_actor_rigid_body_properties(env_ptr, table_handle)
        table_props[0].mass = 100.0
        self.gym.set_actor_rigid_body_properties(env_ptr, table_handle,table_props)

        '''
        # Table creation code
        col_group = env_id 
        col_filter = 0
        segmentation_id = 0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(1.0, 0.0, 0.5)  # Adjust position as needed
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Create table top
        table_top_size = gymapi.Vec3(1.5, 1.0, 0.05)
        table_top_asset = self.gym.create_box(self.sim, table_top_size.x, table_top_size.y, table_top_size.z, gymapi.AssetOptions())
        table_top_handle = self.gym.create_actor(env_ptr, table_top_asset, table_pose, "table_top", col_group, col_filter, segmentation_id)

        # Create table legs
        leg_size = gymapi.Vec3(0.05, 0.05, 0.5)
        leg_asset = self.gym.create_box(self.sim, leg_size.x, leg_size.y, leg_size.z, gymapi.AssetOptions())

        leg_positions = [
            (-0.7, -0.45, -0.25), (0.7, -0.45, -0.25),
            (-0.7, 0.45, -0.25), (0.7, 0.45, -0.25)
        ]
        leg_handlers = []
        for i, pos in enumerate(leg_positions):
            leg_pose = gymapi.Transform()
            leg_pose.p = gymapi.Vec3(*pos) + table_pose.p
            leg_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            leg_handle = self.gym.create_actor(env_ptr, leg_asset, leg_pose, f"table_leg_{i}", col_group, col_filter, segmentation_id)
            leg_handlers.append(leg_handle)
        # Set table color
        self.gym.set_rigid_body_color(env_ptr, table_top_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.6, 0.4))

        # Make table static
        table_props = self.gym.get_actor_rigid_body_properties(env_ptr, table_top_handle)
        for prop in table_props:
            prop.mass = 0  # Set mass to 0 to make it static
        self.gym.set_actor_rigid_body_properties(env_ptr, table_top_handle, table_props)

        for i in range(4):
            leg_handle = self.gym.get_actor_handle(env_ptr, leg_handlers[0])
            leg_props = self.gym.get_actor_rigid_body_properties(env_ptr, leg_handle)
            for prop in leg_props:
                prop.mass = 0  # Set mass to 0 to make it static
            self.gym.set_actor_rigid_body_properties(env_ptr, leg_handle, leg_props)
        '''
        return
    #################################

    def _build_pd_action_offset_scale(self):
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_humanoid_collision_filter(self):
        return 1

    def _build_termination_heights(self):
        self._termination_heights = 0.3
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return
    


    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids): #Z10
        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim) 
        self.gym.refresh_net_contact_force_tensor(self.sim)

        return
    
    def _reset_actors(self, env_ids):
        self._reset_humanoid(env_ids)
        return
    
    def _reset_humanoid(self, env_ids):
        # self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids] 
        
        # self._dof_pos[env_ids,6:] = self.init_dof_pos[env_ids,6:] #changed by me  #ohohohohoohohohohooh #chnaged by me ignore change of wrist_x, wrist_y, wrist_z
        # self._dof_vel[env_ids,6:] = self.init_dof_pos_vel[env_ids,6:] #changed by me  #ohohohohoohohohohooh #chnaged by me ignore change of wrist_x, wrist_y, wrist_z
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids] 
        self._dof_pos[env_ids] = self.init_dof_pos[env_ids] 
        self._dof_vel[env_ids] = self.init_dof_pos_vel[env_ids]
        return

    def get_current_pd_targets(self):
        """获取当前 PD 控制器的目标位置"""
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        pd_targets = dof_state.view(-1, 2)[:, 0]  # 提取位置
        return pd_targets.reshape(self.num_envs, -1)

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone() # [num_envs, 56]
        if (self._pd_control) and (not self._use_delta_action) and (not self._use_res_action):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        elif (self._pd_control) and (self._use_delta_action):
            scaled_action = action_scale(self.actions) if self.use_action_scale else self.actions
            cur_pd_tar = self.get_current_pd_targets()
            pd_tar = self._action_to_pd_targets(scaled_action) # delta action
            pd_tar[:, :6] += cur_pd_tar[:, :6] # delta for wrist joints
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        elif (self._pd_control) and (self._use_res_action):
            ref_pd_tar = self._curr_ref_obs[:,7:7+17*3]
            res_pd_tar = self._action_to_respd_targets(self.actions) # res action
            pd_tar = ref_pd_tar + res_pd_tar
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return
    
    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar
    
    def _action_to_respd_targets(self, action):
        pd_tar = self._pd_action_scale * action
        return pd_tar

    def post_physics_step(self):
        self.progress_buf += 1
        self.progress_buf_total += 1

        self._refresh_sim_tensors()

        self._compute_observations() # for policy
        self._compute_reward()
        self._compute_metrics() 
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf 
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return
    
    def _update_proj(self):
        return
    
    def _compute_reward(self):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf) #V1

        return
    
    def _compute_metrics(self):
        self.extras["metrics"]= {}
        self.extras["metrics"]['E_op'], self.extras["metrics"]['E_or'], self.extras["metrics"]['E_h'], self.extras["metrics"]['error_done'] \
            = compute_metrics(self.obs_buf) #V1

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights
                                                   )
        return


    def _compute_observations(self, env_ids=None): # called @ reset & post step
        obs = None
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        obs = humanoid_obs

        if self._enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            obs = torch.cat([obs, task_obs], dim = -1)

        if (env_ids is None): 
            self.obs_buf[:] = obs

        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        ##########################################changed by me
        body_id = torch.tensor([0] +self._key_body_ids.tolist() )  # Shape: (3,)
        if (env_ids is None):
            body_pos = self._rigid_body_pos[:, body_id, :].clone()
            body_rot = self._rigid_body_rot[:, body_id, :].clone()
            body_vel = self._rigid_body_vel[:, body_id, :].clone()
            body_ang_vel = self._rigid_body_ang_vel[:, body_id, :].clone()
            contact_forces = self._contact_forces.clone()
        else:
            body_pos = self._rigid_body_pos[env_ids][:, body_id, :].clone()
            body_rot = self._rigid_body_rot[env_ids][:, body_id, :].clone()
            body_vel = self._rigid_body_vel[env_ids][:, body_id, :].clone()
            body_ang_vel = self._rigid_body_ang_vel[env_ids][:, body_id, :].clone()
            contact_forces = self._contact_forces[env_ids]
        
        obs = compute_humanoid_observations(body_pos, body_rot, body_vel, body_ang_vel, 
                                                self._local_root_obs, self._root_height_obs,
                                                contact_forces, self._contact_body_ids)
        '''
        print("Shapes of body attributes after indexing with env_ids:")
        print("body_pos shape:", body_pos.shape)
        print("body_rot shape:", body_rot.shape)
        print("body_vel shape:", body_vel.shape)
        print("body_ang_vel shape:", body_ang_vel.shape)
        print("contact_forces shape:", contact_forces.shape)
        print("Shapes of body and observation attributes:")
        print("body_pos shape:", body_pos.shape)
        print("body_rot shape:", body_rot.shape)
        print("body_vel shape:", body_vel.shape)
        print("body_ang_vel shape:", body_ang_vel.shape)
        print("local_root_obs shape:", self._local_root_obs)
        print("root_height_obs shape:", self._root_height_obs)
        print("contact_forces shape:", contact_forces.shape)
        print("contact_body_ids shape:", self._contact_body_ids.shape)
        '''
        #######################################################################
        return obs

    def _compute_humanoid_local_obs(self, env_ids=None):
        body_id = torch.tensor([0] +self._key_body_ids.tolist() )  # Shape: (3,)
        if (env_ids is None):
            body_pos = self._rigid_body_pos[:, body_id, :].clone()
            body_rot = self._rigid_body_rot[:, body_id, :].clone()
            body_vel = self._rigid_body_vel[:, body_id, :].clone()
            body_ang_vel = self._rigid_body_ang_vel[:, body_id, :].clone()
            contact_forces = self._contact_forces.clone()
        else:
            body_pos = self._rigid_body_pos[env_ids][:, body_id, :].clone()
            body_rot = self._rigid_body_rot[env_ids][:, body_id, :].clone()
            body_vel = self._rigid_body_vel[env_ids][:, body_id, :].clone()
            body_ang_vel = self._rigid_body_ang_vel[env_ids][:, body_id, :].clone()
            contact_forces = self._contact_forces[env_ids]
        
        obs = compute_humanoid_local_observations(body_pos, body_rot, body_vel, body_ang_vel, 
                                                self._local_root_obs, self._root_height_obs,
                                                contact_forces, self._contact_body_ids)
        # if self._enable_dof_obs:
        #     dof_pos = self.get_current_pd_targets()
        #     obs = torch.cat([obs, dof_pos], dim = -1)
        return obs


    def render(self, sync_frame_time=False):
        if self.viewer:
            self._update_camera()
            self._draw_task()

        super().render(sync_frame_time)
        return

    def _draw_task(self):
        return
    
    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)
        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.50)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        # cam_pos = gymapi.Vec3(0, 
        #                       - 3.0, 
        #                       4.0)
        # cam_target = gymapi.Vec3(0,
        #                          0,
        #                          1.3)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos

        # # # fixed camera
        # new_cam_target = gymapi.Vec3(0, 0.5, 1.0)
        # new_cam_pos = gymapi.Vec3(1, -1, 1.6)
        # self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)
        return
    

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return
    

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_metrics(obs_buf):
    # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    return torch.ones_like(obs_buf[:, 0]), torch.ones_like(obs_buf[:, 0]), torch.ones_like(obs_buf[:, 0]) , torch.ones_like(obs_buf[:, 0]) 

@torch.jit.script
def action_scale(actions):
    """
    Scale the action to the range of the PD controller
    """
    num_joints = actions.shape[-1]
    wrist_action_std = 0.01
    finger_action_std = 0.03
    action_std = torch.cat([torch.ones(6, device=actions.device) * wrist_action_std, \
                            torch.ones(num_joints-6, device=actions.device) * finger_action_std])
    action_mean = torch.zeros(num_joints, device=actions.device)
    scaled_action = actions * action_std + action_mean
    return scaled_action

# @torch.jit.script
def compute_humanoid_observations(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    body_contact_buf = contact_forces[:, contact_body_ids, :].clone().view(contact_forces.shape[0],-1) # changed by me
    # body_contact_buf = contact_forces[:, :, :].clone().view(contact_forces.shape[0],-1)# changed by me
    '''
    print("Shapes of observation and body attributes:")
    print("root_h_obs shape:", root_h_obs.shape)
    print("local_body_pos shape:", local_body_pos.shape)
    print("local_body_rot_obs shape:", local_body_rot_obs.shape)
    print("local_body_vel shape:", local_body_vel.shape)
    print("local_body_ang_vel shape:", local_body_ang_vel.shape)
    print("body_contact_buf shape:", body_contact_buf.shape)
    '''
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, body_contact_buf), dim=-1)
    return obs

# @torch.jit.script
# body_pos: [root, ..., right_wrist]
def compute_humanoid_local_observations(body_pos, body_rot, body_vel, body_ang_vel, local_wrist_obs, wrist_height_obs, contact_forces, contact_body_ids):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor) -> Tensor
    # exclude the root from the observation
    body_pos, body_rot, body_vel, body_ang_vel = body_pos[:, 1:], body_rot[:, 1:], body_vel[:, 1:], body_ang_vel[:, 1:]
    num_envs, num_joints, dim = body_pos.shape[0], body_pos.shape[1], body_pos.shape[2]

    wrist_pos = body_pos[:, -1, :] # right_wrist global xyz
    wrist_rot = body_rot[:, -1, :] # right_wrist global quat rotation
    wrist_obs = torch.cat([wrist_pos, wrist_rot], dim=-1) # [num_envs, 7]

    heading_rot = torch_utils.calc_heading_quat_inv(wrist_rot)
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    wrist_pos_expand = wrist_pos.unsqueeze(-2)
    local_body_pos = body_pos - wrist_pos_expand # [num_envs, 16, 3]
    flat_local_body_pos = local_body_pos.reshape(num_envs*num_joints, -1)
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(num_envs, -1)
    # remove wrist pos
    local_body_pos = local_body_pos[..., :-3] # [num_envs, 16*3-3]

    flat_body_rot = body_rot.reshape(num_envs*num_joints, -1)
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(num_envs, -1) # [num_envs, 16*6]
    # remove wrist rot
    local_body_rot_obs = local_body_rot_obs[..., :-6] # [num_envs, 15*6]


    flat_body_vel = body_vel.reshape(num_envs*num_joints, -1)
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(num_envs, -1) # [num_envs, 16*3]
    
    flat_body_ang_vel = body_ang_vel.reshape(num_envs*num_joints, -1) # [num_envs*num_joints, 3]
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel) # [num_envs*num_joints, 3]
    local_body_ang_vel = flat_local_body_ang_vel.reshape(num_envs, -1)
    body_contact_buf = contact_forces[:, contact_body_ids, :].clone().view(num_envs,-1) # [num_envs, 5 * 3]
    obs = torch.cat((wrist_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, body_contact_buf), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, rigid_body_pos,
                           
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        body_height = rigid_body_pos[:, 0, 2] # root height
        body_fall = body_height < termination_heights # [4096] 
        has_failed = body_fall.clone()
        has_failed *= (progress_buf > 1)
        
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated