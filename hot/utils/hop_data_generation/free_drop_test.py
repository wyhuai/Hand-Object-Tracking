from isaacgym import gymapi
import torch
import os
import shutil

import numpy as np
from isaacgym import gymtorch
import time as python_time
from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import *

class FindStablePoseTest:
    def __init__(self, asset_path=None,
                 stable_pos_path=None,
                 save_images = True,
                 headless = False
                ):

        self.sim = None
        self.gym = None
        self.viewer = None
        self.target_asset = None
        self.envs = []
        self.target_handles = []
        self.headless = headless
        self._cam_prev_char_pos = np.zeros(3, dtype=float)
        self.num_envs = 1000
        self.save_images = save_images
        self.projtype = "Mouse"
        self.gym = gymapi.acquire_gym()
        sim_params = self.parse_sim_params()
        
        graphics_device_id = 0
        if self.headless == True:
            graphics_device_id = -1

        self.sim = self.gym.create_sim(0, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
                print("*** Failed to create sim")
                quit()


        self._create_ground_plane()


        if asset_path is None:
            asset_root = "../../data/assets/urdf/automate/urdf/" #projectname
            asset_file = "00004_plug.urdf" #"ball.urdf"stick
            #asset_root = "../../data/assets/urdf" #projectname
            #asset_file = "box02.urdf" #"ball.urdf"stick
            asset_root = "../../data/assets/urdf/box003" #projectname
            asset_file = "box003.urdf" #"ball.urdf"stick
            #asset_root = "../../data/assets/urdf/Camera_f1558b9324187ad5ee9f6e6cbbdc436f" #projectname
            #asset_file = "Camera_f1558b9324187ad5ee9f6e6cbbdc436f.urdf" #"ball.urdf"stick
            #asset_root = "../../data/assets/urdf/581" #projectname
            #asset_file = "00581_plug.urdf" #"ball.urdf"stick
            asset_root = "../../data/assets/urdf/WineGlass_2d89d2b3b6749a9d99fbba385cc0d41d" #projectname
            asset_file = "WineGlass_2d89d2b3b6749a9d99fbba385cc0d41d.urdf" #"ball.urdf"stick
        else:
            asset_root = os.path.dirname(asset_path)  # This gets the directory path
            asset_file = os.path.basename(asset_path)  # This gets the file name with extension 
        


        self.object_name =  asset_file.split('.urdf')[0]

        if stable_pos_path is None:
            self.stable_pos_old_path = f"../../data/free_drop_image/{self.object_name}/unique_quats.txt"
        else:
            self.stable_pos_old_path = stable_pos_path

        self.target_asset = self._load_target_asset(asset_root, asset_file)
        self._create_envs(num_envs = self.num_envs , spacing = 0.5 , num_per_row = int(np.sqrt(self.num_envs)))
        self.actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.actor_root_state) # shape = [16, 13] (num_env, actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")

            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 2.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)
    
        self._build_proj_tensors()
        self.gym.prepare_sim(self.sim)


    def compute_pose(self):
        # get root position
        num_actors = self.root_states.shape[0] // self.num_envs
        object_root_states = self.root_states.view(self.num_envs, num_actors, self.actor_root_state.shape[-1])[..., 0, :] 

        unique_rotation_angle = self.load_unique_quats(self.stable_pos_old_path)
        for key in unique_rotation_angle:
            object_root_states[key, 0:3] =  unique_rotation_angle[key]["pos"].clone()
            object_root_states[key, 3:7] = unique_rotation_angle[key]["quat"].clone()
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        init_root_pos = object_root_states[0, 0:3].clone()
        _cam_prev_char_pos = init_root_pos.cpu().numpy()
    

        #while True:
        for i in range(120): 
                self._update_proj()
                # warning don't know if need to call some update here to update object_root_states
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                # refresh the root state tensors
                self.gym.refresh_actor_root_state_tensor(self.sim)
                #gym.refresh_rigid_body_state_tensor(sim)
                #gym.refresh_net_contact_force_tensor(sim)
                self._update_camera(object_root_states[0, 0:3].cpu().numpy())
                self.gym.step_graphics(self.sim)            
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
                wait_time = 0.005 #changed by me
                python_time.sleep(wait_time) #changed by me 

                if self.save_images and i == 100: 
                    unique_rotation_angle = dict(sorted(unique_rotation_angle.items(), key=lambda item: len(item[1]), reverse=True))
                    final_quat = {}
                    for key in unique_rotation_angle:
                        if len(final_quat) >= 30: # at most save 30 stable poses
                            break
                        print(key, " ", object_root_states[key, 3:7],unique_rotation_angle[key]["quat"].clone(), self.quat_equal(object_root_states[key, 3:7].clone().cpu(), unique_rotation_angle[key]["quat"].clone()))
                        if (self.quat_equal(object_root_states[key, 3:7].clone().cpu(), unique_rotation_angle[key]["quat"].clone())):
                            final_quat[key] = {}
                            final_quat[key]["pos"] = object_root_states[key, 0:3].clone()
                            final_quat[key]["quat"] = object_root_states[key, 3:7].clone()
                            self._zoom_camera_to_object(object_root_states[key, 0:3].clone().cpu().numpy(), self.envs[key])
                            wait_time = 0.005 #changed by me
                            python_time.sleep(wait_time) #changed by me 
                            if self.save_images:
                                self.gym.draw_viewer(self.viewer, self.sim, True)
                                self.gym.sync_frame_time(self.sim)
                                image_dir = "../../data/free_drop_test_image/" + self.object_name + "/final"
                                os.makedirs(image_dir, exist_ok=True)
                                rgb_filename = image_dir + "/%s_group%d.png" % (self.object_name, key)
                                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename) 
                        else:
                            self._zoom_camera_to_object(object_root_states[key, 0:3].clone().cpu().numpy(), self.envs[key])
                            wait_time = 0.005 #changed by me
                            python_time.sleep(wait_time) #changed by me 
                            if self.save_images:
                                self.gym.draw_viewer(self.viewer, self.sim, True)
                                self.gym.sync_frame_time(self.sim)
                                image_dir = "../../data/free_drop_test_image/" + self.object_name + "/final"
                                os.makedirs(image_dir, exist_ok=True)
                                rgb_filename = image_dir + "/%s_group%d.png" % (self.object_name, key)
                                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename) 
                            
                    self.gym.destroy_viewer(self.viewer)
                    self.gym.destroy_sim(self.sim)
                    # Save the unique quat for stable pose
                    self.save_unique_quats(final_quat, f"../../data/free_drop_test_image/{self.object_name}/unique_quats.txt")
                    return os.path.abspath(f"../../data/free_drop_test_image/{self.object_name}/unique_quats.txt")
                
    def release_GPU(self):
            torch.cuda.empty_cache()
                            
    def remove_non_empty_directory(self, directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' and all its contents removed successfully.")
        except OSError as e:
            print(f"Error: {e.strerror}")

    def quat_conjugate(self, q):
        # Compute the conjugate of a quaternion. The input q is a quaternion or an array of quaternions.
        # The format of the quaternion is [x, y, z, w], where w is the real part.
        q_conj = q.clone()  # Clone q to avoid modifying the original data.
        q_conj[..., 0:3] = -q_conj[..., 0:3]  # Negate the x, y, z components.
        return q_conj

    def quat_multiply(self, q1, q2):
        # Compute the product of two quaternions.
        # The input quaternion format is [x, y, z, w], where w is the real part.
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

        return torch.stack((x, y, z, w), dim=-1)

    def save_first_quats(self, rotation_groups, object_root_states, filename):
        """
        Save the first quaternion and position of each rotation group to a text file.

        Parameters:
        rotation_groups (dict): Dictionary containing rotation groups.
        object_root_states (np.ndarray): Array containing the object's root states including positions and quaternions.
        filename (str): The path to the output text file.
        """
        with open(filename, 'w') as f:
            for key, indices in rotation_groups.items():
                # Get the first element's quaternion and position from the first index in the group
                first_element_env_index = indices[0]
                first_quat = object_root_states[first_element_env_index, 3:7]  # Quaternions at indices 3 to 6
                first_pos = object_root_states[first_element_env_index, 0:3]    # Position at indices 0 to 2
                
                # Write the position and quaternion to the file separated by a semicolon
                f.write(f"Group {key}: {first_pos[0]:.6f}, {first_pos[1]:.6f}, {first_pos[2]:.6f}; "
                        f"{first_quat[0]:.6f}, {first_quat[1]:.6f}, {first_quat[2]:.6f}, {first_quat[3]:.6f}\n")
        
        print(f"First positions and quaternions saved to {filename}")

    def save_unique_quats(self, rotation_groups, filename):
        """
        Save the unique rotation of each rotation group to a text file.

        Parameters:
        rotation_groups (dict): Dictionary containing rotation groups.
        object_root_states (np.ndarray): Array containing the object's root states including quaternions.
        filename (str): The path to the output text file.
        """
        with open(filename, 'w') as f:
            for key, pose in rotation_groups.items():
                # Write the quaternion to the file
                pos = pose["pos"]
                quat = pose["quat"]
                f.write(f"Group {key}: {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}; "
                        f"{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}\n")
        
        

    #def is_only_z_rotation(q_initial, q_current, first_element_env_index, atol=1e-3):
    def is_only_z_rotation(self, q_initial, q_current, atol=5e-2):
        # Compute the relative quaternion
        q_initial_inv = self.quat_conjugate(q_initial)
        q_relative = self.quat_multiply(q_current, q_initial_inv) # represent current orientation (q_current) relative to the initial orientation's (q_initial) coordinate system, i.e. transforms q_current into the local coordinate system defined by q_initial
        # Check if q_relative represents a Z-axis rotation
        is_z_rotation = (
            torch.isclose(q_relative[0], torch.tensor(0.0), atol=atol) and  # x component should be close to 0
            torch.isclose(q_relative[1], torch.tensor(0.0), atol=atol)   # y component should be close to 0
            #and np.isclose(np.sqrt(q_relative[0]**2 + q_relative[1]**2 + q_relative[2]**2 + q_relative[3]**2), 1, atol=atol)  # must be a unit quaternion
        )
        if is_z_rotation:
            return True
        else:
            return False

    def quat_equal(self, q1: torch.Tensor, q2: torch.Tensor, atol: float = 5e-2) -> bool:
        """
        Check if two quaternions (xyzw order) are numerically equal within a tolerance.
        
        Args:
            q1 (torch.Tensor): First quaternion tensor.
            q2 (torch.Tensor): Second quaternion tensor.
            atol (float, optional): Absolute tolerance. Defaults to 1e-6.
        
        Returns:
            bool: True if quaternions are equal within the tolerance, False otherwise.
        """
        if q1.shape != q2.shape:
            return False
        return torch.allclose(q1, q2, atol=atol, rtol=0.0) or torch.allclose(q1, -q2, atol=atol, rtol=0.0)

    def parse_sim_params(self):
        # initialize sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        
        physics_engine = gymapi.SIM_PHYSX

        if physics_engine == gymapi.SIM_FLEX:
            sim_params.flex.shape_collision_margin = 0.01
            sim_params.flex.num_outer_iterations = 4
            sim_params.flex.num_inner_iterations = 10
        elif physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 4
            sim_params.physx.num_velocity_iterations = 0
            sim_params.physx.num_threads = 4
            sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        return sim_params


    def _create_ground_plane(self):
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = 1.0
            plane_params.dynamic_friction = 1.0
            plane_params.restitution = 0.0 # as required
            self.gym.add_ground(self.sim, plane_params)
            return

    def _load_target_asset(self, asset_root, asset_file): # smplx
            asset_root = asset_root
            asset_file = asset_file
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01
            asset_options.max_angular_velocity = 100.0
            asset_options.density = 1000 #85.0#*6
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = 10
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000

            self.target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            return self.target_asset


    def _build_target(self, env_id, env_ptr):
            col_group = env_id
            col_filter = 0
            segmentation_id = 0

            default_pose = gymapi.Transform()

            target_handle = self.gym.create_actor(env_ptr, self.target_asset, default_pose, "target", col_group, col_filter, segmentation_id)

            ball_props =  self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
            # Modify the properties
            for b in ball_props:
                b.restitution = 0.0 # as required
            self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, ball_props)  
            
            # set ball color
            if self.headless == False:
                self.gym.set_rigid_body_color(env_ptr, target_handle, 0, gymapi.MESH_VISUAL,
                                            gymapi.Vec3(1.5, 1.5, 1.5))
                                            # gymapi.Vec3(0., 1.0, 1.5))
                h = self.gym.create_texture_from_file(self.sim, '../../data/assets/urdf/box.png') #projectname
                self.gym.set_rigid_body_texture(env_ptr, target_handle, 0, gymapi.MESH_VISUAL, h)

            ballSize = 1
            self.target_handles.append(target_handle)
            self.gym.set_actor_scale(env_ptr, target_handle, ballSize)
            return

    def _update_camera(self, object_root_pos):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            char_root_pos = object_root_pos
            
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
            cam_delta = cam_pos - self._cam_prev_char_pos
            new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
            new_cam_pos = gymapi.Vec3(char_root_pos[0] + 1, 
                                    char_root_pos[1] + 1, 
                                    cam_pos[2])


            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

            self._cam_prev_char_pos[:] = char_root_pos

            return

    def _zoom_camera_to_object(self, object_root_pos, env_num):
            self.gym.refresh_actor_root_state_tensor(self.sim)
            char_root_pos = object_root_pos
            
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
            cam_delta = [0,1]
            new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
            new_cam_pos = gymapi.Vec3(char_root_pos[0], 
                                    char_root_pos[1]+ 0.2, 
                                    char_root_pos[2]+ 0.3)


            self.gym.viewer_camera_look_at(self.viewer, env_num, new_cam_pos, new_cam_target)

            self._cam_prev_char_pos[:] = char_root_pos

            return
    
    def load_unique_quats(self, filename):
        """
        Load the unique rotation of each rotation group from a text file.

        Parameters:
        filename (str): The path to the input text file.

        Returns:
        dict: Dictionary containing rotation groups with positions and quaternions.
        """
        rotation_groups = {}
        
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split(':')
                if len(parts) > 1:
                    group_id = int(parts[0].split()[1])  # Extract group ID
                    values = parts[1].strip().split(';')
                    
                    # Parse position and quaternion
                    pos_values = list(map(float, values[0].strip().split(',')))  # Position
                    quat_values = list(map(float, values[1].strip().split(',')))  # Quaternion
                    
                    # Store in the rotation_groups dictionary
                    rotation_groups[group_id] = {
                        'pos': torch.tensor(pos_values),
                        'quat': torch.tensor(quat_values)
                    }
        
        return rotation_groups
    
    def _create_envs(self, num_envs, spacing, num_per_row):
            lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            upper = gymapi.Vec3(spacing, spacing, spacing)
        
            sensor_pose = gymapi.Transform()

            max_agg_bodies =  10
            max_agg_shapes =  10
            if self.projtype == "Mouse" or self.projtype == "Auto":
                self._proj_handles = []
                self._load_proj_asset()
            for i in range(num_envs):
                # create env instance
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
                #self._build_env(i, env_ptr, humanoid_asset)
                self._build_target(i, env_ptr)
                if self.projtype == "Mouse" or self.projtype == "Auto":
                    self._build_proj(i, env_ptr)
                    self.gym.end_aggregate(env_ptr)

                self.envs.append(env_ptr)



            return
    
    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        PERTURB_PROJECTORS = [
            ["small", 60],
            # ["large", 60],
        ]
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
    
    def _build_proj_tensors(self):
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40
        PERTURB_PROJECTORS = [
            ["small", 60],
            # ["large", 60],
        ]
        num_actors = 2
        num_objs = len(PERTURB_PROJECTORS)
        self._proj_states = self.root_states.view(self.num_envs, num_actors, self.root_states.shape[-1])[..., (num_actors - num_objs):, :]
        
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, dtype=torch.int32)
        
        bodies_per_env = 2
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        print(contact_force_tensor.shape)
        self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]

        self._calc_perturb_times()
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space_shoot") #ZC0
        self.gym.subscribe_viewer_mouse_event(self.viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")
        
        return
    
    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        PERTURB_PROJECTORS = [
            ["small", 60],
            # ["large", 60],
        ]
        for i, obj in enumerate(PERTURB_PROJECTORS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

        return
    
    def _load_proj_asset(self):
        asset_root = "../../data/assets/urdf" #projectname
        small_asset_file = "block_projectile.urdf"
        
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 1000.0
        # small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

        return
    
    def quat_from_euler_xyz_extrinsic(self, x, y, z):
        quat = R.from_euler('XYZ',[x,y,z]).as_quat()
        quat = torch.tensor(quat).float()
        return quat

    def quat2rotmat(self, quat):

        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat)

        quat = quat / torch.norm(quat, dim=-1, keepdim=True)

        x, y, z, w = torch.unbind(quat, dim=-1)

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        rotmat = torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), w2 - x2 + y2 - z2, 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), w2 - x2 - y2 + z2
        ], dim=-1)

        output_shape = quat.shape[:-1] + (3, 3)
        rotmat = rotmat.reshape(output_shape)

        return rotmat

    def random_quaternion(self, num_samples):
        # Generate random angles
        u1 = np.random.rand(num_samples)
        u2 = np.random.rand(num_samples)
        u3 = np.random.rand(num_samples)

        # Compute quaternion components
        x = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        y = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        z = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        w = np.sqrt(u1) * np.cos(2 * np.pi * u3)

        return np.array([x, y, z, w]).T  # Shape: (num_samples, 4)

    def _update_proj(self):

        self.evts = list(self.gym.query_viewer_action_events(self.viewer))
        if self.projtype == 'Mouse':
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
                    self._proj_states[..., 3:6] = 0.0
                    self._proj_states[..., 6] = 1.0
                    self._proj_states[..., 10:13] = 0.0
                    self._proj_states[..., 0] = spawn.x
                    self._proj_states[..., 1] = spawn.y
                    self._proj_states[..., 2] = spawn.z
                    self._proj_states[..., 7] = vel.x
                    self._proj_states[..., 8] = vel.y
                    self._proj_states[..., 9] = vel.z

                    self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                            gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                            len(self._proj_actor_ids))
        return

# Usage Example
if __name__ == "__main__":
    processor = FindStablePoseTest(asset_path = "../../data/assets/urdf/Box/Box.urdf")
    processor.compute_pose()
    print("\nAll processing completed!")