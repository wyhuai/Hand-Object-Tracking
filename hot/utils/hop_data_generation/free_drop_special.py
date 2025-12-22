from isaacgym import gymapi
import torch
import os
import shutil

import numpy as np
from isaacgym import gymtorch
import time as python_time
from scipy.spatial.transform import Rotation as R

class FindStablePose:
    def __init__(self, asset_path=None,
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
        self.num_envs = 16
        self.save_images = save_images

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

        self.target_asset = self._load_target_asset(asset_root, asset_file)
        self._create_envs(num_envs = self.num_envs , spacing = 2 , num_per_row = int(np.sqrt(self.num_envs)))
        
        self.gym.prepare_sim(self.sim)

    def compute_pose(self):
        # get root position
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(actor_root_state) # shape = [16, 13] (num_env, actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        num_actors = root_states.shape[0] // self.num_envs
        object_root_states = root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :] 

        # give some height so that it can fall
        object_root_states[:, 2] = 0.199

        init_root_pos = object_root_states[0, 0:3].clone()
        _cam_prev_char_pos = init_root_pos.cpu().numpy()
            # bottle standing: 0.7071067811865475, 0.0, 0.0, 0.7071067811865476
            # sword standing: 0,0,0.199 0.7071067811865475, 0.0, 0.0, 0.7071067811865476
        # Generate random quaternions in [x, y, z, w] format

        random_quats = np.array([0.7071067811865475, 0.0, 0.0, 0.7071067811865476])
        object_root_states[:, 3:7] = torch.from_numpy(random_quats).to(object_root_states.device)

        # update root_state
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))

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
        self._update_camera(_cam_prev_char_pos)
        
        rotation_groups = {} 
        unique_rotation_angle = {}                 

        #while True:
        for i in range(1000): 
                # warning don't know if need to call some update here to update object_root_states
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                # refresh the root state tensors
                self.gym.refresh_actor_root_state_tensor(self.sim)
                #gym.refresh_rigid_body_state_tensor(sim)
                #gym.refresh_net_contact_force_tensor(sim)
                wait_time = 0.01 #changed by me
                python_time.sleep(wait_time) #changed by me 
                self._update_camera(object_root_states[0, 0:3].cpu().numpy())
                self.gym.step_graphics(self.sim)            
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
                
                # without lines below will black screen
                if i == 200: # i=200 when most object stablise on the ground                 
                    rotation_groups[0] = [0]
                    unique_rotation_angle[0] =  object_root_states[0, 3:7].clone()
                    self._zoom_camera_to_object(object_root_states[0, 0:3].clone().cpu().numpy(), self.envs[0])
                    wait_time = 0.005 #changed by me
                    python_time.sleep(wait_time) #changed by me 
                    self.gym.draw_viewer(self.viewer, self.sim, True)
                    self.gym.sync_frame_time(self.sim)

                    if self.save_images:
                        image_dir = "../../data/free_drop_image/" + self.object_name + "/gp_" + str(0)
                        rgb_filename = image_dir + "/%s_env%d.png" % (self.object_name, 0)
                    for e in range(1, self.num_envs):
                        current_env_rot_quat = object_root_states[e, 3:7]

                        find_match = None
                        num_keys = len(rotation_groups)
                        for key in rotation_groups:
                            if self.is_only_z_rotation(unique_rotation_angle[key], current_env_rot_quat): #only need to check against the first element in the group
                                rotation_groups[key].append(e)
                                find_match = key
                                break
                        
                        num_keys = len(rotation_groups)
                        # if after going through whole loop no match, create a new group for this env
                        if find_match == None:
                            rotation_groups[num_keys] = [e]
                            unique_rotation_angle[num_keys] =  object_root_states[e, 3:7].clone()

                        self._zoom_camera_to_object(object_root_states[e, 0:3].clone().cpu().numpy(), self.envs[e])
                        wait_time = 0.005 #changed by me
                        python_time.sleep(wait_time) #changed by me 
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        self.gym.sync_frame_time(self.sim)



                #Check if pose really stable
                elif self.save_images and i >= 801 and i < 899: # need to set the pose for a few iteration so it can hold
                    for key in rotation_groups:
                        object_root_states[key, 0:2] = 0
                        object_root_states[key, 7:13] = 0
                        object_root_states[key, 3:7] =  unique_rotation_angle[key].clone()
                        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))

                if self.save_images and i == 900: 
                    unique_rotation_angle = dict(sorted(unique_rotation_angle.items(), key=lambda item: len(item[1]), reverse=True))
                    final_quat = {}
                    for key in unique_rotation_angle:
                        if len(final_quat) >= 30: # at most save 30 stable poses
                             break
                        if (self.quat_equal(object_root_states[key, 3:7], unique_rotation_angle[key].clone())):
                            final_quat[key] = {}
                            final_quat[key]["pos"] = object_root_states[key, 0:3].clone()
                            final_quat[key]["quat"] = object_root_states[key, 3:7].clone()
                            self._zoom_camera_to_object(object_root_states[key, 0:3].clone().cpu().numpy(), self.envs[key])
                            wait_time = 0.005 #changed by me

                        else:
                            self._zoom_camera_to_object(object_root_states[key, 0:3].clone().cpu().numpy(), self.envs[key])
                            wait_time = 0.005 #changed by me
                            python_time.sleep(wait_time) #changed by me 

                    self.gym.destroy_viewer(self.viewer)
                    self.gym.destroy_sim(self.sim)
                    # Save the unique quat for stable pose
                    return os.path.abspath(f"../../data/free_drop_image/{self.object_name}/unique_quats.txt")
                
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
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -3.9)
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

    def _create_envs(self, num_envs, spacing, num_per_row):
            lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            upper = gymapi.Vec3(spacing, spacing, spacing)
        
            sensor_pose = gymapi.Transform()

            max_agg_bodies =  10
            max_agg_shapes =  10
            
            for i in range(num_envs):
                # create env instance
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
                #self._build_env(i, env_ptr, humanoid_asset)
                self._build_target(i, env_ptr)
                self.gym.end_aggregate(env_ptr)

                self.envs.append(env_ptr)


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


# Usage Example
if __name__ == "__main__":
    processor = FindStablePose(asset_path = "../../data/assets/urdf/Sword/Sword.urdf")
    processor.compute_pose()
    print("\nAll processing completed!")