import numpy as np
import os
import torch
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict

from .vec_task import VecTask
from utils.low_level_model import ActorCritic

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
from torch import Tensor
import torchvision.transforms as transforms


class B1PickLowFreq(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        self.cfg = cfg
        
        self.num_torques = 18
        if sim_device == "cpu":
            self.sim_id = 0
        else:
            self.sim_id = int(sim_device.split(":")[1])
        
        self.debug_vis = self.cfg["env"]["enableDebugVis"]
        
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        
        self.enable_camera = self.cfg["sensor"].get("enableCamera", False)
        if self.enable_camera:
            self.resize_transform = transforms.Resize((self.cfg["sensor"]["resized_resolution"][1], self.cfg["sensor"]["resized_resolution"][0]))
        if self.enable_camera and self.cfg["env"]["numEnvs"] > 10:
            self.cfg["env"]["numEnvs"] = 256
        
        self._setup_obs_and_action_info()
        
        self.reward_scales = self.cfg["reward"]["scales"]
        
        self.randomize = False
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        self.low_level_policy = self._load_low_level_model()
        
        self._prepare_reward_function()
        
        self.dt = self.control_freq_inv * self.sim_params.dt
        
        self._init_tensors()
        
        self.global_step_counter = 0
        
        if self.viewer is not None:
            self._init_camera()

    def _init_camera(self):
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return
    
    def _setup_obs_and_action_info(self):
        _num_action = 9
        _num_obs = 7 + 12 + 13 + 13 + _num_action - 6 - 10 + 3
        self.cfg["env"]["numObservations"] = _num_obs
        self.cfg["env"]["numActions"] = _num_action
        if self.enable_camera:
            _num_states = self.cfg["sensor"]["resized_resolution"][0] * self.cfg["sensor"]["resized_resolution"][1] * 2 * 2 + 25 + 9 #+ 7, * 2 for segmentation
            self.cfg["env"]["numStates"] = _num_states
    
    def _init_tensors(self):
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        _jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        
        self._root_states = gymtorch.wrap_tensor(_actor_root_state)
        num_actors = 3
        
        self._robot_root_states = self._root_states.view(self.num_envs, num_actors, _actor_root_state.shape[-1])[..., 0, :]
        self._initial_robot_root_states = self._robot_root_states.clone()
        self._initial_robot_root_states[:, 7:13] = 0.0
        
        self._table_root_states = self._root_states.view(self.num_envs, num_actors, _actor_root_state.shape[-1])[..., 1, :]
        self._initial_table_root_states = self._table_root_states.clone()
        self._initial_table_root_states[:, 7:13] = 0.0
        
        self._cube_root_states = self._root_states.view(self.num_envs, num_actors, _actor_root_state.shape[-1])[..., 2, :]
        self._initial_cube_root_states = self._cube_root_states.clone()
        self._initial_cube_root_states[:, 7:13] = 0.0
        
        self._robot_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._table_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1
        self._cube_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 2
        
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        dof_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dof_per_env, 2)[..., :self.num_dofs, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dof_per_env, 2)[..., :self.num_dofs, 1]
        self._last_dof_vel = self._dof_vel.clone()
        
        self._initial_dof_pos = self._dof_pos.clone()
        self._initial_dof_vel = self._dof_vel.clone()
        
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        
        self._rigid_body_pos = rigid_body_state_reshaped[..., 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., 10:13]
        
        self.jacobian_whole = gymtorch.wrap_tensor(_jacobian_tensor)
        self.gripper_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "wx250s/ee_gripper_link", gymapi.DOMAIN_ENV)
        self.ee_j_eef = self.jacobian_whole[:, self.gripper_idx, :6, -(6 + 2):-2]
        self.ee_pos = rigid_body_state_reshaped[:, self.gripper_idx, 0:3]
        self.ee_orn = rigid_body_state_reshaped[:, self.gripper_idx, 3:7]
        
        self.left_finger_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "wx250s/left_finger_link", gymapi.DOMAIN_ENV)
        self.right_finger_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "wx250s/right_finger_link", gymapi.DOMAIN_ENV)
        
        self.wrist_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "link06", gymapi.DOMAIN_ENV)
        
        self.left_finger_pos = torch.ones(self.num_envs, 1, device=self.device) * 0.037
        self.right_finger_pos = -torch.ones(self.num_envs, 1, device=self.device) * 0.037
        
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)
        
        contact_force_tensor = gymtorch.wrap_tensor(_contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)
        
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, -1, 6)
        self.foot_contacts_from_sensor = self.force_sensor_tensor.norm(dim=-1) > 1.5
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = self._terminate_buf.clone()
        
        self.last_low_actions = torch.zeros(self.num_envs, 18, device=self.device, dtype=torch.float32)
        
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.low_obs_history_buf = torch.zeros(self.num_envs, 10, self.num_proprio, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.motor_strength = torch.ones(self.num_envs, 18, device=self.device, dtype=torch.float32)
        low_action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        self.low_action_scale = torch.tensor(low_action_scale, device=self.device)
        
        self.p_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_torques, dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_torques):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg["env"]["asset"]["control"]["stiffness"].keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg["env"]["asset"]["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["env"]["asset"]["control"]["damping"][dof_name]
                    found = True
                if not found:
                    self.p_gains[i] = 0.
                    self.d_gains[i] = 0.
        
        self.default_dof_pos_wo_gripper = self._initial_dof_pos[:, :-self.num_gripper_joints]
        self.dof_pos_wo_gripper = self._dof_pos[:, :-self.num_gripper_joints]
        self.dof_vel_wo_gripper = self._dof_vel[:, :-self.num_gripper_joints]
        self.gripper_torques_zero = torch.zeros(self.num_envs, self.num_gripper_joints, device=self.device, dtype=torch.float32)
        
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.ee_goal_center_offset = torch.tensor([0.3, 0.0, 0.7], device=self.device).repeat(self.num_envs, 1)
        
        self.closest_dist = -torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.curr_dist = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.lifted_threshold = self.cfg["env"]["liftedThreshold"]
        self.lifted_object = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        self.highest_object = -torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.curr_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.masked_forward = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.masked_wrist = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        
        # --------- get camera image -------- #
        self.camera_sensor_dict = defaultdict(list)
        if self.enable_camera:
            for env_i, env_handle in enumerate(self.envs):
                self.camera_sensor_dict["forward_depth"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][0],
                        gymapi.IMAGE_DEPTH,
                )))
                self.camera_sensor_dict["forward_color"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][0],
                        gymapi.IMAGE_COLOR,
                )))
                self.camera_sensor_dict["forward_seg"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][0],
                        gymapi.IMAGE_SEGMENTATION,
                )))
                self.camera_sensor_dict["wrist_depth"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][1],
                        gymapi.IMAGE_DEPTH,
                )))
                self.camera_sensor_dict["wrist_color"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][1],
                        gymapi.IMAGE_COLOR,
                )))
                self.camera_sensor_dict["wrist_seg"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.camera_handles[env_i][1],
                        gymapi.IMAGE_SEGMENTATION,
                )))
                
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.sim_id, self.sim_id, self.physics_engine, self.sim_params)
        
        self._create_grond_plane()
        self._create_envs()
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        return
    
    def _create_grond_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return
        
    def _create_envs(self):
        spacing = self.cfg["env"]["envSpacing"]
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        num_per_row = int(np.sqrt(self.num_envs))
        
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file_robot = self.cfg["env"]["asset"]["assetFileRobot"]
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 3
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 1000.0
        asset_options.angular_damping = 0.
        asset_options.linear_damping = 0.
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file_robot, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        dof_props_asset['driveMode'][12:].fill(gymapi.DOF_MODE_POS)  # set arm to pos control
        dof_props_asset['stiffness'][12:].fill(400.0)
        dof_props_asset['damping'][12:].fill(40.0)
        self.dof_limits_lower, self.dof_limits_upper, self.torque_limits = [], [], []
        for i in range(self.num_dofs):
            self.dof_limits_lower.append(dof_props_asset['lower'][i])
            self.dof_limits_upper.append(dof_props_asset['upper'][i])
            self.torque_limits.append(dof_props_asset['effort'][i])
            
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.torque_limits = to_torch(self.torque_limits, device=self.device)
        
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names_to_idx = self.gym.get_asset_rigid_body_dict(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.dof_names_to_idx = self.gym.get_asset_dof_dict(robot_asset)
        # self.gripper_body_idx = self.body_names_to_idx["w250s/ee_gripper_link"]
        
        for i in range(len(rigid_shape_props_asset)):
            rigid_shape_props_asset[i].friction = 2.0
        self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props_asset)
        
        # table
        self.table_dimz = 0.25
        table_dims = gymapi.Vec3(0.6, 1.0, self.table_dimz)
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_options)
        table_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        table_rigid_shape_props[0].friction = 0.5
        self.gym.set_asset_rigid_shape_properties(table_asset, table_rigid_shape_props)
        
        # cube
        cube_size = 0.03
        cube_opts = gymapi.AssetOptions()
        cube_dims = [cube_size, cube_size, 3*cube_size]
        # cube_asset = self.gym.create_box(self.sim, *([cube_size] * 3), cube_opts)
        cube_asset = self.gym.create_box(self.sim, *cube_dims, cube_opts)
        cube_asset_props = self.gym.get_asset_rigid_shape_properties(cube_asset)
        cube_asset_props[0].friction = 2.0
        self.gym.set_asset_rigid_shape_properties(cube_asset, cube_asset_props)
        
        # default dof
        default_joint_angles = { # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,   # [rad]
        'FL_thigh_joint': 0.8,     # [rad]
        'FL_calf_joint': -1.5,   # [rad]

        'RL_hip_joint': 0.1,   # [rad]
        'RL_thigh_joint': 0.8,   # [rad]
        'RL_calf_joint': -1.5,    # [rad]

        'FR_hip_joint': -0.1 ,  # [rad]
        'FR_thigh_joint': 0.8,     # [rad]
        'FR_calf_joint': -1.5,  # [rad]

        'RR_hip_joint': -0.1,   # [rad]
        'RR_thigh_joint': 0.8,   # [rad]
        'RR_calf_joint': -1.5,    # [rad]

        'z1_waist': 0.0,
        'z1_shoulder': 1.48,
        'z1_elbow': -0.63,
        'z1_wrist_angle': -0.84,
        'z1_forearm_roll': 0.0,
        'z1_wrist_rotate': 0.0,
        'widow_left_finger': 0.037,
        'widow_right_finger': -0.037,
        }
        robot_dof_dict = self.gym.get_asset_dof_dict(robot_asset)
        initial_pos = np.zeros(self.num_dofs, dtype=np.float32)
        for k, v in default_joint_angles.items():
            initial_pos[robot_dof_dict[k]] = v
        initial_dof_state = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
        initial_dof_state['pos'] = initial_pos
        self.initial_pos = to_torch(initial_pos, device=self.device)
        
        robot_body_dict = self.gym.get_asset_rigid_body_dict(robot_asset)
        robot_body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        feet_names = [s for s in robot_body_names if "foot" in s]
        self.sensor_indices = []
        for name in feet_names:
            foot_idx = robot_body_dict[name]
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, -0.05))
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)
            self.sensor_indices.append(sensor_idx)
        
        table_pos = [0.0, 0.0, 0.5 * table_dims.z]
        
        self.robot_handles, self.table_handles, self.cube_handles = [], [], []
        self.camera_handles = []
        self.envs = []
        
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            col_group = i
            col_filter = 0
            segmentation_id = 0

            robot_start_pose = gymapi.Transform()
            robot_start_pose.p = gymapi.Vec3(-1.55, 0, 0.55) # 0.95 - 1.35
            robot_start_pose.r = gymapi.Quat(0, 0, 0, 1)
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", col_group, col_filter, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props_asset)
            self.gym.set_actor_dof_states(env_ptr, robot_handle, initial_dof_state, gymapi.STATE_ALL)
            
            if self.enable_camera:
                camera_handle = self._create_onboard_cameras(env_ptr, robot_handle)
                wrist_camera_handle = self._create_wrist_cameras(env_ptr, robot_handle)
                self.camera_handles.append([camera_handle, wrist_camera_handle])
            
            table_start_pose = gymapi.Transform()
            table_start_pose.p = gymapi.Vec3(*table_pos)
            table_start_pose.r = gymapi.Quat(0, 0, 0, 1)
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", col_group, col_filter, 1)
            
            cube_start_pose = gymapi.Transform()
            cube_start_pose.p.x = table_start_pose.p.x + np.random.uniform(-0.1, 0.1)
            cube_start_pose.p.y = table_start_pose.p.y + np.random.uniform(-0.1, 0.1)
            cube_start_pose.p.z = table_dims.z + 3*cube_size / 2
            cube_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-np.pi, np.pi))
            cube_handle = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", col_group, col_filter, 2)
            cube_color = gymapi.Vec3(np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0), np.random.uniform(0.3, 1.0))
            # cube_color = gymapi.Vec3(0.5, 0.5, 0.5)
            self.gym.set_rigid_body_color(env_ptr, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cube_color)
            
            self.robot_handles.append(robot_handle)
            self.table_handles.append(table_handle)
            self.cube_handles.append(cube_handle)
            self.envs.append(env_ptr)
            
    def _create_onboard_cameras(self, env_handle, actor_handle):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = self.cfg["sensor"]["onboard_camera"]["resolution"][1]
        camera_props.width = self.cfg["sensor"]["onboard_camera"]["resolution"][0]
        # if hasattr(getattr(self.cfg.sensor, sensor_name), "horizontal_fov"):
        if self.cfg["sensor"]["onboard_camera"].get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = np.random.uniform(
                self.cfg["sensor"]["onboard_camera"]["horizontal_fov"][0],
                self.cfg["sensor"]["onboard_camera"]["horizontal_fov"][1]
            ) if isinstance(self.cfg["sensor"]["onboard_camera"]["horizontal_fov"], (list, tuple)) else self.cfg["sensor"]["onboard_camera"]["horizontal_fov"]
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*self.cfg["sensor"]["onboard_camera"]["position"])
        local_transform.r = gymapi.Quat.from_euler_zyx(*self.cfg["sensor"]["onboard_camera"]["rotation"])
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        return camera_handle
    
    def _create_wrist_cameras(self, env_handle, actor_handle):
        wrist_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "link06")
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = self.cfg["sensor"]["wrist_camera"]["resolution"][1]
        camera_props.width = self.cfg["sensor"]["wrist_camera"]["resolution"][0]
        if self.cfg["sensor"]["wrist_camera"].get("horizontal_fov", None) is not None:
            camera_props.horizontal_fov = np.random.uniform(
                self.cfg["sensor"]["wrist_camera"]["horizontal_fov"][0],
                self.cfg["sensor"]["wrist_camera"]["horizontal_fov"][1]
            ) if isinstance(self.cfg["sensor"]["wrist_camera"]["horizontal_fov"], (list, tuple)) else self.cfg["sensor"]["wrist_camera"]["horizontal_fov"]
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*self.cfg["sensor"]["wrist_camera"]["position"])
        local_transform.r = gymapi.Quat.from_euler_zyx(*self.cfg["sensor"]["wrist_camera"]["rotation"])
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            wrist_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        return camera_handle
    
    def _load_low_level_model(self, stochastic=False):
        low_level_kwargs = {
            "continue_from_last_std": True,
            "init_std": [[0.8, 1.0, 1.0] * 4 + [1.0] * 6],
            "actor_hidden_dims": [128],
            "critic_hidden_dims": [128],
            "activation": 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            "output_tanh": False,
            "leg_control_head_hidden_dims": [128, 128],
            "arm_control_head_hidden_dims": [128, 128],
            "priv_encoder_dims": [64, 20],
            "num_leg_actions": 12,
            "num_arm_actions": 6,
            "adaptive_arm_gains": False,
            "adaptive_arm_gains_scale": 10.0
        }
        num_actions = 18
        self.num_gripper_joints = 2
        self.num_priv = 5 + 1 + 18
        self.num_proprio = 2 + 3 + (18 + self.num_gripper_joints) + (18 + self.num_gripper_joints) + 18 + 4 + 3 + 3 + 3
        self.history_len = 10
        low_actor_critic: ActorCritic = ActorCritic(self.num_proprio,
                                                    self.num_proprio,
                                                    num_actions,
                                                    **low_level_kwargs,
                                                    num_priv=self.num_priv,
                                                    num_hist=self.history_len,
                                                    num_prop=self.num_proprio,
                                                    )
        policy_path = self.cfg["env"]["low_policy_path"]
        loaded_dict = torch.load(policy_path, map_location=self.device)
        low_actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        low_actor_critic = low_actor_critic.to(self.device)
        low_actor_critic.eval()
        print("Low level pretrained policy loaded!")
        if not stochastic:
            return low_actor_critic.act_inference
        else:
            return low_actor_critic.act
    
    def reset(self):
        """override the vec env reset
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        return super().reset()
    
    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
            
    def _reset_envs(self, env_ids):
        if len(env_ids) > 0:
            self._reset_actors(env_ids)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            
            init_sphere = [0.5, np.pi/8, 0]
            self.curr_ee_goal_sphere[env_ids, :] = torch.tensor(init_sphere, device=self.device)
            
        return
    
    def _reset_actors(self, env_ids):
        self._robot_root_states[env_ids] = self._initial_robot_root_states[env_ids]
        self._robot_root_states[env_ids, :2] += torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device) # small randomization
        rand_yaw_robot = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        self._robot_root_states[env_ids, 3:7] = quat_from_euler_xyz(0*rand_yaw_robot, 0*rand_yaw_robot, 0*rand_yaw_robot)
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        
        self._table_root_states[env_ids] = self._initial_table_root_states[env_ids]
        
        # self._cube_root_states[env_ids] = self._initial_cube_root_states[env_ids]
        self._cube_root_states[env_ids, 0] = 0.0
        self._cube_root_states[env_ids, 1] = 0.0
        self._cube_root_states[env_ids, :2] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)
        self._cube_root_states[env_ids, 2] = self.table_dimz + 3*0.03 / 2
        rand_yaw_box = torch_rand_float(-1, 1, (len(env_ids), 1), device=self.device).squeeze(1)
        self._cube_root_states[env_ids, 3:7] = quat_from_euler_xyz(0*rand_yaw_box, 0*rand_yaw_box, rand_yaw_box)
        self._cube_root_states[env_ids, 7:13] = 0.

        return
    
    def _reset_env_tensors(self, env_ids):
        robot_ids_int32 = self._robot_actor_ids[env_ids]
        table_ids_int32 = self._table_actor_ids[env_ids]
        cube_ids_int32 = self._cube_actor_ids[env_ids]
        multi_ids_int32 = torch.cat([robot_ids_int32, table_ids_int32, cube_ids_int32], dim=0)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(multi_ids_int32), len(multi_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(robot_ids_int32), len(robot_ids_int32))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        
        self.last_actions[env_ids, :] = 0
        self.last_low_actions[env_ids, :] = 0
        self.commands[env_ids, :] = 0
        
        self.lifted_object[env_ids] = 0
        
        self.curr_dist[env_ids] = 0.
        self.closest_dist[env_ids] = -1.
        self.curr_height[env_ids] = 0.
        self.highest_object[env_ids] = -1.

        return
    
    def _refresh_sim_tensors(self):
        self._last_dof_vel = self._dof_vel.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        self.foot_contacts_from_sensor = self.force_sensor_tensor.norm(dim=-1) > 1.5
        self._update_curr_dist()
        self._update_base_yaw_quat()
        return
    
    def _update_curr_dist(self):
        left_finger_pos = self._rigid_body_pos[:, self.left_finger_idx, :]
        right_finger_pos = self._rigid_body_pos[:, self.right_finger_idx, :]
        d1 = torch.norm(self.ee_pos - self._cube_root_states[:, :3], dim=-1)
        d2 = torch.norm(left_finger_pos - self._cube_root_states[:, :3], dim=-1)
        d3 = torch.norm(right_finger_pos - self._cube_root_states[:, :3], dim=-1)
        self.curr_dist[:] = (d1 + d2 + d3) / 3.
        self.closest_dist = torch.where(self.closest_dist < 0, self.curr_dist, self.closest_dist)
        
        self.curr_height[:] = self._cube_root_states[:, 2] - self.table_dimz - 0.03 / 2
        self.highest_object = torch.where(self.highest_object < 0, self.curr_height, self.highest_object)
    
    def _compute_observations(self, env_ids=None):            
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        obs = self._compute_robot_obs(env_ids)
        self.obs_buf[env_ids] = torch.cat([obs, self.last_actions[env_ids, :]], dim=-1)
        if self.enable_camera: # as the high level policy freq now is 10Hz.
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.gym.end_access_image_tensors(self.sim)
            depth_image_flat, color_image_flat, wrist_depth_image_flat, wrist_color_image_flat, forward_mask, wrist_mask = self._get_camera_obs()
            # self.states_buf[:] = torch.cat([depth_image_flat, wrist_depth_image_flat, self.masked_forward, self.masked_wrist, self.obs_buf[:, :]], dim=-1) # self.obs_buf[:, 7:], now for debugging
            self.states_buf[:] = torch.cat([depth_image_flat, wrist_depth_image_flat, forward_mask, wrist_mask, self.obs_buf[:, 7:]], dim=-1)
        elif self.enable_camera:
            # self.states_buf[:] = torch.cat([self.states_buf[:, :-(34 + 7)], self.obs_buf[:, :]], dim=-1) # self.obs_buf[:, 7:], now for debugging
            self.states_buf[:] = torch.cat([self.states_buf[:, :-34], self.obs_buf[:, 7:]], dim=-1)
    
    def _compute_robot_obs(self, env_ids=None):
        if env_ids is None:
            robot_root_state = self._robot_root_states
            table_root_state = self._table_root_states
            cube_root_state = self._cube_root_states
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            commands = self.commands
            table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(self.num_envs, 1).to(self.device)
            base_quat_yaw = self.base_yaw_quat
            spherical_center = self.get_ee_goal_spherical_center()
        else:
            robot_root_state = self._robot_root_states[env_ids]
            table_root_state = self._table_root_states[env_ids]
            cube_root_state = self._cube_root_states[env_ids]
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            commands = self.commands[env_ids]
            table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(len(env_ids), 1).to(self.device)
            base_quat_yaw = self.base_yaw_quat[env_ids]
            spherical_center = self.get_ee_goal_spherical_center()[env_ids]
            
        gripper_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.robot_handles[0], "wx250s/ee_gripper_link", gymapi.DOMAIN_ENV)
        
        obs = compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos,
                                         body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, gripper_idx, table_dim)
        
        return obs
    
    def _compute_low_level_observations(self):
        base_ang_vel = quat_rotate_inverse(self._robot_root_states[:, 3:7], self._robot_root_states[:, 10:13])
        # obs_dof_pos = torch.zeros_like(self._dof_pos)
        # obs_dof_pos[:, :12] = self._dof_pos[:, :12]
        # obs_dof_pos[:, 12:] = self._initial_dof_pos[:, 12:]
        # obs_dof_vel = torch.zeros_like(self._dof_vel)
        # obs_dof_vel[:, :12] = self._dof_vel[:, :12]
        # obs_dof_vel[:, 12:] = 0.
        # # print("obs_dof_vel:")
        # # print(obs_dof_vel)
        # # print("dof_vel:")
        # # print(self._dof_vel)
        # init_sphere = [0.5, np.pi/8, 0]
        # self.curr_ee_goal_sphere[:] = torch.tensor(init_sphere, device=self.device)
        low_level_obs_buf = torch.cat((self.get_body_orientation(), # dim 2
                                       base_ang_vel, # dim 3
                                       reindex_all(self._dof_pos - self._initial_dof_pos), # dim 19 or 20
                                       reindex_all(self._dof_vel * 0.05),
                                    #    reindex_all(obs_dof_pos - self._initial_dof_pos), # dim 19 or 20
                                    #    reindex_all(obs_dof_vel * 0.05),
                                    #    reindex_all(torch.zeros_like(self.last_low_actions, device=self.device)),
                                       reindex_all(self.last_low_actions),
                                       reindex_feet(self.foot_contacts_from_sensor),
                                       self.commands[:, :3],
                                       self.curr_ee_goal_sphere,
                                       0*self.curr_ee_goal_sphere
                                       ), dim=-1)
        self.low_obs_history_buf = torch.where(
            (self.progress_buf < 1)[:, None, None],
            torch.stack([low_level_obs_buf] * 10, dim=1),
            self.low_obs_history_buf
        )
        
        self.low_obs_buf = torch.cat([low_level_obs_buf, self.low_obs_history_buf.view(self.num_envs, -1)], dim=-1)
        
        self.low_obs_history_buf = torch.cat([
            self.low_obs_history_buf[:, 1:],
            low_level_obs_buf.unsqueeze(1)
        ], dim=1)
    
    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self._robot_root_states[:, 3:7])
        body_angles = torch.stack([r, p, y], dim=-1)
        if not return_yaw:
            return body_angles[:, :-1]
        else:
            return body_angles
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        
        # -------------------------receive commands----------------------------
        self.actions[:] = action_tensor[:]
        
        self.curr_ee_goal_sphere[:] = self.curr_ee_goal_sphere + torch.tanh(actions[:, :3]) * 0.10
        ee_goal_local_cart = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart = quat_apply(self.base_yaw_quat, ee_goal_local_cart) + self.get_ee_goal_spherical_center()
        
        ee_goal_orn_delta_rpy = torch.tanh(actions[:, 3:6]) * 0.5
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, ee_goal_local_cart)
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + 0.38
        ee_goal_orn_quat = quat_from_euler_xyz(ee_goal_orn_delta_rpy[:, 0], default_pitch + ee_goal_orn_delta_rpy[:, 1], ee_goal_orn_delta_rpy[:, 2] + default_yaw)
        
        u_gripper = actions[:, 6].unsqueeze(-1)
        self.left_finger_pos[:] = torch.where(u_gripper >= 0, self.dof_limits_lower[-2].item(), self.dof_limits_upper[-2].item())
        self.right_finger_pos[:] = torch.where(u_gripper >= 0, self.dof_limits_upper[-1].item(), self.dof_limits_lower[-1].item())
        
        self.commands[:, 0] = torch.tanh(actions[:, 7]) * 0.7
        self.commands[:, 2] = torch.tanh(actions[:, 8]) * 0.7

        self.commands[:, 0] = torch.clip(self.commands[:, 0], -0.5, 0.5)
        self.commands[:, 2] = torch.clip(self.commands[:, 2], -0.5, 0.5)
        # -------------------------receive commands----------------------------
        
        # -------------------------query low level and simulate--------------------------
        for _ in range(5):
            dpos = ee_goal_cart - self.ee_pos
            drot = orientation_error(ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
            dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
            arm_pos_targets = self.control_ik(dpose) + self._dof_pos[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints]
            all_pos_targets = torch.zeros_like(self._dof_pos)
            all_pos_targets[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints] = arm_pos_targets
            all_pos_targets[:, -self.num_gripper_joints:] = torch.cat([self.left_finger_pos, self.right_finger_pos], dim=-1)
            
            self._compute_low_level_observations()
            with torch.no_grad():
                low_actions = self.low_level_policy(self.low_obs_buf.detach(), hist_encoding=True)
            low_actions = reindex_all(low_actions)
            self.torques = self._compute_torques(self.last_low_actions)
            self.last_low_actions[:] = low_actions[:]

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            # step physics and render each frame
            for i in range(self.control_freq_inv):
                self.gym.simulate(self.sim)
            if not self.headless:
                self.render()
            self._refresh_sim_tensors()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    # def pre_physics_step(self, actions: Tensor):
        self.actions[:] = actions[:]
        
        # curr_ee_orn_rpy = torch.stack(euler_from_quat(self.ee_orn), dim=-1)
        # goal_ee_orn_rpy = curr_ee_orn_rpy + torch.tanh(actions[:, 3:6]) * 0.15
        # ee_goal_orn_quat = quat_from_euler_xyz(goal_ee_orn_rpy[:, 0], goal_ee_orn_rpy[:, 1], goal_ee_orn_rpy[:, 2])
        
        # dpos = torch.tanh(actions[:, :3]) * 0.15
        # ee_goal_cart = self.ee_pos + dpos
        # ee_goal_local_cart = quat_rotate_inverse(self.base_yaw_quat, ee_goal_cart - self.get_ee_goal_spherical_center())
        # self.curr_ee_goal_sphere[:] = cart2sphere(ee_goal_local_cart)
        self.curr_ee_goal_sphere[:] = self.curr_ee_goal_sphere + torch.tanh(actions[:, :3]) * 0.15
        ee_goal_local_cart = sphere2cart(self.curr_ee_goal_sphere)
        ee_goal_cart = quat_apply(self.base_yaw_quat, ee_goal_local_cart) + self.get_ee_goal_spherical_center()
        dpos = ee_goal_cart - self.ee_pos
        
        ee_goal_orn_delta_rpy = torch.tanh(actions[:, 3:6]) * 0.5
        ee_goal_cart_yaw_global = quat_apply(self.base_yaw_quat, ee_goal_local_cart)
        default_yaw = torch.atan2(ee_goal_cart_yaw_global[:, 1], ee_goal_cart_yaw_global[:, 0])
        default_pitch = -self.curr_ee_goal_sphere[:, 1] + 0.38
        ee_goal_orn_quat = quat_from_euler_xyz(ee_goal_orn_delta_rpy[:, 0], default_pitch + ee_goal_orn_delta_rpy[:, 1], ee_goal_orn_delta_rpy[:, 2] + default_yaw)
        
        drot = orientation_error(ee_goal_orn_quat, self.ee_orn / torch.norm(self.ee_orn, dim=-1).unsqueeze(-1))
        dpose = torch.cat([dpos, drot], dim=-1).unsqueeze(-1)
        arm_pos_targets = self.control_ik(dpose) + self._dof_pos[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints]
        
        u_gripper = actions[:, 6].unsqueeze(-1)
        self.left_finger_pos[:] = torch.where(u_gripper >= 0, self.dof_limits_lower[-2].item(), self.dof_limits_upper[-2].item())
        self.right_finger_pos[:] = torch.where(u_gripper >= 0, self.dof_limits_upper[-1].item(), self.dof_limits_lower[-1].item())
        
        all_pos_targets = torch.zeros_like(self._dof_pos)
        all_pos_targets[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints] = arm_pos_targets
        all_pos_targets[:, -self.num_gripper_joints:] = torch.cat([self.left_finger_pos, self.right_finger_pos], dim=-1)
        
        self.commands[:, 0] = torch.tanh(actions[:, 7]) * 1.5
        self.commands[:, 2] = torch.tanh(actions[:, 8]) * 1.5

        self.commands[:, 0] = torch.clip(self.commands[:, 0], -1.2, 1.2)
        self.commands[:, 2] = torch.clip(self.commands[:, 2], -1.2, 1.2)
        
        # ----------------- low level actions -----------------
        self._compute_low_level_observations()
        with torch.no_grad():
            low_actions = self.low_level_policy(self.low_obs_buf.detach(), hist_encoding=True)
        low_actions = reindex_all(low_actions)
        self.torques = self._compute_torques(self.last_low_actions)
        self.last_low_actions[:] = low_actions[:]
        
        
        # ----------------- low level actions -----------------
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(all_pos_targets))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
    
    def get_ee_goal_spherical_center(self):
        center = torch.cat([self._robot_root_states[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        center = center + quat_apply(self.base_yaw_quat, self.ee_goal_center_offset)
        return center
    
    def _compute_torques(self, actions):
        actions_scaled = actions * self.motor_strength * self.low_action_scale

        default_torques = self.p_gains * (actions_scaled + self.default_dof_pos_wo_gripper - self.dof_pos_wo_gripper) - self.d_gains * self.dof_vel_wo_gripper
        default_torques[:, -6:] = 0
        torques = torch.cat([default_torques, self.gripper_torques_zero], dim=-1)
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
        
    def control_ik(self, dpose):
        j_eef_T = torch.transpose(self.ee_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        u = (j_eef_T @ torch.inverse(self.ee_j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u
    
    def post_physics_step(self):
        self.progress_buf += 1
        self.global_step_counter += 1
        self._refresh_sim_tensors()
        self.check_termination()
        self.compute_reward()
        self.last_actions[:] = self.actions[:]
        self._compute_observations()
        
        if self.debug_vis:
            self._draw_camera_sensors()
        
        self.extras["terminate"] = self._terminate_buf.to(self.rl_device)
        return
    
    def check_termination(self):
        base_quat = self._robot_root_states[:, 3:7]
        r, p, _ = euler_from_quat(base_quat)
        z = self._robot_root_states[:, 2]
        
        r_term = torch.abs(r) > 0.8
        p_term = torch.abs(p) > 0.8
        z_term = z < 0.1
        
        z_cube = self._cube_root_states[:, 2]
        cube_falls = (z_cube < (self.table_dimz + 0.03 / 2 - 0.05))
        self.timeout_buf = self.progress_buf >= self.max_episode_length
        self.reset_buf[:] = self.timeout_buf | r_term | p_term | z_term | cube_falls
        
        # add early stop to prevent the agent from gaining too much useless data
        if self.enable_camera:
            robot_head_dir = quat_apply(self.base_yaw_quat, torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1))
            cube_dir = self._cube_root_states[:, :3] - self._robot_root_states[:, :3]
            cube_dir[:, 2] = 0
            cube_dir = cube_dir / torch.norm(cube_dir, dim=-1).unsqueeze(-1)
            # check if dot product is negative
            deviate_much = torch.sum(robot_head_dir * cube_dir, dim=-1) < 0.
            
            fov_camera_pos = self._robot_root_states[:, :3] + quat_apply(self._robot_root_states[:, 3:7], torch.tensor(self.cfg["sensor"]["onboard_camera"]["position"], device=self.device).repeat(self.num_envs, 1))
            too_close_table = (fov_camera_pos[:, 0] > -0.2)
            
            self.reset_buf = self.reset_buf | deviate_much | too_close_table
        
    def _draw_camera_sensors(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=(0.329, 0.831, 0.29))
        relative_camera_pos = to_torch(self.cfg["sensor"]["onboard_camera"]["position"], device=self.device)
        camera_pos = self._robot_root_states[:, :3] + quat_apply(self._robot_root_states[:, 3:7], relative_camera_pos)
        
        relative_wrist_camera_pos = to_torch(self.cfg["sensor"]["wrist_camera"]["position"], device=self.device)
        wrist_pos = self._rigid_body_pos[:, self.wrist_idx]
        wrist_rot = self._rigid_body_rot[:, self.wrist_idx]
        wrist_camera_pos = wrist_pos + quat_apply(wrist_rot, relative_wrist_camera_pos)
        
        for i in range(self.num_envs):
            heights = camera_pos[i].cpu().numpy()
            x = heights[0]
            y = heights[1]
            z = heights[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        for i in range(self.num_envs):
            wrist_heights = wrist_camera_pos[i].cpu().numpy()
            x = wrist_heights[0]
            y = wrist_heights[1]
            z = wrist_heights[2]
            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
    
    def _get_camera_obs(self):
        """ Retrieve the camera images from camera sensors and normalize both depth and rgb images;
        """
        depth_image = torch.stack(self.camera_sensor_dict["forward_depth"]).to(self.device)
        depth_image[depth_image < -2] = -2
        depth_image *= -1
        normalized_depth = (depth_image - 0.) / (2. - 0.)
        
        rgb_image = torch.stack(self.camera_sensor_dict["forward_color"]).to(self.device)
        if torch.max(rgb_image).item() > 1:
            rgb_image = rgb_image.float() / 255.0
        rgb_image = rgb_image.permute(0, 3, 1, 2)
        
        wirst_depth_image = torch.stack(self.camera_sensor_dict["wrist_depth"]).to(self.device)
        wirst_depth_image[wirst_depth_image < -2] = -2
        wirst_depth_image *= -1
        normalized_wrist_depth = (wirst_depth_image - 0.) / (2. - 0.)
        
        wrist_rgb_image = torch.stack(self.camera_sensor_dict["wrist_color"]).to(self.device)
        if torch.max(wrist_rgb_image).item() > 1:
            wrist_rgb_image = wrist_rgb_image.float() / 255.0
        wrist_rgb_image = wrist_rgb_image.permute(0, 3, 1, 2)
        
        normalized_depth = self.resize_transform(normalized_depth)
        rgb_image = self.resize_transform(rgb_image)
        normalized_wrist_depth = self.resize_transform(normalized_wrist_depth)
        wrist_rgb_image = self.resize_transform(wrist_rgb_image)
        
        seg_image = torch.stack(self.camera_sensor_dict["forward_seg"]).to(self.device)
        forward_mask = (seg_image == 2)
        x_coords = torch.arange(seg_image.size(2)).float().to(self.device)
        y_coords = torch.arange(seg_image.size(1)).float().to(self.device)
        x_grid, y_grid = torch.meshgrid(y_coords, x_coords)
        sum_mask = forward_mask.sum(dim=(1,2)).float()
        mean_x = (x_grid * forward_mask).sum(dim=(1,2)).float() / torch.where(sum_mask > 0, sum_mask, torch.ones_like(sum_mask))
        mean_y = (y_grid * forward_mask).sum(dim=(1,2)).float() / torch.where(sum_mask > 0, sum_mask, torch.ones_like(sum_mask))
        mean_x[sum_mask == 0] = -1
        mean_y[sum_mask == 0] = -1
        self.masked_forward[:, 0] = mean_x
        self.masked_forward[:, 1] = mean_y
        
        wrist_seg_image = torch.stack(self.camera_sensor_dict["wrist_seg"]).to(self.device)
        wrist_mask = (wrist_seg_image == 2)
        wrist_sum_mask = wrist_mask.sum(dim=(1,2)).float()
        wrist_mean_x = (x_grid * wrist_mask).sum(dim=(1,2)).float() / torch.where(wrist_sum_mask > 0, wrist_sum_mask, torch.ones_like(wrist_sum_mask))
        wrist_mean_y = (y_grid * wrist_mask).sum(dim=(1,2)).float() / torch.where(wrist_sum_mask > 0, wrist_sum_mask, torch.ones_like(wrist_sum_mask))
        self.masked_wrist[:, 0] = wrist_mean_x
        self.masked_wrist[:, 1] = wrist_mean_y
        wrist_mask_image = wrist_mask.float()
        forward_mask_image = forward_mask.float()
        wrist_mask_image = self.resize_transform(wrist_mask_image)
        forward_mask_image = self.resize_transform(forward_mask_image)
        
        return normalized_depth.flatten(start_dim=1), rgb_image.flatten(start_dim=1), normalized_wrist_depth.flatten(start_dim=1), \
            wrist_rgb_image.flatten(start_dim=1), forward_mask_image.flatten(start_dim=1), wrist_mask_image.flatten(start_dim=1)
    
    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
        
        self.reward_functions, self.reward_names = [], []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))
        
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.episode_metric_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                                    for name in list(self.reward_scales.keys()) + list(self.reward_scales.keys())}
        
    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew, metric = self.reward_functions[i]()
            rew *= self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metric_sums[name] += metric
        if self.cfg["reward"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf, min=0.0)
        
        if "termination" in self.reward_scales:
            rew, metric = self._reward_termination()
            rew *= self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.episode_metric_sums["termination"] += metric
        
    # def get_observations(self):
    #     return self.obs_buf
        
# --------------------------------- reward functions ---------------------------------
    def _reward_approaching(self):
        dist_delta = self.closest_dist - self.curr_dist
        self.closest_dist = torch.minimum(self.closest_dist, self.curr_dist)
        dist_delta = torch.clip(dist_delta, 0., 10.)
        reward = torch.tanh(10.0 * dist_delta) * ~self.lifted_object
        
        return reward, reward
    
    def _reward_lifting(self):
        height_delta = self.curr_height - self.highest_object
        self.highest_object = torch.maximum(self.highest_object, self.curr_height)
        height_delta = torch.clip(height_delta, 0., 10.)
        lifting_rew = torch.tanh(10.0 * height_delta)
        
        reward = torch.where(self.lifted_object, torch.zeros_like(lifting_rew), lifting_rew)
        
        return reward, reward
    
    def _reward_pick_up(self):
        table_height = self._table_root_states[:, 2] * 2.0
        cube_height = self._cube_root_states[:, 2]
        box_pos = self._cube_root_states[:, :3]
        d1 = torch.norm(box_pos - self.ee_pos, dim=-1)
        self.lifted_object = torch.logical_and((cube_height - table_height) > (0.03 / 2 + self.lifted_threshold), d1 < 0.05)

        reward = torch.where(self.lifted_object, torch.ones_like(self.reset_buf, dtype=torch.float), torch.zeros_like(self.reset_buf, dtype=torch.float))
        
        self.reset_buf = torch.where(self.lifted_object, torch.ones_like(self.reset_buf), self.reset_buf) # reset the picked envs
        return reward, reward
    
    def _reward_acc_penalty(self):
        arm_vel = self._dof_vel[:, -(6 + 2):-2]
        last_arm_vel = self._last_dof_vel[:, -(6 + 2):-2]
        penalty = torch.norm(arm_vel - last_arm_vel, dim=-1) / self.dt
        reward = 1 - torch.exp(-penalty)
        return reward, reward
    
    def _reward_command_penalty(self):
        penalty = torch.norm(self.commands[:, :3], dim=-1)
        reward = 1 - torch.exp(-10.*penalty)
        return reward, reward
    
    def _update_base_yaw_quat(self):
        base_yaw = euler_from_quat(self._robot_root_states[:, 3:7])[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        

# --------------------------------- reward functions ---------------------------------

# --------------------------------- jit functions ---------------------------------

@torch.jit.script
def compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos, 
                               body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, gripper_idx, table_dim):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor) -> Tensor
    cube_pos = cube_root_state[:, :3]
    cube_orn = cube_root_state[:, 3:7]
    
    ee_pos = body_pos[..., gripper_idx, :]
    ee_rot = body_rot[..., gripper_idx, :]
    ee_vel = body_vel[..., gripper_idx, :]
    ee_ang_vel = body_ang_vel[..., gripper_idx, :]
    # box pos and orientation  3+4=7
    # dof pos + vel  6+6=12
    # ee state  13
    arm_dof_pos = dof_pos[..., 12:-2]
    arm_dof_vel = dof_vel[..., 12:-2]
    
    cube_pos_local = quat_rotate_inverse(base_quat_yaw, cube_pos - spherical_center)
    cube_orn_local = quat_mul(quat_conjugate(base_quat_yaw), cube_orn)
    
    table_pos_local = quat_rotate_inverse(base_quat_yaw, table_root_state[:, :3] - spherical_center)
    table_orn_local = quat_mul(quat_conjugate(base_quat_yaw), table_root_state[:, 3:7])
    table_dim_local = quat_rotate_inverse(base_quat_yaw, table_dim)
    
    ee_pos_local = quat_rotate_inverse(base_quat_yaw, ee_pos - spherical_center)
    ee_rot_local = quat_mul(quat_conjugate(base_quat_yaw), ee_rot)
    
    robot_vel_local = quat_rotate_inverse(base_quat_yaw, robot_root_state[:, 7:10])
    
    # dim 3 + 4 + 3 + 4 + 12 + 3 + 3 = 32
    obs = torch.cat((cube_pos_local, cube_orn_local, ee_pos_local, ee_rot_local, arm_dof_pos, arm_dof_vel, 
                     commands, robot_vel_local), dim=-1)
    
    return obs
    
@torch.jit.script
def reindex_all(vec):
    # type: (Tensor) -> Tensor
    return torch.hstack((vec[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]], vec[:, 12:]))

@torch.jit.script
def reindex_feet(vec):
    return vec[:, [1, 0, 3, 2]]
