import numpy as np
import os
import torch
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict

from .vec_task import VecTask

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
from torch import Tensor
import torchvision.transforms as transforms


class B1Manip(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False):
        self.cfg = cfg
                        
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        
        # if sim_device == "cpu":
        #     self.sim_id = 0
        # else:
        #     self.sim_id = int(sim_device.split(":")[1])
        
        self.debug_vis = self.cfg["env"]["enableDebugVis"]
        
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
    
        self.reward_scales = self.cfg["reward"]["scales"]
        
        self.randomize = False
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
                
        self._prepare_reward_function()
        
        self.dt = self.control_freq_inv * self.sim_params.dt
        
        self._init_tensors()
        
        self.global_step_counter = 0
        
        if self.viewer is not None:
            self._init_camera()

 
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
        cube_size = 0.07
        # Let us add new assets
        obj_asset_root = self.cfg["env"]["objasset"]["objassetRoot"]
        obj_asset_file_robot = self.cfg["env"]["objasset"]["object"]
        
        obj_asset_options = gymapi.AssetOptions()
        obj_asset_options.replace_cylinder_with_capsule = True
        obj_asset_options.flip_visual_attachments = False
        obj_asset_options.fix_base_link = True
        obj_asset_options.disable_gravity = False
        obj_asset_options.use_mesh_materials = True
        
        cube_asset = self.gym.load_asset(self.sim, obj_asset_root, obj_asset_file_robot, obj_asset_options)
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

            
            self.robot_handles.append(robot_handle)
            self.table_handles.append(table_handle)
            self.cube_handles.append(cube_handle)
            self.envs.append(env_ptr)
            
 
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
