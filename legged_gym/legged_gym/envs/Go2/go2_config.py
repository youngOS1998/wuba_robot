# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ):
    class env:
        num_envs = 4096
        num_observations = 49
        num_privileged_obs = 256 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        # add
        num_observation_history = 10
        priv_observe_contact_forces = True
        priv_observe_base_lin_vel = True
        priv_observe_base_height = True
        priv_observe_foot_clearance = True

        observe_gait_commands = True
        frequencies = 1.5
        offsets = 0.5  # 2
        bounds = 0.5  # 3
        phases = 0.0  # 4
        durations = 0.5

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 100 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # terrain_proportions = [0.1, 0.0, 0.1, 0.1, 0.1, 0, 0, 0, 0.6]
        #terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75

    class commands:
        curriculum = True
        max_curriculum = 1.7
        num_commands = 4 + 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error), base_height, jump_signal
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        pacing_offset = False
        height_command = True # if true: compute height command from base height
        camera_follow = False

        class ranges:
            lin_vel_x = [-1.2, 1.2] # min max [m/s]
            lin_vel_y = [-0.7, 0.7]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]
            height = [0.1, 0.3]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 0.8

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on =["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # flip_visual_attachments = False
        fix_base_link = False
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        # collapse_fixed_joints = True #! 
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        # replace collision cylinders with capsules, leads to faster/more stable simulation
        replace_cylinder_with_capsule = True
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        # friction
        rand_interval_s = 4
        randomize_friction = True
        friction_range = [0.1, 3.0]
        # base mass
        randomize_base_mass = True
        added_mass_range = [-2.0, 2.0]
        # push robot
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.5
        # com displacement
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        # motor strength
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        # Kp
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        # Kd
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]
        # gravity
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99
        # restitution
        randomize_restitution = True
        restitution_range = [0, 0.4]
        # motor offset
        randomize_motor_offset = True
        motor_offset_range = [-0.04, 0.04]
        # lag timesteps
        randomize_lag_timesteps = True
        lag_timesteps = 2
        # observation lag buffer
        randomize_obs_lag_timesteps = False
        obs_lag_timesteps = 2

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]

    class normalization:
        contact_force_range = [0.0, 100.0]#385 
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 1.0
            action = 1
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            action = 0.04
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.6
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -0.2
            dof_acc = -2.5e-7
            base_height = -1.0
            # base_height_command = 1.0
            # base_height_enhance = 1.0
            collision = -1
            feet_stumble = -0.2
            action_rate = -0.01
            torques_rate = -1.0e-7
            dof_pos_limits = -5.0
            dof_vel_limits = -5.0

            joint_power = -2e-5
            default_pos = -1e-1
            power_distribution = -1e-7
            smoothness= -0.02
            torques = -1.0e-7
            feet_slip = -0.01
            feet_impact_vel = -0.2
            low_speed = 0.2

            contacts_shaped_force = 0.01
            contacts_shaped_vel = 0.01
            feet_clearance_cmd_linear = -1.0

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30 # x20: 1.0   go1: 0.4 
        desired_feet_height = 0.10 #0.1: x20 #0.07: go1
        max_contact_force = 400. # forces above this value are penalized
        clearance_height_target = -0.30

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'