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

class GO1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # target angles when action = 0.0
        default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
        }

    class control( LeggedRobotCfg.control ):
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.6}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        control_type = 'P' #'P'  # P: position, V: velocity, T: torques
        decimation = 4
        hip_reduction = 0.8

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on =["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        fix_base_link = False
        disable_gravity = False
        # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        collapse_fixed_joints = True #! 
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
        max_push_vel_xy = 2.5
        # com displacement
        randomize_com_displacement = True
        com_displacement_range = [-0.15, 0.15]
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
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        class scales:
            # termination = -0.0
            # tracking_lin_vel = 1.0 #
            # tracking_ang_vel = 0.5
            # orthogonal_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # dof_acc = -2.5e-7
            # base_height = -1.0
            # dof_pos_limits = -10.0
            # orientation = -0.2
            # dof_vel_limits = -0.5
            # feet_air_time = 0.3
            # feet_clearance= -0.08
            # # feet_stumble = -0.5
            # feet_slip = -0.0
            # # stand_still = -1.0
            # power_distribution=-2e-5
            # action_rate = -0.01
            # # action_rate_smoothness = -0.01
            # action_smoothness_2=-0.01
            # action_smoothness_1=-0.01
            # # joint_power=-2e-5
            # torques = -1e-6
            # dof_vel = 0.0
            # collision = -1

            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -1e-7
            base_height = -2.0
            collision = -1
            action_rate = -0.01
            smoothness= -0.01
            dof_pos_limits = -5.0
            dof_vel_limits = -5.0
            torque_limits = -5.0

            joint_power=-2e-5
            default_pos = -1e-3
            power_distribution=-1e-5
        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.34 # x20: 1.0   go1: 0.4 
        desired_feet_height = -0.1 #0.1: x20 #0.07: go1
        max_contact_force = 400. # forces above this value are penalized
        clearance_height_target = -0.1

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
            

class GO1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_GO1'