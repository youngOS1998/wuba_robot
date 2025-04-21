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

class X20RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.521] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': -0.1,   # [rad]
            'RL_hip_joint': -0.1,   # [rad]
            'FR_hip_joint': 0.1,  # [rad]
            'RR_hip_joint': 0.1,   # [rad]

            'FL_thigh_joint': -0.8,     # [rad]
            'RL_thigh_joint': -0.8,   # [rad]
            'FR_thigh_joint': -0.8,     # [rad]
            'RR_thigh_joint': -0.8,   # [rad]

            'FL_calf_joint': 1.5,   # [rad]
            'RL_calf_joint': 1.5,    # [rad]
            'FR_calf_joint': 1.5,  # [rad]
            'RR_calf_joint': 1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 150.}  # [N*m/rad]
        # damping = {'joint': 3.}     # [N*m*s/rad]
        stiffness = { 'hip_joint': 150.0, 'thigh_joint': 150., 'calf_joint': 150.}  # [N*m/rad]
        damping = { 'hip_joint': 3.0, 'thigh_joint': 3.0, 'calf_joint': 3.0}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x20/urdf/x20_v2.urdf'
        name = "x20"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        # terminate_after_contacts_on = []
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        collapse_fixed_joints = True

    class domain_rand:
        # friction
        rand_interval_s = 8
        randomize_friction = True
        friction_range = [0.1, 1.5]
        # base mass
        randomize_base_mass = True
        added_mass_range = [-5.0, 15.0]
        # push robot
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.5
        # com displacement
        randomize_com_displacement = True
        com_displacement_range = [-0.10, 0.10]
        # motor strength
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        # Kp
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        # Kd
        randomize_Kd_factor = True
        Kd_factor_range = [0.3, 2.0]
        # gravity
        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99
        # restitution
        randomize_restitution = True
        restitution_range = [0, 0.3]
        # motor offset
        randomize_motor_offset = True
        motor_offset_range = [-0.01, 0.01]
        # lag timesteps
        randomize_lag_timesteps = True
        lag_timesteps = 1
        # observation lag buffer
        # randomize_obs_lag_timesteps = False
        # obs_lag_timesteps = 2

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]

# my reward left
    class rewards(LeggedRobotCfg.rewards):
        class scales:

            # normal walking
            termination = -0.0
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.6
            lin_vel_z = -2.0
            ang_vel_xy = -0.1
            orientation = -0.2
            dof_acc = -2.5e-7
            base_height = -1.0
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

            contacts_shaped_force = 0.01
            contacts_shaped_vel = 0.01
            feet_clearance_cmd_linear = -1.0

            # recovery
            # termination = -0.0
            # ang_vel_y = -0.1
            # orientation = -0.2
            # dof_acc = -2.5e-7
            # base_height = -3.0
            # action_rate = -0.01
            # torques_rate = -1.0e-7
            # dof_pos_limits = -5.0
            # dof_vel_limits = -5.0
            # torque_limits = -5.0

            # joint_power = -2e-5
            # default_pos = -1e-1
            # smoothness= -0.01
            # torques = -1.0e-7
            # recovery = 0.3
            # feet_contact = 0.2

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.49 # x20: 1.0   go1: 0.4 
        desired_feet_height = 0.20 #0.1: x20 #0.07: go1
        max_contact_force = 400. # forces above this value are penalized
        clearance_height_target = -0.30

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
        noise_level = 1.5 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            action = 0.0

class X20RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_x20'

