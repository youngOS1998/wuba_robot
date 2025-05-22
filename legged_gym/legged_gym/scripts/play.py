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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
import sys
print(sys.path)
from legged_gym.envs import *
from legged_gym.utils.helpers import  get_args, export_policy_as_jit_actor,export_policy_as_jit_encoder,export_policy_as_jit_whole_model,class_to_dict
from legged_gym.utils.logger import Logger
from legged_gym.utils.task_registry import task_registry
from legged_gym.envs.wrappers.history_wrapper import HistoryWrapper
import numpy as np
import torch
import pickle
import logging
import time

def play(args, x_vel=2.0, y_vel=0.0, yaw_vel=0.0):

    # logging 
    log_name = '/home/byang/Project_byang/wuba_robot/legged_gym/legged_gym/scripts/sim_logs/sim_isaac.log'
    if os.path.exists(log_name):
        os.remove(log_name)

    logger_play = logging.getLogger(__name__)
    logger_play.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_name, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger_play.addHandler(file_handler)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    class_to_dict(env_cfg)
    class_to_dict(train_cfg)
    
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(class_to_dict(env_cfg), f)
    with open('train_cfg.pkl', 'wb') as f:
        pickle.dump(train_cfg, f)
    # override some parameters for testing
    env_cfg.terrain.border_size = 0
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.center_robots = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.test = True
    env_cfg.terrain.selected = False
    env_cfg.terrain.terrain_kwargs = \
    {'type':"room_terrain",
        'step_width':0.3,
        'step_height':0.3} # None # Dict of arguments for selected terrain
    
    env_cfg.x_command = 1.0
    env_cfg.y_command = 0.0
    env_cfg.yaw_command = 0.0
    env_cfg.height_command = 0.3
    
    env_cfg.play_mode = True
    
    # env_cfg.sim_params = "cuda:0"

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    observe = env.get_observations()
    policy = ppo_runner.get_inference_policy(device=env.device)

    print(ppo_runner.alg.actor_critic.net_l2_norm(ppo_runner.alg.actor_critic.encoder_module))
    print(ppo_runner.alg.actor_critic.net_l2_norm(ppo_runner.alg.actor_critic.cv_vel))
    # print(xxx)

    # export policy as a jit module (used to run it from C++)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit_actor(ppo_runner.alg.actor_critic, path)
    export_policy_as_jit_encoder(ppo_runner.alg.actor_critic, path)
    # export_policy_as_jit_whole_model(ppo_runner.alg.actor_critic, path)
    print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.5, 0, 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    for i in range(50*int(env.max_episode_length)):
        # breakpoint()
        start_time = time.time()
        actions, vel = policy(observe['obs'], observe['obs_history'])
        # stance_time = env.stance_time.detach().cpu().numpy()
        # air_time = env.air_time.detach().cpu().numpy()
        # stance_time_max = env.stance_time_max.detach().cpu().numpy()
        # air_time_max = env.air_time_max.detach().cpu().numpy()
        # stance_time_string = np.array2string(stance_time, separator=',', max_line_width=np.inf)
        # air_time_string = np.array2string(air_time, separator=',', max_line_width=np.inf)
        # stance_time_max_string = np.array2string(stance_time_max, separator=',', max_line_width=np.inf)
        # air_time_max_string = np.array2string(air_time_max, separator=',', max_line_width=np.inf)
        # logger_play.info('stance: array({0}, dtype=float32)'.format(stance_time_string))
        # logger_play.info('air: array({0}, dtype=float32)'.format(air_time_string))
        # logger_play.info('stance_max: array({0}, dtype=float32)'.format(stance_time_max_string))
        # logger_play.info('air_max: array({0}, dtype=float32)'.format(air_time_max_string))
        env_vel = env.base_lin_vel# * env.obs_scales.lin_vel

        # desire_state = env.desired_contact_states.detach().cpu().numpy()
        # desire_state_string = np.array2string(desire_state, separator=',', max_line_width=np.inf)
        # logger_play.info('desire_state: array({0}, dtype=float32)'.format(desire_state_string))
        # foot_forces = torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1).detach().cpu().numpy()
        # foot_forces_string = np.array2string(foot_forces, separator=',', max_line_width=np.inf)
        # reward = np.zeros((1, 4))
        # for i in range(4):
        #     reward[:, i] = - (1 - desire_state[:, i]) * (1 - np.exp(-1 * foot_forces[:, i] ** 2 / 50.0))
        # reward_state_string = np.array2string(reward, separator=',', max_line_width=np.inf)
        # logger_play.info('reward_state: array({0}, dtype=float32)'.format(reward_state_string))
        # logger_play.info('foot_forces: array({0}, dtype=float32)'.format(foot_forces_string))

        # foot_velocities = torch.norm(env.foot_velocities, dim=2).view(env.num_envs, -1).detach().cpu().numpy()
        # reward_vel = np.zeros((1, 4))
        # for i in range(4):
        #     reward_vel[:, i] = - (desire_state[:, i] * (
        #             1 - np.exp(-1 * foot_velocities[:, i] ** 2 / 1.25)))
        # foot_vel_string = np.array2string(foot_velocities, separator=',', max_line_width=np.inf)
        # reward_vel_string = np.array2string(reward_vel, separator=',', max_line_width=np.inf)
        # logger_play.info('reward_vel_state: array({0}, dtype=float32)'.format(reward_vel_string))
        # logger_play.info('foot_vel: array({0}, dtype=float32)'.format(foot_vel_string))

        ck = 1
        observe, rewards, dones, infos = env.step(actions.detach(), ck)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        stop_time = time.time()
        # print('policy(HZ): ', 1 / (stop_time - start_time))

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=1.5, y_vel=0.0, yaw_vel=0.0)

