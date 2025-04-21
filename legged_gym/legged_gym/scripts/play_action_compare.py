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

def load_policy(logdir):
    actor = torch.load(logdir + '/base_actor.pt').to("cuda")
    encoder = torch.load(logdir + '/waq_encoder.pt').to("cuda")
    fc_mu = torch.load(logdir + '/waq_encoder_mu.pt').to("cuda")
    fc_var = torch.load(logdir + '/waq_encoder_var.pt').to("cuda")
    fc_vel = torch.load(logdir + '/waq_encoder_vel.pt').to("cuda")
    
    def policy(obs_history, obs, lin_vel=None):
        h = encoder(obs_history)
        mu = fc_mu(h)
        vel = fc_vel(h)
        # log_var = fc_var(h)
        
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        latent = mu# + eps * std
        
        action = actor(torch.cat([vel, latent, obs], dim=-1))
        return action, vel
    
    return policy 

def play(args, x_vel=2.0, y_vel=0.0, yaw_vel=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    class_to_dict(env_cfg)
    class_to_dict(train_cfg)
    
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(class_to_dict(env_cfg), f)
    with open('train_cfg.pkl', 'wb') as f:
        pickle.dump(train_cfg, f)
    # override some parameters for testing
    env_cfg.terrain.border_size = 10
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.center_robots = True
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    env.commands[:, 0] = 1.0
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    observe = env.get_observations()
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit_actor(ppo_runner.alg.actor_critic, path)
    export_policy_as_jit_encoder(ppo_runner.alg.actor_critic, path)
    # export_policy_as_jit_whole_model(ppo_runner.alg.actor_critic, path)
    print('Exported policy as jit script to: ', path)
    
    print(ppo_runner.alg.actor_critic.net_l2_norm(ppo_runner.alg.actor_critic.encoder_module))
    print(ppo_runner.alg.actor_critic.net_l2_norm(ppo_runner.alg.actor_critic.cv_vel))
    
    model_path = "/home/lenovo/Project/DreamWAQ_yiming_changes/legged_gym_58/logs/rough_x20/exported/policies"

    policy_save = load_policy(model_path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 200 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.5, 0, 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    obs_list = []
    hist_obs_list = []
    action_list = []
    vae_vel_list = []
    lin_vel_list = []
    for i in range(50*int(env.max_episode_length)):
        # breakpoint()
        actions, vel = policy(observe['obs'], observe['obs_history'])
        action_save, vel_save = policy_save(observe['obs_history'], observe['obs'])
        print((actions - action_save).abs().sum().item())
        # breakpoint()
        # print(f"vae_vel: {vel}")
        # env_vel = env.base_lin_vel# * env.obs_scales.lin_vel
        # print(f"vel error: {torch.pow((vel - env_vel), 2).mean()}")
        # print(f"true vel: {env.base_lin_vel}")
        # print(f"vae_vel: {vel.abs().mean()}")
        
        obs_list.append(observe['obs'].cpu().detach().numpy().squeeze())
        hist_obs_list.append(observe['obs_history'].cpu().detach().numpy().squeeze())
        action_list.append(actions.cpu().detach().numpy().squeeze())
        vae_vel_list.append(vel.cpu().detach().numpy().squeeze())
        lin_vel_list.append(env.base_lin_vel.cpu().detach().numpy().squeeze())

        # if len(action_list) >= 100:
        #     filename = 'play_data'
        #     filename = f"{filename}_rl.npz" 
        #     np.savez(filename, vae_vel=np.array(vae_vel_list) , obs=np.array(obs_list), hist_obs=np.array(hist_obs_list), 
        #              action=np.array(action_list), lin_vel=np.array(lin_vel_list))
        #     print(f"{filename}")
        #     break
        
        ck = 1
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        observe, rewards, dones, infos = env.step(actions.detach(), ck)
        # obs_tensor = act_dict['obs']
        # privileged_obs = act_dict['privileged_obs']
        # obs_history = act_dict['obs_history']

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

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=1.5, y_vel=0.0, yaw_vel=0.0)

