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
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
from torch import nn
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

import pygame

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    # breakpoint()
    args.web = False
    args.use_joy = True
    args.nodelay = True
    args.save = False
    
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    # exptid = args.exptid
    # log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    log_pth = "/home/lenovo/Project/DreamWAQ_yiming_changes/legged_gym_58/logs/rough_x20/exported/policies"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.play.use_joy = args.use_joy
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 64 if not args.save else 64
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 20
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.07, 0.1]
    env_cfg.terrain.vertical_scale = 0.001
    env_cfg.terrain.terrain_dict = {"smooth slope": 0.0,
                                    "rough slope up": 0.1,
                                    "rough slope down": 0.1,
                                    "rough stairs up": 0.1,
                                    "rough stairs down": 0.1,
                                    "discrete": 0.1,
                                    "stepping stones": 0.0,
                                    "gaps": 0.,
                                    "smooth flat": 0.1,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.0,
                                    "large stairs up": 0.0,
                                    "large stairs down": 0.,
                                    "parkour": 0.,
                                    "parkour_hurdle": 0.,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.,
                                    "parkour_gap": 0.,
                                    "demo": 0.0,
                                    "rough stairs up 25cm": 0.1,
                                    "flat": 0.0}
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    # env_cfg.terrain.max_difficulty = False
    # env_cfg.terrain.

    # env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.asset.terminate_after_contacts_on = ['base']
    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args,
                                                                   train_cfg=train_cfg, return_log_dir=True)
    export_path = os.path.join(log_pth, 'export/')
    os.makedirs(export_path, exist_ok=True)
    import copy
    if env.cfg.depth.use_camera:
        model = copy.deepcopy(ppo_runner.alg.actor_critic_stud.actor).to('cpu')
        # adap_model = copy.deepcopy(ppo_runner.alg.actor_critic_stud.actor.adapt_module).to('cpu')
        # actor_model = copy.deepcopy(ppo_runner.alg.actor_critic_stud.actor.actor_backbone).to('cpu')
        # path = os.path.join(log_pth, 'export/model_0_jit_adap.pt')
        # traced_script_module = torch.jit.script(adap_model)
        # traced_script_module.save(path)
        # path = os.path.join(log_pth, 'export/model_0_jit_actor.pt')
        # traced_script_module = torch.jit.script(actor_model)
        # traced_script_module.save(path)
    else:
        model = copy.deepcopy(ppo_runner.alg.actor_critic.actor).to('cpu')
    traced_script_module = torch.jit.script(model)

    path = os.path.join(export_path, 'model_0_jit.pt')
    traced_script_module.save(path)

    logger = Logger(env.dt)

    if args.use_jit:
        # path = os.path.join(log_pth, "traced")
        # model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        # path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)

    # estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        # depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_student = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        ppo_runner.alg.actor_critic_stud = ppo_runner.alg.actor_critic_stud.cpu()
        state_dict = {
            'stud_model_state_dict': ppo_runner.alg.actor_critic_stud.state_dict(),
            # 'lstm': ppo_runner.alg.actor_critic_stud.actor.lstm,
            # 'rnn': ppo_runner.alg.actor_critic_stud.actor.rnn,
            'adapt_module': ppo_runner.alg.actor_critic_stud.actor.adapt_module,
            'actor': ppo_runner.alg.actor_critic_stud.actor.actor_backbone,
            # 'adapt_module': ppo_runner.alg.actor_critic_stud.actor.adapt_module.state_dict(),
            # 'actor': ppo_runner.alg.actor_critic_stud.actor.actor_backbone.state_dict(),
            'iter': 0,
            'infos': 'dummmy',
        }
        path = os.path.join(export_path, 'model_0_lstm.pt')
        torch.save(state_dict, path)
        loaded_state = torch.load(path)
        # lstm = loaded_state['lstm']
        # rnn = loaded_state['rnn']
        actor = loaded_state['actor']
        adap = loaded_state['adapt_module']
        # lstm = lstm.to(env.device)
        # rnn = rnn.to(env.device)
        actor = actor.to(env.device)
        adap = adap.to(env.device)
        # ht = torch.zeros(3, env.num_envs, 256, device=env.device, dtype=torch.float32, requires_grad=False)
        # ct = torch.zeros(3, env.num_envs, 256, device=env.device, dtype=torch.float32, requires_grad=False)
    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    start_state_log = False
    plot_state_log = False

    for i in range(200*int(env.max_episode_length)):
        axes, buttons = get_joystick_data()
        env.commands[env.lookat_id, 0] = -axes[1] * 1.2
        env.commands[env.lookat_id, 1] = -axes[0] * 0.5
        env.commands[env.lookat_id, 2] = -axes[3] * 0.5

        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    # depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    depth_latent = None
                    actions, depth_latent = policy_jit(obs.detach())
                else:
                    # depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    depth_latent = None
                    actions, depth_latent = policy_jit(obs.detach())
            else:
                # obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions, depth_latent = policy_jit(obs.detach())
        else:
            if env.cfg.depth.use_camera:
                # if infos["depth"] is not None:
                #     obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                #     obs_student[:, 6:8] = 0
                #     depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                #     depth_latent = depth_latent_and_yaw[:, :-2]
                #     yaw = depth_latent_and_yaw[:, -2:]
                # obs[:, 6:8] = 1.5*yaw
                depth_latent = None
            else:
                depth_latent = None

            # if hasattr(ppo_runner.alg, "depth_actor"):
            if env.cfg.depth.use_camera:
                # actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
                # actions, _ = policy_student(obs.detach())

                # lstm_in = obs[:, 3:3+45].reshape([1, env.num_envs, 45])
                # lstm_in = obs[:, -45 * 50:].reshape([1, env.num_envs, 45*50])
                # lstm_out, (ht, ct) = lstm(lstm_in, (ht, ct))
                # adap_latent = adap(lstm_out.squeeze())

                # rnn_in = obs[:, 3:3+45].reshape([1, env.num_envs, 45])
                # rnn_out, ht = rnn(rnn_in, ht.detach())
                # adap_latent = adap(rnn_out.squeeze())
                adap_latent = adap(obs[:, -45*5:])
                backbone_input = torch.cat((obs[:, 3:3+45], adap_latent), dim=-1)
                actions = actor(backbone_input)
            else:
                actions, adap_latent = policy(obs.detach())  # , hist_encoding=True, scandots_latent=depth_latent

        obs, _, rews, dones, infos = env.step(actions.detach())
        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
            print("time:", env.episode_length_buf[env.lookat_id].item() / 50,
                  "cmd v", env.commands[env.lookat_id, 0].item(), env.commands[env.lookat_id, 1].item(), env.commands[env.lookat_id, 2].item(),
                  "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )

        # print(buttons)

        if buttons[0] > 0 and start_state_log == False:
            start_state_log = True
        #
        if buttons[1] > 0 and plot_state_log == False:
            plot_state_log = True

        if start_state_log:
            logger.log_states(
                {
                    #'dof_pos_target': actions[env.lookat_id, 0].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 0].item(),
                    'dof_pos_target': actions[env.lookat_id, 0].item() + env.default_dof_pos[0, 0].item(),
                    'dof_pos': env.dof_pos[env.lookat_id, 0].item(),
                    'dof_vel': env.dof_vel[env.lookat_id, 0].item(),
                    'dof_torque': env.torques[env.lookat_id, 0].item(),
                    'command_x': env.commands[env.lookat_id, 0].item(),
                    'command_y': env.commands[env.lookat_id, 1].item(),
                    'command_yaw': env.commands[env.lookat_id, 2].item(),
                    'base_vel_x': env.base_lin_vel[env.lookat_id, 0].item(),
                    'base_vel_y': env.base_lin_vel[env.lookat_id, 1].item(),
                    'base_vel_z': env.base_lin_vel[env.lookat_id, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[env.lookat_id, 2].item(),
                    'contact_forces_z': env.contact_forces[env.lookat_id, env.feet_indices, 2].cpu().numpy(),
                    'obs_cmd_x': obs[env.lookat_id, 3].item() / env.cfg.normalization.obs_scales.lin_vel,
                    'obs_cmd_y': obs[env.lookat_id, 4].item() / env.cfg.normalization.obs_scales.lin_vel,
                    'obs_cmd_z': obs[env.lookat_id, 5].item() / env.cfg.normalization.obs_scales.ang_vel,
                    'obs_ang_vel_x': obs[env.lookat_id, 6].item(),
                    'obs_ang_vel_y': obs[env.lookat_id, 7].item(),
                    'obs_ang_vel_z': obs[env.lookat_id, 8].item(),
                    'obs_g_x': obs[env.lookat_id, 9].item(),
                    'obs_g_y': obs[env.lookat_id, 10].item(),
                    'obs_g_z': obs[env.lookat_id, 11].item(),

                    'obs_dof_0':  env.dof_pos[env.lookat_id, 0].item(),
                    'obs_dof_1':  env.dof_pos[env.lookat_id, 1].item(),
                    'obs_dof_2':  env.dof_pos[env.lookat_id, 2].item(),
                    'obs_dof_3':  env.dof_pos[env.lookat_id, 3].item(),
                    'obs_dof_4':  env.dof_pos[env.lookat_id, 4].item(),
                    'obs_dof_5':  env.dof_pos[env.lookat_id, 5].item(),
                    'obs_dof_6':  env.dof_pos[env.lookat_id, 6].item(),
                    'obs_dof_7':  env.dof_pos[env.lookat_id, 7].item(),
                    'obs_dof_8':  env.dof_pos[env.lookat_id, 8].item(),
                    'obs_dof_9':  env.dof_pos[env.lookat_id, 9].item(),
                    'obs_dof_10':  env.dof_pos[env.lookat_id, 10].item(),
                    'obs_dof_11':  env.dof_pos[env.lookat_id, 11].item(),

                    'dof_pos_target_0': actions[env.lookat_id, 0].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 0].item(),
                    'dof_pos_target_1': actions[env.lookat_id, 1].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 1].item(),
                    'dof_pos_target_2': actions[env.lookat_id, 2].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 2].item(),
                    'dof_pos_target_3': actions[env.lookat_id, 3].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 3].item(),
                    'dof_pos_target_4': actions[env.lookat_id, 4].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 4].item(),
                    'dof_pos_target_5': actions[env.lookat_id, 5].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 5].item(),
                    'dof_pos_target_6': actions[env.lookat_id, 6].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 6].item(),
                    'dof_pos_target_7': actions[env.lookat_id, 7].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 7].item(),
                    'dof_pos_target_8': actions[env.lookat_id, 8].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 8].item(),
                    'dof_pos_target_9': actions[env.lookat_id, 9].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 9].item(),
                    'dof_pos_target_10': actions[env.lookat_id, 10].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 10].item(),
                    'dof_pos_target_11': actions[env.lookat_id, 11].item() * env.cfg.control.action_scale + env.default_dof_pos[0, 11].item(),

                    'obs_dof_vel_0':  env.dof_vel[env.lookat_id, 0].item(),
                    'obs_dof_vel_1':  env.dof_vel[env.lookat_id, 1].item(),
                    'obs_dof_vel_2':  env.dof_vel[env.lookat_id, 2].item(),
                    'obs_dof_vel_3':  env.dof_vel[env.lookat_id, 3].item(),
                    'obs_dof_vel_4':  env.dof_vel[env.lookat_id, 4].item(),
                    'obs_dof_vel_5':  env.dof_vel[env.lookat_id, 5].item(),
                    'obs_dof_vel_6':  env.dof_vel[env.lookat_id, 6].item(),
                    'obs_dof_vel_7':  env.dof_vel[env.lookat_id, 7].item(),
                    'obs_dof_vel_8':  env.dof_vel[env.lookat_id, 8].item(),
                    'obs_dof_vel_9':  env.dof_vel[env.lookat_id, 9].item(),
                    'obs_dof_vel_10':  env.dof_vel[env.lookat_id, 10].item(),
                    'obs_dof_vel_11':  env.dof_vel[env.lookat_id, 11].item(),

                    'scan_latent_0': adap_latent[env.lookat_id,0].item(),
                    'scan_latent_1': adap_latent[env.lookat_id, 1].item(),
                    'scan_latent_2': adap_latent[env.lookat_id, 2].item(),
                    'scan_latent_3': adap_latent[env.lookat_id, 3].item(),
                    'scan_latent_4': adap_latent[env.lookat_id, 4].item(),
                    'scan_latent_5': adap_latent[env.lookat_id, 5].item(),
                    'scan_latent_6': adap_latent[env.lookat_id, 6].item(),
                    'scan_latent_7': adap_latent[env.lookat_id, 7].item(),
                    'scan_latent_8': adap_latent[env.lookat_id, 8].item(),
                    'scan_latent_9': adap_latent[env.lookat_id, 9].item(),
                    'scan_latent_10': adap_latent[env.lookat_id, 10].item(),
                    'scan_latent_11': adap_latent[env.lookat_id, 11].item(),
                    'scan_latent_12': adap_latent[env.lookat_id, 12].item(),
                    'scan_latent_13': adap_latent[env.lookat_id, 13].item(),
                    'scan_latent_14': adap_latent[env.lookat_id, 14].item(),
                    'scan_latent_15': adap_latent[env.lookat_id, 15].item(),
                    'scan_latent_16': adap_latent[env.lookat_id, 16].item(),
                    'scan_latent_17': adap_latent[env.lookat_id, 17].item(),
                    'scan_latent_18': adap_latent[env.lookat_id, 18].item(),
                    'scan_latent_19': adap_latent[env.lookat_id, 19].item(),
                    'scan_latent_20': adap_latent[env.lookat_id, 20].item(),
                    'scan_latent_21': adap_latent[env.lookat_id, 21].item(),
                    'scan_latent_22': adap_latent[env.lookat_id, 22].item(),
                    'scan_latent_23': adap_latent[env.lookat_id, 23].item(),
                    'scan_latent_24': adap_latent[env.lookat_id, 24].item(),
                    'scan_latent_25': adap_latent[env.lookat_id, 25].item(),
                    'scan_latent_26': adap_latent[env.lookat_id, 26].item(),
                    'scan_latent_27': adap_latent[env.lookat_id, 27].item(),
                    'scan_latent_28': adap_latent[env.lookat_id, 28].item(),
                    'scan_latent_29': adap_latent[env.lookat_id, 29].item(),
                    'scan_latent_30': adap_latent[env.lookat_id, 30].item(),
                    'scan_latent_31': adap_latent[env.lookat_id, 31].item(),

                    'priv_latent_0': adap_latent[env.lookat_id, 32 + 0].item(),
                    'priv_latent_1': adap_latent[env.lookat_id, 32 + 1].item(),
                    'priv_latent_2': adap_latent[env.lookat_id, 32 + 2].item(),
                    'priv_latent_3': adap_latent[env.lookat_id, 32 + 3].item(),
                    'priv_latent_4': adap_latent[env.lookat_id, 32 + 4].item(),
                    'priv_latent_5': adap_latent[env.lookat_id, 32 + 5].item(),
                    'priv_latent_6': adap_latent[env.lookat_id, 32 + 6].item(),
                    'priv_latent_7': adap_latent[env.lookat_id, 32 + 7].item(),
                    'priv_latent_8': adap_latent[env.lookat_id, 32 + 8].item(),
                    'priv_latent_9': adap_latent[env.lookat_id, 32 + 9].item(),
                    'priv_latent_10': adap_latent[env.lookat_id, 32 + 10].item(),
                    'priv_latent_11': adap_latent[env.lookat_id, 32 + 11].item(),
                    'priv_latent_12': adap_latent[env.lookat_id, 32 + 12].item(),
                    'priv_latent_13': adap_latent[env.lookat_id, 32 + 13].item(),
                    'priv_latent_14': adap_latent[env.lookat_id, 32 + 14].item(),
                    'priv_latent_15': adap_latent[env.lookat_id, 32 + 15].item(),
                }
            )
        if plot_state_log:
            logger.plot_states()
            plot_state_log = False
DEADZONE = 0.01  # Adjust this value as needed

def apply_deadzone(value):
    """Apply the deadzone to a joystick axis value."""
    if abs(value) < DEADZONE:
        return 0.0
    return value

def get_joystick_data():
    pygame.event.pump()
    axes = [apply_deadzone(joystick.get_axis(i)) for i in range(joystick.get_numaxes())]
    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
    return axes, buttons

def print_joystick_data(axes, buttons):
    print(f"Axes: {axes}")
    print(f"Buttons: {buttons}")
    print("-------")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # Initialize Pygame for Joystick
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    args = get_args()
    play(args)

