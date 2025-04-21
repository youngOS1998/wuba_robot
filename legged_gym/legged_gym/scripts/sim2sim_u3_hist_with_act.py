import os
from copy import deepcopy
import mujoco as mj
import numpy as np
from mujoco_base import MuJoCoBase
from joystickX import JoystickController
import mujoco_viewer
from tqdm import tqdm
from mujoco.glfw import glfw
from datetime import datetime
import threading
import time
import copy
import pygame
import queue
from motion_example import MotionExample
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import math
import torch
import logging

def load_policy(logdir):
    actor = torch.load(logdir + '/base_actor.pt')
    encoder = torch.load(logdir + '/waq_encoder.pt')
    fc_mu = torch.load(logdir + '/waq_encoder_mu.pt')
    fc_var = torch.load(logdir + '/waq_encoder_var.pt')
    fc_vel = torch.load(logdir + '/waq_encoder_vel.pt')
    # body_height = torch.load(logdir + '/waq_body_height.pt')
    # feet_height = torch.load(logdir + '/waq_feet_height.pt')
    
    def policy(obs_history, obs, lin_vel=None):
        h = encoder(obs_history)
        mu = fc_mu(h)
        vel = fc_vel(h)
        log_var = fc_var(h)
        if lin_vel is not None:
            _vel = lin_vel
        else:
            _vel = vel        
        
        # body_height_est = body_height(h)
        # feet_height_est = feet_height(h)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent = mu + eps * std
        
        # action = actor(torch.cat([_vel, body_height_est, feet_height_est, latent, obs], dim=-1))
        action = actor(torch.cat([_vel, latent, obs], dim=-1))
        return action, latent, vel

    
    return policy 

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.float32)
    dq = data.qvel.astype(np.float32)
    quat = data.sensor('body_quat').data[[1, 2, 3, 0]].astype(np.float32)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.float32)  # In the base frame
    omega = data.sensor('body_gyro').data.astype(np.float32)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32)
    return (q, dq, quat, v, omega, gvec, r)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def runmj(policy):
    mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x20/xml/scenex20.xml'
    model = mj.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = 0.001
    data = mj.MjData(model)
    mj.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros(12, dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)

    hist_obs = deque()
    for _ in range(10):
        hist_obs.append(np.zeros([1, 49], dtype=np.float32))

    defalut_joint_pos = np.array([
                    0.0, -0.8, 1.5, 
                    0.0, -0.8, 1.5, 
                    0.0, -0.8, 1.5, 
                    0.0, -0.8, 1.5], dtype=np.float32)

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05

    # policy 
    # kpP = np.array([100, 100, 150, 100, 100, 150, 100, 100, 150, 100, 100, 150], dtype=np.float32)
    kpP = 1.0 * np.array([150] * 12, dtype=np.float32)
    # kpP = np.array([80, 80, 100, 80, 80, 100, 80, 80, 100, 80, 80, 100], dtype=np.float32)
    # kdP = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=np.float32)
    kdP = 1.0 * np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32)

    count_lowlevel = 0
    joy = JoystickController()

    sim_ts_list = []
    jq_list = []
    jdq_list = []
    jtor_list = []
    taucmd_list = []
    target_q_list = []
    obs_list = []
    action_list = []
    latent_list = []
    # hist_obs_list = []
    vae_vel_list = []
    lin_vel_list = []
    
    
    data_index = 0
    # run_id = datetime.now().strftime('%b%d_%H-%M-%S')
    # data_save_path = os.path.join(os.getcwd(), str(run_id))
    # if not os.path.exists(data_save_path):
    #     os.makedirs(data_save_path, exist_ok=True)
    wake_flag = False
    gait_indices = torch.zeros((1, 1))
    clock_inputs = torch.zeros(1, 4, dtype=torch.float)
    dt = 0.02
    frequencies = 1.5
    offsets = 0.5
    bounds = 0.5
    phases = 0.0
    start_time = 0
    first_forward = True


    policy_0 = time.time()
    for _ in range(int(600 / 0.001)):
        step_start_time = time.time()
        
        start_time += 1
        joy.update_cmd()
        # Obtain an observation
        q, dq, quat, v, omega, gvec, r = get_obs(data)
        q = q[-12:]
        dq = dq[-12:]
        
        lin_vel = dq[:3]
        ang_vel = dq[3:6]
        base_lin_vel = r.apply(lin_vel, inverse=True).astype(np.float32)
        base_ang_vel = r.apply(ang_vel, inverse=True).astype(np.float32)
        
        # print(joy.cmd)

        # 1000hz -> 100hz
        

        if count_lowlevel % 20 == 0:
            count_lowlevel = 0
            
        # if policy_time<=0.02 + 0.0001  and policy_time>=0.02 - 0.0008:

            gait_indices = torch.remainder(gait_indices + dt * frequencies, 1.0)
            foot_indices = [gait_indices,
                            gait_indices + offsets,
                            gait_indices + bounds,
                            gait_indices + phases]
            
            foot_indices_tmp = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
            clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0]).reshape(1)  #
            clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1]).reshape(1)  #
            clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2]).reshape(1)  #
            clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3]).reshape(1)
            vel = np.sqrt(joy.cmd['vx']**2 + joy.cmd['vy']**2)
            if vel < 0.2 and (abs(joy.cmd['dyaw']) < 0.2):
                clock_inputs = torch.zeros(1, 4, dtype=torch.float)
            
            clock_inputs_in = clock_inputs.squeeze(0)
            
            
            obs = np.zeros([1, 49], dtype=np.float32)

            # obs[0, 0] = -joy.cmd['vx'] * obs_scales.lin_vel
            # obs[0, 1] = -joy.cmd['vy'] * obs_scales.lin_vel
            # obs[0, 2] = -joy.cmd['dyaw'] * obs_scales.ang_vel
            # obs[0, 3:6] = omega * obs_scales.ang_vel
            # obs[0, 6:9] = gvec
            # obs[0, 9:21] = (q - defalut_joint_pos) * obs_scales.dof_pos
            # obs[0, 21:33] = dq * obs_scales.dof_vel
            # # !CAREFUL! ACTION IS SCALED OR CLIPED?
            # obs[0, 33:45] = action
            # obs = np.clip(obs, -40, 40)
            
            # obs[0, 0:3] = omega * 0.25
            # obs[0, 3:6] = gvec * 1
            
            # # if wake_flag:
            # #     joy.cmd['vx'] = 1.0
            # #     joy.cmd['vy'] = 0.0
            # #     joy.cmd['dyaw'] = 0.0
            # print(joy.cmd)
            
            # obs[0, 6] = joy.cmd['vx'] * 2.0
            # obs[0, 7] = joy.cmd['vy'] * 2.0
            # obs[0, 8] = joy.cmd['dyaw'] * 0.25
            # obs[0, 9:21] = (q - defalut_joint_pos)
            # obs[0, 21:33] = dq * 0.05
            # # !CAREFUL! ACTION IS SCALED OR CLIPED?
            # # action = np.zeros(12, dtype=np.float32)
            # # action[[0, 3, 6, 9]] = 0.0
            # obs[0, 33:45] = action

            obs[0, 0:3] = omega * 0.25
            obs[0, 3:7] = clock_inputs_in
            
            if wake_flag:
                joy.cmd['vx'] = 1.0
                joy.cmd['vy'] = 0.0
                joy.cmd['dyaw'] = 0.0
            # print(joy.cmd)
            cmd = 0.7
            obs[0, 7:10] = gvec
            obs[0, 10] = joy.cmd['vx'] * 2.0
            obs[0, 11] = joy.cmd['vy'] * 2.0
            obs[0, 12] = joy.cmd['dyaw'] * 0.25
            obs[0, 13:25] = (q - defalut_joint_pos)
            obs[0, 25:37] = dq * 0.05
            # !CAREFUL! ACTION IS SCALED OR CLIPED?
            # action = np.zeros(12, dtype=np.float32)
            # action[[0, 3, 6, 9]] = 0.0
            obs[0, 37:49] = action

            
            # Update the history of observations
            hist_obs.popleft()#!
            # hist_obs.append(obs)
            hist_obs.append(obs[:]) 
            # Flatten the history of observations
            hist_obs_np = np.concatenate(list(hist_obs), axis=0)
            hist_obs_np_flat = hist_obs_np.reshape(-1)
            hist_obs_tensor = torch.tensor(hist_obs_np_flat, dtype=torch.float32)
            obs_np = obs
            obs_np_flat = obs_np.reshape(-1)
            obs_tensor = torch.tensor(obs_np_flat, dtype=torch.float32)
            # Save the current observation and action
            obs_list.append(obs_np_flat.copy())
            # hist_obs_list.append(hist_obs_np_flat.copy())
            
            action, latent, vel = policy(hist_obs_tensor, obs_tensor, lin_vel=None)
            action, latent, vel_est_np = action.detach().numpy(), latent.detach().numpy(), vel.detach().numpy()

            obs_string = np.array2string(hist_obs_np, separator=',', max_line_width=np.inf)
            action_string = np.array2string(action, separator=',', max_line_width=np.inf)
            vel_est_string = np.array2string(vel_est_np, separator=',', max_line_width=np.inf)
            real_lin_vel_string = np.array2string(v, separator=',', max_line_width=np.inf)
            # logger.info('obs: array({0}, dtype=float32)'.format(obs_string))
            logger.info('action: array([{0}], dtype=float32)'.format(action_string))
            logger.info('vel_est: array([{0}], dtype=float32)'.format(vel_est_string))
            logger.info('lin_vel_real: array([{0}], dtype=float32)'.format(real_lin_vel_string))
            
            actions_cliped = np.clip(action, -100, 100)
            actions_scaled = actions_cliped * 0.25
            target_q =  actions_scaled + defalut_joint_pos
            target_dq = np.zeros((12), dtype=np.float32)
            policy_1 = time.time()
            # print('Policy (HZ): ', 1 / (policy_1 - policy_0))
            policy_0 = time.time()

        # target_dq = np.zeros((12), dtype=np.float32)
        # Generate PD control
        tau = pd_control(target_q, q, kpP, target_dq, dq, kdP)  # Calc torques

        data.ctrl = tau
        mj.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1
        step_elapsed_time = time.time() - step_start_time
        # step_sleep_time = max(0, 0.001 - step_elapsed_time)
        # time.sleep(step_sleep_time)
        # print('Control (HZ): ', 1 / (time.time() - step_start_time))
        # print('time: ', step_elapsed_time)


    viewer.close()

if __name__ == "__main__":

    policy = load_policy('/home/rl/Project/legged_gym_baseline/legged_gym/logs/rough_x20/exported/policies')

    if policy is None:
        print("Policy is None. Check the load_policy function.")
    else:
        print("Policy loaded successfully.")

    log_name = '/home/rl/Project/legged_gym_baseline/legged_gym/legged_gym/scripts/sim_mujoco_x20.log'
    if os.path.exists(log_name):
        os.remove(log_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_name, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    runmj(policy)

