import sys
import os
# #current folder path
# current_folder_path = os.path.abspath(__file__)
# print(current_folder_path)
# #parent folder path
# parent_folder_path = os.path.dirname(current_folder_path)
# #lib folder path
# lib_folder_path = "../lib/python/"
# pyblib = os.path.join(parent_folder_path, lib_folder_path)
# if os.path.exists(pyblib):
#     sys.path.insert(0, pyblib)
# else:
#     raise ImportError("Specified library path does not exist")
# model_folder_path = "../model_pt"
# model_path = os.path.join(parent_folder_path, model_folder_path)
from copy import deepcopy
from datetime import datetime
import mujoco as mj
import numpy as np
from mujoco_base import MuJoCoBase
# from joystickYSC import JoystickController
from joystickX import JoystickController


from mujoco.glfw import glfw
import threading
import time
import queue
from motion_example import MotionExample
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR

import torch

def load_policy(logdir):
    actor = torch.load(logdir + '/base_actor.pt')
    encoder = torch.load(logdir + '/waq_encoder.pt')
    fc_mu = torch.load(logdir + '/waq_encoder_mu.pt')
    fc_var = torch.load(logdir + '/waq_encoder_var.pt')
    fc_vel = torch.load(logdir + '/waq_encoder_vel.pt')
    
    def policy(obs_history, obs, lin_vel=None):
        h = encoder(obs_history)
        mu = fc_mu(h)
        vel = fc_vel(h)
        # log_var = fc_var(h)
        if lin_vel is not None:
            _vel = lin_vel
        else:
            _vel = vel        
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        latent = mu# + eps * std
        
        action = actor(torch.cat([_vel, latent, obs], dim=-1))
        return action, latent, vel
    
    # def policy(obs_history, obs):
    #     h = encoder(obs_history)
    #     mu = fc_mu(h)
    #     # vel = fc_vel(h)
    #     # log_var = fc_var(h)
    #     # std = torch.exp(0.5 * log_var)
    #     # eps = torch.randn_like(std)
    #     latent = mu# + eps * std
    #     action = actor(torch.cat((latent, obs), dim=-1))
    #     return action, latent
    
    return policy 

def get_formatted_timestamp():
    return f"{time.time():.3f} s"

# Function to find the next available file index
def get_next_file_index(base_filename, extension="npz"):
    index = 1
    while os.path.exists(f"{base_filename}_{index}.{extension}"):
        index += 1
    return index

class Go1Sim(MuJoCoBase):
  def __init__(self, xml_path, shared_state):
    super().__init__(xml_path)
    self.simend = 1000.0
    self.shared_state = shared_state
    self.ini = 0
    self.i = 0
    self.calibrated = False
    self.robot_set_up_demo = MotionExample()
    
    self.sim_ts_list = []
    self.jq_list = []
    self.jdq_list = []
    self.taucmd_list = []
    self.target_q_list = []
    
    self.run_id = datetime.now().strftime('%b%d_%H-%M-%S')
    self.data_save_path = os.path.join(os.getcwd(), str(self.run_id))
    if not os.path.exists(self.data_save_path):
        os.makedirs(self.data_save_path, exist_ok=True)
        
    self.mode_1_flag = False
    self.mode_2_flag = False
    self.data_index = 0
    
    mj.mj_forward(self.model, self.data)
    # enable contact force visualization
    self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        self.window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    # Update scene and render    
    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
    
    mj.mjr_render(viewport, self.scene, self.context)
    
  def reset(self):
    # Set camera configuration
    self.cam.azimuth = 89.608063
    self.cam.elevation = -11.588379
    self.cam.distance = 5.0
    self.cam.lookat = np.array([0.0, 0.0, 1.5])
    self.model.vis.scale.contactwidth = 0.1
    self.model.vis.scale.contactheight = 0.03
    self.model.vis.scale.forcewidth = 0.1
    self.model.vis.map.force = 0.01
    self.model.opt.disableactuator = 3
    mj.mj_resetDataKeyframe(self.model, self.data, 0)
    mj.mj_forward(self.model, self.data)
    

  def pd_control(self, target_q, current_q, kp, target_dq, current_dq, kd):
    '''Calculates torques from position commands
    '''
    # cmd_torque = (target_q - current_q) * kp + (target_dq - current_dq) * kd
    cmd_torque = (target_q - current_q) * kp - current_dq * kd
    cmd_torque_cliped = np.clip(cmd_torque, -100., 100)
    return cmd_torque_cliped


  def contact_force_callback(self, foot_id):
    cfrc = np.zeros(6)
    force_torque = np.zeros(6)
    force_global = np.zeros(3)

    for i in range(self.data.ncon):
        contact = self.data.contact[i]
        body1 = self.model.geom_bodyid[contact.geom1]
        body2 = self.model.geom_bodyid[contact.geom2]


        if body1 == foot_id or body2 == foot_id:
            mj.mj_contactForce(self.model, self.data, i, force_torque) 
            mj.mju_rotVecMatT(force_global, force_torque[:3], contact.frame)

            for j in range(3):
                cfrc[j] += -force_global[j] if body1 == foot_id else force_global[j]
      
    return cfrc[2]

  def simulate(self):
    ctrl_enabled = False
    print("Simulation started.")
    
    while not glfw.window_should_close(self.window):
      simstart = self.data.time
      ctrl_enabled = not self.pause_flag  
      while (self.data.time - simstart <= 1.0/60.0 and not self.pause_flag):
        step_start_time = time.time()
        # Step simulation environment
        mj.mj_step(self.model, self.data)

        if self.i % 10 == 0:
            motor_cmd_received = self.shared_state.get_motor_cmd()
            target_q = motor_cmd_received['joint_pos_target']
            target_dq = motor_cmd_received['joint_vel_target']
            kp = motor_cmd_received['kp']
            kd = motor_cmd_received['kd']
            motor_mode = motor_cmd_received['motor_mode']
            
        q_full = self.data.qpos.astype(np.float32)
        q = q_full[-12:]
        dq_full = self.data.qvel.astype(np.float32)
        # breakpoint()
        lin_vel = dq_full[:3]
        ang_vel = dq_full[3:6]
        dq = dq_full[-12:]
        quat = self.data.sensor('body_quat').data[[1, 2, 3, 0]].astype(np.float32)
        r = R.from_quat(quat)
        v = r.apply(self.data.qvel[:3], inverse=True).astype(np.float32)  # In the base frame
        omega = self.data.sensor('body_gyro').data.astype(np.float32)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32)
        is_registered = ctrl_enabled
        calibrated = self.calibrated
        # breakpoint()
        base_lin_vel = r.apply(lin_vel, inverse=True).astype(np.float32)
        base_ang_vel = r.apply(ang_vel, inverse=True).astype(np.float32)
        obs_raw = {'q': q, 'dq': dq, 'quat': quat, 'v': v, 'omega': omega, 'gvec': gvec, 'is_registered': is_registered, 'calibrated': calibrated,
                   'lin_vel': base_lin_vel, 'ang_vel': omega}  #! 'ang_vel': omega???
        self.shared_state.update_obs_raw(obs_raw)
            # print(self.i)
        self.i += 1
        # print(self.i)
        # print(motor_mode)
        self.mode_1_flag = False
        self.mode_2_flag = False 
        if motor_mode == 1: #position
            if not self.calibrated:

                self.model.opt.disableactuator = 0
                now_time = self.data.time    
                nominal_stand_pos = np.array([0, -0.7330, 1.3614, 0, -0.7330, 1.3614, 0, -0.7330, 1.3614, 0, -0.7330, 1.3614], dtype=np.float32) #!stand pos
                # nominal_stand_pos = np.array([0, -0.7330, 1.4414, 0, -0.7330, 1.4414, 0, -0.7230, 1.5814, 0, -0.7230, 1.5814], dtype=np.float32) #!stand pos
                if np.max(np.abs(q - nominal_stand_pos)) < 0.085:
                    self.calibrated =True
                    print("Calibration done.")
                    target_q = nominal_stand_pos
                    self.ini = 0 
                else:
                    if self.ini == 0:
                        self.robot_set_up_demo.GetInitData(q, now_time)
                        self.ini = 1
                    target_q = self.robot_set_up_demo.StandUp(now_time)
                    # target_q = q
                    # self.ini = 1
                             
            else:
                if np.max(np.abs(q - nominal_stand_pos)) >= 0.085:
                    self.calibrated = False
                    print("To recalibrate the robot...")
                    self.ini = 0
                    target_q = q
                else:
                    #print("Calibrated. Stand pos is reached")
                    self.model.opt.disableactuator = 0
                    target_q = nominal_stand_pos
                    self.ini = 0

               
            tau_cmd = np.zeros(12, dtype=np.float32)
            self.mode_1_flag = True
        elif motor_mode == 2: #torque
            self.model.opt.disableactuator = 2
            # print("Torque control is enabled.")
            tau_cmd = self.pd_control(target_q, q, kp, target_dq, dq, kd)
            sim_ts = get_formatted_timestamp()
            self.sim_ts_list.append(sim_ts)
            self.jq_list.append(q)
            self.jdq_list.append(dq)
            self.taucmd_list.append(tau_cmd)
            self.target_q_list.append(target_q)
            self.mode_2_flag = True
            # print("Torque control is enabled. ctrl torque is ", tau_cmd)        
        else:# Damping 
            self.model.opt.disableactuator = 3
            self.calibrated = False
            self.ini = 0
            target_q = q
            tau_cmd = np.zeros(12, dtype=np.float32) 

        if ctrl_enabled:
            ctrl_full = np.concatenate((tau_cmd, target_q))
            self.data.ctrl = ctrl_full
        else:
            self.data.ctrl = np.zeros(24, dtype=np.float32)

        if len(self.target_q_list) >= 5000:  
            sim_filename = 'u1_sim_data_sim'
            # sim_save_index = get_next_file_index(sim_filename)
            if self.mode_1_flag:
                sim_filename = f"{self.run_id}/{sim_filename}_carli_{self.data_index}.npz"   
            elif self.mode_2_flag:
                self.data_index += 1
                sim_filename = f"{self.run_id}/{sim_filename}_rl_{self.data_index}.npz"
            else:
                sim_filename = f"{self.run_id}/done.npz"   
            # np.savez(sim_filename, sim_ts=np.array(self.sim_ts_list) , jq=np.array(self.jq_list), jdq=np.array(self.jdq_list), taucmd=np.array(self.taucmd_list), target_q=np.array(self.target_q_list))
            self.sim_ts_list.clear()
            self.jq_list.clear()
            self.jdq_list.clear()
            self.taucmd_list.clear()
            self.target_q_list.clear()
            # print(f"{sim_filename}")
        
            
        # Calculate elapsed time for the control loop execution
        step_elapsed_time = time.time() - step_start_time
        # print("Elapsed time: ", step_elapsed_time)
        # Calculate remaining time to sleep to maintain the desired loop frequency
        step_sleep_time = max(0, 0.001 - step_elapsed_time)
                
        # Sleep for the remaining time
        time.sleep(step_sleep_time)  
 
 
      if self.data.time >= self.simend:
          break
      # get framebuffer viewport
      viewport_width, viewport_height = glfw.get_framebuffer_size(
          self.window)
      
      viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

      # Update scene and render
      mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                          mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        
      mj.mjr_render(viewport, self.scene, self.context)
      

      # swap OpenGL buffers (blocking call due to v-sync)
      glfw.swap_buffers(self.window)

      # process pending GUI events, call GLFW callbacks
      glfw.poll_events()

    glfw.terminate()

def rlcontroller(policy, cmd_queue, shared_state):

    obs = np.zeros([1, 45], dtype=np.float32)
    action = np.zeros((12), dtype=np.float32)
    actions_scaled = np.zeros((12), dtype=np.float32)
    actions_cliped = np.zeros((12), dtype=np.float32)
    
    hist_obs = deque()
    for _ in range(5): #! hist length
        hist_obs.append(np.zeros([1, 45], dtype=np.float32))
    
    # FL,FR,RL,RR
    defalut_joint_pos = np.array([
                    0.0, -0.8, 1.5, 
                    0.0, -0.8, 1.5, 
                    0.0, -0.8, 1.5, 
                    0.0, -0.8, 1.5], dtype=np.float32)
    
    # defalut_joint_pos = np.array([-0.1, -0.66, 1.27, -0.1, -0.66, 1.27, 0.1, -0.66, 1.27, 0.1, -0.66, 1.27], dtype=np.float32)
    # Damping control (zero ctrl torque in sim )
    kpD = np.zeros(12, dtype=np.float32)
    kdD = np.zeros(12, dtype=np.float32)
    # kpS kdS are not applied.  Motionexample in SDK
    kpS = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40], dtype=np.float32)
    kdS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    final_goal = np.zeros(12)
    
    # policy 
    # kpP = np.array([100, 100, 150, 100, 100, 150, 100, 100, 150, 100, 100, 150], dtype=np.float32)
    kpP = 1.0 * np.array([170]*12, dtype=np.float32)
    # kpP = np.array([80, 80, 100, 80, 80, 100, 80, 80, 100, 80, 80, 100], dtype=np.float32)
    # kdP = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=np.float32)
    kdP = 1.0 * np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32)


 
    # loop_period = cfg.sim_config.dt * cfg.sim_config.decimation
    loop_period = 0.02 #! robotcontrol runs at 500hz which is different from issacgym (dt is 0.005)
    
    running = False
    calibrated = False
    
    # time.sleep(3.0)
    
    # Initialize lists to store obs and actions
    policy_ts_list = []
    obs_list = []
    latent_list = []
    action_list = []
    hist_obs_list = []
    vae_vel_list = []
    lin_vel_list = []
    
    data_index = 0
    enter_rl_flag = False
    
    run_id = datetime.now().strftime('%b%d_%H-%M-%S')
    data_save_path = os.path.join(os.getcwd(), str(run_id))
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path, exist_ok=True)
    print("Entering the main control loop.")
    while True:
        start_time = time.time()
        obs_raw = shared_state.get_obs_raw()
        # Get the latest proprioceptive observation
        q = obs_raw['q']
        dq = obs_raw['dq']
        quat = obs_raw['quat']
        r = R.from_quat(quat)
        v = obs_raw['v']
        omega = obs_raw['omega']
        gvec = obs_raw['gvec']
        
        lin_vel = obs_raw['lin_vel']
        ang_vel = obs_raw['ang_vel']
    
        
        # clip last action
        # print('action: ')
        # print(action)
        action = np.clip(action, -100, 100)#!clip
        
        # Check if the low level controller is registered
        running = obs_raw['is_registered']
        
        calibrated = obs_raw['calibrated']
    
        # Get the latest command
        try:
            cmd = cmd_queue.get_nowait()  
            # print("Command received.") 
        except queue.Empty:
            pass
            # cmd = {'vx': 0.0, 'vy': 0.0, 'dyaw': 0.0, 'mode': 0}
    
        # Compose latest observation
        obs[0, 0:3] = omega * 0.25 #!cfg.normalization.obs_scales.lin_vel
        obs[0, 3:6] = gvec * 1
        
        cmd['vx'] = 1.0
        cmd['vy'] = 0.0
        cmd['dyaw'] = 0.0
        
        obs[0, 6] = cmd['vx'] * 2.0 #!cfg.normalization.obs_scales.lin_vel
        obs[0, 7] = cmd['vy'] * 1.0 #!cfg.normalization.obs_scales.lin_vel
        obs[0, 8] = cmd['dyaw'] * 1.0#!cfg.normalization.obs_scales.ang_vel
        obs[0, 9:21] = (q - defalut_joint_pos) #!cfg.normalization.obs_scales.dof_pos
        obs[0, 21:33] = dq * 0.05 #!cfg.normalization.obs_scales.dof_vel
        #!CAREFUL! ACTION IS SCALED OR CLIPED?
        obs[0, 33:45] = action
        # print('obs: ')
        # print(obs)
        
        mode_1_flag = False
        mode_2_flag = False
        
        if running:
            if cmd['mode'] == 1: # Calibration mode
                # print("Excecuting the standup demo to calibrate the robot using Postion Control")
                action = np.zeros((12), dtype=np.float32)
                actions_scaled = np.zeros((12), dtype=np.float32)
                actions_cliped = np.zeros((12), dtype=np.float32)
                joint_pos_target = np.zeros((12), dtype=np.float32)
                joint_vel_target = np.zeros((12), dtype=np.float32)
                new_motor_cmd = {'kp': kpS, 'kd': kdS, 'joint_pos_target': joint_pos_target, 'joint_vel_target': joint_vel_target, 'motor_mode': 1}
                shared_state.update_motor_cmd(new_motor_cmd)
                mode_1_flag = True                                       
            elif cmd['mode'] == 2: # Walk mode
                if calibrated:
                    if not enter_rl_flag:
                        policy_ts_list.clear()
                        obs_list.clear()
                        hist_obs_list.clear()
                        action_list.clear()
                        latent_list.clear()
                        vae_vel_list.clear()
                        lin_vel_list.clear()
                        enter_rl_flag = True
                    timestamp1 = get_formatted_timestamp()
                    policy_ts_list.append(timestamp1)
                    # Update the history of observations
                    hist_obs.popleft()
                    hist_obs.append(obs) 
                    
                    # Flatten the history of observations
                    hist_obs_np = np.concatenate(list(hist_obs), axis=0)
                    hist_obs_np_flat = hist_obs_np.reshape(-1)
                    hist_obs_tensor = torch.tensor(hist_obs_np_flat, dtype=torch.float32) 
                    obs_np_flat = obs.reshape(-1)
                    obs_tensor = torch.tensor(obs_np_flat, dtype=torch.float32)

                    oracle_vel = torch.tensor(lin_vel, dtype=torch.float32)
                    # breakpoint()
                    _action, _latent, _vel = policy(hist_obs_tensor, obs_tensor, lin_vel=oracle_vel)
                    action[:], latent, vel = _action.detach().numpy(), _latent.detach().numpy(), _vel.detach().numpy()
                    # print(f"vel error: {np.power((vel - lin_vel), 2).mean()}")
                    # print(f"vae_vel: {np.abs(vel).mean()}")
                    # breakpoint()
                    # print(action)
                    # breakpoint()
                    # action = np.zeros_like(action) # ! default
                    # ac_reshape = action.reshape(-1).squeeze()  
                    obs_list.append(obs_np_flat.copy())
                    hist_obs_list.append(hist_obs_np_flat.copy())
                    action_list.append(deepcopy(action))
                    latent_list.append(deepcopy(latent))
                    vae_vel_list.append(deepcopy(vel))
                    lin_vel_list.append(deepcopy(lin_vel))
                    
                    # print(len(action_list))
                    
                    actions_cliped = np.clip(action, -100, 100)
                    actions_scaled= actions_cliped * 0.25
                    joint_pos_target =  actions_scaled + defalut_joint_pos
                    # joint_pos_target =  defalut_joint_pos
                    joint_vel_target = np.zeros((12), dtype=np.float32)
                    new_motor_cmd = {'kp': kpP, 'kd': kdP, 'joint_pos_target': joint_pos_target, 'joint_vel_target': joint_vel_target, 'motor_mode': 2}
                    shared_state.update_motor_cmd(new_motor_cmd)
                    mode_2_flag = True                  
                else:
                    # print("Press B to calibrate the controller first.")
                    action = np.zeros((12), dtype=np.float32)
                    actions_scaled = np.zeros((12), dtype=np.float32)
                    actions_cliped = np.zeros((12), dtype=np.float32)
                    joint_pos_target = np.zeros((12), dtype=np.float32)
                    joint_vel_target = np.zeros((12), dtype=np.float32)
                    new_motor_cmd = {'kp': kpD, 'kd': kdD, 'joint_pos_target': joint_pos_target, 'joint_vel_target': joint_vel_target, 'motor_mode': 0}
                    shared_state.update_motor_cmd(new_motor_cmd)
            else: # Standby mode
                # print("Mode not recognized. Motor commands are disabled. Press B to calibrate the controller first.")
                action = np.zeros((12), dtype=np.float32)
                actions_scaled = np.zeros((12), dtype=np.float32)
                actions_cliped = np.zeros((12), dtype=np.float32)
                joint_pos_target = np.zeros((12), dtype=np.float32)
                joint_vel_target = np.zeros((12), dtype=np.float32)
                new_motor_cmd= {'kp': kpD, 'kd': kdD, 'joint_pos_target': joint_pos_target, 'joint_vel_target': joint_vel_target, 'motor_mode': 0}
                shared_state.update_motor_cmd(new_motor_cmd)                  
        else:
            # print("low level controller is not registered.")
            action = np.zeros((12), dtype=np.float32)
            actions_scaled = np.zeros((12), dtype=np.float32)
            actions_cliped = np.zeros((12), dtype=np.float32)
            joint_pos_target = np.zeros((12), dtype=np.float32)
            joint_vel_target = np.zeros((12), dtype=np.float32)
            new_motor_cmd = {'kp': kpD, 'kd': kdD, 'joint_pos_target': joint_pos_target, 'joint_vel_target': joint_vel_target, 'motor_mode': 0}
            shared_state.update_motor_cmd(new_motor_cmd)                  
            calibrated = False  
            
        # Calculate elapsed time for the control loop execution
        elapsed_time = time.time() - start_time
        # Calculate remaining time to sleep to maintain the desired loop frequency
        sleep_time = max(0, loop_period - elapsed_time)
        
        # Save the collected data periodically
        if len(action_list) >= 200:
            sim_filename = 'u1_sim_data'
            # if mode_1_flag:
            #     sim_filename = f"{run_id}/{sim_filename}_carli_{data_index}.npz"   
            # elif mode_2_flag:
            #     data_index += 1
            #     sim_filename = f"{run_id}/{sim_filename}_rl_{data_index}.npz"
            # else:
            #     sim_filename = f"{run_id}/done.npz"   
            sim_filename = f"{sim_filename}_rl_{data_index}.npz" 
            np.savez(sim_filename, vae_vel=np.array(vae_vel_list) , obs=np.array(obs_list), hist_obs=np.array(hist_obs_list), 
                     action=np.array(action_list), latent=np.array(latent_list), lin_vel=np.array(lin_vel_list))
            policy_ts_list.clear()
            obs_list.clear()
            hist_obs_list.clear()
            action_list.clear()
            latent_list.clear()
            vae_vel_list.clear()
            lin_vel_list.clear()
            print(f"{sim_filename}")
            break
        # Sleep for the remaining time
        time.sleep(sleep_time)         



class SharedState:
    def __init__(self):
        # Initialize queues for obs_raw and motor_cmd with a max size to prevent memory overflow
        # If the queue is full, the oldest items will be discarded
        self.obs_raw_queue = queue.Queue(maxsize=10)
        self.motor_cmd_queue = queue.Queue(maxsize=10)
        self.last_motor_cmd = self._init_motor_cmd()  # Initialize with a default command
        # Pre-fill queues with initial data to ensure they're never empty
        for _ in range(4):
            self.obs_raw_queue.put(self._init_obs_raw())
            self.motor_cmd_queue.put(self._init_motor_cmd())

    def _init_obs_raw(self):
        """Helper method to initialize an observation dictionary."""
        q = np.zeros(12, dtype=np.float32)
        dq = np.zeros(12, dtype=np.float32)
        quat = np.array([0., 0., 0., 1.], dtype=np.float32)
        v = np.zeros(3, dtype=np.float32)
        omega = np.zeros(3, dtype=np.float32)
        gvec = np.array([0., 0., -9.81], dtype=np.float32)
        lin_vel = np.array([0., 0., 0.], dtype=np.float32)
        ang_vel = np.array([0., 0., 0.], dtype=np.float32)
        return {'q': q, 'dq': dq, 'quat': quat, 'v': v, 'omega': omega, 'gvec': gvec, 'is_registered': False, 'calibrated': False,
                'lin_vel': lin_vel, 'ang_vel': omega}

    def _init_motor_cmd(self):
        """Helper method to initialize a motor command dictionary."""
        kp = np.zeros(12, dtype=np.float32)
        kd = np.zeros(12, dtype=np.float32)
        joint_pos_target = np.zeros(12, dtype=np.float32)
        joint_vel_target = np.zeros(12, dtype=np.float32)
        motor_mode = 0
        return {'kp': kp, 'kd': kd, 'joint_pos_target': joint_pos_target, 'joint_vel_target': joint_vel_target, 'motor_mode': motor_mode}

    def update_obs_raw(self, new_obs_raw):
        if self.obs_raw_queue.full():
            try:
                self.obs_raw_queue.get_nowait()  
            except queue.Empty:
                pass  
        self.obs_raw_queue.put(new_obs_raw.copy())

    def get_obs_raw(self):
        try:
            return self.obs_raw_queue.get_nowait()  
        except queue.Empty:
            return self._init_obs_raw()  

    def update_motor_cmd(self, new_motor_cmd):
        # get rid of the oldest command if the queue is full
        if self.motor_cmd_queue.full():
            try:
                self.motor_cmd_queue.get_nowait()  
            except queue.Empty:
                pass
        self.motor_cmd_queue.put(new_motor_cmd.copy())

    def get_motor_cmd(self):
        try:
            # Attempt to get the latest command without waiting
            self.last_motor_cmd = self.motor_cmd_queue.get_nowait()
        except queue.Empty:
            # If the queue is empty, return the last fetched command
            pass
        return self.last_motor_cmd

    
if __name__ == "__main__":
    
    model_path = "/home/lenovo/Project/DreamWAQ_yiming_changes/legged_gym_58/logs/rough_x20/exported/policies"

    policy = load_policy(model_path)

    if policy is None:
        print("Policy is None. Check the load_policy function.")
    else:
        print("Policy loaded successfully.")
        
    shared_state = SharedState()
    joystick_controller = JoystickController()    
    
    mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x20/xml/scenex20.xml'
    
    sim = Go1Sim(mujoco_model_path, shared_state)
    sim.reset()


    joy_thread = threading.Thread(target=joystick_controller.run)
    policy_thread = threading.Thread(target=rlcontroller, args=(policy, joystick_controller.cmd_queue, shared_state))
    
    joy_thread.start()
    policy_thread.start()

    sim.simulate()


