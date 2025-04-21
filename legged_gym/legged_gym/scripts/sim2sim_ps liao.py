import mujoco as mj
import numpy as np
from mujoco_base import MuJoCoBase
from joystickX import JoystickController

from mujoco.glfw import glfw
import threading
import time
import copy
import pygame
import queue
from motion_example import MotionExample
from collections import deque
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from collections import defaultdict
from multiprocessing import Process, Value
import logging

from legged_gym import LEGGED_GYM_ROOT_DIR

import torch

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
    
    def log_state(self, key, value):
        self.state_log[key].append(value)
    
    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)
    
    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()
    
    def _plot(self):
        nb_rows = 3
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value)*self.dt, len(value))
            break
        log = self.state_log
        a = axs[0, 0]
        if log["ang_vel_x"]: a.plot(time, log["ang_vel_x"], label='measured')
        a.set(xlabel='time [s]', ylabel='Velocity [m/s]', title='Robot Angular Velocity X')
        a.legend()
        # if log["ang_vel_y"]: a.plot(time, log["ang_vel_x"], label='measured')
        # a.set(xlabel='time [s]', ylabel='Velocity [m/s]', title='Robot Angular Velocity Y')
        # a.legend()
        # if log["ang_vel_z"]: a.plot(time, log["ang_vel_z"], label='measured')
        # a.set(xlabel='time [s]', ylabel='Velocity [m/s]', title='Robot Angular Velocity Z')
        # a.legend()
        plt.show()

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()


def quat_to_euler(quat):

    # Extract the values from the quaternion
    x, y, z, w = quat

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.pi / 2 * np.sign(sinp)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def load_policy(logdir):
    actor = torch.jit.load(logdir + '/policy.pt')
    
    def policy(obs):
        action = actor(obs)
        return action

    return policy 

class Go1Sim(MuJoCoBase):
  def __init__(self, xml_path, shared_state):
    super().__init__(xml_path)
    self.simend = 1000.0
    self.shared_state = shared_state
    self.ini = 0
    self.i = 0
    self.calibrated = False
    self.robot_set_up_demo = MotionExample()
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
    # self.model.vis.scale.contactwidth = 0.01
    # self.model.vis.scale.contactheight = 0.01
    # self.model.vis.scale.forcewidth = 0.01
    # self.model.vis.map.force = 0.005
    self.model.vis.map.force = 0
    self.model.opt.disableactuator = 3
    mj.mj_resetDataKeyframe(self.model, self.data, 0)
    mj.mj_forward(self.model, self.data)
    

  def pd_control(self, target_q, current_q, kp, target_dq, current_dq, kd):
    '''Calculates torques from position commands
    '''
    # cmd_torque = (target_q - current_q) * kp + (target_dq - current_dq) * kd
    cmd_torque = (target_q - current_q) * kp - current_dq * kd
    # print(cmd_torque)
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
        dq = dq_full[-12:]
        quat = self.data.sensor('body_quat').data[[1, 2, 3, 0]].astype(np.float32)
        r = R.from_quat(quat)
        v = r.apply(self.data.qvel[:3], inverse=True).astype(np.float32)  # In the base frame
        omega = self.data.sensor('body_gyro').data.astype(np.float32)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32)
        is_registered = ctrl_enabled
        calibrated = self.calibrated
        obs_raw = {'q': q, 'dq': dq, 'quat': quat, 'v': v, 'omega': omega, 'gvec': gvec, 'is_registered': is_registered, 'calibrated': calibrated}                             
        self.shared_state.update_obs_raw(obs_raw)
            # print(self.i)
        self.i += 1
        # print(self.i)
        # print(motor_mode)    
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
        elif motor_mode == 2: #torque
            self.model.opt.disableactuator = 2
            # print("Torque control is enabled.")
            tau_cmd = self.pd_control(target_q, q, kp, target_dq, dq, kd)
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
    for _ in range(6): #! hist length
        hist_obs.append(np.zeros([1, 45], dtype=np.float32))
    
    # FL,FR,RL,RR
    defalut_joint_pos = np.array([
                    -0.1, -0.8, 1.5, 
                    -0.1, -0.8, 1.5, 
                    0.1, -0.8, 1.5, 
                    0.1, -0.8, 1.5], dtype=np.float32)
    
    # defalut_joint_pos = np.array([-0.1, -0.66, 1.27, -0.1, -0.66, 1.27, 0.1, -0.66, 1.27, 0.1, -0.66, 1.27], dtype=np.float32)
    # Damping control (zero ctrl torque in sim )
    kpD = np.zeros(12, dtype=np.float32)
    kdD = np.zeros(12, dtype=np.float32)
    # kpS kdS are not applied.  Motionexample in SDK
    kpS = np.array([40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40], dtype=np.float32)
    kdS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    final_goal = np.zeros(12)
    
    # policy 
    kpP = np.array([100, 100, 150, 100, 100, 150, 100, 100, 150, 100, 100, 150], dtype=np.float32)
    # kpP = np.array([80, 80, 100, 80, 80, 100, 80, 80, 100, 80, 80, 100], dtype=np.float32)
    kdP = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=np.float32)
    # kdP = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
    
  
 
    # loop_period = cfg.sim_config.dt * cfg.sim_config.decimation
    loop_period = 0.02 #! robotcontrol runs at 500hz which is different from issacgym (dt is 0.005)
    
    running = False
    calibrated = False
    
    print("Entering the main control loop.")
    
    # # define the log
    # logger = Logger(1)
    # stop_state_log = 300
    # i = 0

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename='/home/rl/Project/DreamWAQ_yiming_changes/legged_gym_58/legged_gym/scripts/sim.log', mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    while True:

        start_time = time.time()
        obs_raw = shared_state.get_obs_raw()
        # Get the latest proprioceptive observation
        q = obs_raw['q']
        dq = obs_raw['dq']
        # print(dq)
        quat = obs_raw['quat']
        r = R.from_quat(quat)
        v = obs_raw['v']
        omega = obs_raw['omega']
        gvec = obs_raw['gvec']

        # clip last action
        action = np.clip(action, -100, 100)#!clip
        # print(action)
        
        # Check if the low level controller is registered
        running = obs_raw['is_registered']
        
        calibrated = obs_raw['calibrated']

        
        # if i < stop_state_log:
        #     logger.log_states(
        #         {
        #             'ang_vel_x': omega[0],
        #             'ang_vel_y': omega[1],
        #             'ang_vel_z': omega[2]
        #         }
        #     )
    
        # elif i==stop_state_log:
        #     logger.plot_states()

        # Get the latest command
        try:
            cmd = cmd_queue.get_nowait()  
            # print("Command received.") 
        except queue.Empty:
            pass
            # cmd = {'vx': 0.0, 'vy': 0.0, 'dyaw': 0.0, 'mode': 0}
    
        # Compose latest observation
        # print(omega)
        # obs[0, 0:3] = omega * 0.25 / 50 #!cfg.normalization.obs_scales.lin_vel
        # obs[0, 0] = omega[0] * 0.25
        # obs[0, 1] = omega[1] * 0.25
        # obs[0, 2] = omega[2] * 0.25
        # # obs[0, 0:3] = omega * 0.25
        # obs[0, 3:4] = gvec[0]
        # obs[0, 4:5] = gvec[1]
        # obs[0, 5:6] = gvec[2]
        # # obs[0, 3:6] = gvec * 1
        # import random
        # noise = random.random() 
        # obs[0, 6] = cmd['vx'] * 2.0 #!cfg.normalization.obs_scales.lin_vel
        # obs[0, 7] = cmd['vy'] * 2.0 #!cfg.normalization.obs_scales.lin_vel
        # obs[0, 8] = cmd['dyaw'] * 0.25#!cfg.normalization.obs_scales.ang_vel
        # obs[0, 9:21] = (q - defalut_joint_pos) #!cfg.normalization.obs_scales.dof_pos
        # obs[0, 21:33] = dq * 0.05 #!cfg.normalization.obs_scales.dof_vel
        # #!CAREFUL! ACTION IS SCALED OR CLIPED?
        # obs[0, 33:45] = action

        obs[0, 0] = cmd['vx'] * 2.0 
        obs[0, 1] = cmd['vy'] * 2.0
        obs[0, 2] = cmd['dyaw'] * 0.25
        # obs[0, 0:3] = omega * 0.25
        obs[0, 3:4] = omega[0] * 0.25
        obs[0, 4:5] = omega[1] * 0.25
        obs[0, 5:6] = omega[2] * 0.25
        # obs[0, 3:6] = gvec * 1
        import random
        noise = random.random() 
        obs[0, 6] = gvec[0] #!cfg.normalization.obs_scales.lin_vel
        obs[0, 7] = gvec[1] #!cfg.normalization.obs_scales.lin_vel
        obs[0, 8] = gvec[2]#!cfg.normalization.obs_scales.ang_vel
        obs[0, 9:21] = (q - defalut_joint_pos) #!cfg.normalization.obs_scales.dof_pos
        obs[0, 21:33] = dq * 0.05 #!cfg.normalization.obs_scales.dof_vel
        #!CAREFUL! ACTION IS SCALED OR CLIPED?
        obs[0, 33:45] = action

        # Update the history of observations
        hist_obs.popleft()
        hist_obs.append(obs)        
        
        # Flatten the history of observations
        hist_obs_np = np.concatenate(list(hist_obs), axis=0)
        hist_obs_np_flat = hist_obs_np.reshape(-1)
        hist_obs_tensor = torch.tensor(hist_obs_np_flat, dtype=torch.float32) 
        obs_np_flat = obs.reshape(-1)
        obs_tensor = torch.tensor(obs_np_flat, dtype=torch.float32)
       

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
            elif cmd['mode'] == 2: # Walk mode
                if calibrated:   
                    action[:] = policy(hist_obs_tensor).detach().numpy()
                    obs_string = np.array2string(obs, separator=',', max_line_width=np.inf)
                    action_string = np.array2string(action, separator=',', max_line_width=np.inf)
                    logger.info('Observation: array({0}, dtype=float32)'.format(obs_string))
                    logger.info('action: array({0}, dtype=float32)'.format(action_string))
                    actions_cliped = np.clip(action, -100, 100)
                    actions_scaled= actions_cliped * 0.25
                    joint_pos_target =  actions_scaled + defalut_joint_pos
                    # joint_pos_target =  defalut_joint_pos
                    joint_vel_target = np.zeros((12), dtype=np.float32)
                    new_motor_cmd = {'kp': kpP, 'kd': kdP, 'joint_pos_target': joint_pos_target, 'joint_vel_target': joint_vel_target, 'motor_mode': 2}
                    shared_state.update_motor_cmd(new_motor_cmd)                  
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
            print("low level controller is not registered.")
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
        return {'q': q, 'dq': dq, 'quat': quat, 'v': v, 'omega': omega, 'gvec': gvec, 'is_registered': False, 'calibrated': False}

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

    policy = load_policy('/home/rl/Project/legged_gym_initial/legged_gym/logs/policies_HIM')

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


