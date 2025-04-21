from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import re
import matplotlib.pyplot as plt

def parse_logs(file_path):
    # Initialize a list to store the NumPy arrays

    with open(file_path, 'r') as file:
        file_content = file.read()  # Read the entire file content into a string

    desire_state = np.zeros((1, 4), dtype=np.float32)
    reward_state = np.zeros((1, 4), dtype=np.float32)
    foot_forces = np.zeros((1, 4), dtype=np.float32)
    reward_vel_state = np.zeros((1, 4), dtype=np.float32)
    foot_vel = np.zeros((1, 4), dtype=np.float32)

    info_list = ['desire_state', 'reward_state', 'foot_forces', 'reward_vel_state', 'foot_vel']


    res = {'desire_state': desire_state, 'reward_state': reward_state, 'foot_forces': foot_forces, 'reward_vel_state': reward_vel_state, 'foot_vel': foot_vel}
    
    pattern_all = ': array\(\[\[(.*?)\]\]'

    key_list = list(res.keys())
    for i in range(len(info_list)):
        shape_in = res[key_list[i]].shape
        info_list[i] += pattern_all
        pattern_index = re.compile(info_list[i], re.DOTALL)
        info_raw_arrays = pattern_index.findall(file_content)
        for array_str in info_raw_arrays:
            array_list = eval(array_str)
            temp = np.array(array_list, dtype=np.float32).reshape(shape_in[0], shape_in[1])
            res[key_list[i]] = np.concatenate((res[key_list[i]], temp), axis=0)

    return res



if __name__ == "__main__":


    file_name_isaac = "/home/rl/Project/legged_gym_baseline/legged_gym/legged_gym/scripts/sim_logs/" + "sim_isaac" + ".log"
    res_is = parse_logs(file_name_isaac)

    obs_ac = res_is['desire_state'][:, 0]

    log_dir = log_dir_path = f"/home/rl/Project/legged_gym_baseline/legged_gym/legged_gym/scripts/sim_logs/tensorboard_logs/Sim_{datetime.now().strftime('%b%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    for i in range(obs_ac.shape[0]):
        writer.add_scalars('foot_1',  {'desire_state': res_is['desire_state'][i, 0], 'reward_state': res_is['reward_state'][i, 0], 'reward_vel_state': res_is['reward_vel_state'][i, 0]}, i)
        writer.add_scalars('foot_1_force',  {'foot_forces': res_is['foot_forces'][i, 0]}, i)
        writer.add_scalars('foot_1_vel',  {'foot_vel': res_is['foot_vel'][i, 0]}, i)
        writer.add_scalars('foot_2',  {'desire_state': res_is['desire_state'][i, 1], 'reward_state': res_is['reward_state'][i, 1], 'reward_vel_state': res_is['reward_vel_state'][i, 1]}, i)
        writer.add_scalars('foot_2_force',  {'foot_forces': res_is['foot_forces'][i, 1]}, i)
        writer.add_scalars('foot_2_vel',  {'foot_vel': res_is['foot_vel'][i, 1]}, i)
        writer.add_scalars('foot_3',  {'desire_state': res_is['desire_state'][i, 2], 'reward_state': res_is['reward_state'][i, 2], 'reward_vel_state': res_is['reward_vel_state'][i, 2]}, i)
        writer.add_scalars('foot_3_force',  {'foot_forces': res_is['foot_forces'][i, 2]}, i)
        writer.add_scalars('foot_3_vel',  {'foot_vel': res_is['foot_vel'][i, 2]}, i)
        writer.add_scalars('foot_4',  {'desire_state': res_is['desire_state'][i, 3], 'reward_state': res_is['reward_state'][i, 3], 'reward_vel_state': res_is['reward_vel_state'][i, 3]}, i)
        writer.add_scalars('foot_4_force',  {'foot_forces': res_is['foot_forces'][i, 3]}, i)
        writer.add_scalars('foot_4_vel',  {'foot_vel': res_is['foot_vel'][i, 3]}, i)

    writer.close()
    print('Done!')
