from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import re
import matplotlib.pyplot as plt

def parse_logs(file_path):
    # Initialize a list to store the NumPy arrays

    with open(file_path, 'r') as file:
        file_content = file.read()  # Read the entire file content into a string

    stance_time = np.zeros((1, 4), dtype=np.float32)
    air_time = np.zeros((1, 4), dtype=np.float32)
    stance_time_max = np.zeros((1, 4), dtype=np.float32)
    air_time_max = np.zeros((1, 4), dtype=np.float32)

    info_list = ['stance', 'air', 'stance_max', 'air_max']


    res = {'stance': stance_time, 'air': air_time, 'stance_max': stance_time_max, 'air_max': air_time_max}
    
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


    file_name_isaac = "/home/rl/Project/legged_gym_cjw/quadrl/legged_gym_58/legged_gym/scripts/sim_logs/" + "sim_isaac" + ".log"
    res_is = parse_logs(file_name_isaac)

    obs_ac = res_is['stance'][:, 0]

    log_dir = log_dir_path = f"/home/rl/Project/legged_gym_cjw/quadrl/legged_gym_58/legged_gym/scripts/sim_logs/tensorboard_logs/Sim_{datetime.now().strftime('%b%d_%H-%M-%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    for i in range(obs_ac.shape[0]):
        writer.add_scalars('foot_1',  {'stance': res_is['stance'][i, 0], 'air': res_is['air'][i, 0], 'stance_max': res_is['stance_max'][i, 0], 'air_max': res_is['air_max'][i, 0]}, i)
        writer.add_scalars('foot_2',  {'stance': res_is['stance'][i, 1], 'air': res_is['air'][i, 1], 'stance_max': res_is['stance_max'][i, 1], 'air_max': res_is['air_max'][i, 1]}, i)
        writer.add_scalars('foot_3',  {'stance': res_is['stance'][i, 2], 'air': res_is['air'][i, 2], 'stance_max': res_is['stance_max'][i, 2], 'air_max': res_is['air_max'][i, 2]}, i)
        writer.add_scalars('foot_4',  {'stance': res_is['stance'][i, 3], 'air': res_is['air'][i, 3], 'stance_max': res_is['stance_max'][i, 3], 'air_max': res_is['air_max'][i, 3]}, i)

    writer.close()
    print('Done!')
