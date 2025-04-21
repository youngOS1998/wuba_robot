import re
import json
import numpy as np
import matplotlib.pyplot as plt


def parse_logs(file_path):
    observations = np.zeros((1, 45), dtype=np.float32)
    actions = np.zeros((1, 12), dtype=np.float32)

    # Regular expressions to match log entries
    obs_pattern = re.compile(r"Observation: (.*)")
    action_pattern = re.compile(r"joint_pos_target: (.*)")

    with open(file_path, 'r') as file:
        for line in file:
            obs_match = obs_pattern.search(line)
            action_match = action_pattern.search(line)

            if obs_match:
                # Convert the JSON string back to a list and then to a numpy array
                obs_list = json.loads(obs_match.group(1))
                obs = np.array(obs_list)
                observations = np.concatenate((observations, obs), axis=0)
                
            if action_match:
                # Convert the matched string back to a dictionary or the original data structure
                # action = eval(action_match.group(1))
                # actions.append(action)
                action_list = json.loads(action_match.group(1))
                action = np.array(action_list)
                # actions.append(action)
                # actions = np.append(actions, action)
                actions = np.concatenate((actions, action), axis=0)
    return observations, actions

if __name__ == "__main__":
    file_name = "/home/rl/Project/DreamWAQ_yiming_changes/legged_gym_58/legged_gym/scripts/" + "01" + ".log"
    observations, actions = parse_logs(file_name)
    print("Observations:", observations)
    print("Observations shape:", observations.shape)
    obs_ac = observations[:50, 33:45]
    print("obs_ac shape:", obs_ac.shape[0])
    x = np.arange(0, obs_ac.shape[0], 1)
    plt.plot(x, obs_ac[:, 0].reshape(-1, 1), label='ac1')
    # # plt.plot(x, obs_ac[:, 1], label='ac2')
    plt.show()
    
    
    
    
    # print("Actions:", actions)
    # print("last 10 Actions:", actions[-10:])
