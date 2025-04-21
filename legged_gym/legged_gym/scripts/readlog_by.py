import re
import numpy as np
import matplotlib.pyplot as plt


def parse_logs(file_path):
    # Initialize a list to store the NumPy arrays
    observations = np.zeros((1, 45), dtype=np.float32)
    actions = np.zeros((1,12), dtype=np.float32)
    
    # Regular expression to match log entries for observations
    obs_pattern = re.compile(r'Observation: array\(\[\[(.*?)\]\]', re.DOTALL)
    action_pattern = re.compile(r'action: array\(\[(.*?)\]', re.DOTALL)

    # Open the log file and read its contents
    with open(file_path, 'r') as file:
        file_content = file.read()  # Read the entire file content into a string

    # Find all matches in the log content
    obs_raw_arrays = obs_pattern.findall(file_content)

    for array_str in obs_raw_arrays:
        # Convert the string representation of the array to a list
        array_list = eval('[' + array_str + ']')  # Convert the string to a list
        # Convert the list to a NumPy array and append to the list of arrays
        obs = np.array(array_list, dtype=np.float32).reshape(1,45)
        # print(obs.shape)
        observations = np.concatenate((observations, obs), axis=0)
    
    action_raw_arrays = action_pattern.findall(file_content)

    for array_str in action_raw_arrays:
        # Convert the string representation of the array to a list
        array_list = eval('[' + array_str + ']')  # Convert the string to a list
        # Convert the list to a NumPy array and append to the list of arrays
        action = np.array(array_list, dtype=np.float32).reshape(1,12)
        # print(action.shape)
        actions = np.concatenate((actions, action), axis=0)

    return observations,actions 

if __name__ == "__main__":
    file_name = "/home/rl/Project/legged_gym_initial/legged_gym/legged_gym/scripts/" + "sim" + ".log"
    observations,actions = parse_logs(file_name)
    # print("Observations:", observations)
    print("Observations shape:", observations.shape)
    print("Actions shape:", actions.shape)
    obs_ac = observations[:, 33:45]
    # print("obs_ac shape:", obs_ac.shape[0])
    x = np.arange(0, obs_ac.shape[0], 1)
    nb_rows = 2
    nb_cols = 3
    fig, axs = plt.subplots(nb_rows, nb_cols)

    # for i in range(4):
    #     # plt.plot(x, obs_ac[:, i], '*',  label=f'Observation Feature {i+33}')
    #     plt.plot(x, obs_ac[:, i], '*',  label=f'Action {i}')

    # Plotting the first six features from 'observations' with stars

    # plot base_ang velocity
    a = axs[0, 0]
    for i in range(0, 3):
        a.plot(x, observations[:, i],label=f'Observation {i}') 
    a.set(xlabel='time [s]', ylabel='Ang_vel', title='ang vel')
    a.grid()
    # plot projected_gravity
    a = axs[0, 1]
    for i in range(3, 6):
        a.plot(x, observations[:, i],label=f'Observation {i}') 
    a.set(xlabel='time [s]', ylabel='Projected gravity', title='Projected gravity')
    a.grid()
    # plot command
    a = axs[0, 2]
    a.plot(x, observations[:, 6],label='lin_vel_x')
    a.plot(x, observations[:, 7],label='lin_vel_y')
    a.plot(x, observations[:, 8],label='ang_vel_z')
    a.set(xlabel='time [s]', ylabel='Command', title='Command')
    a.grid()
    a.legend()
    # plot dof_pos offset
    a = axs[1, 0]
    for i in range(9, 21):
        a.plot(x, observations[:, i],label=f'Observation {i}')
    a.set(xlabel='time [s]', ylabel='Dof pos offset', title='Dof pos offset')
    a.grid()
    # plot dof_vel
    a = axs[1, 1]
    for i in range(21, 33):
        a.plot(x, observations[:, i],label=f'Observation {i}')
    a.set(xlabel='time [s]', ylabel='Dof_vel', title='Dof_vel')
    a.grid()
    # plot action
    a = axs[1, 2]
    for i in range(33, 45):
        a.plot(x, observations[:, i],label=f'Observation {i}')
    a.set(xlabel='time [s]', ylabel='Action', title='Action')
    a.grid()

    # plt.plot(x, observations[:, 6],label=f'Observation {6}')
    # plt.plot(x, observations[:, 16],label=f'Observation {16}')
    # plt.plot(x, observations[:, 26],label=f'Observation {26}')

    # Adding the legend
    # plt.legend()

    # Show the plot
    plt.show()
    
