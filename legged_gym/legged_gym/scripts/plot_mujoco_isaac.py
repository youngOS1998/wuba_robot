from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    log_dir_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/plot_data/{datetime.now().strftime('%b%d_%H-%M-%S')}"
    isaac_data_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/legged_gym_58/legged_gym/scripts/play_data_rl.npz"
    mujoco_data_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/legged_gym_58/legged_gym/scripts/u3_sim_data_rl_0.npz"
    
    mujoco_data = np.load(mujoco_data_path)
    isaac_data = np.load(isaac_data_path)

    mujoco_data = {k: v for k, v in mujoco_data.items()}
    isaac_data = {k: v for k, v in isaac_data.items()}

    writer = SummaryWriter(log_dir=log_dir_path)
    
    total_steps = isaac_data['action'].shape[0]
    for k in ['obs', 'action']:
        dims = isaac_data[k].shape[1]
        for dim in range(dims):
            for step in range(total_steps):
                if k == "obs":
                    if dim in list(range(0, 3)):
                        writer.add_scalars(f"ang_vel/{dim-0}", {'mujoco': mujoco_data[k][step][dim], 'isaac': isaac_data[k][step][dim]}, step)
                    elif dim in list(range(3, 6)):
                        writer.add_scalars(f"projected_gravity/{dim-3}", {'mujoco': mujoco_data[k][step][dim], 'isaac': isaac_data[k][step][dim]}, step)
                    elif dim in list(range(6, 9)):
                        writer.add_scalars(f"commands/{dim-6}", {'mujoco': mujoco_data[k][step][dim], 'isaac': isaac_data[k][step][dim]}, step)
                    elif dim in list(range(9, 21)):
                        writer.add_scalars(f"dof_pos/{dim-9}", {'mujoco': mujoco_data[k][step][dim], 'isaac': isaac_data[k][step][dim]}, step)
                    elif dim in list(range(21, 33)):
                        writer.add_scalars(f"dof_vel/{dim-21}", {'mujoco': mujoco_data[k][step][dim], 'isaac': isaac_data[k][step][dim]}, step)
                    else:
                        continue
                else:
                    writer.add_scalars(f"{k}/{dim}", {'mujoco': mujoco_data[k][step][dim], 'isaac': isaac_data[k][step][dim]}, step)
    # vel plot
    for dim in range(3):
        for step in range(total_steps):
            writer.add_scalars(f"mujoco_vel/{dim}", {'vae_vel': mujoco_data['vae_vel'][step][dim], 'lin_vel': mujoco_data['lin_vel'][step][dim]}, step)
            writer.add_scalars(f"isaac_vel/{dim}", {'vae_vel': isaac_data['vae_vel'][step][dim], 'lin_vel': isaac_data['lin_vel'][step][dim]}, step)
    
    writer.close()
    print("done!")
    