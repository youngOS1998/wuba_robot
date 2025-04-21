from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    log_dir_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/plot_test/{datetime.now().strftime('%b%d_%H-%M-%S')}"
    sim_data_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/legged_gym_58/legged_gym/scripts/Jun17_16-04-21/u1_sim_data_rl_1.npz"
    real_data_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/real_data_yb_stand_still/u1_real_data_rl_1.npz"
    
    sim_data = np.load(sim_data_path)
    real_data = np.load(real_data_path)

    sim_data = {k: v for k, v in sim_data.items()}
    real_data = {k: v for k, v in real_data.items()}

    writer = SummaryWriter(log_dir=log_dir_path)
    
    total_steps = real_data['action'].shape[0]
    for k in ['obs', 'action', 'taucmd', 'jdq', 'jq', 'target_q', "latent"]:
    # for k in ['obs', 'action']:
        dims = real_data[k].shape[1]
        for dim in range(dims):
            for step in range(total_steps):
                if k == "obs":
                    if dim in list(range(0, 3)):
                        writer.add_scalars(f"ang_vel/{dim-0}", {'sim': sim_data[k][step][dim], 'real': real_data[k][step][dim]}, step)
                    elif dim in list(range(3, 6)):
                        writer.add_scalars(f"projected_gravity/{dim-3}", {'sim': sim_data[k][step][dim], 'real': real_data[k][step][dim]}, step)
                    elif dim in list(range(6, 9)):
                        writer.add_scalars(f"commands/{dim-6}", {'sim': sim_data[k][step][dim], 'real': real_data[k][step][dim]}, step)
                    elif dim in list(range(9, 21)):
                        writer.add_scalars(f"dof_pos/{dim-9}", {'sim': sim_data[k][step][dim], 'real': real_data[k][step][dim]}, step)
                    elif dim in list(range(21, 33)):
                        writer.add_scalars(f"dof_vel/{dim-21}", {'sim': sim_data[k][step][dim], 'real': real_data[k][step][dim]}, step)
                    else:
                        continue
                else:
                    writer.add_scalars(f"{k}/{dim}", {'sim': sim_data[k][step][dim], 'real': real_data[k][step][dim]}, step)
    writer.close()
    print("done!")
    