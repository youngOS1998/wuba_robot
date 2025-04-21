from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    log_dir_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/plot_test/{datetime.now().strftime('%b%d_%H-%M-%S')}"
    real_data_path = f"/home/lenovo/Project/DreamWAQ_yiming_changes/plot_data/real_data_yb_walking/done.npz"
    real_data = np.load(real_data_path)
    real_data = {k: v for k, v in real_data.items()}

    writer = SummaryWriter(log_dir=log_dir_path)
    
    total_steps = real_data['action'].shape[0]
    for k in ['obs', 'action']:
        if k == "action":
            for step in range(total_steps):
                dims = real_data[k].shape[1]
                action_dict = {}
                for dim in range(dims):
                    action_dict[str(dim)] = real_data[k][step][dim]
                writer.add_scalars(f"{k}", action_dict, step)
        elif k == "obs":
            for step in range(total_steps):
                dims = real_data[k].shape[1]
                ang_vel_dict, projected_gravity_dict, commands_dict, dof_pos_dict, dof_vel_dict = {}, {}, {}, {}, {}
                
                for dim in range(0, 3):
                    ang_vel_dict[str(dim)] = real_data[k][step][dim]
                writer.add_scalars(f"ang_vel", ang_vel_dict, step)
                
                for dim in range(3, 6):
                    projected_gravity_dict[str(dim)] = real_data[k][step][dim]
                writer.add_scalars(f"projected_gravity", projected_gravity_dict, step)
                
                for dim in range(6, 9):
                    commands_dict[str(dim)] = real_data[k][step][dim]
                writer.add_scalars(f"commands", commands_dict, step)
                
                for dim in range(9, 21):
                    dof_pos_dict[str(dim)] = real_data[k][step][dim]
                writer.add_scalars(f"dof_pos", dof_pos_dict, step)
                
                for dim in range(21, 33):
                    dof_vel_dict[str(dim)] = real_data[k][step][dim]
                writer.add_scalars(f"dof_vel", dof_vel_dict, step)

    writer.close()
    print("done!")
    