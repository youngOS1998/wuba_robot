import time
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.wrappers.history_wrapper import HistoryWrapper
import torch

def is_recovered(env, threshold=0.2):
    # 判断是否恢复正常步态（如线速度误差小于阈值）
    # 你可以根据实际reward或观测定义更复杂的判据
    lin_vel_error = torch.abs(env.base_lin_vel[:, 0] - env.commands[:, 0])
    return (lin_vel_error < threshold).all().item()

def test_recovery_time(args, push_step=100, max_steps=500, threshold=0.2, repeat=10):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env = HistoryWrapper(env)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    actor_critic = ppo_runner.alg.actor_critic
    actor_critic.eval()  # 推理模式

    recovery_times = []
    for trial in range(repeat):
        obs_dict = env.reset()
        obs = obs_dict['obs']
        obs = obs[0]
        obs_history = obs_dict['obs_history']
        privileged_obs = obs_dict['privileged_obs']
        recovered = True
        step = 0
        recovery_time = None

        while step < max_steps:
            # 推搡
            if step == push_step:
                env._push_robots()
                recovered = False

            # 推理
            with torch.no_grad():
                action = actor_critic.act_inference(obs, obs_history)[0]
            observe, rewards, dones, infos = env.step(action, 0)
            obs = observe['obs']
            obs_history = observe['obs_history']

            # 检查是否恢复
            if not recovered:
                if is_recovered(env, threshold):
                    recovery_time = step - push_step
                    recovery_times.append(recovery_time)
                    print(f"Trial {trial+1}: 恢复用时 {recovery_time} 步")
                    break
            step += 1

        if recovery_time is None:
            print(f"Trial {trial+1}: 未在最大步数内恢复")
            recovery_times.append(max_steps - push_step)

    print(f"\n平均恢复步数: {sum(recovery_times)/len(recovery_times):.2f}")

if __name__ == '__main__':
    args = get_args()
    test_recovery_time(args)