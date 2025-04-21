from copy import deepcopy

import isaacgym
assert isaacgym
import torch
import gym

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.obs_history_length = self.env.cfg.env.num_observation_history

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs
        self.i=0
        self.rew = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.float)

    def step(self, action, ck):

        # privileged information and observation history are stored in info
        obs, obs_no_noise, privileged_obs, rew_constant, rew_increase, done, info= self.env.step(action)
        self.rew[:] = rew_constant[:] + ck * rew_increase[:]
        
        if self.env.cfg.rewards.only_positive_rewards:
            self.rew[:] = torch.clip(self.rew[:], min=0.)
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        self.i+=1
        return {'obs': obs, 'obs_no_noise': obs_no_noise, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, self.rew, done, info

    # def step(self, action, ck):
    #     # privileged information and observation history are stored in info
    #     obs, obs_no_noise, privileged_obs, rew1, rew2, done, info= self.env.step(action)
    #     self.rew[:] = rew1[:] + ck * rew2[:]
        
    #     if self.env.cfg.rewards.only_positive_rewards:
    #         self.rew[:] = torch.clip(self.rew[:], min=0.)
    #     # breakpoint()
    #     self.obs_history = torch.cat((self.obs_history[:, self.chunk_size:], obs[:, :-12]), dim=-1)
    #     self.i += 1
    #     return {'obs': obs, 'obs_no_noise': obs_no_noise, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, self.rew, done, info
   
    def get_observations(self):
        obs, obs_no_noise = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        # breakpoint()
        return {'obs': obs, 'obs_no_noise': obs_no_noise, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    # def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
    #     ret = super().reset_idx(env_ids)
    #     self.obs_history[env_ids, :] = 0.
    #     self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
    #     return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    import ml_logger as logger

    from legged_gym.envs.wrappers.history_wrapper import HistoryWrapper

