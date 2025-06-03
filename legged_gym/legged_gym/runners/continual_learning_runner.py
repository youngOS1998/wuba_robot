import os
import time
from collections import deque
import statistics
from typing import Dict, List, Optional

from torch.utils.tensorboard import SummaryWriter
import torch
import shutil

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv
from legged_gym.skills.ewc import EWC
from legged_gym.skills.skill_manager import SkillManager

class ContinualLearningRunner(OnPolicyRunner):
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        """Initialize the continual learning runner.
        
        Args:
            env: The environment
            train_cfg: Training configuration
            log_dir: Directory for logging
            device: Device to use for training
        """
        super().__init__(env, train_cfg, log_dir, device)
        
        # Initialize EWC and Skill Manager
        self.ewc = EWC(self.alg.actor_critic, device)
        self.skill_manager = SkillManager(self.alg.actor_critic, device)
        self.current_skill = None
        
    def add_skill(self, skill_name: str, skill_config: Dict):
        """Add a new skill to the skill manager."""
        self.skill_manager.add_skill(skill_name, skill_config)
        
    def switch_skill(self, skill_name: str):
        """Switch to a different skill."""
        if self.current_skill is not None:
            # Save current skill's Fisher information
            self.ewc.save_fisher_info(os.path.join(self.log_dir, f"{self.current_skill}_fisher.pt"))
            
        self.current_skill = skill_name
        
        # Load the new skill's Fisher information
        try:
            self.ewc.load_fisher_info(os.path.join(self.log_dir, f"{skill_name}_fisher.pt"))
        except FileNotFoundError:
            print(f"No saved Fisher information found for skill {skill_name}")
            
        # Update environment configuration for current skill
        skill_config = self.skill_manager.get_skill_config(skill_name)
        self.env.cfg.commands.ranges = skill_config['command_ranges']
        self.env.cfg.rewards.scales = skill_config['reward_scales']
        
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """Learn with continual learning support."""
        if self.current_skill is None:
            print("Warning: No skill selected. Please use switch_skill() before learning.")
            return
            
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, 
                                                           high=int(self.env.max_episode_length))
            
        get_dict = self.env.get_observations()
        obs = get_dict['obs'].clone().detach()
        privileged_obs = get_dict['privileged_obs'].clone().detach()
        obs_history = get_dict['obs_history'].clone().detach()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(self.device)          
        self.alg.actor_critic.train() 

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        ck = 0.001
        d = 0.99
        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # Rollout
            ck = ck**d
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, obs_history)
                    act_dict, rewards, dones, infos = self.env.step(actions, ck)
                    obs = act_dict['obs'].clone().detach()
                    obs_no_noise = act_dict['obs_no_noise'].clone().detach()
                    privileged_obs = act_dict['privileged_obs'].clone().detach()
                    obs_history = act_dict['obs_history'].clone().detach()
                    self.alg.transition.next_observations = obs_no_noise

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(self.device),\
                        obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs, privileged_obs)
            
            # Update with EWC loss
            mean_value_loss, mean_surrogate_loss, mean_ratio, mean_entropy, mean_kl, \
               mean_body_vel_loss, mean_recons_loss, kld_loss, mean_body_h_loss, mean_feet_h_loss, \
                   encoder_l2_norm, cv_vel_l2_norm = self.alg.update()
                   
            # Compute and update skill performance
            if it % 100 == 0:
                performance = self.evaluate_skill()
                self.skill_manager.update_skill_performance(self.current_skill, performance)
                
            stop = time.time()
            learn_time = stop - start
            
            if self.log_dir is not None:
                self.log(locals())
                
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                self.ewc.save_fisher_info(os.path.join(self.log_dir, f"{self.current_skill}_fisher.pt"))
                    
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        self.ewc.save_fisher_info(os.path.join(self.log_dir, f"{self.current_skill}_fisher.pt"))
            
    def evaluate_skill(self) -> float:
        """Evaluate the performance of the current skill."""
        num_episodes = 10
        total_reward = 0
        
        with torch.no_grad():
            for _ in range(num_episodes):
                obs_dict = self.env.reset()
                obs = obs_dict['obs'].to(self.device)
                privileged_obs = obs_dict['privileged_obs'].to(self.device)
                obs_history = obs_dict['obs_history'].to(self.device)
                done = False
                episode_reward = 0
                
                while not done:
                    actions = self.alg.act(obs, privileged_obs, obs_history)
                    act_dict, rewards, dones, _ = self.env.step(actions)
                    obs = act_dict['obs'].to(self.device)
                    privileged_obs = act_dict['privileged_obs'].to(self.device)
                    obs_history = act_dict['obs_history'].to(self.device)
                    episode_reward += rewards.mean().item()
                    done = dones.any().item()
                    
                total_reward += episode_reward
                
        return total_reward / num_episodes
        
    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'current_skill': self.current_skill,
            'skill_manager': self.skill_manager,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        self.current_skill = loaded_dict.get('current_skill')
        if 'skill_manager' in loaded_dict:
            self.skill_manager = loaded_dict['skill_manager']
        return loaded_dict['infos'] 