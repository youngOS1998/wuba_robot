import torch
import numpy as np
from typing import Dict, List, Optional
from legged_gym.skills.ewc import EWC

class SkillManager:
    def __init__(self, model, device: str):
        """Skill manager for continual learning.
        
        Args:
            model: The neural network model
            device: Device to store the skills
        """
        self.model = model
        self.device = device
        self.skills: Dict[str, Dict] = {}
        self.current_skill = None
        self.ewc = EWC(model, device)
        
    def add_skill(self, skill_name: str, skill_config: Dict):
        """Add a new skill.
        
        Args:
            skill_name: Name of the skill
            skill_config: Configuration for the skill
        """
        self.skills[skill_name] = {
            'config': skill_config,
            'fisher_info': None,
            'optpar': None,
            'performance': 0.0
        }
        
    def switch_skill(self, skill_name: str):
        """Switch to a different skill.
        
        Args:
            skill_name: Name of the skill to switch to
        """
        if skill_name not in self.skills:
            raise ValueError(f"Skill {skill_name} not found")
            
        if self.current_skill is not None:
            # Save current skill's Fisher information
            self.ewc.save_fisher_info(f"skills/{self.current_skill}_fisher.pt")
            
        self.current_skill = skill_name
        
        # Load the new skill's Fisher information
        try:
            self.ewc.load_fisher_info(f"skills/{skill_name}_fisher.pt")
        except FileNotFoundError:
            print(f"No saved Fisher information found for skill {skill_name}")
            
    def update_skill_performance(self, skill_name: str, performance: float):
        """Update the performance of a skill.
        
        Args:
            skill_name: Name of the skill
            performance: Performance metric
        """
        if skill_name in self.skills:
            self.skills[skill_name]['performance'] = performance
            
    def get_best_skill(self) -> Optional[str]:
        """Get the best performing skill.
        
        Returns:
            Name of the best performing skill, or None if no skills exist
        """
        if not self.skills:
            return None
            
        return max(self.skills.items(), key=lambda x: x[1]['performance'])[0]
        
    def get_skill_config(self, skill_name: str) -> Dict:
        """Get the configuration for a skill.
        
        Args:
            skill_name: Name of the skill
            
        Returns:
            Configuration for the skill
        """
        if skill_name not in self.skills:
            raise ValueError(f"Skill {skill_name} not found")
            
        return self.skills[skill_name]['config']
        
    def compute_fisher_info(self, dataloader, num_batches: int = 100):
        """Compute Fisher information for the current skill.
        
        Args:
            dataloader: DataLoader containing the current skill data
            num_batches: Number of batches to use for computing Fisher information
        """
        self.ewc.compute_fisher_info(dataloader, num_batches)
        
    def save_skills(self, path: str):
        """Save all skills information.
        
        Args:
            path: Path to save the skills information
        """
        torch.save({
            'skills': self.skills,
            'current_skill': self.current_skill
        }, path)
        
    def load_skills(self, path: str):
        """Load all skills information.
        
        Args:
            path: Path to load the skills information from
        """
        checkpoint = torch.load(path)
        self.skills = checkpoint['skills']
        self.current_skill = checkpoint['current_skill']