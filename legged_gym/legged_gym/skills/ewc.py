import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class EWC:
    def __init__(self, model: nn.Module, device: str, ewc_lambda: float = 0.4):
        """Elastic Weight Consolidation (EWC) implementation.
        
        Args:
            model: The neural network model
            device: Device to store the Fisher information
            ewc_lambda: Weight of the EWC loss term
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optpar: Dict[str, torch.Tensor] = {}
        
    def compute_fisher_info(self, dataloader, num_batches: int = 100):
        """Compute Fisher Information for the current task.
        
        Args:
            dataloader: DataLoader containing the current task data
            num_batches: Number of batches to use for computing Fisher information
        """
        self.model.eval()
        self.fisher_info = {}
        self.optpar = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_info[name] = torch.zeros_like(param.data)
                self.optpar[name] = param.data.clone()
        
        # Compute Fisher information
        for i, (obs, actions) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            
            # Compute log probabilities
            log_probs = self.model.get_log_probs(obs, actions)
            
            # Compute gradients
            self.model.zero_grad()
            log_probs.backward()
            
            # Update Fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.fisher_info[name] += param.grad.data ** 2 / num_batches
        
        # Average Fisher information
        for name in self.fisher_info:
            self.fisher_info[name] = self.fisher_info[name].to(self.device)
            self.optpar[name] = self.optpar[name].to(self.device)
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute the EWC loss term.
        
        Returns:
            EWC loss term
        """
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                ewc_loss += torch.sum(self.fisher_info[name] * (param - self.optpar[name]) ** 2)
        return self.ewc_lambda * ewc_loss
    
    def save_fisher_info(self, path: str):
        """Save Fisher information and optimal parameters.
        
        Args:
            path: Path to save the Fisher information
        """
        torch.save({
            'fisher_info': self.fisher_info,
            'optpar': self.optpar
        }, path)
    
    def load_fisher_info(self, path: str):
        """Load Fisher information and optimal parameters.
        
        Args:
            path: Path to load the Fisher information from
        """
        checkpoint = torch.load(path)
        self.fisher_info = checkpoint['fisher_info']
        self.optpar = checkpoint['optpar']