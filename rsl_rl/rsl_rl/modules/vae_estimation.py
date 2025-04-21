import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class VAEEstimator(nn.Module):
    def __init__(self,
                 mlp_input_dim,
                 enc_hidden_dims,
                 mlp_output_dim,
                 dec_hidden_dims,
                 latent_dim=19,
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(VAEEstimator, self).__init__()
        activation = get_activation(activation)
        self.beta = 1.0 

        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.mlp_output_dim_d = mlp_output_dim

        # Encoder
        self.enc_input_dim = mlp_input_dim       
        encoder_layers = []
        encoder_layers.append(nn.Linear(self.enc_input_dim, enc_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(enc_hidden_dims)):
            if l == len(enc_hidden_dims) - 1: #map the final hidden layer's output to the parameters (mean and variance) of the latent space distribution
                self.cv_mu = nn.Linear(enc_hidden_dims[l], latent_dim)
                self.cv_var = nn.Linear(enc_hidden_dims[l], latent_dim)
            else:
                encoder_layers.append(nn.Linear(enc_hidden_dims[l], enc_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder_module = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, dec_hidden_dims[0]))   
        decoder_layers.append(activation)
        for l in range(len(dec_hidden_dims)):
            if l == len(dec_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(dec_hidden_dims[l], self.mlp_output_dim_d))
            else:
                decoder_layers.append(nn.Linear(dec_hidden_dims[l], dec_hidden_dims[l + 1]))
                decoder_layers.append(activation)
        self.decoder_module = nn.Sequential(*decoder_layers)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def encode(self, observation_history):
        h = self.encoder_module(observation_history)
        mu, log_var = self.cv_mu(h), self.cv_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent):
        return self.decoder_module(latent)
    
    # define forward function for only the vae
    def vae_forward(self, observation_history):
        mu, log_var = self.encode(observation_history)
        latent = self.reparameterize(mu, log_var)
        return latent,self.decode(latent),mu,log_var
    
    
    def update(self, observation_history, base_lin_batch, next_obs_batch):
        encoder_pred,recons,vae_mu,vae_log_var=self.vae_forward(observation_history)
        body_vel_pred = encoder_pred[:, :3]
        vel = base_lin_batch[:]
        next_obs = next_obs_batch[:]  # notice the next obs does not contain command but contain vel
        body_vel_loss = F.mse_loss(body_vel_pred, vel) #torch.mean(torch.square(body_vel_pred - vel).sum(1)) #  F.mse_loss(body_vel_pred, vel)
        recons_loss = F.mse_loss(recons, next_obs) # torch.mean(torch.square(recons - next_obs).sum(1))  # F.mse_loss(recons, next_obs)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + vae_log_var - vae_mu.pow(2) - vae_log_var.exp(),1),0)
        vae_loss = body_vel_loss + recons_loss + self.beta * kld_loss
        self.optimizer.zero_grad()
        vae_loss.backward()
        self.optimizer.step()
        return body_vel_loss.item(), recons_loss.item(), kld_loss.item()
    
    
    def forward(self,observation_history):
        mu, log_var = self.encode(observation_history)
        latent = self.reparameterize(mu, log_var)
        vel = latent[:, :3]
        z = latent[:,3:]
        return vel.detach(),z.detach()
    
    # inference
    def inference(self, observation_history):
        mu, log_var = self.encode(observation_history)
        latent = self.reparameterize(mu, log_var)
        return latent
    


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None