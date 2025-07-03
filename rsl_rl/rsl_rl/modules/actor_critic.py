# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from rsl_rl.modules.vae_estimation import VAEEstimator


# 新增VQ量化层实现
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-0.1, 0.1)
        self.codebook.weight.requires_grad = False  # EMA不需要梯度

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_codebook', self.codebook.weight.data.clone())

    def forward(self, inputs):
        # 计算距离
        flat_inputs = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        quantized = torch.matmul(encodings, self.codebook.weight)
        quantized = quantized.view_as(inputs)
        quantized = inputs + (quantized - inputs).detach()

        # VQ Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # EMA更新
        if self.training:
            with torch.no_grad():
                # 统计每个码本被用到的次数
                cluster_size = encodings.sum(0)
                # EMA更新
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                embed_sum = torch.matmul(encodings.t(), flat_inputs)
                self.ema_codebook.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # 归一化
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps) * n
                )
                # 更新码本
                self.codebook.weight.data.copy_(
                    self.ema_codebook / cluster_size.unsqueeze(1)
                )

        return quantized, encoding_indices, vq_loss
    

class STDPVectorQuantizer(VectorQuantizer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, 
                 decay=0.99, eps=1e-5, tau_stdp=10.0, a_plus=0.01, a_minus=0.01, learning_rate=0.01):
        super().__init__(num_embeddings, embedding_dim, commitment_cost, decay, eps)
        # STDP参数
        self.tau_stdp = tau_stdp  # 时间常数
        self.a_plus = a_plus      # LTP幅度
        self.a_minus = a_minus    # LTD幅度
        self.learning_rate = learning_rate  
        self.online_learning = True

        # 上次激活时间记录
        self.register_buffer('last_active', torch.zeros(num_embeddings))

    def forward(self, inputs):
        # 计算距离
        flat_inputs = inputs.view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.codebook.weight.t())
            + torch.sum(self.codebook.weight ** 2, dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        quantized = torch.matmul(encodings, self.codebook.weight)
        quantized = quantized.view_as(inputs)
        quantized = inputs + (quantized - inputs).detach()

        # VQ Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = torch.mean((quantized - inputs.detach()) ** 2)
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # EMA更新
        if self.training or self.online_learning:

            current_time = self.last_active.max() + 1
            time_diff = current_time - self.last_active[encoding_indices]

            # STDP规则：近期的活跃码本增强
            stdp_factor = torch.where(
                time_diff > 0,
                self.a_plus * torch.exp(-time_diff / self.tau_stdp),
                -self.a_minus * torch.exp(time_diff / self.tau_stdp)
            )

            # print(stdp_factor)

            with torch.no_grad():

                # 只更新被使用的码本向量
                # 先将stdp_factor扩展为4096×1，然后与encodings逐元素相乘
                weighted_encodings = encodings * stdp_factor.unsqueeze(1)  # 4096×256
                # 再转置
                weighted_encodings_t = weighted_encodings.t()  # 256×4096
                # 再与flat_inputs做矩阵乘法
                stdp_update = torch.matmul(weighted_encodings_t, flat_inputs)  # 256×19
                self.codebook.weight.data += self.learning_rate * stdp_update


                # 统计每个码本被用到的次数
                cluster_size = encodings.sum(0)
                # EMA更新
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                embed_sum = torch.matmul(encodings.t(), flat_inputs)
                self.ema_codebook.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # 归一化
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps) * n
                )
                # 更新码本
                self.codebook.weight.data.copy_(
                    self.ema_codebook / cluster_size.unsqueeze(1)
                )

            # 更新时间戳
            self.last_active[encoding_indices] = current_time

        return quantized, encoding_indices, vq_loss

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_obs_history,
                        num_actions,
                        encoder_hidden_dims=[128,64, 19],
                        decoder_hidden_dims=[64,128,48],
                        latent_dim=19,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        num_embeddings = 256,
                        commitment_cost=0.1,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs + latent_dim + 1 + 4 + 3 
        mlp_input_dim_c = num_critic_obs + num_actor_obs

        mlp_input_dim_e = num_obs_history
        mlp_output_dim_d = num_actor_obs

        # Estimator
        # self.estimator = VAEEstimator(mlp_input_dim_e, encoder_hidden_dims, mlp_output_dim_d, decoder_hidden_dims)

        # 新增VQ-VAE组件
        self.encoder_fc = nn.Linear(encoder_hidden_dims[-1], latent_dim) 

        # VQ码本 (核心组件)
        # self.vq_layer = STDPVectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        # # 修改解码器输入维度适配VQ
        # self.decoder_input_fc = nn.Linear(latent_dim, decoder_hidden_dims[0])

        # Encoder
        self.enc_input_dim = mlp_input_dim_e
        encoder_layers = []
        encoder_layers.append(nn.Linear(self.enc_input_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1: #map the final hidden layer's output to the parameters (mean and variance) of the latent space distribution
                self.cv_vel = nn.Linear(encoder_hidden_dims[l], 3)
                self.body_height = nn.Linear(encoder_hidden_dims[l], 1)
                self.feet_height = nn.Linear(encoder_hidden_dims[l], 4)
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.encoder_module = nn.Sequential(*encoder_layers)

        # Decoder
        self.mlp_output_dim = mlp_output_dim_d
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, decoder_hidden_dims[0]))   
        decoder_layers.append(activation)
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], self.mlp_output_dim))
            else:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l + 1]))
                decoder_layers.append(activation)
        self.decoder_module = nn.Sequential(*decoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        # self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.actor_logstd = nn.Parameter(torch.zeros(num_actions,))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    # VAE
    def encode(self, observation_history):
        h = self.encoder_module(observation_history)

        h_flatten = h.view(h.size(0), -1)
        
        # 获取连续编码
        continuous_z = self.encoder_fc(h_flatten)
        continuous_z = torch.tanh(continuous_z)

        # VQ量化
        quantized_z, encoding_indices, vq_loss = self.vq_layer(continuous_z)

        vel = self.cv_vel(h)
        body_h = self.body_height(h)
        feet_h = self.feet_height(h)

        return quantized_z, encoding_indices, vq_loss, vel, body_h, feet_h

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent):
        return self.decoder_module(latent)
    
    # define forward function for only the vae
    def vae_forward(self, observation_history):
        # eps = torch.normal(mean=torch.zeros_like(observation_history, dtype=torch.float32), 
        #                    std=torch.ones_like(observation_history, dtype=torch.float32)).to(observation_history.device)
        # noised_hist = observation_history + 2 * eps * observation_history.std(0)
        # vel, mu, log_var = self.encode(noised_hist)
        
        quantized_z, _, vq_loss, vel, body_h, feet_h = self.encode(observation_history)
        # print('quantized_z', quantized_z)

        # 解码器前处理
        # decoder_input = self.decoder_input_fc(quantized_z)
        recons = self.decoder_module(quantized_z)

        return quantized_z, recons, vq_loss, vel, body_h, feet_h
    
    def net_l2_norm(self, network, mean=False):
        weights = 0
        param_num = 0
        for item in list(network.parameters()):
            if item.requires_grad:
                weights += item.pow(2).sum()
                param_num += np.prod(list(item.data.shape))
        if mean:
            param_num = max(param_num, 1)
            weights = weights / param_num
        return weights

    # Actor
    def update_distribution(self, observations, observation_history):
        # with torch.no_grad():
        quantized_z, recons, vq_loss, vel, body_h, feet_h = self.vae_forward(observation_history)
        
        # eps = torch.normal(mean=torch.zeros_like(vae_vel, dtype=torch.float32), std=torch.ones_like(vae_vel, dtype=torch.float32)).to(vae_vel.device)
        # noised_vel = vae_vel + (eps * vae_vel.std(0)).detach()
        mean = self.actor(torch.cat([vel.detach(), body_h.detach(), feet_h.detach(), quantized_z, observations], dim=-1))
        
        # mean = self.actor(torch.cat([vae_vel, latent, observations], dim=-1))
        
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        self.distribution = Normal(mean, std)
        return quantized_z, recons, vq_loss, vel, body_h, feet_h

    def act(self, observations,observation_history):
        quantized_z, recons, vq_loss, vel, body_h, feet_h = self.update_distribution(observations, observation_history)
        action = self.distribution.sample()                                                                                        
        return action, quantized_z, recons, vel, body_h, feet_h
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, observation_history):
        # 编码历史观测，获得VQ-VAE的量化向量和辅助特征
        quantized_z, encoding_indices, _, vel, body_h, feet_h = self.encode(observation_history)
        # 拼接特征送入actor
        action = self.actor(torch.cat((vel, body_h, feet_h, quantized_z, observations), dim=-1))
        return action, vel, encoding_indices

    def evaluate(self, observations,privileged_observations):
        value = self.critic(torch.cat((observations,privileged_observations),dim=-1))
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
