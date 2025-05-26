import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleDiffusion(nn.Module):
    def __init__(self, dim, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.dim = dim
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.model = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def forward(self, x, t):
        """
        x: [batch, seq_len, dim]
        t: [batch] - time step indices
        """
        t = t.float().unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), 1)  # [batch, seq_len, 1]
        xt = torch.cat([x, t], dim=2)  # concat time to each vector
        return self.model(xt)

    def compute_loss(self, x_start):
        batch_size = x_start.size(0)
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.forward(x_noisy, t)
        return F.mse_loss(noise_pred, noise)
