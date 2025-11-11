import torch
import torch.nn as nn
import math

class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor):
        # sinusoidal embedding
        half = t.shape[0]
        device = t.device
        # simple learned embedding: just scale t to [0,1]
        t = t.float().unsqueeze(-1)
        return self.fc(t)

class DenoiserMLP(nn.Module):
    def __init__(self, latent_dim=256, cond_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + 1, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x, cond, t):
        t = t.float().unsqueeze(-1) / 100.0
        h = torch.cat([x, cond, t], dim=-1)
        return self.net(h)

class SimpleLatentDiffusion(nn.Module):
    def __init__(self, latent_dim=256, cond_dim=512, timesteps=100, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.denoiser = DenoiserMLP(latent_dim, cond_dim)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alphas_cumprod[t].unsqueeze(-1)
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    def forward(self, x0, cond):
        # training loss: predict noise
        bs = x0.size(0)
        t = torch.randint(0, self.timesteps, (bs,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.denoiser(xt, cond, t)
        loss = ((pred - noise)**2).mean()
        return loss

    @torch.no_grad()
    def sample(self, cond, steps=None):
        steps = steps or self.timesteps
        x = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        for i in reversed(range(steps)):
            t = torch.full((cond.size(0),), i, device=cond.device, dtype=torch.long)
            a = self.alphas_cumprod[i]
            beta = self.betas[i]
            pred_noise = self.denoiser(x, cond, t)
            x = (x - (1 - a).sqrt() * pred_noise) / a.sqrt()
            if i > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        return x
