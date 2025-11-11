import torch.nn as nn
import torch

class SDFDecoder(nn.Module):
    """Map latent to an implicit representation (placeholder).
    For demo, just apply an MLP to produce another 256-D vector.
    """
    def __init__(self, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
