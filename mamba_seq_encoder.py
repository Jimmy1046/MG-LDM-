import torch
import torch.nn as nn

class SimpleMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        h = self.proj_in(x)                          # (B,T,D)
        h = h.transpose(1, 2)                        # (B,D,T)
        h = self.conv(h)                             # depthwise conv
        h = h.transpose(1, 2)                        # (B,T,D)
        g = torch.sigmoid(self.gate(x))
        h = self.proj_out(self.act(h)) * g
        return residual + h

class MambaSeqEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([SimpleMambaBlock(hidden_dim) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2)
        x = self.input_proj(x)       # (B,T,H)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)        # (B,H,T)
        x = self.pool(x).squeeze(-1) # (B,H)
        return self.out_proj(x)      # (B,H)
