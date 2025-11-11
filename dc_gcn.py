import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock3D(nn.Module):
    def __init__(self, in_ch, growth, layers=3):
        super().__init__()
        ch = in_ch
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm3d(ch),
                nn.SiLU(),
                nn.Conv3d(ch, growth, kernel_size=3, padding=1)
            ))
            ch += growth
        self.out_ch = ch

    def forward(self, x):
        feats = [x]
        h = x
        for layer in self.layers:
            y = layer(h)
            feats.append(y)
            h = torch.cat(feats, dim=1)
        return h

class DCGCN3D(nn.Module):
    """Dense-connected 3D conv encoder as a placeholder for DC-GCN.
    Input: (B,7,32,32,32) -> Output: (B, hidden_dim)
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.stem = nn.Conv3d(7, 16, kernel_size=3, padding=1)
        self.db1 = DenseBlock3D(16, growth=16, layers=3)
        self.tr1 = nn.Conv3d(self.db1.out_ch, 32, kernel_size=1)
        self.pool1 = nn.MaxPool3d(2)

        self.db2 = DenseBlock3D(32, growth=32, layers=3)
        self.tr2 = nn.Conv3d(self.db2.out_ch, 64, kernel_size=1)
        self.pool2 = nn.MaxPool3d(2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.tr1(self.db1(x))
        x = self.pool1(x)
        x = self.tr2(self.db2(x))
        x = self.pool2(x)
        x = self.head(x)
        return x
