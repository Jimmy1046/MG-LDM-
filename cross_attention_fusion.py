import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """Fuse three modality embeddings (seq, mesh, para) into a single vector.
    Uses simple multi-head attention over a 3-token sequence.
    """
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.proj = nn.Linear(256+256+128, dim)  # concat then project
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, e_seq, e_mesh, e_para):
        x = torch.cat([e_seq, e_mesh, e_para], dim=-1)  # (B, 256+256+128)
        x = self.proj(x).unsqueeze(1)                  # (B,1,dim)
        attn_out, _ = self.attn(x, x, x)               # self-attend
        h = self.fc(self.norm(attn_out))
        return h.squeeze(1)                            # (B, dim)
