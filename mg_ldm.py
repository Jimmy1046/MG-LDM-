import torch
import torch.nn as nn
from .mamba_seq_encoder import MambaSeqEncoder
from .dc_gcn import DCGCN3D
from .mlp_encoder import MLPEncoder
from .cross_attention_fusion import CrossAttentionFusion
from .diffusion_model import SimpleLatentDiffusion
from .sdf_network import SDFDecoder

class MGLDM(nn.Module):
    def __init__(self,
                 seq_hidden_dim=256,
                 mesh_hidden_dim=256,
                 para_hidden_dim=128,
                 fused_dim=512,
                 diffusion_latent_dim=256,
                 sdf_hidden_dim=256,
                 timesteps=100, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.seq_enc = MambaSeqEncoder(2, seq_hidden_dim, num_layers=4)
        self.mesh_enc = DCGCN3D(hidden_dim=mesh_hidden_dim)
        self.para_enc = MLPEncoder(6, para_hidden_dim, para_hidden_dim)
        self.fuse = CrossAttentionFusion(dim=fused_dim)
        self.diffusion = SimpleLatentDiffusion(latent_dim=diffusion_latent_dim,
                                               cond_dim=fused_dim,
                                               timesteps=timesteps,
                                               beta_start=beta_start,
                                               beta_end=beta_end)
        self.sdf = SDFDecoder(latent_dim=diffusion_latent_dim, hidden_dim=sdf_hidden_dim)

    def forward_loss(self, seq, mesh, para, target):
        e_seq = self.seq_enc(seq)      # (B,256)
        e_mesh = self.mesh_enc(mesh)   # (B,256)
        e_para = self.para_enc(para)   # (B,128)
        cond = self.fuse(e_seq, e_mesh, e_para)  # (B,512)

        # diffusion loss on target latent
        loss = self.diffusion(target, cond)
        return loss

    @torch.no_grad()
    def infer_latent(self, seq, mesh, para, steps=None):
        e_seq = self.seq_enc(seq)
        e_mesh = self.mesh_enc(mesh)
        e_para = self.para_enc(para)
        cond = self.fuse(e_seq, e_mesh, e_para)
        z = self.diffusion.sample(cond, steps=steps)
        sdf_feat = self.sdf(z)
        return z, sdf_feat
