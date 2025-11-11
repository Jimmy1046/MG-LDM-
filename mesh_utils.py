import torch

def dummy_mesh_from_latent(latent: torch.Tensor):
    # Placeholder: normalize latent as a pseudo 'mesh embedding'
    return (latent - latent.mean(dim=-1, keepdim=True)) / (latent.std(dim=-1, keepdim=True) + 1e-6)
