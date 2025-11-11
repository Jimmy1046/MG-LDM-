import os, yaml, torch, numpy as np
from torch.utils.data import DataLoader

from .data.dataset import FSSDataset
from .models import MGLDM

def load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    val_set = FSSDataset(cfg["data"]["root"], cfg["data"]["val_file"])
    loader = DataLoader(val_set, batch_size=2, shuffle=False)

    model = MGLDM(**cfg["model"],
                  timesteps=cfg["diffusion"]["timesteps"],
                  beta_start=cfg["diffusion"]["beta_start"],
                  beta_end=cfg["diffusion"]["beta_end"]).to(device)

    # If a checkpoint exists, load it
    ckpt = os.path.join(cfg["train"]["save_dir"], "mg_ldm_demo.pt")
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"])
        print(f"Loaded checkpoint from {ckpt}")
    model.eval()

    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(loader))
        seq = batch["seq"].to(device)
        mesh = batch["mesh"].to(device)
        para = batch["para"].to(device)

        z, sdf_feat = model.infer_latent(seq, mesh, para, steps=20)
        np.save(os.path.join(out_dir, "latent.npy"), z.cpu().numpy())
        np.save(os.path.join(out_dir, "sdf_feat.npy"), sdf_feat.cpu().numpy())
        print(f"Saved outputs to {out_dir}")

if __name__ == "__main__":
    main()
