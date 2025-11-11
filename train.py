import os, yaml, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .data.dataset import FSSDataset
from .models import MGLDM
from .utils import set_seed

def load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    set_seed(42)

    train_set = FSSDataset(cfg["data"]["root"], cfg["data"]["train_file"])
    loader = DataLoader(train_set, batch_size=cfg["data"]["batch_size"], shuffle=True)

    model = MGLDM(**cfg["model"],
                  timesteps=cfg["diffusion"]["timesteps"],
                  beta_start=cfg["diffusion"]["beta_start"],
                  beta_end=cfg["diffusion"]["beta_end"]).to(device)

    optim = AdamW(model.parameters(), lr=cfg["train"]["lr"])

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    step = 0
    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        for batch in loader:
            seq = batch["seq"].to(device)
            mesh = batch["mesh"].to(device)
            para = batch["para"].to(device)
            target = batch["target"].to(device)

            loss = model.forward_loss(seq, mesh, para, target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optim.step()

            if step % cfg["train"]["log_interval"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

    ckpt = os.path.join(cfg["train"]["save_dir"], "mg_ldm_demo.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt)
    print(f"Saved checkpoint to {ckpt}")

if __name__ == "__main__":
    main()
