import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FSSDataset(Dataset):
    def __init__(self, root, filename):
        path = os.path.join(root, filename)
        data = np.load(path)
        self.seq = data["seq"]    # (N, T, 2)
        self.mesh = data["mesh"]  # (N, 7, 32, 32, 32)
        self.para = data["para"]  # (N, 6)
        self.target = data["target"]  # (N, 256)

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        return {
            "seq": torch.from_numpy(self.seq[idx]),
            "mesh": torch.from_numpy(self.mesh[idx]),
            "para": torch.from_numpy(self.para[idx]),
            "target": torch.from_numpy(self.target[idx]),
        }
