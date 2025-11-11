import os
import numpy as np

def main():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy")
    os.makedirs(root, exist_ok=True)

    n_train, n_val, T = 64, 16, 1024

    def make_split(n, fname):
        seq = np.random.rand(n, T, 2).astype("float32")           # stress–strain (T,2)
        mesh = np.random.rand(n, 7, 32, 32, 32).astype("float32") # 7-ch 3D fields
        para = np.random.rand(n, 6).astype("float32")             # printing params
        target = np.random.rand(n, 256).astype("float32")         # latent target
        np.savez(os.path.join(root, fname), seq=seq, mesh=mesh, para=para, target=target)

    make_split(n_train, "train.npz")
    make_split(n_val, "val.npz")
    print(f"Dummy data saved to {root}")

if __name__ == "__main__":
    main()
