# MG-LDM: Multimodal Guided Latent Diffusion Model (Minimal Reference Implementation)

This repository provides a minimal, runnable reference implementation of the MG-LDM
framework. It follows the architectural spirit of the paper while using synthetic dummy
data for confidentiality and to ensure out-of-the-box execution.

## Quick Start

```bash
pip install -r requirements.txt
python -m mg_ldm.data.generate_dummy_data
python -m mg_ldm.train
python -m mg_ldm.infer_demo
```

## Data Shapes

- Stress–strain sequences: `(T, 2)` with `T=1024`
- 7-channel 3D stress fields: `(7, 32, 32, 32)`
- Printing parameters: `(6,)`
- Target latent (supervision for diffusion): `(256,)`

## Notes

- The code is a *minimal* educational scaffold. Replace dummy data, losses,
  and evaluation with your domain specifics.
