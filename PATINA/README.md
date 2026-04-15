# PATINA

PATINA is the final project repository for the manuscript **"PATINA: A Geometric Alignment and Frequency-Aware Synergy Network for Ceramic Artifact Image Inpainting"**.

The framework keeps a stable SEM-style encoder-decoder backbone and organizes the paper method around three named corrective branches:

- `MRDA`: Mask-aware Rearranged Downsampling Adapter
- `LCBC`: Latent Context Borrowing Correction
- `DFCC`: Dual-stage Frequency Corrective Coupling

The current repository is organized for journal release and reproducible training. Legacy internal names have been removed from the public-facing configs and entrypoints.

## Repository Layout

```text
PATINA/
├── src/                          # Core model, loss, dataset, and runtime code
├── script/                       # Smoke tests, dataset tools, and validation helpers
├── checkpoints/                  # Default config and optional local weights
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

## Environment

The repository has been tested with Python 3.10 and CUDA-enabled PyTorch:

```bash
conda create -n patina python=3.10 -y
conda activate patina
pip install -e .
```

If you prefer installing from a plain requirements file instead of editable mode:

```bash
pip install -r requirements.txt
```

## Main Training

Edit `checkpoints/config.yml` to match your dataset flists and output locations, then run:

```bash
python main.py --mode 1 --path ./checkpoints
```

To evaluate an existing run:

```bash
python main.py --mode 2 --path ./checkpoints --resume_from /path/to/run/checkpoints/best.pth
```

Default paper-aligned config:

- `checkpoints/config.yml`

## Pretrained Weights

The local scripts currently assume the following checkpoint names under `PATINA/checkpoints/`:

- `InpaintingModel_gen.pth`: generator initialization / local default pretrain
- `InpaintingModel_dis.pth`: discriminator checkpoint when paired adversarial weights are needed

Reference download links for the released SEM-Net backbone initialization weights:

- SEM-Net released weights (root): <https://drive.google.com/drive/folders/1zlFqhm9JMYs4J0WaAHPSL_N1QkL8q479?usp=drive_link>
- CelebA-HQ weights: <https://drive.google.com/drive/folders/1L-Tt3mTgbJ_8Ki8jQIZE6xmddhHNl3YN?usp=drive_link>
- Places2 weights: <https://drive.google.com/drive/folders/1sgJRu-Vf6u6taZY-RpY8cFXRAH09i4fu?usp=drive_link>

## Smoke Tests

Run one GPU training step without starting a full experiment:

```bash
python script/smoke_train_step.py --offline-vgg
```

Run the lightweight PATINA preset validation:

```bash
python script/smoke_patina.py --pretrain checkpoints/InpaintingModel_gen.pth
```

## Core Paper Modules

The main paper-specific code paths are:

- `src/mrda_module.py`
- `src/lcbc_module.py`
- `src/dfcc_module.py`
- `src/adaptive_fusion_module.py`
- `src/networks.py`

## Notes

- `checkpoints/config.yml` is the editable default config used by the public entrypoints.
- Dataset paths in the config are written as relative examples and should be adjusted to your environment.
- `run_baseline.sh` uses the local `PATINA/checkpoints/InpaintingModel_gen.pth` weight file by default.

## License

This repository currently retains the upstream MIT license distributed with the released codebase. See `LICENSE`.
