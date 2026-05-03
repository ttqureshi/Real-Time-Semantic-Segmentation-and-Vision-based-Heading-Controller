# PIDNet (inference-only vendored copy)

This directory contains a **minimal subset** of [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) kept for loading **PIDNet-S** and running forward passes in our pipeline. Training scripts, datasets, benchmark configs, and demo assets were removed to reduce size.

**Included:** `configs/default.py`, `configs/__init__.py`, `models/pidnet.py`, `models/model_utils.py`, `models/__init__.py`, and `LICENSE`.

**Model weights** are not stored here; use `checkpoints/segmentation/pidNet_wieghts.pt` (or pass `--weights`). **YAML** for our six-class head lives under `src/segmentation/configs/pidnet_robotics_6class.yaml`.

To restore the full upstream tree (e.g. for retraining on Cityscapes), clone the official repository separately or use a git submodule with sparse checkout.
