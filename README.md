# Real-Time Semantic Segmentation + Vision-Based Heading Controller

## Overview

Student project pipeline: **PIDNet** semantic segmentation → **vision-based heading and speed control** (optical-flow proximity, PD yaw) → annotated video and per-frame CSV. Perception and control are modular so each stage can be tested on its own.

## Repository layout

| Path | Purpose |
|------|--------|
| `run_full_pipeline.py` | **Main entry:** video → PIDNet masks → heading controller → `results/full_pipeline_run/` |
| `src/segmentation/` | PIDNet inference helpers (`pidnet_inference.py`), robotics YAML config |
| `src/heading_controller/` | Controller loop (`main.py`), PD heading, braking, optical flow, visualization |
| `third_party/PIDNet/` | **Minimal** upstream PIDNet code (model + yacs defaults only); see `third_party/PIDNet/README.md` |
| `checkpoints/segmentation/` | Fine-tuned weights (`pidNet_wieghts.pt`) |
| `src/pipeline/` | Dataset prep (`prepare_dataset.py`), camera calibration (`calibrate_camera.py`) |
| `notebooks/main.ipynb` | Interactive experiments (imports project modules) |
| `data/` | Raw/processed datasets and calibration files |
| `results/` | Run outputs (often gitignored except placeholders) |
| `docs/` | Internal notes |

Legacy placeholders under `src/segmentation/` / `src/heading_controller/` (unused `train.py` stubs, empty `utils.py`) may still exist; the supported paths above are what the team uses.

## Quick start

1. **Environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. **Run the full pipeline** (place your input `.mp4` where you like; defaults assume project-relative paths):

   ```bash
   python run_full_pipeline.py --video path/to/video.mp4 --no-viz
   ```

   Common options:

   - `--weights checkpoints/segmentation/pidNet_wieghts.pt`
   - `--cfg src/segmentation/configs/pidnet_robotics_6class.yaml`
   - `--calibration data/calibration/calibration_data.npz` (undistort + calibrated heading; expects `K`, `dist` in the archive)
   - `--seg-every N` / `--flow-every N` for faster CPU runs
   - `--output-dir results/full_pipeline_run`

   Outputs: `controller_output.csv`, `controller_output.mp4`; optional `--save-masks` / `--save-seg-png`.

3. **Heading controller only** (precomputed masks or heuristic segmenter):

   ```bash
   python src/heading_controller/main.py --video path/to/video.mp4 --masks path/to/masks_dir
   ```

4. **Dataset download / layout** (for training or replication):

   ```bash
   python src/pipeline/prepare_dataset.py
   ```

5. **Notebook:** open `notebooks/main.ipynb` for step-by-step experiments.

## Team workflow

- Implement logic under `src/` and call it from `run_full_pipeline.py` or the notebook—avoid large amounts of logic only in notebooks.
- Large input videos are usually **not** committed; use `.gitignore` patterns as needed.

## Citation

PIDNet: see upstream [PIDNet](https://github.com/XuJiacong/PIDNet) and the paper referenced there. Our fork retains `LICENSE` under `third_party/PIDNet/`.
