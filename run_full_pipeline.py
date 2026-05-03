#!/usr/bin/env python3
"""
Full end-to-end pipeline
========================
Video  →  PIDNet segmentation  →  Heading controller  →  CSV + annotated video

Setup (one-time)
----------------
1. Clone PIDNet (provides the model code — weights are stored separately):

       git clone https://github.com/XuJiacong/PIDNet.git third_party/PIDNet

2. Install dependencies:

       pip install torch torchvision opencv-python numpy yacs tqdm

Usage (minimal — uses auto-detected weights and default YAML)
--------------------------------------------------------------
    python run_full_pipeline.py --video segmentation.mp4

Usage (explicit paths)
----------------------
    python run_full_pipeline.py \\
        --video     segmentation.mp4 \\
        --weights   checkpoints/segmentation/pidNet_wieghts.pt \\
        --cfg       src/segmentation/configs/pidnet_robotics_6class.yaml

Outputs (written to results/full_pipeline_run/ by default)
----------------------------------------------------------
  controller_output.csv      — per-frame v_cmd, omega_cmd, distances, etc.
  controller_output.mp4      — annotated video (segmentation overlay + gauges)
  masks_npy/frame_XXXXXX.npy — (optional, --save-masks)  raw label maps 0-5
  seg_png/frame_XXXXXX.png   — (optional, --save-seg-png) colorised segmentation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

HEADING_DIR = PROJECT_ROOT / "src" / "heading_controller"
SEG_DIR     = PROJECT_ROOT / "src" / "segmentation"

for _p in (str(HEADING_DIR), str(SEG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Colors for visualisation (class 0-5, RGB order)
# ---------------------------------------------------------------------------
_CLASS_COLORS_RGB = np.array(
    [
        [20,  24,  33],   # 0 background
        [51,  122, 183],  # 1 human
        [64,  145, 108],  # 2 obstacle
        [229, 126, 49],   # 3 road
        [214, 48,  49],   # 4 sidewalk
        [243, 196, 15],   # 5 speed breaker
    ],
    dtype=np.uint8,
)

DEFAULT_PIDNET_ROOT = PROJECT_ROOT / "third_party" / "PIDNet"
DEFAULT_WEIGHTS     = PROJECT_ROOT / "checkpoints" / "segmentation" / "pidNet_wieghts.pt"
DEFAULT_CFG         = PROJECT_ROOT / "src" / "segmentation" / "configs" / "pidnet_robotics_6class.yaml"
DEFAULT_VIDEO       = PROJECT_ROOT / "segmentation.mp4"
DEFAULT_OUT_DIR     = PROJECT_ROOT / "results" / "full_pipeline_run"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colorize_label_bgr(label: np.ndarray) -> np.ndarray:
    """Return a BGR uint8 image coloured by class."""
    safe = np.clip(label, 0, len(_CLASS_COLORS_RGB) - 1)
    rgb  = _CLASS_COLORS_RGB[safe]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resolve(path_str: str, must_exist: bool = False) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        # Try CWD first, then project root
        cwd_p = Path.cwd() / p
        root_p = PROJECT_ROOT / p
        p = cwd_p if cwd_p.exists() else root_p
    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser(
        description="Video → PIDNet → heading controller (full pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    pa.add_argument(
        "--video", default=str(DEFAULT_VIDEO),
        help=f"Input .mp4 path (default: {DEFAULT_VIDEO.name}).",
    )
    pa.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help="Trained PIDNet .pt checkpoint.",
    )
    pa.add_argument(
        "--cfg", default=str(DEFAULT_CFG),
        help="PIDNet YAML config (must match training).",
    )
    pa.add_argument(
        "--pidnet-root", default=str(DEFAULT_PIDNET_ROOT),
        help="Root of cloned XuJiacong/PIDNet repo. "
             "Override via env var PIDNET_ROOT.",
    )
    pa.add_argument(
        "--output-dir", default=str(DEFAULT_OUT_DIR),
        help="Directory for all outputs (created if missing).",
    )
    pa.add_argument(
        "--calibration", default=None,
        help="Path to calibration_data.npz (keys: K, dist).  When provided, "
             "frames are undistorted before segmentation and heading error is "
             "computed using the calibrated camera matrix.",
    )
    pa.add_argument(
        "--save-masks", action="store_true",
        help="Save per-frame label masks as frame_XXXXXX.npy.",
    )
    pa.add_argument(
        "--save-seg-png", action="store_true",
        help="Save per-frame colorised segmentation PNGs.",
    )
    pa.add_argument(
        "--no-viz", action="store_true",
        help="Disable the live OpenCV preview window.",
    )
    pa.add_argument(
        "--flow-every", type=int, default=1,
        help="Compute optical flow every N frames (default 1).",
    )
    pa.add_argument(
        "--seg-every", type=int, default=1,
        help="Run PIDNet segmentation every N frames, reusing the previous mask "
             "for the frames in between.  Speeds up CPU inference significantly "
             "(e.g. --seg-every 3 gives ~3× throughput at some accuracy cost).",
    )
    return pa.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # ------------------------------------------------------------------
    # 1. Resolve paths
    # ------------------------------------------------------------------
    pidnet_root = Path(os.environ.get("PIDNET_ROOT", args.pidnet_root)).expanduser().resolve()
    weights_path = _resolve(args.weights, must_exist=True)
    cfg_path     = _resolve(args.cfg,     must_exist=True)
    video_path   = _resolve(args.video,   must_exist=True)
    out_dir      = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks_npy_dir = out_dir / "masks_npy"
    seg_png_dir   = out_dir / "seg_png"
    if args.save_masks:
        masks_npy_dir.mkdir(parents=True, exist_ok=True)
    if args.save_seg_png:
        seg_png_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Validate PIDNet clone
    # ------------------------------------------------------------------
    if not (pidnet_root / "models").is_dir():
        print(
            "[ERROR] PIDNet code not found.\n"
            f"  Expected:   {pidnet_root}/models/\n"
            "  Fix:        git clone https://github.com/XuJiacong/PIDNet.git third_party/PIDNet\n"
            "  Or pass:    --pidnet-root /path/to/PIDNet\n"
            "  Or set env: PIDNET_ROOT=/path/to/PIDNet",
            file=sys.stderr,
        )
        return 1

    pr = str(pidnet_root)
    if pr not in sys.path:
        sys.path.insert(0, pr)

    # ------------------------------------------------------------------
    # 3. Build PIDNet model
    # ------------------------------------------------------------------
    import torch
    import torch.nn.functional as F

    from configs import config as pidnet_cfg
    from configs import update_config
    import models  # from PIDNet repo

    from pidnet_inference import choose_device, load_pidnet_checkpoint, predict_mask_logits

    class _CfgArgs:
        def __init__(self, cfg_file: str) -> None:
            self.cfg  = cfg_file
            self.opts: list = []

    update_config(pidnet_cfg, _CfgArgs(str(cfg_path)))

    device       = choose_device()
    model_name   = pidnet_cfg.MODEL.NAME
    num_classes  = pidnet_cfg.DATASET.NUM_CLASSES
    align_corners = bool(getattr(pidnet_cfg.MODEL, "ALIGN_CORNERS", True))
    out_idx      = int(pidnet_cfg.TEST.OUTPUT_INDEX)
    input_size   = (int(pidnet_cfg.TEST.IMAGE_SIZE[0]), int(pidnet_cfg.TEST.IMAGE_SIZE[1]))

    print(f"[INFO] Building model '{model_name}' with {num_classes} classes …")
    model = models.pidnet.get_pred_model(model_name, num_classes)
    n_loaded = load_pidnet_checkpoint(model, weights_path, device)
    if n_loaded == 0:
        print(
            "[WARN] 0 tensors loaded from checkpoint!\n"
            "       Check that MODEL.NAME in the YAML matches how the model was trained.",
            file=sys.stderr,
        )
    else:
        print(f"[INFO] Loaded {n_loaded} tensors from {weights_path.name}")
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 4. Define the per-frame mask function
    # ------------------------------------------------------------------
    def mask_fn(frame_idx: int, frame_bgr: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            logits = predict_mask_logits(
                model, frame_bgr, input_size, device, align_corners, out_idx
            )
            h, w = frame_bgr.shape[:2]
            logits_up = F.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=align_corners
            )
            label = torch.argmax(logits_up.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)

        # Optional disk dumps
        if args.save_masks:
            np.save(str(masks_npy_dir / f"frame_{frame_idx:06d}.npy"), label.astype(np.int32))
        if args.save_seg_png:
            bgr_vis = _colorize_label_bgr(label)
            cv2.imwrite(str(seg_png_dir / f"frame_{frame_idx:06d}.png"), bgr_vis)

        return label.astype(np.int32)

    # ------------------------------------------------------------------
    # 5. Camera calibration (optional) — precompute undistortion maps once
    # ------------------------------------------------------------------
    undistort_maps = None
    new_K = None

    if args.calibration:
        cal_path = _resolve(args.calibration, must_exist=True)
        cal = np.load(str(cal_path))
        K_cam = cal["K"].astype(np.float64)
        dist  = cal["dist"].astype(np.float64)

        # We need the video dimensions to compute the maps.
        # Peek at the video quickly (VideoCapture is not yet open in this scope).
        _cap_peek = cv2.VideoCapture(str(video_path))
        _vid_w = int(_cap_peek.get(cv2.CAP_PROP_FRAME_WIDTH))
        _vid_h = int(_cap_peek.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _cap_peek.release()

        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K_cam, dist, (_vid_w, _vid_h), alpha=1, newImgSize=(_vid_w, _vid_h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K_cam, dist, None, new_K, (_vid_w, _vid_h), cv2.CV_32FC1
        )
        undistort_maps = (map1, map2)
        print(f"[INFO] Calibration loaded: {cal_path.name}")
        print(f"[INFO] Undistortion maps precomputed for {_vid_w}×{_vid_h}")
        print(f"[INFO] new_K  fx={new_K[0,0]:.1f}  fy={new_K[1,1]:.1f}  "
              f"cx={new_K[0,2]:.1f}  cy={new_K[1,2]:.1f}")

    # ------------------------------------------------------------------
    # 6. Run the heading controller pipeline
    # ------------------------------------------------------------------
    from heading_controller import DEFAULT_CONFIG
    from main import run_pipeline

    print()
    print("=" * 62)
    print("  Pipeline summary")
    print("=" * 62)
    print(f"  Video        : {video_path}")
    print(f"  Weights      : {weights_path}")
    print(f"  Config       : {cfg_path}")
    print(f"  Device       : {device}")
    print(f"  Input size   : {input_size[0]}×{input_size[1]}")
    print(f"  Calibration  : {'YES  (' + str(args.calibration) + ')' if args.calibration else 'no'}")
    print(f"  seg_every    : {args.seg_every}")
    print(f"  flow_every   : {args.flow_every}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Save masks   : {args.save_masks}")
    print(f"  Save PNGs    : {args.save_seg_png}")
    print("=" * 62)
    print()

    run_pipeline(
        video_path     = str(video_path),
        masks_dir      = None,
        output_csv     = str(out_dir / "controller_output.csv"),
        output_video   = str(out_dir / "controller_output.mp4"),
        visualise      = not args.no_viz,
        flow_every     = args.flow_every,
        config         = dict(DEFAULT_CONFIG),
        mask_fn        = mask_fn,
        undistort_maps = undistort_maps,
        new_K          = new_K,
        seg_every      = args.seg_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
