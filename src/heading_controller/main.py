"""
Main processing loop — vision-based heading and velocity controller.

Expected input format
---------------------
Video      : 1280×720 RGB/BGR, 30 FPS  (.mp4)
Seg masks  : (720, 1280) int32 NumPy arrays, one file per frame,
             stored as  <masks_dir>/frame_XXXXXX.npy  (zero-padded 6 digits).
             Class labels follow _classes.csv:
               0=background  1=Human  2=Obstacle
               3=Road        4=Sidewalk  5=SpeedBreaker

Usage
-----
python main.py --video path/to/video.mp4 --masks path/to/masks/ [options]

Key arguments
-------------
--video          : Path to the input .mp4 video file (required).
--masks          : Directory with pre-computed .npy segmentation masks
                   (frame_XXXXXX.npy, one per frame, shape 720×1280 int32).
                   Omit to activate the heuristic AutoSegmenter fallback.
--output         : CSV log path   (default: results/controller_output.csv).
--output-video   : Annotated output video path
                   (default: results/controller_output.mp4).
--no-viz         : Disable the live OpenCV preview window.
--flow-every N   : Reuse flow for N-1 frames between Farneback calls
                   (default 1 = compute every frame).

Pipeline per frame
------------------
1. Read 1280×720 BGR frame from video.
2. Load pre-computed (720, 1280) int32 .npy mask  OR  run AutoSegmenter.
3. Downscale frame to 640×360 → compute Farneback optical flow.
4. Compute road-relative proximity scores for obstacle/human/speed-breaker.
5. compute_velocity_commands() → v_cmd, omega_cmd.
6. FrameRenderer.render() → annotated BGR frame (banner + gauges + overlay).
7. Write annotated frame to output video.
8. Accumulate CSV row.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make sure sibling modules are importable regardless of CWD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from auto_segment import AutoSegmenter
from braking_logic import compute_velocity_commands
from heading_controller import DEFAULT_CONFIG, HeadingController
from optical_flow_module import (
    compute_optical_flow,
    frame_to_gray,
    resize_for_flow,
    visualize_flow,
)
from visualizer import FrameRenderer


# ---------------------------------------------------------------------------
# Mask loader
# ---------------------------------------------------------------------------

def load_mask(
    masks_dir:    Optional[Path],
    frame_idx:    int,
    frame_bgr:    np.ndarray,
    auto_seg:     Optional[AutoSegmenter],
) -> np.ndarray:
    """
    Resolve segmentation mask for one frame (priority order):

    1. Pre-computed .npy file  <masks_dir>/frame_XXXXXX.npy
       Expected shape: (720, 1280) int32, labels 0-5 from _classes.csv.
    2. Heuristic AutoSegmenter  (colour + spatial priors on the raw frame)
    3. Last-resort dummy all-road mask  (should never be reached)
    """
    if masks_dir is not None:
        for mask_path in (
            masks_dir / f"frame_{frame_idx:06d}.npy",
            masks_dir / f"{frame_idx}.npy",
        ):
            if mask_path.exists():
                return np.load(str(mask_path)).astype(np.int32)

    if auto_seg is not None:
        return auto_seg.segment(frame_bgr)

    # Absolute fallback
    h, w = frame_bgr.shape[:2]
    return np.full((h, w), 3, dtype=np.int32)


# ---------------------------------------------------------------------------
# VideoWriter factory
# ---------------------------------------------------------------------------

def _make_writer(
    path: str,
    fps: float,
    frame_size: tuple,          # (width, height)
) -> cv2.VideoWriter:
    """Create a VideoWriter, trying mp4v then XVID codec as fallback."""
    os.makedirs(Path(path).parent, exist_ok=True)
    for fourcc_str in ("mp4v", "XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if writer.isOpened():
            print(f"[INFO] VideoWriter opened with '{fourcc_str}' -> '{path}'")
            return writer
    raise RuntimeError(f"Could not open VideoWriter for '{path}'")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _v_cmd_from_scores(distance_like: Dict[str, float], config: dict) -> float:
    """Compute linear velocity from proximity scores (braking logic, no heading)."""
    v_nominal    = float(config.get("v_nominal",            DEFAULT_CONFIG["v_nominal"]))
    t_emergency  = float(config.get("threshold_emergency",  DEFAULT_CONFIG["threshold_emergency"]))
    t_caution    = float(config.get("threshold_caution",    DEFAULT_CONFIG["threshold_caution"]))
    brake_factor = float(config.get("brake_factor",         DEFAULT_CONFIG["brake_factor"]))
    sb_brake     = float(config.get("sb_brake_factor",      DEFAULT_CONFIG["sb_brake_factor"]))
    sb_crawl     = float(config.get("sb_crawl_factor",      DEFAULT_CONFIG["sb_crawl_factor"]))

    d_obs = distance_like.get("obstacle", 1.0)
    d_hum = distance_like.get("human", 1.0)
    d_sb  = distance_like.get("speed_breaker", 1.0)

    v = v_nominal
    if d_obs < t_emergency or d_hum < t_emergency:
        v = 0.0
    elif d_obs < t_caution or d_hum < t_caution:
        v = v_nominal * brake_factor
    if v > 0.0:
        if d_sb < t_emergency:
            v = min(v, v_nominal * sb_crawl)
        elif d_sb < t_caution:
            v = min(v, v_nominal * sb_brake)
    return float(v)


def run_pipeline(
    video_path: str,
    masks_dir: Optional[str],
    output_csv: str,
    output_video: str,
    visualise: bool,
    flow_every: int,
    config: dict,
    mask_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    new_K: Optional[np.ndarray] = None,
    heading_fn: Optional[Callable[[np.ndarray, Optional[np.ndarray]], Tuple[float, float]]] = None,
    seg_every: int = 1,
) -> None:
    """
    Full frame-by-frame processing loop.

    Parameters
    ----------
    video_path      : path to input .mp4
    masks_dir       : directory with .npy masks, or None
    output_csv      : path for per-frame CSV log
    output_video    : path for annotated output video (.mp4)
    visualise       : show live OpenCV window
    flow_every      : run Farneback optical flow every N frames (reuse otherwise)
    config          : controller configuration dict
    mask_fn         : optional (frame_idx, frame_bgr) -> (H,W) int32 mask
    undistort_maps  : optional (map1, map2) from cv2.initUndistortRectifyMap;
                      when set each frame is remapped before any processing
    new_K           : (3,3) calibrated camera matrix for the undistorted frame;
                      forwarded to heading_fn and used for calibrated error
    heading_fn      : optional (seg_mask, new_K) -> (omega_cmd, e_psi);
                      plug in your own heading logic here; when set the built-in
                      HeadingController is used only for proximity scores and
                      v_cmd — omega_cmd comes entirely from heading_fn
    seg_every       : run segmentation every N frames and reuse mask otherwise
                      (speeds up CPU inference; set to 1 for every frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Opened '{video_path}'  "
          f"{vid_w}x{vid_h}  frames={total_frames}  FPS={video_fps:.1f}")
    if undistort_maps is not None:
        print("[INFO] Camera undistortion maps loaded — remapping every frame.")
    if heading_fn is not None:
        print("[INFO] Custom heading_fn provided — built-in HeadingController used only for proximity scores.")
    if seg_every > 1:
        print(f"[INFO] seg_every={seg_every} — segmentation runs on 1 of every {seg_every} frames.")

    masks_path = Path(masks_dir) if masks_dir else None
    os.makedirs(Path(output_csv).parent, exist_ok=True)

    # ---- Controllers, auto-segmenter & renderer -------------------------
    controller = HeadingController(config)
    renderer   = FrameRenderer(config, seg_alpha=0.38)

    auto_seg: Optional[AutoSegmenter] = None
    if masks_path is None and mask_fn is None:
        auto_seg = AutoSegmenter()
        print("[INFO] No masks directory or mask_fn given — using heuristic AutoSegmenter.")

    # ---- VideoWriter -------------------------------------------------------
    writer = _make_writer(output_video, video_fps, (vid_w, vid_h))

    # ---- State -----------------------------------------------------------
    prev_gray_flow: Optional[np.ndarray] = None
    cached_flow:    Optional[np.ndarray] = None
    cached_seg:     Optional[np.ndarray] = None
    frame_idx:  int   = 0
    fps_display: float = 0.0
    csv_rows = []

    # Per-stage timing accumulators (milliseconds)
    _T = {"undistort": 0.0, "seg": 0.0, "flow": 0.0,
          "ctrl": 0.0, "render": 0.0, "write": 0.0, "total": 0.0}

    print(f"[INFO] Writing annotated video -> '{output_video}'")
    print(f"[INFO] Starting processing loop ({total_frames} frames) ...")
    print(f"{'Frame':>7}  {'undist':>7}  {'seg':>8}  {'flow':>7}  "
          f"{'ctrl':>6}  {'rendr':>6}  {'write':>6}  {'total':>8}  {'FPS':>5}  "
          f"v_cmd  omega_cmd")

    while True:
        t0 = time.perf_counter()

        ret, frame_bgr = cap.read()
        if not ret:
            break

        # ---- 1. Undistort (precomputed maps, very fast) ------------------
        t1 = time.perf_counter()
        if undistort_maps is not None:
            frame_bgr = cv2.remap(frame_bgr, undistort_maps[0], undistort_maps[1],
                                  cv2.INTER_LINEAR)
        t_undistort = (time.perf_counter() - t1) * 1000

        # ---- 2. Segmentation mask (reuse when seg_every > 1) -------------
        t2 = time.perf_counter()
        seg_reused = (seg_every > 1 and frame_idx % seg_every != 0
                      and cached_seg is not None)
        if seg_reused:
            seg_mask = cached_seg
        elif mask_fn is not None:
            seg_mask = np.asarray(mask_fn(frame_idx, frame_bgr)).astype(np.int32)
            cached_seg = seg_mask
        else:
            seg_mask = load_mask(masks_path, frame_idx, frame_bgr, auto_seg)
            cached_seg = seg_mask
        t_seg = (time.perf_counter() - t2) * 1000

        # ---- 3. Optical flow (reuse when flow_every > 1) -----------------
        t3 = time.perf_counter()
        small     = resize_for_flow(frame_bgr)
        curr_gray = frame_to_gray(small)
        if prev_gray_flow is None:
            flow = np.zeros((*curr_gray.shape, 2), dtype=np.float32)
        elif frame_idx % flow_every == 0:
            flow = compute_optical_flow(prev_gray_flow, curr_gray)
            cached_flow = flow
        else:
            flow = cached_flow if cached_flow is not None else \
                   np.zeros((*curr_gray.shape, 2), dtype=np.float32)
        prev_gray_flow = curr_gray
        t_flow = (time.perf_counter() - t3) * 1000

        # ---- 4. Proximity scores + velocity commands ---------------------
        t4 = time.perf_counter()
        distance_like = controller.compute_distance_like_scores(seg_mask, flow)

        if heading_fn is not None:
            # Custom heading: get omega from plug-in, v from braking logic
            v_cmd = _v_cmd_from_scores(distance_like, config)
            omega_cmd, e_psi = heading_fn(seg_mask, new_K)
        else:
            # Default: built-in HeadingController (pixel or calibrated path)
            v_cmd, omega_cmd = compute_velocity_commands(
                seg_mask, flow, distance_like, controller, config
            )
            if new_K is not None:
                omega_cmd, e_psi = controller.compute_omega_cmd_calibrated(seg_mask, new_K)
            else:
                lane_center_x = controller.compute_lane_center(seg_mask)
                e_psi = lane_center_x - seg_mask.shape[1] / 2.0
        t_ctrl = (time.perf_counter() - t4) * 1000

        # ---- 5. Render annotated frame -----------------------------------
        t5 = time.perf_counter()
        lane_center_x, lane_horizon_y = controller.compute_lane_centerline(seg_mask)
        annotated = renderer.render(
            frame_bgr      = frame_bgr,
            seg_mask       = seg_mask,
            v_cmd          = v_cmd,
            omega_cmd      = omega_cmd,
            distance_like  = distance_like,
            frame_idx      = frame_idx,
            fps            = fps_display,
            lane_center_x  = lane_center_x,
            lane_horizon_y = lane_horizon_y,
            e_psi          = e_psi,
        )
        t_render = (time.perf_counter() - t5) * 1000

        # ---- 6. Write to output video ------------------------------------
        t6 = time.perf_counter()
        writer.write(annotated)
        t_write = (time.perf_counter() - t6) * 1000

        # ---- 7. Optional live preview ------------------------------------
        if visualise:
            try:
                cv2.imshow("Heading Controller", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User pressed 'q', stopping.")
                    break
            except cv2.error:
                print("[WARN] cv2.imshow not available in this environment — disabling live preview.")
                visualise = False

        # ---- 8. Timing bookkeeping ---------------------------------------
        t_total = (time.perf_counter() - t0) * 1000
        fps_display = 0.9 * fps_display + 0.1 * (1000.0 / max(t_total, 1e-3))
        _T["undistort"] += t_undistort
        _T["seg"]       += t_seg
        _T["flow"]      += t_flow
        _T["ctrl"]      += t_ctrl
        _T["render"]    += t_render
        _T["write"]     += t_write
        _T["total"]     += t_total

        # ---- 9. CSV row --------------------------------------------------
        csv_rows.append({
            "frame":          frame_idx,
            "v_cmd":          round(v_cmd, 4),
            "omega_cmd":      round(omega_cmd, 4),
            "e_psi":          round(e_psi, 4),
            "d_obstacle":     round(distance_like.get("obstacle", 1.0), 4),
            "d_human":        round(distance_like.get("human", 1.0), 4),
            "d_speed_breaker":round(distance_like.get("speed_breaker", 1.0), 4),
            "motion_state":   _state_label(v_cmd, omega_cmd, distance_like, config),
            "seg_reused":     int(seg_reused),
            "proc_ms":        round(t_total, 2),
        })

        # ---- 10. Per-frame console log (every 30 frames) ----------------
        if frame_idx % 30 == 0:
            n = max(frame_idx, 1)
            print(
                f"[{frame_idx:5d}/{total_frames}]"
                f"  {_T['undistort']/n:6.1f}ms"
                f"  {_T['seg']/n:7.1f}ms"
                f"  {_T['flow']/n:6.1f}ms"
                f"  {_T['ctrl']/n:5.1f}ms"
                f"  {_T['render']/n:5.1f}ms"
                f"  {_T['write']/n:5.1f}ms"
                f"  {_T['total']/n:7.1f}ms"
                f"  {fps_display:5.1f}"
                f"  {v_cmd:+.3f}  {omega_cmd:+.4f}"
            )

        frame_idx += 1

    # ---- Cleanup ---------------------------------------------------------
    cap.release()
    writer.release()

    # ---- Timing summary --------------------------------------------------
    n = max(frame_idx, 1)
    print()
    print("=" * 62)
    print(f"  Timing summary  ({frame_idx} frames)")
    print("=" * 62)
    for stage, label in [
        ("undistort", "undistort"),
        ("seg",       "segmentation"),
        ("flow",      "optical flow"),
        ("ctrl",      "controller"),
        ("render",    "rendering"),
        ("write",     "video write"),
        ("total",     "TOTAL"),
    ]:
        avg = _T[stage] / n
        tot = _T[stage] / 1000.0
        print(f"  {label:<14}: avg {avg:7.2f} ms   total {tot:6.1f} s")
    print(f"  {'effective FPS':<14}: {1000.0 / (_T['total'] / n):>6.2f}")
    print("=" * 62)

    # ---- Write CSV -------------------------------------------------------
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        os.makedirs(Path(output_csv).parent, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)
        print(f"[INFO] Saved {len(csv_rows)} rows -> '{output_csv}'")

    print(f"[INFO] Done.  Processed {frame_idx} frames.")
    print(f"[INFO] Annotated video  -> '{output_video}'")

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


# ---------------------------------------------------------------------------
# Tiny helper — grab label string for CSV column
# ---------------------------------------------------------------------------

def _state_label(
    v_cmd: float,
    omega_cmd: float,
    distance_like: Dict[str, float],
    config: dict,
) -> str:
    from visualizer import get_motion_state
    label, _ = get_motion_state(v_cmd, omega_cmd, distance_like, config)
    return label


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision-based heading and velocity controller — annotated video output."
    )
    parser.add_argument("--video", required=True,
                        help="Input .mp4 video file.")
    parser.add_argument("--masks", default=None,
                        help="Directory with .npy segmentation masks "
                             "(frame_XXXXXX.npy).  Omit for dummy all-road mask.")
    parser.add_argument("--output", default="results/controller_output.csv",
                        help="Output CSV path.")
    parser.add_argument("--output-video",
                        default="results/controller_output.mp4",
                        help="Annotated output video path "
                             "(default: results/controller_output.mp4).")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable live OpenCV preview window.")
    parser.add_argument("--flow-every", type=int, default=1,
                        help="Compute optical flow every N frames (default 1).")
    parser.add_argument("--kp",          type=float, default=None)
    parser.add_argument("--kd",          type=float, default=None)
    parser.add_argument("--v-nominal",   type=float, default=None)
    parser.add_argument("--t-emergency", type=float, default=None)
    parser.add_argument("--t-caution",   type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.kp          is not None: cfg["K_p"]                  = args.kp
    if args.kd          is not None: cfg["K_d"]                  = args.kd
    if args.v_nominal   is not None: cfg["v_nominal"]            = args.v_nominal
    if args.t_emergency is not None: cfg["threshold_emergency"]  = args.t_emergency
    if args.t_caution   is not None: cfg["threshold_caution"]    = args.t_caution

    run_pipeline(
        video_path   = args.video,
        masks_dir    = args.masks,
        output_csv   = args.output,
        output_video = args.output_video,
        visualise    = not args.no_viz,
        flow_every   = args.flow_every,
        config       = cfg,
    )
