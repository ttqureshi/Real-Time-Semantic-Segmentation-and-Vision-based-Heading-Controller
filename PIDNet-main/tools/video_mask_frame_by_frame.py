#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import _init_paths
import models
from configs import config
from configs import update_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "background",
    "human",
    "obstacle",
    "road",
    "sidewalk",
    "speed_breaker",
]

# RGB colors tied to the label IDs used by the robotics_semantic dataset.
CLASS_COLORS_RGB = np.array(
    [
        [20, 24, 33],
        [51, 122, 183],
        [64, 145, 108],
        [229, 126, 49],
        [214, 48, 49],
        [243, 196, 15],
    ],
    dtype=np.uint8,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_path(path: str, must_exist: bool = False) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        cwd_candidate = Path.cwd() / candidate
        root_candidate = PROJECT_ROOT / candidate
        candidate = cwd_candidate if cwd_candidate.exists() else root_candidate
    candidate = candidate.resolve()
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained PIDNet model to a video frame by frame and save a masked video."
    )
    parser.add_argument(
        "--cfg",
        default="configs/robotics_semantic/pidnet_small_robotics_semantic_refined.yaml",
        help="PIDNet YAML config.",
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="checkpoint",
        default="output/robotics_semantic/pidnet_small_robotics_semantic_refined/best.pt",
        help="Trained model checkpoint path.",
    )
    parser.add_argument("--output", required=True, help="Output masked video path.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Mask opacity. 0 shows only the original frame, 1 shows only the mask.",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=0,
        help="Model input width. Default uses TEST.IMAGE_SIZE from the config.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=0,
        help="Model input height. Default uses TEST.IMAGE_SIZE from the config.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    parser.add_argument(
        "--mask-dir",
        default="",
        help="Optional directory to save the color mask PNG for each processed frame.",
    )
    parser.add_argument(
        "--label-dir",
        default="",
        help="Optional directory to save raw single-channel label-ID PNG masks.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N frames. Use this for quick tests. 0 means full video.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not draw the class legend on the output video.",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Optional JSON summary path. Default is output path with .json suffix.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Optional config overrides.",
    )

    args = parser.parse_args()
    args.cfg = str(resolve_path(args.cfg, must_exist=True))
    update_config(config, args)
    return args


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_bgr_frame(frame_bgr: np.ndarray, input_size: tuple[int, int]) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, input_size, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - MEAN) / STD
    chw = normalized.transpose(2, 0, 1).copy()
    return torch.from_numpy(chw).unsqueeze(0)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> int:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_state = model.state_dict()
    filtered_state = {}
    for raw_key, value in checkpoint.items():
        key = raw_key
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key in model_state and model_state[key].shape == value.shape:
            filtered_state[key] = value

    model_state.update(filtered_state)
    model.load_state_dict(model_state, strict=False)
    return len(filtered_state)


def build_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = models.pidnet.get_pred_model(config.MODEL.NAME, config.DATASET.NUM_CLASSES)
    loaded_tensors = load_checkpoint(model, checkpoint_path)
    if loaded_tensors == 0:
        raise RuntimeError(f"No tensors from {checkpoint_path} matched the PIDNet model.")
    model = model.to(device)
    model.eval()
    print(f"Loaded {loaded_tensors} tensors from {checkpoint_path}")
    return model


def predict_mask(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    input_size: tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    frame_h, frame_w = frame_bgr.shape[:2]
    tensor = normalize_bgr_frame(frame_bgr, input_size).to(device)
    logits = model(tensor)
    if isinstance(logits, (list, tuple)):
        output_index = config.TEST.OUTPUT_INDEX
        if output_index < 0:
            output_index = 1 if len(logits) > 1 else 0
        logits = logits[output_index]
    logits = F.interpolate(
        logits,
        size=(frame_h, frame_w),
        mode="bilinear",
        align_corners=config.MODEL.ALIGN_CORNERS,
    )
    return torch.argmax(logits.squeeze(0), dim=0).cpu().numpy().astype(np.uint8)


def colorize_mask(label_mask: np.ndarray) -> np.ndarray:
    safe_mask = np.clip(label_mask, 0, len(CLASS_COLORS_RGB) - 1)
    mask_rgb = CLASS_COLORS_RGB[safe_mask]
    return cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)


def draw_legend(frame_bgr: np.ndarray) -> np.ndarray:
    out = frame_bgr.copy()
    x0, y0 = 16, 16
    row_h = 24
    box_w = 212
    box_h = row_h * len(CLASS_NAMES) + 14
    cv2.rectangle(out, (x0 - 8, y0 - 8), (x0 + box_w, y0 + box_h), (18, 18, 18), -1)
    cv2.rectangle(out, (x0 - 8, y0 - 8), (x0 + box_w, y0 + box_h), (235, 235, 235), 1)
    for class_id, name in enumerate(CLASS_NAMES):
        y = y0 + class_id * row_h
        color_bgr = tuple(int(channel) for channel in CLASS_COLORS_RGB[class_id][::-1])
        cv2.rectangle(out, (x0, y), (x0 + 16, y + 16), color_bgr, -1)
        cv2.putText(
            out,
            name,
            (x0 + 25, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ("mp4v", "avc1", "XVID"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer for {output_path}")


def main() -> int:
    args = parse_args()
    video_path = resolve_path(args.video, must_exist=True)
    checkpoint_path = resolve_path(args.checkpoint, must_exist=True)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary) if args.summary else output_path.with_suffix(".json")
    mask_dir = resolve_path(args.mask_dir) if args.mask_dir else None
    label_dir = resolve_path(args.label_dir) if args.label_dir else None
    if mask_dir:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if label_dir:
        label_dir.mkdir(parents=True, exist_ok=True)

    input_w = args.input_width or int(config.TEST.IMAGE_SIZE[0])
    input_h = args.input_height or int(config.TEST.IMAGE_SIZE[1])
    input_size = (input_w, input_h)
    alpha = min(1.0, max(0.0, float(args.alpha)))
    device = choose_device(args.device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = open_video_writer(output_path, fps, width, height)
    model = build_model(checkpoint_path, device)

    class_pixel_totals = np.zeros(config.DATASET.NUM_CLASSES, dtype=np.int64)
    frames_with_class = np.zeros(config.DATASET.NUM_CLASSES, dtype=np.int64)
    processed = 0

    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            label_mask = predict_mask(model, frame_bgr, input_size, device)
            color_mask = colorize_mask(label_mask)
            blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_mask, alpha, 0.0)
            if not args.no_legend:
                blended = draw_legend(blended)

            cv2.putText(
                blended,
                f"Frame {processed + 1}",
                (16, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(blended)

            counts = np.bincount(label_mask.ravel(), minlength=config.DATASET.NUM_CLASSES)
            class_pixel_totals += counts[: config.DATASET.NUM_CLASSES]
            frames_with_class += (counts[: config.DATASET.NUM_CLASSES] > 0).astype(np.int64)

            if mask_dir:
                cv2.imwrite(str(mask_dir / f"frame_{processed + 1:06d}.png"), color_mask)
            if label_dir:
                cv2.imwrite(str(label_dir / f"frame_{processed + 1:06d}.png"), label_mask)

            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed}/{total_frames if total_frames > 0 else '?'} frames")
            if args.max_frames > 0 and processed >= args.max_frames:
                break

    cap.release()
    writer.release()

    total_pixels = int(class_pixel_totals.sum())
    class_summary = {}
    for class_id, name in enumerate(CLASS_NAMES[: config.DATASET.NUM_CLASSES]):
        pixels = int(class_pixel_totals[class_id])
        class_summary[name] = {
            "pixels": pixels,
            "pixel_percent": round((pixels / total_pixels * 100.0) if total_pixels else 0.0, 4),
            "frames_present": int(frames_with_class[class_id]),
        }

    summary = {
        "input_video": str(video_path),
        "model": str(checkpoint_path),
        "config": str(Path(args.cfg).resolve()),
        "output_video": str(output_path),
        "frames_processed": processed,
        "source_frames_reported": total_frames,
        "fps": fps,
        "frame_size": {"width": width, "height": height},
        "model_input_size": {"width": input_w, "height": input_h},
        "device": str(device),
        "alpha": alpha,
        "class_summary": class_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved masked video: {output_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
