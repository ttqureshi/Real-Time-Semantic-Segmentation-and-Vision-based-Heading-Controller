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


CLASS_NAMES = [
    "background",
    "human",
    "obstacle",
    "road",
    "sidewalk",
    "speed_breaker",
]

# Label colors stay tied to numeric IDs so they match the annotated masks.
CLASS_COLORS = np.array(
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

BOX_CLASSES = {
    1: {"min_area": 90, "min_score": 0.40},
    2: {"min_area": 280, "min_score": 0.45},
    5: {"min_area": 80, "min_score": 0.35},
}

NOISE_FILTERS = {
    1: 50,
    2: 120,
    5: 40,
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refined PIDNet robotics segmentation on a video.")
    parser.add_argument(
        "--cfg",
        default="configs/robotics_semantic/pidnet_small_robotics_semantic_refined.yaml",
        type=str,
        help="PIDNet config file.",
    )
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to the annotated output video.",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/robotics_semantic/pidnet_small_robotics_semantic_refined/best.pt",
        type=str,
        help="Fine-tuned checkpoint path.",
    )
    parser.add_argument(
        "--alpha",
        default=0.42,
        type=float,
        help="Overlay alpha for the segmentation mask.",
    )
    parser.add_argument(
        "--resize-width",
        default=0,
        type=int,
        help="Optional inference width. Set 0 to derive sizes from the source frame width.",
    )
    parser.add_argument(
        "--resize-height",
        default=0,
        type=int,
        help="Optional inference height. Set 0 to derive sizes from the source frame height.",
    )
    parser.add_argument(
        "--scales",
        default="0.85,1.0,1.15",
        type=str,
        help="Comma-separated multi-scale inference factors.",
    )
    parser.add_argument(
        "--flip-test",
        action="store_true",
        help="Average predictions from original and horizontally flipped inputs.",
    )
    parser.add_argument(
        "--temporal-alpha",
        default=0.65,
        type=float,
        help="EMA factor for temporal smoothing across adjacent frames.",
    )
    parser.add_argument(
        "--max-frames",
        default=0,
        type=int,
        help="Optional cap for smoke tests. Set 0 to process the full video.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Additional config overrides.",
    )
    args = parser.parse_args()
    update_config(config, args)
    return args


def normalize_rgb(rgb: np.ndarray) -> torch.Tensor:
    normalized = rgb.astype(np.float32) / 255.0
    normalized = (normalized - MEAN) / STD
    normalized = normalized.transpose((2, 0, 1)).copy()
    return torch.from_numpy(normalized).unsqueeze(0)


def parse_scales(raw_scales: str) -> list[float]:
    scales = sorted({float(item.strip()) for item in raw_scales.split(",") if item.strip()})
    if not scales:
        raise ValueError("At least one inference scale is required.")
    return scales


def scaled_size(
    width: int,
    height: int,
    scale: float,
    explicit_size: tuple[int, int] | None,
) -> tuple[int, int]:
    if explicit_size is not None:
        base_w, base_h = explicit_size
    else:
        base_w, base_h = width, height
    scaled_w = max(32, int(round(base_w * scale / 32.0) * 32))
    scaled_h = max(32, int(round(base_h * scale / 32.0) * 32))
    return scaled_w, scaled_h


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model_dict = model.state_dict()
    filtered = {}
    for key, value in checkpoint.items():
        stripped = key
        if stripped.startswith("model."):
            stripped = stripped[6:]
        if stripped in model_dict and value.shape == model_dict[stripped].shape:
            filtered[stripped] = value
    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(filtered)} tensors from {checkpoint_path}")
    return model


def build_model(device: torch.device, checkpoint_path: str) -> torch.nn.Module:
    model = models.pidnet.get_pred_model(config.MODEL.NAME, config.DATASET.NUM_CLASSES)
    model = load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def infer_once(
    model: torch.nn.Module,
    device: torch.device,
    frame_bgr: np.ndarray,
    target_size: tuple[int, int],
    output_size: tuple[int, int],
    flip_test: bool,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    tensor = normalize_rgb(resized).to(device)
    logits = model(tensor)
    logits = F.interpolate(
        logits,
        size=(output_size[1], output_size[0]),
        mode="bilinear",
        align_corners=config.MODEL.ALIGN_CORNERS,
    )

    if flip_test:
        flipped = normalize_rgb(resized[:, ::-1, :]).to(device)
        flipped_logits = model(flipped)
        flipped_logits = torch.flip(flipped_logits, dims=[3])
        flipped_logits = F.interpolate(
            flipped_logits,
            size=(output_size[1], output_size[0]),
            mode="bilinear",
            align_corners=config.MODEL.ALIGN_CORNERS,
        )
        logits = 0.5 * (logits + flipped_logits)

    probs = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()
    return probs


def infer_multiscale(
    model: torch.nn.Module,
    device: torch.device,
    frame_bgr: np.ndarray,
    base_size: tuple[int, int] | None,
    scales: list[float],
    flip_test: bool,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    accumulated = np.zeros((config.DATASET.NUM_CLASSES, height, width), dtype=np.float32)
    for scale in scales:
        target_size = scaled_size(width, height, scale, base_size)
        accumulated += infer_once(
            model=model,
            device=device,
            frame_bgr=frame_bgr,
            target_size=target_size,
            output_size=(width, height),
            flip_test=flip_test,
        )
    accumulated /= float(len(scales))
    return accumulated


def remove_small_components(prediction: np.ndarray) -> np.ndarray:
    refined = prediction.copy()
    for class_id, min_area in NOISE_FILTERS.items():
        binary = (refined == class_id).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_id in range(1, component_count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area >= min_area:
                continue
            component_mask = labels == component_id
            refined[component_mask] = 0
    return refined


def build_color_mask(prediction: np.ndarray) -> np.ndarray:
    rgb_mask = CLASS_COLORS[prediction]
    return cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)


def draw_legend(frame: np.ndarray) -> np.ndarray:
    legend = frame.copy()
    x0, y0 = 18, 18
    cv2.rectangle(legend, (x0 - 10, y0 - 10), (x0 + 220, y0 + 26 * len(CLASS_NAMES) + 12), (15, 15, 15), -1)
    cv2.rectangle(legend, (x0 - 10, y0 - 10), (x0 + 220, y0 + 26 * len(CLASS_NAMES) + 12), (255, 255, 255), 1)
    for index, name in enumerate(CLASS_NAMES):
        y = y0 + index * 26
        color_bgr = tuple(int(channel) for channel in CLASS_COLORS[index][::-1])
        cv2.rectangle(legend, (x0, y), (x0 + 18, y + 18), color_bgr, -1)
        cv2.putText(
            legend,
            name,
            (x0 + 28, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return legend


def component_boxes(
    prediction: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, float | int | str]]:
    boxes: list[dict[str, float | int | str]] = []
    for class_id, rules in BOX_CLASSES.items():
        binary = (prediction == class_id).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        class_prob = probabilities[class_id]
        for component_id in range(1, component_count):
            x, y, w, h, area = stats[component_id]
            if int(area) < int(rules["min_area"]):
                continue
            component_mask = labels == component_id
            score = float(class_prob[component_mask].mean())
            if score < float(rules["min_score"]):
                continue
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": int(area),
                    "score": score,
                }
            )
    return boxes


def draw_boxes(frame: np.ndarray, boxes: list[dict[str, float | int | str]]) -> np.ndarray:
    annotated = frame.copy()
    for box in boxes:
        class_id = int(box["class_id"])
        color = tuple(int(channel) for channel in CLASS_COLORS[class_id][::-1])
        x = int(box["x"])
        y = int(box["y"])
        w = int(box["w"])
        h = int(box["h"])
        label = f"{box['class_name']} {float(box['score']):.2f}"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y0 = max(0, y - text_h - 8)
        cv2.rectangle(annotated, (x, text_y0), (x + text_w + 8, text_y0 + text_h + 8), color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 4, text_y0 + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    args = parse_args()
    scales = parse_scales(args.scales)
    device = torch.device("cuda" if torch.cuda.is_available() and len(config.GPUS) > 0 else "cpu")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open input video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    inference_size = None
    if args.resize_width > 0 and args.resize_height > 0:
        inference_size = (args.resize_width, args.resize_height)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {output_path}")

    model = build_model(device=device, checkpoint_path=args.checkpoint)
    processed_frames = 0
    box_counts = {CLASS_NAMES[class_id]: 0 for class_id in BOX_CLASSES}
    smoothed_probs: np.ndarray | None = None

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            probs = infer_multiscale(
                model=model,
                device=device,
                frame_bgr=frame_bgr,
                base_size=inference_size,
                scales=scales,
                flip_test=args.flip_test,
            )
            if smoothed_probs is None:
                smoothed_probs = probs
            else:
                smoothed_probs = args.temporal_alpha * smoothed_probs + (1.0 - args.temporal_alpha) * probs

            pred = np.argmax(smoothed_probs, axis=0).astype(np.uint8)
            pred = remove_small_components(pred)

            mask_bgr = build_color_mask(pred)
            overlay = cv2.addWeighted(frame_bgr, 1.0 - args.alpha, mask_bgr, args.alpha, 0.0)
            boxes = component_boxes(prediction=pred, probabilities=smoothed_probs)
            for box in boxes:
                box_counts[str(box["class_name"])] += 1
            annotated = draw_boxes(overlay, boxes)
            annotated = draw_legend(annotated)
            cv2.putText(
                annotated,
                f"Frame {processed_frames + 1}/{total_frames if total_frames > 0 else '?'}",
                (18, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)

            processed_frames += 1
            if processed_frames % 25 == 0:
                print(f"Processed {processed_frames}/{total_frames if total_frames > 0 else '?'} frames")
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

    cap.release()
    writer.release()

    summary = {
        "video": str(Path(args.video).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output": str(output_path.resolve()),
        "frames_processed": processed_frames,
        "fps": fps,
        "frame_size": {"width": width, "height": height},
        "base_inference_size": {
            "width": inference_size[0] if inference_size is not None else width,
            "height": inference_size[1] if inference_size is not None else height,
        },
        "scales": scales,
        "flip_test": bool(args.flip_test),
        "temporal_alpha": float(args.temporal_alpha),
        "box_counts": box_counts,
        "box_classes": [CLASS_NAMES[class_id] for class_id in BOX_CLASSES],
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved annotated video to {output_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
