"""
Undistort one BGR frame using camera calibration,
run PIDNet, clean noisy detections, and return the class-id segmentation mask.

Use:
    from mask_from_frame import mask_from_frame

    mask = mask_from_frame(frame_bgr, output_size=(432, 432), apply_calibration=False)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent

SEG_DIR = ROOT / "src" / "segmentation"
PIDNET_ROOT = ROOT / "third_party" / "PIDNet"
PIDNET_LIB = PIDNET_ROOT / "lib"

# Clear wrong cached configs in notebooks
if "configs" in sys.modules:
    del sys.modules["configs"]

# Put PIDNet paths before local segmentation configs
for _p in (str(PIDNET_LIB), str(PIDNET_ROOT), str(SEG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from configs import config as pidnet_cfg  # noqa: E402
from configs import update_config  # noqa: E402
import models  # noqa: E402

from pidnet_inference import (  # noqa: E402
    choose_device,
    load_pidnet_checkpoint,
    predict_mask_logits,
)


# --------------------------------------------------
# Default files
# --------------------------------------------------
CALIB_NPZ = ROOT / "data" / "calibration" / "calibration_data.npz"
WEIGHTS = ROOT / "checkpoints" / "segmentation" / "pidNet_wieghts.pt"
CFG_YAML = ROOT / "src" / "segmentation" / "configs" / "pidnet_robotics_6class.yaml"


CLASS_NAMES = {
    0: "background",
    1: "Human",
    2: "Obstacle",
    3: "Road",
    4: "Sidewalk",
    5: "Speed Breaker",
}


# --------------------------------------------------
# Global cached model
# --------------------------------------------------
_model = None
_cfg = None  # (device, input_size, align_corners, out_idx)


def _load_pidnet_once() -> None:
    """
    Loads PIDNet model only once.
    """

    global _model, _cfg

    if _model is not None:
        return

    if not CFG_YAML.is_file():
        raise FileNotFoundError(f"PIDNet config YAML not found: {CFG_YAML}")

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"PIDNet weights not found: {WEIGHTS}")

    class _Args:
        def __init__(self) -> None:
            self.cfg = str(CFG_YAML)
            self.opts: list = []

    update_config(pidnet_cfg, _Args())

    device = choose_device()

    model = models.pidnet.get_pred_model(
        pidnet_cfg.MODEL.NAME,
        pidnet_cfg.DATASET.NUM_CLASSES,
    )

    loaded_tensors = load_pidnet_checkpoint(model, WEIGHTS, device)

    if loaded_tensors == 0:
        raise RuntimeError(f"No tensors loaded from weights file: {WEIGHTS}")

    model = model.to(device)
    model.eval()

    input_size = (
        int(pidnet_cfg.TEST.IMAGE_SIZE[0]),
        int(pidnet_cfg.TEST.IMAGE_SIZE[1]),
    )

    align_corners = bool(getattr(pidnet_cfg.MODEL, "ALIGN_CORNERS", True))
    out_idx = int(pidnet_cfg.TEST.OUTPUT_INDEX)

    _model = model
    _cfg = (device, input_size, align_corners, out_idx)

    print("PIDNet loaded successfully")
    print("Device:", device)
    print("Input size:", input_size)
    print("Number of classes:", pidnet_cfg.DATASET.NUM_CLASSES)


def _undistort_frame(
    frame_bgr: np.ndarray,
    calibration_npz: Path | str | None = None,
) -> np.ndarray:
    """
    Applies camera calibration/undistortion to original frame size.
    """

    cal_path = Path(calibration_npz or CALIB_NPZ)

    if not cal_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {cal_path}")

    cal = np.load(cal_path)

    K = cal["K"] if "K" in cal.files else cal["camera_matrix"]
    dist = cal["dist"] if "dist" in cal.files else cal["dist_coeffs"]

    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)

    h, w = frame_bgr.shape[:2]

    new_K, _ = cv2.getOptimalNewCameraMatrix(
        K,
        dist,
        (w, h),
        alpha=1,
        newImgSize=(w, h),
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        new_K,
        (w, h),
        cv2.CV_32FC1,
    )

    undistorted = cv2.remap(
        frame_bgr,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
    )

    return undistorted


def print_mask_class_counts(mask: np.ndarray) -> None:
    """
    Prints class pixel counts for debugging.
    """

    unique_values, counts = np.unique(mask, return_counts=True)

    print("\nClasses found in mask:")
    for class_id, count in zip(unique_values, counts):
        class_name = CLASS_NAMES.get(int(class_id), "unknown")
        print(f"Class ID: {int(class_id)} | Name: {class_name} | Pixels: {int(count)}")


def remove_small_regions_for_class(
    mask: np.ndarray,
    class_id: int,
    min_pixels: int,
    replace_with: int = 0,
) -> np.ndarray:
    """
    Removes small disconnected noisy regions for one class.
    """

    cleaned_mask = mask.copy()

    class_region = (mask == class_id).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        class_region,
        connectivity=8,
    )

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_pixels:
            cleaned_mask[labels == label_id] = replace_with

    return cleaned_mask


def remove_small_regions_for_class(
    mask: np.ndarray,
    class_id: int,
    min_pixels: int,
    replace_with: int = 0,
) -> np.ndarray:
    """
    Removes small disconnected regions for one class.
    """

    cleaned_mask = mask.copy()

    class_region = (mask == class_id).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        class_region,
        connectivity=8,
    )

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_pixels:
            cleaned_mask[labels == label_id] = replace_with

    return cleaned_mask


def clean_safety_classes_by_danger_zone(mask):
    """
    Strong cleaning for safety controller.

    It removes false Human / Obstacle / Speed Breaker detections
    if they are not inside the useful driving danger zone.

    Classes:
        1 = Human
        2 = Obstacle
        5 = Speed Breaker
    """

    cleaned_mask = mask.copy()

    h, w = cleaned_mask.shape[:2]

    # Only lower part of image is important for immediate robot safety
    danger_y_start = int(h * 0.45)

    # Class-specific rules
    rules = {
        1: {  # Human
            "min_area": 450,
            "min_height": 18,
            "must_touch_danger_zone": True,
        },
        2: {  # Obstacle
            "min_area": 450,
            "min_height": 15,
            "must_touch_danger_zone": True,
        },
        5: {  # Speed Breaker
            "min_area": 700,
            "min_width": 35,
            "must_touch_danger_zone": True,
        },
    }

    for class_id, rule in rules.items():
        class_region = (cleaned_mask == class_id).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            class_region,
            connectivity=8,
        )

        for label_id in range(1, num_labels):
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]

            bottom_y = y + height

            remove_component = False

            # Rule 1: remove small components
            if area < rule.get("min_area", 0):
                remove_component = True

            # Rule 2: remove too-flat / too-small height components
            if "min_height" in rule and height < rule["min_height"]:
                remove_component = True

            # Rule 3: speed breaker should have some horizontal width
            if "min_width" in rule and width < rule["min_width"]:
                remove_component = True

            # Rule 4: must appear in lower danger zone
            if rule.get("must_touch_danger_zone", False):
                if bottom_y < danger_y_start:
                    remove_component = True

            if remove_component:
                cleaned_mask[labels == label_id] = 0

    return cleaned_mask

def print_mask_class_counts(mask: np.ndarray) -> None:
    """
    Prints class pixel counts for debugging.
    """

    unique_values, counts = np.unique(mask, return_counts=True)

    print("\nClasses found in mask:")
    for class_id, count in zip(unique_values, counts):
        class_name = CLASS_NAMES.get(int(class_id), "unknown")
        print(f"Class ID: {int(class_id)} | Name: {class_name} | Pixels: {int(count)}")


def clean_safety_classes_by_danger_zone(
    mask: np.ndarray,
    danger_y_ratio: float = 0.55,
    danger_left_ratio: float = 0.20,
    danger_right_ratio: float = 0.80,
) -> np.ndarray:
    """
    Cleans false detections for safety classes using danger zone.

    It only keeps Human, Obstacle, and Speed Breaker detections if they are:
    1. large enough
    2. inside lower driving danger zone

    Classes:
        1 = Human
        2 = Obstacle
        5 = Speed Breaker
    """

    cleaned_mask = mask.copy()

    h, w = cleaned_mask.shape[:2]

    danger_y_start = int(h * danger_y_ratio)
    danger_x_start = int(w * danger_left_ratio)
    danger_x_end = int(w * danger_right_ratio)

    rules = {
        1: {  # Human
            "min_area": 900,
            "min_height": 25,
            "must_touch_danger_zone": True,
        },
        2: {  # Obstacle
            "min_area": 700,
            "min_height": 20,
            "must_touch_danger_zone": True,
        },
        5: {  # Speed Breaker
            "min_area": 900,
            "min_width": 40,
            "must_touch_danger_zone": True,
        },
    }

    for class_id, rule in rules.items():
        class_region = (cleaned_mask == class_id).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            class_region,
            connectivity=8,
        )

        for label_id in range(1, num_labels):
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]

            x_center = x + width // 2
            bottom_y = y + height

            remove_component = False

            # Remove small components
            if area < rule.get("min_area", 0):
                remove_component = True

            # Remove too small height objects
            if "min_height" in rule and height < rule["min_height"]:
                remove_component = True

            # Speed breaker should be wide
            if "min_width" in rule and width < rule["min_width"]:
                remove_component = True

            # Must be in lower driving zone
            if rule.get("must_touch_danger_zone", False):
                if bottom_y < danger_y_start:
                    remove_component = True

            # Must be near center driving region
            if x_center < danger_x_start or x_center > danger_x_end:
                remove_component = True

            if remove_component:
                cleaned_mask[labels == label_id] = 0

    return cleaned_mask


def mask_from_frame(
    frame_bgr: np.ndarray,
    calibration_npz: Path | str | None = None,
    apply_calibration: bool = False,
    output_size: tuple[int, int] | None = None,
    clean_mask: bool = True,
    debug: bool = False,
) -> np.ndarray:
    """
    Converts one BGR frame into a class-id mask.

    Parameters
    ----------
    frame_bgr:
        Original OpenCV BGR frame.
        Example shape: (720, 1280, 3)

    calibration_npz:
        Optional path to calibration_data.npz.

    apply_calibration:
        True = undistort frame before PIDNet.
        False = skip camera calibration.

    output_size:
        None = return mask same size as original frame.
        (432, 432) = return mask resized to 432x432.

    clean_mask:
        True = apply danger-zone cleaning.

    debug:
        True = print class counts before and after cleaning.

    Returns
    -------
    np.ndarray:
        2D class-id mask.
    """

    if frame_bgr is None:
        raise ValueError("frame_bgr is None")

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected BGR frame with shape (H, W, 3), got {frame_bgr.shape}"
        )

    _load_pidnet_once()

    assert _model is not None
    assert _cfg is not None

    device, input_size, align_corners, out_idx = _cfg

    original_h, original_w = frame_bgr.shape[:2]

    # 1. Optional calibration
    if apply_calibration:
        input_frame = _undistort_frame(
            frame_bgr=frame_bgr,
            calibration_npz=calibration_npz,
        )
    else:
        input_frame = frame_bgr

    # 2. Run PIDNet
    with torch.inference_mode():
        logits = predict_mask_logits(
            _model,
            input_frame,
            input_size,
            device,
            align_corners,
            out_idx,
        )

        logits = F.interpolate(
            logits,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=align_corners,
        )

        mask = torch.argmax(logits.squeeze(0), dim=0)
        mask = mask.cpu().numpy().astype(np.int32)

    # 3. Resize mask to 432x432 if needed
    if output_size is not None:
        mask = cv2.resize(
            mask,
            output_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

    if debug:
        print("\nBefore danger-zone cleaning:")
        print_mask_class_counts(mask)

    # 4. Correct cleaner used here
    if clean_mask:
        mask = clean_safety_classes_by_danger_zone(mask)

    if debug:
        print("\nAfter danger-zone cleaning:")
        print_mask_class_counts(mask)

    return mask.astype(np.int32)

# --------------------------------------------------
# Command line usage
# --------------------------------------------------
if __name__ == "__main__":
    # Examples:
    # python mask_from_frame.py frame.png mask_out.npy
    # python mask_from_frame.py frame.png mask_out.npy no_calib 432
    # python mask_from_frame.py frame.png mask_out.npy calib 432

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python mask_from_frame.py <image.png> [out.npy] [calib|no_calib] [432]\n\n"
            "Examples:\n"
            "  python mask_from_frame.py frame.png mask_out.npy\n"
            "  python mask_from_frame.py frame.png mask_out.npy no_calib 432\n"
            "  python mask_from_frame.py frame.png mask_out.npy calib 432\n\n"
            "Import usage:\n"
            "  from mask_from_frame import mask_from_frame\n"
            "  mask = mask_from_frame(frame, output_size=(432, 432), apply_calibration=False)\n"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    out_npy = sys.argv[2] if len(sys.argv) > 2 else "mask_out.npy"

    calib_mode = sys.argv[3] if len(sys.argv) > 3 else "no_calib"
    apply_calibration = calib_mode == "calib"

    output_size = None
    if len(sys.argv) > 4 and sys.argv[4] == "432":
        output_size = (432, 432)

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    mask = mask_from_frame(
        frame_bgr=frame,
        apply_calibration=False,
        output_size=(432, 432),
        clean_mask=True,
        debug=True,
    )

    np.save(out_npy, mask)

    print(f"\nSaved mask: {out_npy}")
    print("Mask shape:", mask.shape)
    print("Unique values:", np.unique(mask))