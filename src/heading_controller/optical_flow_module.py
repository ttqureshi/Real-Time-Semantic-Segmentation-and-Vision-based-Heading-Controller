"""
Optical flow computation module.

Uses OpenCV Farneback dense optical flow at a downscaled resolution so that
the full pipeline can sustain ~30 FPS on a modern GPU.

Public API
----------
compute_optical_flow(prev_gray, curr_gray) -> flow (H, W, 2)
frame_to_gray(frame)                       -> gray (H, W)
resize_for_flow(frame)                     -> downscaled frame
align_mask_to_flow(seg_mask, flow_shape)   -> mask at flow resolution
flow_magnitude(flow)                       -> magnitude map (H, W)
visualize_flow(flow)                       -> HSV-encoded BGR image
"""

from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default processing resolution for optical flow.
# Exactly half the expected input resolution (1280×720), giving a good
# speed/accuracy trade-off while keeping the flow grid aligned with the
# segmentation mask aspect ratio.
# ---------------------------------------------------------------------------
FLOW_WIDTH: int  = 640   # half of input width  (1280 → 640)
FLOW_HEIGHT: int = 360   # half of input height  (720 → 360)

# Farneback algorithm parameters (tuned for real-time performance)
_FARNEBACK_PARAMS: dict = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def resize_for_flow(
    frame: np.ndarray,
    width: int = FLOW_WIDTH,
    height: int = FLOW_HEIGHT,
) -> np.ndarray:
    """Downscale a BGR/RGB frame to the flow resolution."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def frame_to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR or RGB frame to an 8-bit grayscale image."""
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Already grayscale
    return frame.astype(np.uint8)


def compute_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    params: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute dense Farneback optical flow between two grayscale frames.

    Parameters
    ----------
    prev_gray : (H, W) uint8  previous grayscale frame
    curr_gray : (H, W) uint8  current  grayscale frame
    params    : Farneback parameter dict; defaults to _FARNEBACK_PARAMS

    Returns
    -------
    flow : (H, W, 2) float32
        flow[..., 0] = horizontal displacement u
        flow[..., 1] = vertical   displacement v
    """
    if params is None:
        params = _FARNEBACK_PARAMS
    flow: np.ndarray = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, **params
    )
    return flow


def align_mask_to_flow(
    seg_mask: np.ndarray,
    flow_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Nearest-neighbour resize of a segmentation mask to the flow resolution.

    Parameters
    ----------
    seg_mask   : (H_seg, W_seg) integer class-label array
    flow_shape : (H_flow, W_flow)

    Returns
    -------
    mask_resized : (H_flow, W_flow) int32
    """
    h_flow, w_flow = flow_shape
    if seg_mask.shape == (h_flow, w_flow):
        return seg_mask.astype(np.int32)
    mask_resized = cv2.resize(
        seg_mask.astype(np.uint8),
        (w_flow, h_flow),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask_resized.astype(np.int32)


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Return per-pixel flow magnitude array of shape (H, W)."""
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """
    Encode a flow field as an HSV-coloured BGR image (hue = direction,
    value = magnitude).

    Returns
    -------
    bgr : (H, W, 3) uint8
    """
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32),
                               flow[..., 1].astype(np.float32))
    hsv[..., 0] = (ang * 90.0 / np.pi).astype(np.uint8)   # hue 0-179
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_flow_arrows(
    bgr_frame: np.ndarray,
    flow: np.ndarray,
    step: int = 32,
    scale: float = 3.0,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Overlay sparse flow arrows on a BGR frame (for debugging).

    Parameters
    ----------
    bgr_frame : (H, W, 3) uint8  background image (will be resized to flow)
    flow      : (H_f, W_f, 2) float
    step      : grid spacing in pixels
    scale     : arrow length multiplier
    color     : BGR arrow colour

    Returns
    -------
    canvas : (H_f, W_f, 3) uint8
    """
    h_f, w_f = flow.shape[:2]
    canvas = cv2.resize(bgr_frame, (w_f, h_f), interpolation=cv2.INTER_LINEAR)
    for y in range(0, h_f, step):
        for x in range(0, w_f, step):
            u, v = flow[y, x]
            tip = (int(x + u * scale), int(y + v * scale))
            cv2.arrowedLine(canvas, (x, y), tip, color, 1, tipLength=0.3)
    return canvas
