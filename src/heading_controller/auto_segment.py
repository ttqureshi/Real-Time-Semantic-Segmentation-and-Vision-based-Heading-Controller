"""
Heuristic per-frame semantic segmentation  (fallback for 1280×720 input).

Used automatically when no pre-computed .npy mask files are supplied.
When real .npy masks ARE provided they are loaded directly and this module
is never called.

Produces a (H, W) int32 mask with class labels matching _classes.csv:
    0 = background
    1 = Human
    2 = Obstacle
    3 = Road / navigable path
    4 = Sidewalk / path edge
    5 = Speed Breaker

Expected primary input
----------------------
frame_bgr : (720, 1280, 3) uint8  raw BGR video frame

Algorithm
---------
1. Dynamically sample the road colour from a small patch at the bottom-
   centre of the frame (assumed to always be navigable ground).
2. Compute per-pixel colour distance from the sampled road colour in
   LAB space (perceptually uniform → better colour matching).
3. Threshold + spatial prior → candidate road pixels.
4. Morphological close/open → clean connected road region.
5. Keep only the largest connected component that touches the bottom edge
   → the navigable path.  This shifts left/right as the robot turns,
   producing genuine heading-error variation.
6. Sidewalk / path-margin: a thin dilation halo around the road.
7. Obstacle detection: compact non-road blobs sitting on top of the road
   zone (e.g. people, bins, rocks).
8. Speed-breaker: a near-horizontal bright band across the road zone.
9. Everything above the estimated horizon → background.

Public API
----------
AutoSegmenter(config)
    .segment(frame_bgr) -> (H, W) int32 mask
segment_frame(frame_bgr) -> (H, W) int32 mask   [stateless one-shot call]
"""

from typing import Optional, Tuple

import cv2
import numpy as np

# ---- semantic class IDs ------------------------------------------------
CLASS_BACKGROUND: int = 0
CLASS_HUMAN:      int = 1
CLASS_OBSTACLE:   int = 2
CLASS_ROAD:       int = 3
CLASS_SIDEWALK:   int = 4
CLASS_SPEED_BREAKER: int = 5

# ---- default tuning knobs ----------------------------------------------
DEFAULT_SEG_CONFIG: dict = {
    # Fraction of frame height that defines the horizon
    "horizon_frac": 0.42,
    # Half-width / half-height of the seed patch at bottom-centre
    # (pixels in the *working* downscaled image, 640-wide ≈ half of 1280)
    "seed_patch_hw": 40,
    # Colour distance threshold in LAB space for road membership
    "road_color_thresh": 28.0,
    # Exponential smoothing of sampled road colour across frames (0=no smooth)
    "road_color_alpha": 0.15,
    # Minimum road blob area as fraction of lower-half area
    "min_road_frac": 0.03,
    # Morphology kernel sizes (at working resolution 640×360)
    "morph_close_k": 15,
    "morph_open_k":  9,
    # Sidewalk dilation (pixels at working res)
    "sidewalk_dil_k": 14,
    # Obstacle: min contour area (pixels² at working res)
    "obstacle_min_area": 80,
    # Speed-breaker: bright horizontal band relative to road mean V
    "sb_bright_factor": 1.25,
    "sb_band_min_width_frac": 0.30,   # fraction of frame width
    # Internal working resolution — exactly half of 1280×720, matches flow grid
    "work_width": 640,
}


class AutoSegmenter:
    """
    Stateful heuristic segmenter.  Maintains an exponential moving average
    of the sampled road colour across frames for temporal stability.

    Parameters
    ----------
    config : dict, optional   override any key from DEFAULT_SEG_CONFIG
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = {**DEFAULT_SEG_CONFIG, **(config or {})}
        self._horizon_frac:       float = cfg["horizon_frac"]
        self._seed_hw:            int   = cfg["seed_patch_hw"]
        self._road_thresh:        float = cfg["road_color_thresh"]
        self._alpha:              float = cfg["road_color_alpha"]
        self._min_road_frac:      float = cfg["min_road_frac"]
        self._close_k:            int   = cfg["morph_close_k"]
        self._open_k:             int   = cfg["morph_open_k"]
        self._sdil_k:             int   = cfg["sidewalk_dil_k"]
        self._obs_min:            int   = cfg["obstacle_min_area"]
        self._sb_bright:          float = cfg["sb_bright_factor"]
        self._sb_min_w:           float = cfg["sb_band_min_width_frac"]
        self._work_w:             int   = cfg["work_width"]

        self._road_lab: Optional[np.ndarray] = None   # smoothed road colour (LAB)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def segment(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a (H, W) int32 semantic mask for one frame.

        Parameters
        ----------
        frame_bgr : (H, W, 3) uint8  BGR video frame

        Returns
        -------
        mask : (H, W) int32  class labels 0-5
        """
        H, W = frame_bgr.shape[:2]

        # Downscale for speed
        scale   = self._work_w / W
        work_h  = int(H * scale)
        work_w  = self._work_w
        small   = cv2.resize(frame_bgr, (work_w, work_h),
                             interpolation=cv2.INTER_LINEAR)

        mask_small = self._segment_small(small)

        # Upscale back to original resolution (nearest to preserve labels)
        mask = cv2.resize(mask_small.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(np.int32)
        return mask

    # ------------------------------------------------------------------
    # Internal — operates at work_w × work_h resolution
    # ------------------------------------------------------------------

    def _segment_small(self, small: np.ndarray) -> np.ndarray:
        sh, sw = small.shape[:2]
        horizon_y = int(sh * self._horizon_frac)
        seed_hw   = self._seed_hw

        # ---- Convert to LAB (perceptually uniform colour space) ------
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)

        # ---- 1. Sample road colour from bottom-centre patch ----------
        cy = max(sh - seed_hw, sh - 1)
        cx = sw // 2
        y1 = max(cy - seed_hw, 0);   y2 = min(cy + seed_hw, sh)
        x1 = max(cx - seed_hw, 0);   x2 = min(cx + seed_hw, sw)
        patch_lab = lab[y1:y2, x1:x2].reshape(-1, 3)
        sample_lab = patch_lab.mean(axis=0)  # (L, a, b)

        # Exponential smoothing across frames
        if self._road_lab is None:
            self._road_lab = sample_lab.copy()
        else:
            self._road_lab = (1 - self._alpha) * self._road_lab \
                           + self._alpha * sample_lab

        # ---- 2. Colour distance map (LAB Euclidean) ------------------
        diff = np.linalg.norm(lab - self._road_lab, axis=2)   # (sh, sw)

        # ---- 3. Candidate road pixels --------------------------------
        road_cand = (diff < self._road_thresh).astype(np.uint8) * 255

        # Apply horizon spatial prior (no road above horizon)
        road_cand[:horizon_y] = 0

        # ---- 4. Morphological cleanup --------------------------------
        kc = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._close_k, self._close_k))
        ko = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._open_k,  self._open_k))
        road_cand = cv2.morphologyEx(road_cand, cv2.MORPH_CLOSE, kc)
        road_cand = cv2.morphologyEx(road_cand, cv2.MORPH_OPEN,  ko)

        # ---- 5. Largest connected component touching bottom edge -----
        road_binary = self._largest_bottom_component(road_cand, sh, sw)

        # ---- 6. Sidewalk: dilation halo around road ------------------
        ks = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._sdil_k, self._sdil_k))
        sidewalk_zone = cv2.dilate(road_binary, ks)
        sidewalk_mask = (sidewalk_zone > 0) & (road_binary == 0)
        sidewalk_mask[:horizon_y] = False

        # ---- 7. Obstacle: compact non-road blobs in lower zone -------
        obstacle_mask = self._detect_obstacles(
            small, road_binary, sidewalk_zone, horizon_y, sh, sw)

        # ---- 8. Speed breaker: bright horizontal band across road ----
        sb_mask = self._detect_speed_breaker(hsv, road_binary, sh, sw)

        # ---- 9. Assemble final mask ----------------------------------
        seg = np.zeros((sh, sw), dtype=np.int32)
        seg[:horizon_y] = CLASS_BACKGROUND
        seg[sidewalk_mask]          = CLASS_SIDEWALK
        seg[road_binary > 0]        = CLASS_ROAD
        seg[obstacle_mask]          = CLASS_OBSTACLE
        seg[sb_mask]                = CLASS_SPEED_BREAKER

        return seg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _largest_bottom_component(
        binary: np.ndarray, sh: int, sw: int
    ) -> np.ndarray:
        """
        Keep only the connected component(s) that touch the bottom row.
        Falls back to largest component if none touches the bottom.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        if num_labels <= 1:
            return binary

        result = np.zeros_like(binary)
        bottom_row_labels = set(labels[sh - 1, :].tolist()) - {0}

        if bottom_row_labels:
            for lbl in bottom_row_labels:
                result[labels == lbl] = 255
        else:
            # Fallback: use largest non-background component
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = int(np.argmax(areas)) + 1
            result[labels == largest] = 255

        return result

    def _detect_obstacles(
        self,
        small:        np.ndarray,
        road_bin:     np.ndarray,
        sidewalk_zone: np.ndarray,
        horizon_y:    int,
        sh: int, sw: int,
    ) -> np.ndarray:
        """
        Detect compact non-road blobs sitting on or above the road zone.
        Uses edge density + non-road colour as cues.
        """
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)

        # Non-road, non-sidewalk area within the lower half
        non_road = ((road_bin == 0) & (sidewalk_zone == 0)).astype(np.uint8) * 255
        non_road[:horizon_y] = 0

        # Combine with edge density
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_dense = cv2.dilate(edges, ke) & non_road

        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blobs = cv2.morphologyEx(edge_dense, cv2.MORPH_CLOSE, ko)

        # Filter by size
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            blobs, connectivity=8)
        result = np.zeros((sh, sw), dtype=bool)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self._obs_min:
                result[labels == i] = True

        return result

    def _detect_speed_breaker(
        self,
        hsv:      np.ndarray,
        road_bin: np.ndarray,
        sh: int, sw: int,
    ) -> np.ndarray:
        """
        Detect a near-horizontal bright stripe across the road region
        (speed-breaker / road marking).
        """
        V = hsv[:, :, 2]

        # Road-region V statistics
        road_v = V[road_bin > 0]
        if road_v.size == 0:
            return np.zeros((sh, sw), dtype=bool)
        road_mean_v = road_v.mean()
        road_std_v  = road_v.std()

        # Bright pixels well above mean road brightness
        bright_thresh = road_mean_v + max(road_std_v * 1.8, 25.0)
        bright_mask = (V > bright_thresh) & (road_bin > 0)

        # Keep only rows where bright pixels span a wide fraction of road width
        result = np.zeros((sh, sw), dtype=bool)
        min_span_px = int(self._sb_min_w * sw)

        for row in range(sh):
            row_bright = bright_mask[row]
            if row_bright.sum() >= min_span_px:
                result[row, row_bright] = True

        # Small morphological cleanup
        if result.any():
            kb = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
            result = cv2.morphologyEx(
                result.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kb) > 0

        return result

    def reset(self) -> None:
        """Clear the smoothed road-colour state (call at start of new video)."""
        self._road_lab = None


# ---------------------------------------------------------------------------
# Stateless convenience wrapper
# ---------------------------------------------------------------------------
_default_segmenter: Optional[AutoSegmenter] = None


def segment_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    One-shot heuristic segmentation (uses a module-level AutoSegmenter).
    Convenient for quick tests; prefer AutoSegmenter directly in pipelines.
    """
    global _default_segmenter
    if _default_segmenter is None:
        _default_segmenter = AutoSegmenter()
    return _default_segmenter.segment(frame_bgr)
