"""
Vision-based heading (yaw-rate) controller.

Expected input format
---------------------
seg_mask : (720, 1280) int32  — pre-computed semantic segmentation mask
flow     : (360, 640,  2) float32 — dense optical flow at half resolution

Design
------
* Computes the horizontal centre-of-mass of the drivable region
  (road + sidewalk) from the segmentation mask.
* Feeds the resulting lateral error into a PD controller to produce
  an angular-velocity command omega_cmd (rad/s).
* Also computes per-class optical-flow proximity scores for obstacle,
  human, and speed-breaker, with temporal smoothing.
* Image centre is derived dynamically from seg_mask.shape[1] so the
  controller works correctly for any mask resolution.

Semantic class IDs (from _classes.csv)
---------------------------------------
0  background
1  Human
2  Obstacle
3  Road
4  Sidewalk
5  Speed Breaker

Public API
----------
HeadingController(config)
    .compute_omega_cmd(seg_mask)                           -> (omega_cmd, e_psi)
    .compute_omega_cmd_calibrated(seg_mask, new_K)         -> (omega_cmd, e_psi)
    .compute_distance_like_scores(seg_mask, flow)          -> dict
    .compute_lane_center(seg_mask)                         -> float (horizon x)
    .compute_lane_centerline(seg_mask)                     -> (x_horizon, x_bottom, y_horizon, y_bottom)
    .reset()
"""

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from optical_flow_module import align_mask_to_flow

# ---------------------------------------------------------------------------
# Semantic class constants
# ---------------------------------------------------------------------------
CLASS_BACKGROUND: int = 0
CLASS_HUMAN: int = 1
CLASS_OBSTACLE: int = 2
CLASS_ROAD: int = 3
CLASS_SIDEWALK: int = 4
CLASS_SPEED_BREAKER: int = 5

# ---------------------------------------------------------------------------
# Default configuration (all tunable parameters in one place)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    # Heading PD gains (pixel-error path)
    "K_p": 0.003,           # proportional gain  (rad/s per pixel of error)
    "K_d": 0.001,           # derivative gain    (rad/s per pixel/frame)
    "omega_max": 1.0,       # clamp |omega_cmd| to this value (rad/s)
    # Heading PD gains (calibrated / normalised-error path)
    # e = (road_cx - cx) / fx  [dimensionless ≈ radians lateral offset]
    # Defaults approximate K_p * fx (≈ 0.003 × 640) so behaviour is similar
    # to the pixel path out of the box.  Re-tune after calibration.
    "K_yaw":   1.5,         # proportional gain  (rad/s per rad of lateral error)
    "K_yaw_d": 0.4,         # derivative gain    (rad/s per rad/frame)
    # Proximity score calibration
    "max_calib_score": 50.0,  # score_rel upper bound; scores above this
                              # are clamped to 1.0 (= very far / safe)
    # Temporal smoothing window for distance-like scores
    "smooth_window": 5,
    # Velocity parameters (used by braking_logic but defined here as defaults)
    "v_nominal": 0.5,           # m/s
    "threshold_emergency": 0.25,
    "threshold_caution": 0.50,
    "brake_factor": 0.5,
    "sb_brake_factor": 0.7,
    "sb_crawl_factor": 0.2,
}


# ---------------------------------------------------------------------------
# HeadingController
# ---------------------------------------------------------------------------

class HeadingController:
    """
    PD yaw-rate controller that tracks the drivable-region centre.

    Parameters
    ----------
    config : dict, optional
        Override any key from DEFAULT_CONFIG.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg: dict = {**DEFAULT_CONFIG, **(config or {})}

        self.K_p: float = float(cfg["K_p"])
        self.K_d: float = float(cfg["K_d"])
        self.K_yaw: float = float(cfg["K_yaw"])
        self.K_yaw_d: float = float(cfg["K_yaw_d"])
        self.omega_max: float = float(cfg["omega_max"])
        self.max_calib_score: float = float(cfg["max_calib_score"])
        self.smooth_window: int = int(cfg["smooth_window"])

        # PD controller state
        self._e_psi_prev: float = 0.0

        # Per-class smoothing buffers: class_name -> deque of raw score_rel values
        self._score_buffers: Dict[str, deque] = {
            "obstacle": deque(maxlen=self.smooth_window),
            "human": deque(maxlen=self.smooth_window),
            "speed_breaker": deque(maxlen=self.smooth_window),
        }

    # ------------------------------------------------------------------
    # Shared row-midpoint helper  (used by both control and visualisation)
    # ------------------------------------------------------------------

    def _collect_row_mids(
        self,
        seg_mask: np.ndarray,
        ya: int,
        yb: int,
    ) -> list:
        """
        For each row in [ya, yb) return (y, mid_x) where mid_x is the
        midpoint between the inner edges of the left and right sidewalks.

        Falls back to road left/right boundaries for rows where one or both
        sidewalks are absent.

        Returns [] when fewer than 8 valid rows are found.
        """
        H, W = seg_mask.shape[:2]
        side = (seg_mask == CLASS_SIDEWALK)
        road = (seg_mask == CLASS_ROAD)
        cx   = W / 2.0
        min_px = max(6, int(0.008 * W))
        out = []

        for y in range(max(0, ya), min(H, yb)):
            xs_s = np.where(side[y])[0]
            ls = xs_s[xs_s < cx]
            rs = xs_s[xs_s >= cx]

            if ls.size >= min_px and rs.size >= min_px:
                xl = float(ls.max())    # right edge of left sidewalk
                xr = float(rs.min())    # left  edge of right sidewalk
                if xr - xl > 0.06 * W:
                    out.append((float(y), 0.5 * (xl + xr)))
                    continue

            # Sidewalk not fully visible → use road boundaries
            xs_r = np.where(road[y])[0]
            if xs_r.size >= max(min_px, int(0.06 * W)):
                xl = float(xs_r.min())
                xr = float(xs_r.max())
                if xr - xl > 0.06 * W:
                    out.append((float(y), 0.5 * (xl + xr)))

        return out

    # ------------------------------------------------------------------
    # Lane-centre tracking
    # ------------------------------------------------------------------

    def compute_lane_center(self, seg_mask: np.ndarray) -> float:
        """
        Lane centre used for e_psi (control error) and visualization.

        Uses the SAME far-band midpoint as the green "Road Axis" ray so the
        control command direction always matches the visual ray direction:

          ray leans right  →  e_psi > 0  →  turn right  ✓
          ray leans left   →  e_psi < 0  →  turn left   ✓

        Why far-band (y ∈ [0.35H, 0.55H]):
          A heading controller should minimize the FUTURE cross-track error,
          not the current one.  Sampling a look-ahead band naturally accounts
          for road curvature ahead, making the robot anticipate turns rather
          than react late.

        Fallback chain:
          far-band  →  near-band  →  row-weighted road CoM
        """
        H, W = seg_mask.shape[:2]
        mask_center_x = W / 2.0

        # ---- Primary: far look-ahead band --------------------------------
        far_pts = self._collect_row_mids(
            seg_mask, ya=int(0.35 * H), yb=int(0.55 * H))
        if len(far_pts) >= 8:
            return float(np.median([p[1] for p in far_pts]))

        # ---- Secondary: near band ----------------------------------------
        near_pts = self._collect_row_mids(
            seg_mask, ya=int(0.72 * H), yb=int(0.96 * H))
        if len(near_pts) >= 8:
            return float(np.median([p[1] for p in near_pts]))

        # ---- Final fallback: row-weighted road CoM -----------------------
        y_idx, x_idx = np.where(seg_mask == CLASS_ROAD)
        if x_idx.size == 0:
            y_idx, x_idx = np.where(seg_mask == CLASS_SIDEWALK)
            if x_idx.size == 0:
                return mask_center_x
        weights = (y_idx / max(H - 1, 1)) ** 2
        w_sum = weights.sum()
        if w_sum < 1e-9:
            return float(np.mean(x_idx))
        return float(np.average(x_idx, weights=weights))

    def compute_lane_centerline(
        self, seg_mask: np.ndarray
    ) -> Tuple[float, float]:
        """
        Estimate a lane direction line for visualisation.

        Returns
        -------
        line_x : float
            X-coordinate of the line endpoint near the top of frame (mask coords).
        line_y : float
            Y-coordinate of the line endpoint (typically 0).

        Notes
        -----
        Priority order:
          1) Sidewalk-parallel centreline (requested behaviour)
          2) Road-boundary vanishing-point fallback
          3) Top-centre fallback
        """
        line = self._estimate_sidewalk_parallel_centerline(seg_mask)
        if line is not None:
            return line
        vp = self._estimate_lane_vanishing_point(seg_mask)
        if vp is not None:
            return vp
        return self.compute_lane_center(seg_mask), 0.0

    def _estimate_sidewalk_parallel_centerline(
        self, seg_mask: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Lane-axis direction from per-row sidewalk/road midpoints.

        Uses the shared _collect_row_mids() helper so the visualisation ray
        and the control error e_psi are derived from identical geometry.

        Returns (x_far, y_far) — a concrete image-space anchor in the far
        band that the visualiser extends toward the banner.
        """
        H, W = seg_mask.shape[:2]

        near_pts = self._collect_row_mids(seg_mask, int(0.72 * H), int(0.96 * H))
        far_pts  = self._collect_row_mids(seg_mask, int(0.35 * H), int(0.55 * H))

        if len(near_pts) < 8 or len(far_pts) < 8:
            return None

        x_far = float(np.median([p[1] for p in far_pts]))
        y_far = float(np.median([p[0] for p in far_pts]))

        if not np.isfinite(x_far):
            return None

        x_far = float(np.clip(x_far, -0.1 * W, 1.1 * W))
        return x_far, y_far

    def _estimate_lane_vanishing_point(
        self, seg_mask: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate road vanishing point from left/right road boundaries.

        Method
        ------
        1) Find the row with the largest number of road pixels (dominant row).
        2) Keep only the contiguous road-support band around that dominant row.
           This rejects sparse false positives (e.g., tree tops mislabeled road).
        3) For each kept row, get left-most and right-most road pixel.
        4) Fit two lines x_left(y), x_right(y) in least-squares sense.
        5) Intersect them -> vanishing point.
        """
        H, W = seg_mask.shape[:2]
        road = (seg_mask == CLASS_ROAD)
        if int(road.sum()) < 100:
            return None

        y0 = int(0.20 * H)
        y1 = int(0.97 * H)
        min_row_pixels = max(10, int(0.02 * W))

        # ------------------------------------------------------------------
        # Dominant-row selection (user-requested robustness):
        # use the strongest road-support row as an anchor and only keep the
        # contiguous band around it. This suppresses tiny disconnected regions
        # such as tree tops accidentally segmented as road.
        # ------------------------------------------------------------------
        row_counts = road.sum(axis=1).astype(np.int32)
        if y1 <= y0:
            return None
        roi_counts = row_counts[y0:y1]
        if roi_counts.size == 0:
            return None

        peak_off = int(np.argmax(roi_counts))
        peak_y = y0 + peak_off
        peak_count = int(row_counts[peak_y])
        if peak_count < min_row_pixels:
            return None

        band_thresh = max(min_row_pixels, int(0.15 * peak_count))
        fit_thresh = max(min_row_pixels, int(0.22 * peak_count))

        # Find contiguous band around the dominant row.
        y_top = peak_y
        while y_top > y0 and row_counts[y_top - 1] >= band_thresh:
            y_top -= 1
        y_bot = peak_y
        while y_bot < (y1 - 1) and row_counts[y_bot + 1] >= band_thresh:
            y_bot += 1

        ys = []
        lefts = []
        rights = []
        for y in range(y_top, y_bot + 1):
            xs = np.where(road[y])[0]
            if xs.size < fit_thresh:
                continue
            ys.append(float(y))
            lefts.append(float(xs.min()))
            rights.append(float(xs.max()))

        if len(ys) < 25:
            # Fallback: broaden to all ROI rows (legacy behavior) so VP still
            # exists in difficult frames where the dominant band is too narrow.
            ys = []
            lefts = []
            rights = []
            for y in range(y0, y1):
                xs = np.where(road[y])[0]
                if xs.size < min_row_pixels:
                    continue
                ys.append(float(y))
                lefts.append(float(xs.min()))
                rights.append(float(xs.max()))
            if len(ys) < 25:
                return None

        y_arr = np.asarray(ys, dtype=np.float64)
        l_arr = np.asarray(lefts, dtype=np.float64)
        r_arr = np.asarray(rights, dtype=np.float64)

        # Linear fits: x = m*y + b
        m_l, b_l = np.polyfit(y_arr, l_arr, 1)
        m_r, b_r = np.polyfit(y_arr, r_arr, 1)

        denom = (m_l - m_r)
        if abs(denom) < 1e-6:
            return None

        y_vp = (b_r - b_l) / denom
        x_vp = m_l * y_vp + b_l

        if not (np.isfinite(x_vp) and np.isfinite(y_vp)):
            return None

        # Basic sanity checks:
        # - vp should be above most of the observed lane body
        # - avoid absurd outliers that make the ray unstable
        if y_vp > 0.85 * H:
            return None
        x_vp = float(np.clip(x_vp, -0.5 * W, 1.5 * W))
        y_vp = float(np.clip(y_vp, -0.6 * H, 0.85 * H))
        return x_vp, y_vp

    # ------------------------------------------------------------------
    # PD heading controller
    # ------------------------------------------------------------------

    def compute_omega_cmd(
        self, seg_mask: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute the yaw-rate command from the segmentation mask.

        omega_cmd = K_p * e_psi + K_d * (e_psi - e_psi_prev)

        e_psi is computed by compute_lane_center() which uses the same
        sidewalk-inner-edge midpoint geometry as the green "Road Axis" ray
        in the visualiser, so the error driving the controller and the
        visual indicator are always consistent.

        Parameters
        ----------
        seg_mask : (720, 1280) int32  segmentation mask (any resolution)

        Returns
        -------
        omega_cmd : float  angular velocity command (rad/s), clamped to
                           [-omega_max, +omega_max]
        e_psi     : float  lateral heading error in pixels (signed)
                           positive = road centre is to the right of image centre
        """
        lane_center_x = self.compute_lane_center(seg_mask)
        image_center_x = seg_mask.shape[1] / 2.0
        e_psi = lane_center_x - image_center_x

        omega_cmd = (
            self.K_p * e_psi
            + self.K_d * (e_psi - self._e_psi_prev)
        )
        omega_cmd = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))

        self._e_psi_prev = e_psi
        return omega_cmd, e_psi

    def compute_omega_cmd_calibrated(
        self,
        seg_mask: np.ndarray,
        new_K: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute yaw-rate using calibrated camera intrinsics (undistorted frame).

        Heading error is normalised by focal length so it is scale-invariant
        and approximately in radians:

            e = (road_cx - cx) / fx

        where road_cx is the lane centre column, cx = new_K[0,2],
        fx = new_K[0,0] (from getOptimalNewCameraMatrix).

        Parameters
        ----------
        seg_mask : (H, W) int  segmentation mask (any resolution)
        new_K    : (3, 3) float  calibrated camera matrix for undistorted frame

        Returns
        -------
        omega_cmd : float  angular velocity command (rad/s)
        e_psi     : float  normalised lateral heading error (dimensionless ≈ rad)
                           positive = lane centre is to the right of optical axis
        """
        road_cx = self.compute_lane_center(seg_mask)
        cx = float(new_K[0, 2])
        fx = float(new_K[0, 0])
        e_psi = (road_cx - cx) / fx

        omega_cmd = (
            self.K_yaw * e_psi
            + self.K_yaw_d * (e_psi - self._e_psi_prev)
        )
        omega_cmd = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))

        self._e_psi_prev = e_psi
        return omega_cmd, e_psi

    # ------------------------------------------------------------------
    # Optical-flow-based proximity scores
    # ------------------------------------------------------------------

    def compute_distance_like_scores(
        self,
        seg_mask: np.ndarray,
        flow: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estimate how close each class of interest is using optical-flow
        magnitude relative to the road region.

        Intuition: under forward ego-motion, closer objects produce larger
        flow magnitudes; farther objects produce smaller ones.

        score_norm = 1.0  →  far / safe
        score_norm = 0.0  →  very close / dangerous

        Parameters
        ----------
        seg_mask : (720, 1280) int32  segmentation mask
        flow     : (360,  640, 2) float32  optical flow at half resolution

        Returns
        -------
        distance_like : dict  {"obstacle", "human", "speed_breaker"} → [0, 1]
        """
        flow_h, flow_w = flow.shape[:2]
        mask_f = align_mask_to_flow(seg_mask, (flow_h, flow_w))

        # ----------------------------------------------------------------
        # Road reference magnitude
        # ----------------------------------------------------------------
        road_vecs = flow[mask_f == CLASS_ROAD]
        if road_vecs.size > 0:
            road_mean = float(np.mean(np.linalg.norm(road_vecs, axis=1)))
        else:
            road_mean = 1.0
        road_mean = max(road_mean, 0.5)   # guard against near-zero motion frames

        classes: Dict[str, int] = {
            "obstacle":      CLASS_OBSTACLE,
            "human":         CLASS_HUMAN,
            "speed_breaker": CLASS_SPEED_BREAKER,
        }

        distance_like: Dict[str, float] = {}
        for name, cls_id in classes.items():
            vecs = flow[mask_f == cls_id]
            if vecs.size == 0:
                # Class not visible → maximally safe (score = 1.0)
                raw_score = 1.0
            else:
                mean_mag = float(np.mean(np.linalg.norm(vecs, axis=1)))
                # Road-relative score:
                #   mean_mag == road_mean  → score = 1.0  (same depth as road)
                #   mean_mag == 2×road     → score = 0.5  (caution zone)
                #   mean_mag == 4×road     → score = 0.25 (emergency threshold)
                # This is self-calibrating: no manual max_calib_score needed.
                raw_score = min(road_mean / (mean_mag + 1e-4), 1.0)

            # Temporal smoothing over the last smooth_window frames
            buf = self._score_buffers[name]
            buf.append(raw_score)
            distance_like[name] = float(np.mean(buf))

        return distance_like

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset PD state and all temporal smoothing buffers."""
        self._e_psi_prev = 0.0
        for buf in self._score_buffers.values():
            buf.clear()
