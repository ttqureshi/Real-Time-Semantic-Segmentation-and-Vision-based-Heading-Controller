"""
Frame visualizer for the heading / velocity controller.

Renders a rich HUD on top of the raw video frame showing:
  - Colour-coded motion-state banner  (GO STRAIGHT, TURN LEFT/RIGHT,
    SLOW DOWN, BRAKE, STOP, …)
  - Lane-centre indicator line
  - Steering arrow (direction + magnitude)
  - Speed bar gauge
  - Proximity danger bars (obstacle / human / speed-breaker)
  - Segmentation colour overlay

Public API
----------
FrameRenderer(config)
    .render(frame_bgr, seg_mask, flow, v_cmd, omega_cmd,
            distance_like, frame_idx, fps)  ->  annotated_bgr (same size as frame)
get_motion_state(v_cmd, omega_cmd, distance_like, config)
    ->  (label: str, color_bgr: tuple)
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Semantic class colours (BGR)
# ---------------------------------------------------------------------------
_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (50,  50,  50),    # background    — dark grey
    1: (0,   0,  220),    # Human          — red
    2: (0,  130, 255),    # Obstacle       — orange
    3: (30, 180,  30),    # Road           — green
    4: (200, 200,  0),    # Sidewalk       — cyan-yellow
    5: (0,  220, 220),    # Speed Breaker  — yellow
}
_CLASS_NAMES: Dict[int, str] = {
    0: "BG", 1: "Human", 2: "Obstacle",
    3: "Road", 4: "Sidewalk", 5: "SpeedBreaker",
}

# ---------------------------------------------------------------------------
# Motion-state thresholds
# ---------------------------------------------------------------------------
_OMEGA_STRAIGHT_THRESH: float = 0.04   # rad/s  — below this = "straight"
_OMEGA_STRONG_THRESH:   float = 0.35   # rad/s  — above this = "sharp turn"


# ---------------------------------------------------------------------------
# Motion-state classifier
# ---------------------------------------------------------------------------

def get_motion_state(
    v_cmd: float,
    omega_cmd: float,
    distance_like: Dict[str, float],
    config: dict,
) -> Tuple[str, Tuple[int, int, int]]:
    """
    Classify the robot's current motion into a human-readable label and
    a BGR banner colour.

    Returns
    -------
    (label, bgr_color)
    """
    t_emg: float  = config.get("threshold_emergency", 0.25)
    v_nom: float  = config.get("v_nominal", 0.5)
    d_obs: float  = distance_like.get("obstacle", 1.0)
    d_hum: float  = distance_like.get("human", 1.0)
    d_sb:  float  = distance_like.get("speed_breaker", 1.0)

    # ---- Emergency stop ----
    if v_cmd == 0.0:
        if d_obs < t_emg or d_hum < t_emg:
            return "BRAKE — EMERGENCY", (0, 0, 220)        # vivid red
        return "STOP", (0, 0, 160)

    # ---- Crawl over speed-breaker ----
    if v_cmd <= v_nom * 0.25:
        return "CRAWL — SPEED BREAKER", (0, 200, 255)      # gold

    # ---- Speed-breaker slow ----
    if d_sb < config.get("threshold_caution", 0.5) and v_cmd < v_nom:
        if abs(omega_cmd) > _OMEGA_STRAIGHT_THRESH:
            side = "RIGHT" if omega_cmd > 0 else "LEFT"
            return f"SLOW DOWN + TURN {side}", (0, 160, 230)
        return "SLOW DOWN — SPEED BREAKER", (0, 180, 255)  # amber

    # ---- Obstacle / human caution ----
    if v_cmd < v_nom * 0.6 and (d_obs < 0.8 or d_hum < 0.8):
        if abs(omega_cmd) > _OMEGA_STRAIGHT_THRESH:
            side = "RIGHT" if omega_cmd > 0 else "LEFT"
            return f"BRAKE + TURN {side}", (0, 80, 255)
        return "BRAKE — OBSTACLE", (0, 60, 220)            # red-orange

    # ---- Steering ----
    if abs(omega_cmd) >= _OMEGA_STRONG_THRESH:
        side = "RIGHT" if omega_cmd > 0 else "LEFT"
        return f"SHARP TURN {side}", (255, 120, 0)          # blue-ish
    if abs(omega_cmd) >= _OMEGA_STRAIGHT_THRESH:
        side = "RIGHT" if omega_cmd > 0 else "LEFT"
        return f"TURN {side}", (255, 180, 0)                # cyan-blue

    # ---- Nominal forward ----
    return "GO STRAIGHT", (40, 180, 40)                     # green


# ---------------------------------------------------------------------------
# Helpers — drawing primitives
# ---------------------------------------------------------------------------

def _draw_banner(
    canvas: np.ndarray,
    text: str,
    color_bgr: Tuple[int, int, int],
    height: int = 56,
) -> None:
    """Filled banner across the top of the canvas."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (w, height), color_bgr, -1)
    # Slight dark overlay for legibility
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    # Text centred
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.05
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = (w - tw) // 2
    ty = height // 2 + th // 2
    cv2.putText(canvas, text, (tx, ty), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _draw_speed_bar(
    canvas: np.ndarray,
    v_cmd: float,
    v_nominal: float,
    x: int, y: int, bar_w: int = 200, bar_h: int = 22,
) -> None:
    """Horizontal speed bar (green = slow → red = fast)."""
    ratio = min(v_cmd / max(v_nominal, 1e-6), 1.0)
    filled = int(bar_w * ratio)
    g = int(255 * (1.0 - ratio))
    r = int(255 * ratio)
    # Background
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)
    # Filled portion
    cv2.rectangle(canvas, (x, y), (x + filled, y + bar_h), (0, g, r), -1)
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (180, 180, 180), 1)
    # Label
    label = f"SPEED  {v_cmd:.2f} / {v_nominal:.2f} m/s"
    cv2.putText(canvas, label, (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_proximity_bars(
    canvas: np.ndarray,
    distance_like: Dict[str, float],
    x: int, y: int, bar_w: int = 130, bar_h: int = 18, gap: int = 8,
) -> None:
    """
    Three vertical danger bars for obstacle / human / speed-breaker.
    score_norm = 1.0 (far) → green,  = 0.0 (close) → red.
    """
    items = [
        ("OBS",  distance_like.get("obstacle", 1.0),      (0, 130, 255)),
        ("HUM",  distance_like.get("human", 1.0),         (0,   0, 220)),
        ("S.B.", distance_like.get("speed_breaker", 1.0), (0, 220, 220)),
    ]
    cy = y
    for name, score, accent in items:
        # Danger = 1 - score
        danger = 1.0 - score
        filled = int(bar_w * danger)
        # Choose bar colour based on danger level
        if danger > 0.75:
            bar_col = (0, 0, 220)
        elif danger > 0.5:
            bar_col = (0, 80, 255)
        elif danger > 0.25:
            bar_col = (0, 200, 255)
        else:
            bar_col = (40, 180, 40)
        cv2.rectangle(canvas, (x, cy), (x + bar_w, cy + bar_h), (40, 40, 40), -1)
        if filled > 0:
            cv2.rectangle(canvas, (x, cy), (x + filled, cy + bar_h), bar_col, -1)
        cv2.rectangle(canvas, (x, cy), (x + bar_w, cy + bar_h), (160, 160, 160), 1)
        lbl = f"{name}  {score:.2f}"
        cv2.putText(canvas, lbl, (x, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cy += bar_h + gap


def _draw_steering_arrow(
    canvas: np.ndarray,
    omega_cmd: float,
    omega_max: float = 1.0,
    cx: int = 0, cy: int = 0, radius: int = 38,
) -> None:
    """
    Circular steering indicator: arrow pointing left or right,
    length proportional to |omega_cmd|.
    """
    # Draw dial circle
    cv2.circle(canvas, (cx, cy), radius, (60, 60, 60), -1)
    cv2.circle(canvas, (cx, cy), radius, (160, 160, 160), 2)

    ratio = omega_cmd / (omega_max + 1e-6)
    ratio = max(-1.0, min(1.0, ratio))

    # Arrow: horizontal, pointing right (positive omega) or left (negative)
    arrow_len = int(radius * 0.85 * abs(ratio))
    if arrow_len < 4:
        arrow_len = 4
    direction = 1 if omega_cmd >= 0 else -1
    tip_x = cx + direction * arrow_len
    tip_y = cy
    color = (255, 180, 0) if direction > 0 else (255, 100, 100)
    cv2.arrowedLine(canvas, (cx - direction * 4, cy), (tip_x, tip_y),
                    color, 3, tipLength=0.4)

    # Label
    label = "R" if omega_cmd > _OMEGA_STRAIGHT_THRESH else \
            "L" if omega_cmd < -_OMEGA_STRAIGHT_THRESH else "—"
    cv2.putText(canvas, label, (cx - 5, cy + radius + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_dashed_line(
    canvas: np.ndarray,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 10,
    gap_len: int = 6,
) -> None:
    """Draw a dashed segment between two points."""
    x0, y0 = p0
    x1, y1 = p1
    dist = int(np.hypot(x1 - x0, y1 - y0))
    if dist <= 0:
        return
    for start in range(0, dist, dash_len + gap_len):
        end = min(start + dash_len, dist)
        t0 = start / dist
        t1 = end / dist
        xa = int(x0 + (x1 - x0) * t0)
        ya = int(y0 + (y1 - y0) * t0)
        xb = int(x0 + (x1 - x0) * t1)
        yb = int(y0 + (y1 - y0) * t1)
        cv2.line(canvas, (xa, ya), (xb, yb), color, thickness, cv2.LINE_AA)


def _draw_lane_centre(
    canvas: np.ndarray,
    seg_mask: np.ndarray,
    lane_center_x_mask: float,
    lane_horizon_y_mask: float = 0.0,
) -> None:
    """
    Draw heading-reference rays from a common camera pivot at the bottom.

      - White dashed ray: camera/image forward reference
      - Green solid ray : lane-axis ray (sidewalk-parallel centreline endpoint)
    """
    h, w = canvas.shape[:2]
    mh, mw = seg_mask.shape[:2]
    scale_x = w / max(mw, 1)
    scale_y = h / max(mh, 1)

    # Shared origin (camera pivot) just above bottom info strip
    top_margin = 58
    pivot = (w // 2, h - 30)

    # ---- White dashed ray: straight-ahead reference ----------------------
    white_top = (w // 2, top_margin)
    _draw_dashed_line(canvas, pivot, white_top, (200, 200, 200), thickness=1)
    _line_label(canvas, white_top[0] + 4, top_margin + 12, "Image Ctr", (200, 200, 200))

    # ---- Green lane-axis ray: pivot → anchor → banner -------------------
    # lane_center_x_mask / lane_horizon_y_mask are the anchor point returned
    # by compute_lane_centerline() in mask coordinates.  Convert to canvas.
    anc_x = float(lane_center_x_mask * scale_x)
    anc_y = float(lane_horizon_y_mask * scale_y)

    # Direction vector: from pivot toward anchor
    dx = anc_x - pivot[0]
    dy = anc_y - pivot[1]   # negative = anchor is above pivot (normal)

    if abs(dy) > 1e-6 and dy < 0:
        # Extend the ray from pivot through anchor all the way to top_margin
        t = (top_margin - pivot[1]) / dy          # t > 1  (extend beyond anchor)
        end_x = int(pivot[0] + t * dx)
        end_y = top_margin
    elif abs(dy) <= 1e-6:
        # Horizontal ray — just go to anchor x
        end_x = int(anc_x)
        end_y = int(anc_y)
    else:
        # Anchor is below pivot — draw only to anchor (degenerate)
        end_x = int(anc_x)
        end_y = int(anc_y)

    end_x = int(np.clip(end_x, 0, w - 1))
    end_y = int(np.clip(end_y, top_margin, pivot[1] - 20))

    cv2.line(canvas, pivot, (end_x, end_y), (0, 255, 120), 2, cv2.LINE_AA)
    lbl_x = end_x + 4 if end_x < w - 112 else end_x - 108
    _line_label(canvas, lbl_x, top_margin + 12, "Road Axis", (0, 255, 120))

    # Mark the shared pivot
    cv2.circle(canvas, pivot, 4, (230, 230, 230), -1)
    cv2.circle(canvas, pivot, 4, (30, 30, 30), 1)


def _line_label(
    canvas: np.ndarray,
    x: int, y: int,
    text: str,
    color: Tuple[int, int, int],
) -> None:
    """Draw a small outlined text label — used for line annotations."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0),   thick + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color,        thick,     cv2.LINE_AA)


def _draw_info_row(
    canvas: np.ndarray,
    frame_idx: int,
    fps: float,
    e_psi: float,
    omega_cmd: float,
    y: int,
) -> None:
    """Single-line telemetry row at the bottom."""
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (0, y), (w, y + 26), (20, 20, 20), -1)
    txt = (f"Frame {frame_idx:5d}  |  FPS {fps:5.1f}  |  "
           f"e_psi {e_psi:+.1f} px  |  omega {omega_cmd:+.4f} rad/s")
    cv2.putText(canvas, txt, (10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main renderer class
# ---------------------------------------------------------------------------

class FrameRenderer:
    """
    Renders controller state onto a video frame and returns the annotated BGR
    image at the original frame resolution.

    Parameters
    ----------
    config : dict   controller config (for thresholds / nominal speed)
    seg_alpha : float  segmentation overlay transparency (0 = none, 1 = full)
    """

    def __init__(self, config: dict, seg_alpha: float = 0.40) -> None:
        self.config = config
        self.seg_alpha = seg_alpha
        self._v_nominal: float = float(config.get("v_nominal", 0.5))
        self._omega_max: float = float(config.get("omega_max", 1.0))

    def _colorize_mask(self, seg_mask: np.ndarray,
                       out_wh: Tuple[int, int]) -> np.ndarray:
        h, w = seg_mask.shape
        colour = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, bgr in _CLASS_COLORS.items():
            colour[seg_mask == cls_id] = bgr
        return cv2.resize(colour, out_wh, interpolation=cv2.INTER_NEAREST)

    def render(
        self,
        frame_bgr: np.ndarray,
        seg_mask: np.ndarray,
        v_cmd: float,
        omega_cmd: float,
        distance_like: Dict[str, float],
        frame_idx: int,
        fps: float,
        lane_center_x: Optional[float] = None,
        lane_horizon_y: Optional[float] = None,
        e_psi: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compose the full annotated frame.

        Parameters
        ----------
        frame_bgr      : (H, W, 3) uint8  raw video frame
        seg_mask       : (H_m, W_m) int   segmentation mask
        v_cmd          : float  linear velocity command (m/s)
        omega_cmd      : float  angular velocity command (rad/s)
        distance_like  : dict  proximity scores
        frame_idx      : int   current frame index
        fps            : float current processing FPS
        lane_center_x  : float (optional) lane centre column in mask coords
        lane_horizon_y : float (optional) row of the horizon estimate (mask coords)
        e_psi          : float (optional) controller heading error for telemetry row

        Returns
        -------
        canvas : (H, W, 3) uint8  annotated frame
        """
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()

        # ---- Segmentation overlay ----------------------------------------
        if self.seg_alpha > 0:
            colour_mask = self._colorize_mask(seg_mask, (w, h))
            cv2.addWeighted(colour_mask, self.seg_alpha,
                            canvas, 1.0 - self.seg_alpha, 0, canvas)

        # ---- Lane-centre line --------------------------------------------
        if lane_center_x is not None:
            _draw_lane_centre(canvas, seg_mask, lane_center_x,
                              lane_horizon_y_mask=lane_horizon_y or 0.0)

        # ---- Motion-state banner (top) -----------------------------------
        label, banner_color = get_motion_state(
            v_cmd, omega_cmd, distance_like, self.config
        )
        _draw_banner(canvas, label, banner_color, height=52)

        # ---- Bottom info strip -------------------------------------------
        epsi_draw = e_psi
        if epsi_draw is None:
            epsi_draw = (lane_center_x - seg_mask.shape[1] / 2.0) \
                        if lane_center_x is not None else 0.0
        _draw_info_row(canvas, frame_idx, fps, float(epsi_draw), omega_cmd, y=h - 28)

        # ---- Right-side panel (proximity bars) ---------------------------
        panel_w = 155
        panel_x = w - panel_w - 8
        panel_y = 62
        _draw_proximity_bars(canvas, distance_like,
                             x=panel_x, y=panel_y,
                             bar_w=panel_w, bar_h=20, gap=22)

        # ---- Speed bar (bottom-left) -------------------------------------
        _draw_speed_bar(canvas, v_cmd, self._v_nominal,
                        x=12, y=h - 68, bar_w=220, bar_h=22)

        # ---- Steering dial (bottom-right of speed bar area) -------------
        dial_cx = 270
        dial_cy = h - 57
        _draw_steering_arrow(canvas, omega_cmd, self._omega_max,
                             cx=dial_cx, cy=dial_cy, radius=36)

        # ---- Legend strip (class colours, top-right) --------------------
        self._draw_legend(canvas, x=w - panel_w - 8, y=panel_y + 110)

        return canvas

    def _draw_legend(self, canvas: np.ndarray, x: int, y: int) -> None:
        items = [
            (3, "Road"),
            (4, "Sidewalk"),
            (1, "Human"),
            (2, "Obstacle"),
            (5, "Spd.Breaker"),
        ]
        cy = y
        for cls_id, name in items:
            bgr = _CLASS_COLORS[cls_id]
            cv2.rectangle(canvas, (x, cy), (x + 14, cy + 12), bgr, -1)
            cv2.putText(canvas, name, (x + 18, cy + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
            cy += 18
