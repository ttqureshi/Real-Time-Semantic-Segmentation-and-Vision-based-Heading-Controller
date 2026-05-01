"""
Velocity command generation based on optical-flow proximity scores.

Design
------
* Starts at v_nominal (configurable, default 0.5 m/s).
* Applies a three-tier response for obstacle/human proximity:
    - score < threshold_emergency  →  emergency stop  (v = 0)
    - score < threshold_caution    →  caution brake   (v *= brake_factor)
    - otherwise                    →  nominal speed
* Applies a two-tier response for speed-breaker proximity (lower priority):
    - score < threshold_emergency  →  crawl  (v *= sb_crawl_factor)
    - score < threshold_caution    →  slow   (v *= sb_brake_factor)
* Delegates yaw-rate computation to HeadingController.compute_omega_cmd.

Public API
----------
compute_velocity_commands(seg_mask, flow, distance_like, controller, config)
    -> (v_cmd, omega_cmd)
"""

from typing import Dict, Tuple

import numpy as np

from heading_controller import DEFAULT_CONFIG, HeadingController


def compute_velocity_commands(
    seg_mask: np.ndarray,
    flow: np.ndarray,
    distance_like: Dict[str, float],
    controller: HeadingController,
    config: dict,
) -> Tuple[float, float]:
    """
    Compute linear and angular velocity commands for one frame.

    Parameters
    ----------
    seg_mask      : (720, 1280) int  semantic segmentation mask
    flow          : (H_flow, W_flow, 2) float32  optical flow field
    distance_like : dict  {"obstacle", "human", "speed_breaker"} → [0, 1]
                    1.0 = far/safe, 0.0 = very close/dangerous
    controller    : HeadingController instance (carries PD state)
    config        : configuration dict (see heading_controller.DEFAULT_CONFIG)

    Returns
    -------
    v_cmd     : float  linear velocity  (m/s)
    omega_cmd : float  angular velocity (rad/s)
    """
    # -----------------------------------------------------------------
    # Resolve configuration values (fall back to defaults)
    # -----------------------------------------------------------------
    v_nominal: float = float(config.get("v_nominal",
                                        DEFAULT_CONFIG["v_nominal"]))
    t_emergency: float = float(config.get("threshold_emergency",
                                          DEFAULT_CONFIG["threshold_emergency"]))
    t_caution: float = float(config.get("threshold_caution",
                                        DEFAULT_CONFIG["threshold_caution"]))
    brake_factor: float = float(config.get("brake_factor",
                                           DEFAULT_CONFIG["brake_factor"]))
    sb_brake_factor: float = float(config.get("sb_brake_factor",
                                              DEFAULT_CONFIG["sb_brake_factor"]))
    sb_crawl_factor: float = float(config.get("sb_crawl_factor",
                                              DEFAULT_CONFIG["sb_crawl_factor"]))

    # -----------------------------------------------------------------
    # Extract proximity scores
    # -----------------------------------------------------------------
    d_obs: float = distance_like.get("obstacle", 1.0)
    d_hum: float = distance_like.get("human", 1.0)
    d_sb: float = distance_like.get("speed_breaker", 1.0)

    # -----------------------------------------------------------------
    # A. Default speed
    # -----------------------------------------------------------------
    v_cmd: float = v_nominal

    # -----------------------------------------------------------------
    # B. Obstacle / human response (highest priority)
    # -----------------------------------------------------------------
    if d_obs < t_emergency or d_hum < t_emergency:
        # Emergency stop
        v_cmd = 0.0
    elif d_obs < t_caution or d_hum < t_caution:
        # Caution: reduce speed
        v_cmd = v_nominal * brake_factor

    # -----------------------------------------------------------------
    # C. Speed-breaker response (only when not already stopped/braking)
    # -----------------------------------------------------------------
    if v_cmd > 0.0:
        if d_sb < t_emergency:
            v_cmd = min(v_cmd, v_nominal * sb_crawl_factor)
        elif d_sb < t_caution:
            v_cmd = min(v_cmd, v_nominal * sb_brake_factor)

    # -----------------------------------------------------------------
    # D. Heading controller (yaw rate)
    # -----------------------------------------------------------------
    omega_cmd, _ = controller.compute_omega_cmd(seg_mask)

    return float(v_cmd), float(omega_cmd)
