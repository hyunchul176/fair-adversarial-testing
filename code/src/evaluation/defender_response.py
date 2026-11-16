"""Observable, fixed-size defender response features from a safe probe rollout."""
from __future__ import annotations

import math

import numpy as np


RESPONSE_FEATURE_VERSION = "observable-defender-response-v1"
RESPONSE_FEATURE_NAMES = (
    "initial_speed_40",
    "final_speed_40",
    "speed_change_10",
    "mean_acceleration_5",
    "acceleration_std_5",
    "minimum_acceleration_5",
    "maximum_acceleration_5",
    "lateral_displacement_4",
    "maximum_abs_lateral_speed_5",
    "final_heading_0p5",
    "maximum_abs_heading_0p5",
    "normalized_forward_progress",
    "initial_gap_100",
    "final_gap_100",
    "minimum_gap_100",
    "maximum_closing_speed_30",
    "mean_closing_speed_30",
    "minimum_ttc_10",
)


def _state(row: dict, key: str) -> dict:
    value = row.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"trace row has no {key} state")
    required = ("x", "y", "vx", "vy", "heading")
    if any(name not in value or not math.isfinite(float(value[name])) for name in required):
        raise ValueError(f"invalid {key} state")
    return value


def summarize_observable_response(trace: list[dict], dt: float) -> dict:
    """Summarize only externally observable kinematics from a nominal probe."""
    if len(trace) < 2 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("response probe needs at least two finite-timestep samples")
    defender = [_state(row, "defender") for row in trace]
    generator = [_state(row, "generator") for row in trace]
    speeds = np.asarray([float(row["vx"]) for row in defender], dtype=np.float64)
    lateral = np.asarray([float(row["vy"]) for row in defender], dtype=np.float64)
    headings = np.asarray([float(row["heading"]) for row in defender], dtype=np.float64)
    acceleration = np.diff(speeds) / dt
    gaps = np.asarray([float(target["x"]) - float(actor["x"])
                       for actor, target in zip(generator, defender)], dtype=np.float64)
    closing = np.asarray([float(actor["vx"]) - float(target["vx"])
                          for actor, target in zip(generator, defender)], dtype=np.float64)
    positive = closing > 1e-6
    ttc = np.divide(gaps, closing, out=np.full_like(gaps, np.inf), where=positive)
    finite_ttc = ttc[np.isfinite(ttc) & (ttc >= 0)]
    minimum_ttc = float(np.min(finite_ttc)) if finite_ttc.size else 10.0
    duration = dt * max(1, len(trace) - 1)
    progress = float(defender[-1]["x"]) - float(defender[0]["x"])
    raw = np.asarray([
        speeds[0] / 40.0,
        speeds[-1] / 40.0,
        (speeds[-1] - speeds[0]) / 10.0,
        float(acceleration.mean()) / 5.0,
        float(acceleration.std()) / 5.0,
        float(acceleration.min()) / 5.0,
        float(acceleration.max()) / 5.0,
        (float(defender[-1]["y"]) - float(defender[0]["y"])) / 4.0,
        float(np.max(np.abs(lateral))) / 5.0,
        headings[-1] / 0.5,
        float(np.max(np.abs(headings))) / 0.5,
        progress / max(1e-6, duration * 40.0),
        gaps[0] / 100.0,
        gaps[-1] / 100.0,
        float(gaps.min()) / 100.0,
        max(0.0, float(closing.max())) / 30.0,
        float(closing.mean()) / 30.0,
        min(10.0, minimum_ttc) / 10.0,
    ], dtype=np.float64)
    if raw.shape != (len(RESPONSE_FEATURE_NAMES),) or not np.isfinite(raw).all():
        raise ValueError("invalid response feature vector")
    return {
        "version": RESPONSE_FEATURE_VERSION,
        "names": list(RESPONSE_FEATURE_NAMES),
        "features": np.clip(raw, -1.0, 1.0).astype(np.float32).tolist(),
        "sample_count": len(trace),
        "duration_seconds": duration,
    }
