"""A deterministic longitudinal safety shield for physical defender v1.

The shield leaves steering untouched and only caps acceleration when the
closest laterally overlapping leader has a short headway or predicted contact.
It is intentionally separate from PhysicalDefenderEnv so v1 results remain
reproducible and the intervention can be reported as an explicit ablation.
"""
from __future__ import annotations

import numpy as np

from src.environments.physical_defender_env import PhysicalDefenderEnv
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc, following_headway

ENV_VERSION = "physical-defender-shield-v2"


def shield_action(
    ego: VehicleState,
    traffic: list[VehicleState],
    raw_action,
    *,
    ttc_threshold: float = 4.0,
    headway_threshold: float = 1.2,
    minimum_brake: float = 0.25,
):
    """Return an acceleration-capped action and auditable intervention data."""
    action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0).copy()
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Invalid shield input action")
    if ttc_threshold <= 0 or headway_threshold <= 0 or not 0 < minimum_brake <= 1:
        raise ValueError("Invalid shield threshold")

    leaders = []
    for vehicle in traffic:
        headway = following_headway(ego, vehicle)
        if headway is None:
            continue
        ttc = constant_velocity_ttc(ego, vehicle)
        leaders.append((headway, ttc, vehicle))
    if not leaders:
        return action, {"intervened": False, "headway": None, "ttc": None, "brake_cap": None}

    headway, ttc, _ = min(leaders, key=lambda item: item[0])
    headway_pressure = max(0.0, 1.0 - headway / headway_threshold)
    ttc_pressure = max(0.0, 1.0 - ttc / ttc_threshold) if ttc is not None else 0.0
    pressure = max(headway_pressure, ttc_pressure)
    if pressure <= 0:
        return action, {"intervened": False, "headway": headway, "ttc": ttc, "brake_cap": None}

    brake = max(minimum_brake, min(1.0, pressure))
    cap = -brake
    raw_acceleration = float(action[0])
    action[0] = min(action[0], cap)
    intervened = bool(action[0] < raw_acceleration - 1e-8)
    return action, {"intervened": intervened, "headway": headway, "ttc": ttc, "brake_cap": cap}


class ShieldedPhysicalDefenderEnv(PhysicalDefenderEnv):
    def __init__(self, *, ttc_threshold=4.0, headway_threshold=1.2, minimum_brake=0.25, **kwargs):
        super().__init__(**kwargs)
        self.ttc_threshold = float(ttc_threshold)
        self.headway_threshold = float(headway_threshold)
        self.minimum_brake = float(minimum_brake)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        info["environment_version"] = ENV_VERSION
        return observation, info

    def step(self, action):
        ego_vehicle = self._inner_env.unwrapped.vehicle
        ego = VehicleState.from_vehicle(ego_vehicle)
        traffic = [VehicleState.from_vehicle(v) for v in ego_vehicle.road.vehicles if v is not ego_vehicle]
        applied, shield = shield_action(
            ego,
            traffic,
            action,
            ttc_threshold=self.ttc_threshold,
            headway_threshold=self.headway_threshold,
            minimum_brake=self.minimum_brake,
        )
        observation, reward, terminated, truncated, info = super().step(applied)
        info.update(
            environment_version=ENV_VERSION,
            shield_intervened=shield["intervened"],
            shield_headway=shield["headway"],
            shield_ttc=shield["ttc"],
            shield_brake_cap=shield["brake_cap"],
            raw_action=np.asarray(action, dtype=np.float32).tolist(),
            applied_action=applied.tolist(),
        )
        return observation, reward, terminated, truncated, info
