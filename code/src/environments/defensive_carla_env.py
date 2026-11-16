# -*- coding: utf-8 -*-
"""Defensive RL Environment for CARLA Town04.

This environment ports the highway-env DefensiveRL to CARLA Town04. It was written to
obtain checkpoints trained in the CARLA environment for the victim side of the
measurement model grid (AV × G × c × K).

The highway-env DefensiveRL takes 5-vehicle kinematics as input and
outputs acceleration and steering. This CARLA environment instead reuses the
21D ego-centric observation and the acceleration and lane_offset actions of
AdversarialCarlaEnv as they are. Only the reward function is redefined as a defensive one (collision avoidance + lane keeping + highway
speed maintenance).

Key differences:
    AdversarialCarlaEnv: the ego is the adversarial agent (RL-trained), reward for causing a collision
                          · BRS · severity tracking
    DefensiveCarlaEnv:   the ego is the victim AV (RL-trained), reward for avoiding collisions
                          · lane keeping · keeping highway speed

ARCHITECTURE:
    RL Agent (PPO) ──> action [acceleration, lane_offset] ──> Ego Vehicle
                          │ accel → throttle/brake (no speed PID)
                          │ lane_offset → PID lateral control
                          ▼
                 CARLA Town04 step (TrafficManager IDM+PID NPCs)
                          │
                          ▼
           obs (21D ego-centric), reward (defensive)

REWARD:
    R = R_collision + R_lane_keeping + R_speed
    - R_collision:    -10.0 on crash (terminal)
    - R_lane_keeping: +0.1 · (1 - |lateral_offset| / half_lane_width)
    - R_speed:        +0.1 if 20 ≤ ego_speed ≤ 30 m/s, else -0.1

This reward follows the highway-env DefensiveRL exactly.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.environments.adversarial_carla_env import AdversarialCarlaEnv

logger = logging.getLogger(__name__)


# DefensiveRL reward parameters (same as defensive_env.py)
DEFAULT_COLLISION_PENALTY = -10.0
DEFAULT_LANE_KEEPING_BONUS = 0.1
DEFAULT_SPEED_BONUS = 0.1
DEFAULT_TARGET_SPEED_LOW = 20.0   # m/s
DEFAULT_TARGET_SPEED_HIGH = 30.0  # m/s


class DefensiveCarlaEnv(AdversarialCarlaEnv):
    """Gymnasium environment that trains a defensive AV in CARLA Town04.

    Inherits AdversarialCarlaEnv to reuse the CARLA connection, NPCs, ego_controller and observation,
    and redefines only the reward function as a defensive one. scenario_type and c_level
    have no meaning in this environment (it is not adversarial) but are kept formally for
    compatibility with the parent class.

    The trained checkpoints can be used as the test-taker (AV) side of the measurement model grid.
    """

    def __init__(
        self,
        collision_penalty: float = DEFAULT_COLLISION_PENALTY,
        lane_keeping_bonus: float = DEFAULT_LANE_KEEPING_BONUS,
        speed_bonus: float = DEFAULT_SPEED_BONUS,
        target_speed_low: float = DEFAULT_TARGET_SPEED_LOW,
        target_speed_high: float = DEFAULT_TARGET_SPEED_HIGH,
        **kwargs: Any,
    ):
        """Initialise DefensiveCarlaEnv.

        Parameters
        ----------
        collision_penalty : float
            penalty on collision (default: -10.0)
        lane_keeping_bonus : float
            lane-keeping reward coefficient (default: 0.1)
        speed_bonus : float
            reward for keeping highway speed (default: 0.1)
        target_speed_low, target_speed_high : float
            target range of highway speed (default: 20~30 m/s)
        **kwargs : Any
            arguments of the parent class AdversarialCarlaEnv (CARLA host, port, town etc.)
        """
        super().__init__(**kwargs)
        self._collision_penalty = float(collision_penalty)
        self._lane_keeping_bonus = float(lane_keeping_bonus)
        self._speed_bonus = float(speed_bonus)
        self._target_speed_low = float(target_speed_low)
        self._target_speed_high = float(target_speed_high)
        logger.info(
            "DefensiveCarlaEnv initialised: collision_penalty=%.1f, lane_bonus=%.2f, "
            "speed_bonus=%.2f, target_speed=[%.1f, %.1f] m/s",
            self._collision_penalty, self._lane_keeping_bonus,
            self._speed_bonus, self._target_speed_low, self._target_speed_high,
        )

    def _compute_reward(self, ssm: Optional[dict] = None) -> Tuple[float, Dict[str, float]]:
        """Defensive reward function (same as the highway-env DefensiveRL).

        This function fully replaces the adversarial reward of AdversarialCarlaEnv (causing a collision + BRS +
        severity tracking).

        Returns
        -------
        (total_reward, component_dict)
        """
        r_collision = 0.0
        r_lane = 0.0
        r_speed = 0.0

        # --- collision penalty ---
        # uses AdversarialCarlaEnv's _ego_collision_history as is
        crashed = len(self._ego_collision_history) > 0
        if crashed:
            r_collision = self._collision_penalty

        # --- lane keeping ---
        # lateral offset between the ego position and the lane centre from the CARLA waypoint
        try:
            ego_loc = self.ego_vehicle.get_location()
            waypoint = self._map.get_waypoint(
                ego_loc,
                project_to_road=True,
                lane_type=self._carla.LaneType.Driving,
            )
            if waypoint is not None:
                # distance between the waypoint's lane centre and the ego position (lateral offset)
                wp_loc = waypoint.transform.location
                dx = ego_loc.x - wp_loc.x
                dy = ego_loc.y - wp_loc.y
                # project onto the direction perpendicular to the waypoint's lane direction
                wp_yaw_rad = np.radians(waypoint.transform.rotation.yaw)
                # right vector (perpendicular): (-sin(yaw), cos(yaw))
                lateral_offset = -np.sin(wp_yaw_rad) * dx + np.cos(wp_yaw_rad) * dy
                half_width = float(getattr(self, "_lane_width", 3.5)) / 2.0
                lane_centering = max(0.0, 1.0 - abs(lateral_offset) / max(half_width, 0.1))
                r_lane = self._lane_keeping_bonus * lane_centering
        except Exception as exc:
            logger.debug("lane_keeping reward computation failed (fallback 0): %s", exc)
            r_lane = 0.0

        # --- keeping highway speed ---
        try:
            vel = self.ego_vehicle.get_velocity()
            speed = float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
            if self._target_speed_low <= speed <= self._target_speed_high:
                r_speed = self._speed_bonus
            else:
                r_speed = -self._speed_bonus
        except Exception as exc:
            logger.debug("speed reward computation failed (fallback 0): %s", exc)
            r_speed = 0.0

        total = r_collision + r_lane + r_speed
        components = {
            "r_collision": r_collision,
            "r_lane_keeping": r_lane,
            "r_speed": r_speed,
        }
        return total, components
