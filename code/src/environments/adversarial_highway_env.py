"""
Adversarial Highway-Env Environment for Scenario Generation

PURPOSE:
    Gymnasium environment wrapping highway-env where an RL agent controls the
    ego vehicle as an adversarial driver model.  This is a lightweight
    replacement for the CARLA-based environment (adversarial_carla_env.py),
    providing orders-of-magnitude faster iteration while preserving the
    identical reward structure, observation design, and aggressiveness
    monitoring that the research pipeline requires.

RESEARCH CONTEXT:
    - AuthSim (T-ITS 2025) also used highway-env, making this acceptable for
      AAP publication.
    - The ego vehicle IS the adversarial agent; NPC vehicles use IDM.
    - The aggressiveness constraint (Hellinger distance to NDD) applies to the
      ego's driving behavior, ensuring behavioral realism.

ARCHITECTURE:
    RL Agent (PPO) --> action [acceleration, steering] --> Ego Vehicle
                         |  accel in [-5, 5] m/s^2
                         |  steering in [-0.3, 0.3]
                         v
                highway-env step
                         |
                         v
          obs (14D), reward (THW-based), SSM, aggressiveness features

    NPC vehicles follow IDM (built into highway-env).

OBSERVATION SPACE (14D, flat, normalized to [-1, 1]):
    [ego_speed, ego_accel, ego_lane_idx,
     rel_longitudinal_dist, rel_lateral_dist, rel_speed, target_accel,
     thw, ttc,
     type_rear_end, type_cut_in,
     c_level,
     nearest_other_dist, nearest_other_rel_speed]

ACTION SPACE:
    Continuous 2D: [acceleration, steering]
    acceleration in [-5, 5] m/s^2, steering in [-0.3, 0.3]

REWARD:
    R_base = w_type * R_type + w_sev * R_severity + w_ssm * R_ssm
           + R_near_miss + R_collision
    Collisions are TERMINAL and heavily penalized (R_collision = -3.0).
    This is the adversarial objective ONLY; the Lagrangian constraint
    (-lambda * (D_Hellinger - eps)) is added externally.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization constants (from HighD 95th-percentile statistics)
# ---------------------------------------------------------------------------
_MAX_SPEED: float = 40.0       # m/s (~144 km/h)
_MAX_ACCEL: float = 5.0        # m/s^2
_MAX_DISTANCE: float = 100.0   # m
_MAX_DELTA_V: float = 30.0     # m/s
_MAX_TTC: float = 10.0         # s
_MAX_THW: float = 10.0         # s
_MAX_NUM_LANES: float = 3.0    # highway: 3 lanes

# Sentinel values for undefined SSM
_TTC_INF: float = 999.0
_THW_INF: float = 999.0

# Grace period (steps) — collisions within this window are non-terminal
_GRACE_STEPS: int = 30


def _try_import_highway_env():
    """Lazy-import highway_env to allow importing this module without it installed."""
    try:
        import highway_env  # noqa: F401
        return highway_env
    except ImportError as exc:
        raise ImportError(
            "highway-env not found. Install with: pip install highway-env"
        ) from exc


class AdversarialHighwayEnv(gymnasium.Env):
    """
    Gymnasium environment for adversarial scenario generation on highway-env.

    The RL agent controls the ego vehicle as an adversarial driver model.
    NPC vehicles are IDM-controlled traffic (built into highway-env).
    The goal is to create dangerous interactions (rear-end, cut-in) with
    controllable severity while maintaining behavioral realism.

    Phase 2 CONSTRAINED MODE:
        When constrained=True, the reward is modified to enforce behavioral
        realism.  Two constraint modes are supported:

        1. "lagrangian" (PID-Lagrangian, Stooke et al., 2020):
           R_total = R_base - lambda * (D_Hellinger - epsilon)
           The Lagrange multiplier lambda is LEARNED via PID control,
           providing principled constraint enforcement with theoretical
           guarantees.

        2. "fixed_penalty":
           R_total = R_base - alpha * max(0, D_Hellinger - epsilon)
           A fixed penalty weight alpha penalizes constraint violations.
           Simpler, no adaptive lambda — useful for ablation sweeps.

        In both modes:
        - When D_Hellinger > epsilon: constraint violated, penalty applied.
        - When D_Hellinger < epsilon: constraint satisfied, no penalty.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        scenario_type: Optional[str] = None,
        c_level: Optional[float] = None,
        w_type: float = 1.0,
        w_severity: float = 3.0,
        w_ssm: float = 1.5,
        lanes_count: int = 3,
        vehicles_count: int = 15,
        duration: int = 200,
        simulation_frequency: int = 10,
        policy_frequency: int = 10,
        render_mode: str = "none",
        # Phase 2: Constrained RL parameters
        constrained: bool = False,
        constraint_mode: str = "fixed_penalty",  # "fixed_penalty" or "lagrangian"
        epsilon: float = 0.3,
        penalty_alpha: float = 5.0,  # fixed penalty weight (for fixed_penalty mode)
        pid_kp: float = 0.5,
        pid_ki: float = 0.05,
        pid_kd: float = 0.001,
        lambda_max: float = 10.0,
        lambda_init: float = 0.0,
        # Constraint feature mode: "all" (6D) or "behavioral" (2D: jerk_x, a_x)
        constraint_features: str = "behavioral",
    ):
        super().__init__()

        _try_import_highway_env()

        # Configuration
        self._fixed_scenario_type = scenario_type
        self._fixed_c_level = c_level
        self._w_type = w_type
        self._w_severity = w_severity
        self._w_ssm = w_ssm
        self._lanes_count = lanes_count
        self._vehicles_count = vehicles_count
        self._duration = duration
        self._sim_freq = simulation_frequency
        self._policy_freq = policy_frequency
        self.render_mode = render_mode

        self._dt: float = 1.0 / self._policy_freq  # seconds per policy step

        # Constraint feature mode
        if constraint_features not in ("all", "behavioral"):
            raise ValueError(
                f"constraint_features must be 'all' or 'behavioral', "
                f"got '{constraint_features}'"
            )
        self._constraint_features = constraint_features

        # Phase 2: Constrained RL
        self._constrained = constrained
        if constraint_mode not in ("fixed_penalty", "lagrangian"):
            raise ValueError(
                f"constraint_mode must be 'fixed_penalty' or 'lagrangian', "
                f"got '{constraint_mode}'"
            )
        self._constraint_mode = constraint_mode
        self._penalty_alpha = penalty_alpha
        self._epsilon = epsilon
        self._pid_lagrangian = None
        if constrained:
            if constraint_mode == "lagrangian":
                from src.agents.constrained_ppo import PIDLagrangian
                self._pid_lagrangian = PIDLagrangian(
                    epsilon=epsilon,
                    K_p=pid_kp,
                    K_i=pid_ki,
                    K_d=pid_kd,
                    lambda_max=lambda_max,
                    lambda_init=lambda_init,
                )
                logger.info(
                    f"Constrained mode ENABLED (lagrangian): epsilon={epsilon}, "
                    f"PID=({pid_kp}, {pid_ki}, {pid_kd}), lambda_max={lambda_max}"
                )
            else:
                logger.info(
                    f"Constrained mode ENABLED (fixed_penalty): "
                    f"epsilon={epsilon}, alpha={penalty_alpha}"
                )

        # Aggressiveness calculator for Hellinger distance monitoring
        self._agg_calculator = None
        self._hellinger_distance: float = 0.0       # used for constraint (6D or 2D)
        self._hellinger_distance_6d: float = 0.0    # always 6D, for monitoring

        # Per-episode Hellinger tracking (for PID update at episode end)
        self._episode_hellinger_values: list = []

        # Episode state
        self._current_scenario_type: str = "rear_end"
        self._current_c_level: float = 0.5
        self._step_count: int = 0
        self._prev_ego_speed: Optional[float] = None
        self._prev_ego_accel: Optional[float] = None
        self._prev_target_speed: Optional[float] = None
        self._collision_occurred: bool = False
        self._ttc_min_episode: float = _TTC_INF

        # Target vehicle reference
        self._target_vehicle = None

        # Internal highway-env environment (created on first reset)
        self._inner_env = None

        # Gymnasium spaces — 14D observation
        self.observation_space = spaces.Box(
            low=-np.ones(14, dtype=np.float32),
            high=np.ones(14, dtype=np.float32),
            dtype=np.float32,
        )
        # 2D continuous action: [acceleration, steering]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Initialize AggressivenessCalculator if Phase 0 outputs exist
        self._init_agg_calculator()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init_agg_calculator(self):
        """Load GMM/scaler for Hellinger distance monitoring.

        Preference order for 6D reference:
        1. Sim-calibrated NDD (gmm_model_sim_ndd.pkl) -- matches highway-env IDM traffic
        2. HighD-fitted (gmm_model_highd.pkl) -- original Phase 0 output (fallback)

        When constraint_features="behavioral", also loads a separate 2D GMM/scaler
        for the behavioral-only Hellinger constraint (jerk_x, a_x).  The 6D model
        is still loaded for monitoring even in behavioral mode.
        """
        try:
            import joblib
            # Prefer sim-calibrated NDD reference (matches highway-env IDM traffic)
            sim_gmm_path = os.path.join("models", "gmm_model_sim_ndd.pkl")
            sim_scaler_path = os.path.join("models", "scaler_sim_ndd.pkl")
            highd_gmm_path = os.path.join("models", "gmm_model_highd.pkl")
            highd_scaler_path = os.path.join("models", "scaler_highd.pkl")

            gmm = None
            scaler = None

            if os.path.exists(sim_gmm_path) and os.path.exists(sim_scaler_path):
                gmm = joblib.load(sim_gmm_path)
                scaler = joblib.load(sim_scaler_path)
                logger.info(
                    "Loaded SIM-NDD 6D reference (sim-calibrated IDM distribution)"
                )
            elif os.path.exists(highd_gmm_path) and os.path.exists(highd_scaler_path):
                gmm = joblib.load(highd_gmm_path)
                scaler = joblib.load(highd_scaler_path)
                logger.info(
                    "Loaded HighD 6D reference (fallback)"
                )
            else:
                logger.warning(
                    "No 6D GMM/scaler found (checked sim_ndd and highd). "
                    "Hellinger monitoring disabled."
                )
                return

            # Load optional 2D behavioral models (jerk_x, a_x only)
            behavioral_gmm = None
            behavioral_scaler = None
            if self._constraint_features == "behavioral":
                beh_gmm_path = os.path.join("models", "gmm_model_sim_ndd_2d.pkl")
                beh_scaler_path = os.path.join("models", "scaler_sim_ndd_2d.pkl")
                try:
                    if os.path.exists(beh_gmm_path) and os.path.exists(beh_scaler_path):
                        behavioral_gmm = joblib.load(beh_gmm_path)
                        behavioral_scaler = joblib.load(beh_scaler_path)
                        logger.info(
                            "Loaded 2D behavioral reference "
                            "(gmm_model_sim_ndd_2d.pkl) for constraint"
                        )
                    else:
                        logger.warning(
                            "constraint_features='behavioral' but 2D models not found "
                            f"(checked {beh_gmm_path}, {beh_scaler_path}). "
                            "Behavioral Hellinger will return 0.0 until models are created."
                        )
                except FileNotFoundError:
                    logger.warning(
                        "Failed to load 2D behavioral models. "
                        "Behavioral Hellinger will return 0.0."
                    )

            from src.agents.aggressiveness_calculator import AggressivenessCalculator
            self._agg_calculator = AggressivenessCalculator(
                gmm_model=gmm,
                scaler=scaler,
                window_size=50,
                behavioral_gmm=behavioral_gmm,
                behavioral_scaler=behavioral_scaler,
            )
            logger.info(
                f"AggressivenessCalculator initialized "
                f"(constraint_features={self._constraint_features!r})"
            )
        except Exception as e:
            logger.warning(f"Could not init AggressivenessCalculator: {e}")

    def _make_inner_env(self):
        """Create the underlying highway-env environment."""
        import gymnasium as gym

        config = {
            "lanes_count": self._lanes_count,
            "vehicles_count": self._vehicles_count,
            "duration": self._duration,
            "simulation_frequency": self._sim_freq,
            "policy_frequency": self._policy_freq,
            # Disable ALL built-in rewards — we compute our own
            "collision_reward": 0,
            "right_lane_reward": 0,
            "high_speed_reward": 0,
            "lane_change_reward": 0,
            "reward_speed_range": [0, 100],  # effectively disable speed reward
            "normalize_reward": False,
            "action": {
                "type": "ContinuousAction",
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-0.3, 0.3],
            },
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["x", "y", "vx", "vy", "heading"],
            },
        }

        # highway-env expects render_mode=None for headless, not "none"
        rm = self.render_mode if self.render_mode in ("human", "rgb_array") else None
        env = gym.make("highway-v0", render_mode=rm, config=config)
        return env

    # ------------------------------------------------------------------
    # Target vehicle designation
    # ------------------------------------------------------------------
    def _designate_target(self) -> bool:
        """
        Designate a target vehicle based on the current scenario type.

        rear_end: closest IDM vehicle in same lane, 30-50m ahead of ego.
        cut_in: IDM vehicle in adjacent lane, within +/-20m longitudinally.

        Returns True if a valid target was found.
        """
        ego = self._inner_env.unwrapped.vehicle
        road_vehicles = self._inner_env.unwrapped.road.vehicles[1:]  # skip ego

        if not road_vehicles:
            return False

        ego_x, ego_y = ego.position
        ego_lane = ego.lane_index[2] if hasattr(ego, 'lane_index') else 0

        if self._current_scenario_type == "rear_end":
            # Find closest IDM vehicle in same lane, 30-50m ahead
            candidates = []
            for v in road_vehicles:
                vx, vy = v.position
                v_lane = v.lane_index[2] if hasattr(v, 'lane_index') else -1
                lon_dist = vx - ego_x
                if v_lane == ego_lane and 30.0 <= lon_dist <= 50.0:
                    candidates.append((v, lon_dist))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                self._target_vehicle = candidates[0][0]
                return True
            # Relax: any same-lane vehicle ahead within 10-80m
            for v in road_vehicles:
                vx, vy = v.position
                v_lane = v.lane_index[2] if hasattr(v, 'lane_index') else -1
                lon_dist = vx - ego_x
                if v_lane == ego_lane and 10.0 <= lon_dist <= 80.0:
                    candidates.append((v, lon_dist))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                self._target_vehicle = candidates[0][0]
                return True

        elif self._current_scenario_type == "cut_in":
            # Find IDM vehicle in adjacent lane, within +/-20m longitudinally
            candidates = []
            for v in road_vehicles:
                vx, vy = v.position
                v_lane = v.lane_index[2] if hasattr(v, 'lane_index') else -1
                lon_dist = vx - ego_x
                if abs(v_lane - ego_lane) == 1 and abs(lon_dist) <= 20.0:
                    candidates.append((v, abs(lon_dist)))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                self._target_vehicle = candidates[0][0]
                return True
            # Relax: any adjacent lane vehicle within 50m
            for v in road_vehicles:
                vx, vy = v.position
                v_lane = v.lane_index[2] if hasattr(v, 'lane_index') else -1
                lon_dist = vx - ego_x
                if abs(v_lane - ego_lane) == 1 and abs(lon_dist) <= 50.0:
                    candidates.append((v, abs(lon_dist)))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                self._target_vehicle = candidates[0][0]
                return True

        return False

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Phase 2: Update lambda via PID at episode boundary
        # (skip on the very first reset when no data has been collected)
        # Only applies in lagrangian mode; fixed_penalty has no adaptive lambda.
        if (
            self._constrained
            and self._constraint_mode == "lagrangian"
            and self._pid_lagrangian is not None
            and self._episode_hellinger_values
        ):
            # Use the mean Hellinger distance over the completed episode
            episode_mean_hellinger = float(np.mean(self._episode_hellinger_values))
            old_lambda = self._pid_lagrangian.get_lambda()
            new_lambda = self._pid_lagrangian.update(episode_mean_hellinger)
            logger.debug(
                f"PID lambda update: hellinger={episode_mean_hellinger:.4f}, "
                f"lambda={old_lambda:.4f}->{new_lambda:.4f}, "
                f"violation={episode_mean_hellinger - self._pid_lagrangian.epsilon:.4f}"
            )

        # Reset per-episode Hellinger tracking
        self._episode_hellinger_values = []

        # Sample scenario parameters
        if self._fixed_scenario_type is not None:
            self._current_scenario_type = self._fixed_scenario_type
        else:
            self._current_scenario_type = self.np_random.choice(
                ["rear_end", "cut_in"]
            )

        if self._fixed_c_level is not None:
            self._current_c_level = self._fixed_c_level
        else:
            self._current_c_level = float(self.np_random.uniform(0.0, 0.8))

        # Create or reset inner environment
        max_reset_attempts = 10
        target_found = False
        for attempt in range(max_reset_attempts):
            if self._inner_env is None:
                self._inner_env = self._make_inner_env()
            self._inner_env.reset(seed=seed)

            # Designate target vehicle
            if self._designate_target():
                target_found = True
                break
            else:
                logger.debug(
                    f"No valid target for {self._current_scenario_type}, "
                    f"attempt {attempt + 1}/{max_reset_attempts}"
                )
                # Close and recreate for a fresh traffic layout
                self._inner_env.close()
                self._inner_env = None

        if not target_found:
            # Last resort: ensure env exists and assign any vehicle as target
            if self._inner_env is None:
                self._inner_env = self._make_inner_env()
                self._inner_env.reset(seed=seed)

            # Try switching scenario type
            if self._current_scenario_type == "cut_in":
                self._current_scenario_type = "rear_end"
                if self._designate_target():
                    target_found = True

            if not target_found:
                # Assign nearest vehicle as fallback target
                road_vehicles = self._inner_env.unwrapped.road.vehicles[1:]
                if road_vehicles:
                    self._target_vehicle = road_vehicles[0]
                else:
                    logger.warning("No vehicles available at all")
                    self._target_vehicle = None

        # Reset episode state
        self._step_count = 0
        self._prev_ego_speed = None
        self._prev_ego_accel = None
        self._prev_target_speed = None
        self._collision_occurred = False
        self._ttc_min_episode = _TTC_INF
        self._hellinger_distance = 0.0
        self._hellinger_distance_6d = 0.0

        if self._agg_calculator is not None:
            self._agg_calculator.reset()

        obs = self._get_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # Clip action to [-1, 1] and map to physical ranges
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # highway-env ContinuousAction expects [acceleration_norm, steering_norm]
        # both in [-1, 1], internally mapped via configured ranges
        inner_obs, inner_reward, inner_terminated, inner_truncated, inner_info = (
            self._inner_env.step(action)
        )

        # Check collision via highway-env
        ego = self._inner_env.unwrapped.vehicle
        if ego.crashed:
            self._collision_occurred = True

        # NPC anomaly detection: remove stopped vehicles (speed < 0.5 m/s)
        # that can create unrealistic obstacles
        if self._step_count > _GRACE_STEPS:
            self._remove_anomalous_npcs()

        # Check if target vehicle is still alive/present
        if self._target_vehicle is not None:
            road_vehicles = self._inner_env.unwrapped.road.vehicles
            if self._target_vehicle not in road_vehicles:
                # Target despawned — try to find new target
                self._designate_target()

        # Compute SSM
        ssm = self._compute_ssm()
        self._ttc_min_episode = min(self._ttc_min_episode, ssm["ttc"])

        # Compute reward (our own, ignoring highway-env's)
        reward, reward_components = self._compute_reward(ssm)

        # Aggressiveness features
        agg_features = self._compute_aggressiveness_features(ssm)

        # Feed features to AggressivenessCalculator for Hellinger monitoring
        if self._agg_calculator is not None:
            self._agg_calculator.feature_buffer.append(agg_features)

            # Always compute 6D Hellinger for monitoring/logging
            self._hellinger_distance_6d = (
                self._agg_calculator.compute_hellinger_distance()
            )

            # Constraint Hellinger: behavioral (2D) or full (6D)
            if self._constraint_features == "behavioral":
                self._hellinger_distance = (
                    self._agg_calculator.compute_behavioral_hellinger()
                )
            else:
                self._hellinger_distance = self._hellinger_distance_6d

            # Track per-episode Hellinger values for PID update at reset
            if self._hellinger_distance > 0.0:
                self._episode_hellinger_values.append(self._hellinger_distance)

        # Update previous-step values
        self._prev_ego_speed = float(ego.speed)

        # Termination
        terminated, truncated = self._check_termination(inner_terminated, inner_truncated)

        # Build observation and info
        obs = self._get_observation(ssm=ssm)
        info = self._build_info(ssm, reward_components, agg_features)
        info["hellinger_distance"] = self._hellinger_distance
        info["hellinger_distance_6d"] = self._hellinger_distance_6d

        return obs, reward, terminated, truncated, info

    def render(self):
        """Delegate rendering to inner highway-env."""
        if self._inner_env is not None:
            return self._inner_env.render()

    def close(self):
        """Clean up the inner environment."""
        if self._inner_env is not None:
            self._inner_env.close()
            self._inner_env = None
        logger.info("AdversarialHighwayEnv closed.")

    # ------------------------------------------------------------------
    # NPC anomaly detection
    # ------------------------------------------------------------------
    def _remove_anomalous_npcs(self):
        """Detect and remove NPC vehicles that have stopped (< 0.5 m/s).

        Stopped vehicles on a highway are unrealistic and create false
        collision signals. We despawn them to keep episodes clean.
        """
        road = self._inner_env.unwrapped.road
        ego = self._inner_env.unwrapped.vehicle
        to_remove = []

        for v in road.vehicles[1:]:  # skip ego
            if v.speed < 0.5 and v is not self._target_vehicle:
                to_remove.append(v)

        for v in to_remove:
            road.vehicles.remove(v)
            if v is self._target_vehicle:
                self._target_vehicle = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self, ssm: Optional[dict] = None) -> np.ndarray:
        """
        Build 14D observation vector, all normalized to ~[-1, 1].

        [0]:   ego_speed
        [1]:   ego_accel
        [2]:   ego_lane_idx
        [3]:   rel_longitudinal_dist
        [4]:   rel_lateral_dist
        [5]:   rel_speed
        [6]:   target_accel
        [7]:   thw
        [8]:   ttc
        [9]:   type_rear_end
        [10]:  type_cut_in
        [11]:  c_level
        [12]:  nearest_other_dist
        [13]:  nearest_other_rel_speed
        """
        ego = self._inner_env.unwrapped.vehicle

        # Ego state
        ego_speed = float(ego.speed)
        if self._prev_ego_speed is not None:
            ego_accel = (ego_speed - self._prev_ego_speed) / self._dt
        else:
            ego_accel = 0.0
        ego_lane_idx = ego.lane_index[2] if hasattr(ego, 'lane_index') else 0

        # Target vehicle relative state
        rel_lon_dist = _MAX_DISTANCE
        rel_lat_dist = 0.0
        rel_speed = 0.0
        target_accel = 0.0
        thw = _THW_INF
        ttc = _TTC_INF

        if self._target_vehicle is not None:
            ego_x, ego_y = ego.position
            tv_x, tv_y = self._target_vehicle.position

            rel_lon_dist = tv_x - ego_x
            rel_lat_dist = tv_y - ego_y

            tv_speed = float(self._target_vehicle.speed)
            # Relative speed: positive = ego approaching target
            rel_speed = ego_speed - tv_speed

            if self._prev_target_speed is not None:
                target_accel = (tv_speed - self._prev_target_speed) / self._dt
            self._prev_target_speed = tv_speed

            # SSM from pre-computed values if available
            if ssm is not None:
                thw = ssm["thw"]
                ttc = ssm["ttc"]
            else:
                distance = math.sqrt(rel_lon_dist ** 2 + rel_lat_dist ** 2)
                if ego_speed > 0.1:
                    thw = max(distance, 0.01) / ego_speed
                if rel_speed > 0.01 and distance > 0.1:
                    ttc = distance / rel_speed
        else:
            self._prev_target_speed = None

        # Nearest OTHER vehicle (not target) — for contextual awareness
        nearest_other_dist, nearest_other_rel_speed = self._find_nearest_other()

        # Scenario type one-hot
        type_rear_end = 1.0 if self._current_scenario_type == "rear_end" else 0.0
        type_cut_in = 1.0 if self._current_scenario_type == "cut_in" else 0.0

        obs = np.array([
            np.clip(ego_speed / _MAX_SPEED, -1, 1),
            np.clip(ego_accel / _MAX_ACCEL, -1, 1),
            np.clip(ego_lane_idx / _MAX_NUM_LANES, -1, 1),
            np.clip(rel_lon_dist / _MAX_DISTANCE, -1, 1),
            np.clip(rel_lat_dist / _MAX_DISTANCE, -1, 1),
            np.clip(rel_speed / _MAX_DELTA_V, -1, 1),
            np.clip(target_accel / _MAX_ACCEL, -1, 1),
            np.clip(thw / _MAX_THW, -1, 1),
            np.clip(ttc / _MAX_TTC, -1, 1),
            type_rear_end,
            type_cut_in,
            self._current_c_level,
            np.clip(nearest_other_dist / _MAX_DISTANCE, -1, 1),
            np.clip(nearest_other_rel_speed / _MAX_DELTA_V, -1, 1),
        ], dtype=np.float32)

        return obs

    def _find_nearest_other(self) -> Tuple[float, float]:
        """Find nearest vehicle that is not the target.

        Returns (distance, relative_speed).
        """
        ego = self._inner_env.unwrapped.vehicle
        ego_x, ego_y = ego.position
        ego_speed = float(ego.speed)

        best_dist = _MAX_DISTANCE
        best_rel_speed = 0.0

        for v in self._inner_env.unwrapped.road.vehicles[1:]:
            if v is self._target_vehicle:
                continue
            vx, vy = v.position
            dist = math.sqrt((vx - ego_x) ** 2 + (vy - ego_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_rel_speed = ego_speed - float(v.speed)

        return best_dist, best_rel_speed

    # ------------------------------------------------------------------
    # SSM computation
    # ------------------------------------------------------------------
    def _compute_ssm(self) -> dict:
        """
        Compute Surrogate Safety Measures between ego and target vehicle.

        Returns dict with ttc, thw, drac, distance, closing_speed.
        """
        if self._target_vehicle is None:
            return {
                "ttc": _TTC_INF,
                "thw": _THW_INF,
                "drac": 0.0,
                "distance": _MAX_DISTANCE,
                "closing_speed": 0.0,
            }

        ego = self._inner_env.unwrapped.vehicle
        ego_x, ego_y = ego.position
        tv_x, tv_y = self._target_vehicle.position
        ego_speed = float(ego.speed)

        distance = math.sqrt((tv_x - ego_x) ** 2 + (tv_y - ego_y) ** 2)

        # Closing speed: positive = approaching
        # Project velocity components along longitudinal direction
        closing_speed = ego_speed - float(self._target_vehicle.speed)

        # TTC
        if closing_speed > 0.01 and distance > 0.1:
            ttc = distance / closing_speed
        else:
            ttc = _TTC_INF

        # THW
        if ego_speed > 0.1:
            thw = distance / ego_speed
        else:
            thw = _THW_INF

        # DRAC
        if distance > 0.1 and closing_speed > 0.01:
            drac = (closing_speed ** 2) / (2.0 * distance)
        else:
            drac = 0.0

        return {
            "ttc": min(ttc, _TTC_INF),
            "thw": min(thw, _THW_INF),
            "drac": drac,
            "distance": distance,
            "closing_speed": closing_speed,
        }

    # ------------------------------------------------------------------
    # Aggressiveness features (6D, matching HighD Phase 0)
    # ------------------------------------------------------------------
    def _compute_aggressiveness_features(self, ssm: dict) -> np.ndarray:
        """
        Extract 6D aggressiveness feature vector from ego's current state.

        Features: [jerk_x, a_x, delta_v, delta_d, thw, ttc]
        These match the HighD feature definition exactly for constraint
        computation.
        """
        ego = self._inner_env.unwrapped.vehicle
        ego_speed = float(ego.speed)

        # Acceleration
        if self._prev_ego_speed is not None:
            a_x = (ego_speed - self._prev_ego_speed) / self._dt
        else:
            a_x = 0.0

        # Jerk
        if self._prev_ego_accel is not None:
            jerk_x = (a_x - self._prev_ego_accel) / self._dt
        else:
            jerk_x = 0.0
        self._prev_ego_accel = a_x

        # Relative speed to target (positive = approaching)
        delta_v = ssm.get("closing_speed", 0.0)

        # Distance headway
        delta_d = ssm.get("distance", _MAX_DISTANCE)

        # THW and TTC
        thw = ssm.get("thw", _THW_INF)
        ttc = ssm.get("ttc", _TTC_INF)

        return np.array(
            [jerk_x, a_x, delta_v, delta_d, thw, ttc], dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Reward — THW-based, ported from AdversarialCarlaEnv
    # ------------------------------------------------------------------
    def _compute_reward(self, ssm: dict) -> Tuple[float, dict]:
        """
        Compute base adversarial reward R_base.

        R_base = w_type * R_type + w_sev * R_severity + w_ssm * R_ssm
               + R_near_miss + R_collision

        Collisions are PUNISHED, not rewarded. The goal is sustained
        dangerous-but-realistic interactions that mirror aggressive
        moments observed in HighD.
        """
        ego = self._inner_env.unwrapped.vehicle
        ego_speed = float(ego.speed)

        # R_type: scenario type achievement
        r_type = self._compute_r_type(ssm)

        # R_severity: THW target tracking with speed gating
        #
        # Severity mapping: THW_target = 3.0 - 2.5 * c_level
        #   c=0.0 -> THW=3.0s (conservative)
        #   c=0.5 -> THW=1.75s (moderate — HighD THW median is 1.71s)
        #   c=0.8 -> THW=1.0s (aggressive — HighD aggressive THW < 1.0s)
        thw_target = 3.0 - 2.5 * self._current_c_level
        thw_actual = ssm["thw"]

        # Speed gate: ego_speed < 20 m/s -> r_severity = -1.0
        _MIN_REWARD_SPEED = 20.0
        speed_factor = min(ego_speed / _MIN_REWARD_SPEED, 1.0)

        if ego_speed < _MIN_REWARD_SPEED:
            r_severity = -1.0
        elif ego_speed > 1.0 and ssm["distance"] < _MAX_DISTANCE:
            distance = ssm["distance"]
            proximity_bonus = 0.5 * max(0.0, 1.0 - distance / _MAX_DISTANCE)

            if thw_actual < _MAX_THW:
                thw_error = min(
                    abs(thw_actual - thw_target) / max(thw_target, 0.3), 1.0
                )
                base_severity = -thw_error + proximity_bonus
                # Bonuses for good behavior
                if abs(thw_actual - thw_target) < 0.3:
                    base_severity += 0.5  # Within 0.3s of target THW
                closing_speed = ssm.get("closing_speed", 0.0)
                if abs(closing_speed) < 1.0 and thw_actual < thw_target * 2.0:
                    base_severity += 0.3  # Speed matching bonus
                # Apply speed gate: positive scaled, negative kept full
                if base_severity > 0:
                    r_severity = base_severity * speed_factor
                else:
                    r_severity = base_severity
            else:
                r_severity = -1.0 + proximity_bonus
        else:
            r_severity = -1.0

        # R_ssm: physical plausibility penalties
        r_ssm = 0.0

        # TTC < 0.3s — about to crash unavoidably
        if ssm["ttc"] < 0.3:
            r_ssm -= 3.0

        # Unrealistic deceleration requirement
        if ssm["drac"] > 8.0:
            r_ssm -= 1.0

        # Collision penalty within r_ssm
        if self._collision_occurred:
            r_ssm -= 2.0

        # Speed maintenance: STRONG penalty for slowing on highway
        if ego_speed < 15.0:
            r_ssm -= 2.0 * (1.0 - ego_speed / 15.0)

        # Near-miss bonus: THW-based, rewarding stable close following
        r_near_miss = 0.0
        if not self._collision_occurred and ego_speed > 1.0:
            if thw_actual < thw_target * 1.2 and thw_actual > 0.3:
                r_near_miss = 0.3

        # Collision penalty (terminal)
        r_collision = 0.0
        if self._collision_occurred:
            r_collision = -3.0

        r_base = (
            self._w_type * r_type
            + self._w_severity * r_severity
            + self._w_ssm * r_ssm
            + r_near_miss
            + r_collision
        )

        # Phase 2: Constraint term
        # Two modes:
        #   fixed_penalty:  R_total = R_base - alpha * max(0, H - epsilon)
        #   lagrangian:     R_total = R_base - lambda * (H - epsilon)
        r_constraint = 0.0
        lambda_val = 0.0
        if self._constrained:
            hellinger = self._hellinger_distance
            if self._constraint_mode == "fixed_penalty":
                violation = max(0.0, hellinger - self._epsilon)
                r_constraint = -self._penalty_alpha * violation
                lambda_val = self._penalty_alpha  # report alpha as "lambda" for logging
            elif self._pid_lagrangian is not None:
                lambda_val = self._pid_lagrangian.get_lambda()
                r_constraint = -lambda_val * (hellinger - self._pid_lagrangian.epsilon)

        r_total = r_base + r_constraint

        components = {
            "r_type": r_type,
            "r_severity": r_severity,
            "r_ssm": r_ssm,
            "r_near_miss": r_near_miss,
            "r_collision": r_collision,
            "r_base": r_base,
            "r_constraint": r_constraint,
            "r_total": r_total,
            "thw_target": thw_target,
            "lambda_val": lambda_val,
        }

        return r_total, components

    def _compute_r_type(self, ssm: dict) -> float:
        """
        Compute scenario-type achievement reward based on sustained interaction.

        Collision is PUNISHED (-0.5). Reward is proportional to closeness
        while maintaining distance. Near-misses are valuable.
        """
        distance = ssm["distance"]
        ttc = ssm["ttc"]

        if self._collision_occurred:
            return -0.5

        if self._current_scenario_type == "rear_end":
            # Reward for maintaining close following distance (not crashing)
            if distance < 30.0 and ttc > 1.0:
                return float(np.clip((30.0 - distance) / 30.0, 0.0, 1.0))
            elif 0.0 < ttc < 1.0 and not self._collision_occurred:
                return 0.5  # Near-miss situation
            elif distance < _MAX_DISTANCE:
                return 0.0
            return -0.5

        elif self._current_scenario_type == "cut_in":
            if self._target_vehicle is None:
                return -0.5

            ego = self._inner_env.unwrapped.vehicle
            ego_lane = ego.lane_index[2] if hasattr(ego, 'lane_index') else 0
            tv_lane = (
                self._target_vehicle.lane_index[2]
                if hasattr(self._target_vehicle, 'lane_index') else -1
            )

            same_lane = (ego_lane == tv_lane)
            if same_lane and distance < 15.0:
                # Phase 2: Successfully cut in — same lane, close range
                return float(
                    1.0 * np.clip((15.0 - distance) / 15.0, 0.0, 1.0)
                )
            elif not same_lane and distance < 30.0:
                # Phase 1: Adjacent lane, positioning for cut-in
                return float(
                    0.3 * np.clip((30.0 - distance) / 30.0, 0.0, 1.0)
                )
            return 0.0

        return 0.0

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _check_termination(
        self, inner_terminated: bool, inner_truncated: bool
    ) -> Tuple[bool, bool]:
        """
        Check episode termination conditions.

        Returns (terminated, truncated).
        """
        terminated = False
        truncated = False

        # Grace period: first GRACE_STEPS steps non-terminal for collision
        if self._collision_occurred and self._step_count > _GRACE_STEPS:
            terminated = True

        # highway-env may terminate for off-road or other reasons
        if inner_terminated and not self._collision_occurred:
            terminated = True

        # Duration truncation
        if inner_truncated or self._step_count >= self._duration:
            truncated = True

        return terminated, truncated

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------
    def _build_info(
        self,
        ssm: Optional[dict] = None,
        reward_components: Optional[dict] = None,
        agg_features: Optional[np.ndarray] = None,
    ) -> dict:
        """Build the info dict returned by reset/step."""
        ego = self._inner_env.unwrapped.vehicle
        ego_speed = float(ego.speed)

        # Determine termination reason
        if self._collision_occurred:
            term_reason = "ego_collision"
        elif self._step_count >= self._duration:
            term_reason = "timeout"
        else:
            term_reason = "running"

        info = {
            "scenario_type": self._current_scenario_type,
            "c_level": self._current_c_level,
            "ego_speed": ego_speed,
            "collision": self._collision_occurred,
            "term_reason": term_reason,
            "step_count": self._step_count,
            "ttc_min_episode": self._ttc_min_episode,
        }

        if ssm is not None:
            info["ttc"] = ssm["ttc"]
            info["thw"] = ssm["thw"]
            info["drac"] = ssm["drac"]
            info["distance"] = ssm["distance"]

        if reward_components is not None:
            info["reward_components"] = reward_components

        if agg_features is not None:
            info["aggressiveness_features"] = agg_features

        # Phase 2: Constraint-related metrics
        if self._constrained:
            if self._constraint_mode == "fixed_penalty":
                info["lambda_val"] = self._penalty_alpha
                info["constraint_epsilon"] = self._epsilon
                info["constraint_violation"] = max(
                    0.0, self._hellinger_distance - self._epsilon
                )
                info["constraint_mode"] = "fixed_penalty"
            elif self._pid_lagrangian is not None:
                info["lambda_val"] = self._pid_lagrangian.get_lambda()
                info["constraint_epsilon"] = self._pid_lagrangian.epsilon
                info["constraint_violation"] = max(
                    0.0, self._hellinger_distance - self._pid_lagrangian.epsilon
                )
                info["constraint_mode"] = "lagrangian"

        return info
