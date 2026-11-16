"""
NPC Adapter for Downstream AV Evaluation

PURPOSE:
    Enables evaluation of autonomous vehicle (AV) stacks against adversarial
    NPC traffic learned during Phase 1/2.  The trained PPO policy originally
    controlled the *ego* vehicle; this adapter re-maps that policy onto a
    *single NPC*, while the ego slot is occupied by the AV under test.

RESEARCH CONTEXT:
    After training adversarial scenarios (Phase 1-2), the natural next step is
    to evaluate how well a given AV planner handles those scenarios.  This
    requires inverting the perspective: the adversarial policy now drives an
    NPC, and the AV drives the ego vehicle.

    The 14D observation from AdversarialHighwayEnv is reconstructed from the
    NPC's viewpoint so the loaded policy receives inputs in the same format
    it was trained on, with the NPC taking the role of "self" and the AV
    taking the role of "target".

ARCHITECTURE:
    AV (ego vehicle)  <-- controlled by av_policy ("idm" / external RL)
    NPC (adversarial)  <-- controlled by loaded PPO policy via NPCAdapter
    Other NPCs         <-- IDM (default highway-env behavior)

    Per step:
      1. Build 14D observation from adversarial NPC's perspective
      2. Query loaded PPO policy for action
      3. Override NPC's action dict (after IDM act, before physics step)
      4. Step environment with AV's action on ego
      5. Return AV-centric observation to caller

OBSERVATION SPACE (14D, reinterpreted for NPC perspective):
    [0]:   npc_speed           (was ego_speed)
    [1]:   npc_accel           (was ego_accel)
    [2]:   npc_lane_idx        (was ego_lane_idx)
    [3]:   npc->AV rel_lon     (was ego->target rel_lon)
    [4]:   npc->AV rel_lat     (was ego->target rel_lat)
    [5]:   npc->AV rel_speed   (was ego->target rel_speed)
    [6]:   av_accel             (was target_accel)
    [7]:   thw (NPC perspective)(was thw from ego)
    [8]:   ttc (NPC perspective)(was ttc from ego)
    [9]:   type_rear_end        (scenario one-hot, unchanged)
    [10]:  type_cut_in          (scenario one-hot, unchanged)
    [11]:  c_level              (unchanged)
    [12]:  nearest_other_dist   (from NPC, excluding AV)
    [13]:  nearest_other_rel_speed (from NPC, excluding AV)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc, following_headway

logger = logging.getLogger(__name__)

# Normalization constants — must match AdversarialHighwayEnv exactly
_MAX_SPEED: float = 40.0
_MAX_ACCEL: float = 5.0
_MAX_DISTANCE: float = 100.0
_MAX_DELTA_V: float = 30.0
_MAX_TTC: float = 10.0
_MAX_THW: float = 10.0
_MAX_NUM_LANES: float = 3.0
_TTC_INF: float = 999.0
_THW_INF: float = 999.0


def _try_import_highway_env():
    """Lazy-import highway_env to allow importing this module without it."""
    try:
        import highway_env  # noqa: F401
        return highway_env
    except ImportError as exc:
        raise ImportError(
            "highway-env not found. Install with: pip install highway-env"
        ) from exc


# =========================================================================
# 1. NPCObservationAdapter
# =========================================================================
class NPCObservationAdapter:
    """Construct 14D observation from the NPC's perspective.

    The observation mirrors the layout of AdversarialHighwayEnv._get_observation()
    but with roles swapped: the NPC is "self" and the AV (ego) is "target".

    Parameters
    ----------
    scenario_type : str
        One of "rear_end" or "cut_in".
    c_level : float
        Severity level in [0, 1].
    dt : float
        Time-step for acceleration/jerk estimation (seconds).
    """

    def __init__(
        self,
        scenario_type: str = "rear_end",
        c_level: float = 0.4,
        dt: float = 0.1,
    ) -> None:
        self.scenario_type = scenario_type
        self.c_level = c_level
        self.dt = dt

        # Previous-step caches for finite-difference derivatives
        self._prev_npc_speed: Optional[float] = None
        self._prev_av_speed: Optional[float] = None

    def reset(self) -> None:
        """Clear derivative caches at episode start."""
        self._prev_npc_speed = None
        self._prev_av_speed = None

    def observe(
        self,
        npc_vehicle,
        av_vehicle,
        all_vehicles: list,
    ) -> np.ndarray:
        """Build the 14D observation vector from the NPC's point of view.

        Parameters
        ----------
        npc_vehicle : highway_env Vehicle
            The adversarial NPC.
        av_vehicle : highway_env Vehicle
            The AV under test (sits in the ego slot).
        all_vehicles : list
            All vehicles on the road (for nearest-other computation).

        Returns
        -------
        np.ndarray, shape (14,), dtype float32
            Observation normalized to approximately [-1, 1].
        """
        # --- NPC state (dims 0-2) ---
        npc_speed = float(npc_vehicle.speed)
        if self._prev_npc_speed is not None:
            npc_accel = (npc_speed - self._prev_npc_speed) / self.dt
        else:
            npc_accel = 0.0
        self._prev_npc_speed = npc_speed

        npc_lane_idx = (
            npc_vehicle.lane_index[2]
            if hasattr(npc_vehicle, "lane_index")
            else 0
        )

        # --- NPC -> AV relative state (dims 3-6) ---
        npc_x, npc_y = npc_vehicle.position
        av_x, av_y = av_vehicle.position
        av_speed = float(av_vehicle.speed)

        rel_lon_dist = av_x - npc_x
        rel_lat_dist = av_y - npc_y
        # Positive rel_speed = NPC approaching AV
        rel_speed = npc_speed - av_speed

        # AV acceleration (dim 6)
        if self._prev_av_speed is not None:
            av_accel = (av_speed - self._prev_av_speed) / self.dt
        else:
            av_accel = 0.0
        self._prev_av_speed = av_speed

        # --- SSM from NPC perspective (dims 7-8) ---
        distance = math.sqrt(rel_lon_dist ** 2 + rel_lat_dist ** 2)

        if npc_speed > 0.1 and distance > 0.1:
            thw = distance / npc_speed
        else:
            thw = _THW_INF

        if rel_speed > 0.01 and distance > 0.1:
            ttc = distance / rel_speed
        else:
            ttc = _TTC_INF

        # --- Scenario encoding (dims 9-11) ---
        type_rear_end = 1.0 if self.scenario_type == "rear_end" else 0.0
        type_cut_in = 1.0 if self.scenario_type == "cut_in" else 0.0

        # --- Nearest other vehicle from NPC, excluding AV (dims 12-13) ---
        nearest_dist = _MAX_DISTANCE
        nearest_rel_speed = 0.0
        for v in all_vehicles:
            if v is npc_vehicle or v is av_vehicle:
                continue
            vx, vy = v.position
            d = math.sqrt((vx - npc_x) ** 2 + (vy - npc_y) ** 2)
            if d < nearest_dist:
                nearest_dist = d
                nearest_rel_speed = npc_speed - float(v.speed)

        # --- Assemble & normalize ---
        obs = np.array(
            [
                np.clip(npc_speed / _MAX_SPEED, -1, 1),          # [0]
                np.clip(npc_accel / _MAX_ACCEL, -1, 1),          # [1]
                np.clip(npc_lane_idx / _MAX_NUM_LANES, -1, 1),   # [2]
                np.clip(rel_lon_dist / _MAX_DISTANCE, -1, 1),    # [3]
                np.clip(rel_lat_dist / _MAX_DISTANCE, -1, 1),    # [4]
                np.clip(rel_speed / _MAX_DELTA_V, -1, 1),        # [5]
                np.clip(av_accel / _MAX_ACCEL, -1, 1),           # [6]
                np.clip(thw / _MAX_THW, -1, 1),                  # [7]
                np.clip(ttc / _MAX_TTC, -1, 1),                  # [8]
                type_rear_end,                                    # [9]
                type_cut_in,                                      # [10]
                self.c_level,                                     # [11]
                np.clip(nearest_dist / _MAX_DISTANCE, -1, 1),    # [12]
                np.clip(nearest_rel_speed / _MAX_DELTA_V, -1, 1),# [13]
            ],
            dtype=np.float32,
        )
        return obs


# =========================================================================
# 2. NPCActionExecutor
# =========================================================================
class NPCActionExecutor:
    """Map normalized policy output [-1, 1]^2 to highway-env vehicle actions.

    The PPO policy outputs actions in [-1, 1].  This executor scales them to
    physical units and writes the action dict directly onto the vehicle object,
    bypassing the IDM act() method.

    Action mapping:
        action[0] * 5.0  -> acceleration in [-5, 5] m/s^2
        action[1] * 0.3  -> steering    in [-0.3, 0.3] rad
    """

    @staticmethod
    def execute(npc_vehicle, action: np.ndarray) -> None:
        """Apply the adversarial action to the NPC vehicle.

        This directly sets `npc_vehicle.action`, which is read by
        `Vehicle.step(dt)` during the physics integration.  The action
        must be applied *after* `road.act()` (which runs IDM) so that
        it overrides the IDM decision.

        Parameters
        ----------
        npc_vehicle : highway_env Vehicle
            The adversarial NPC whose action will be overridden.
        action : np.ndarray, shape (2,)
            Policy output in [-1, 1].
        """
        action = np.clip(action, -1.0, 1.0)
        npc_vehicle.action = {
            "acceleration": float(action[0] * 5.0),   # [-5, 5] m/s^2
            "steering": float(action[1] * 0.3),        # [-0.3, 0.3] rad
        }


# =========================================================================
# 3. AdversarialNPCEvalEnv
# =========================================================================
class AdversarialNPCEvalEnv(gym.Wrapper):
    """Gymnasium wrapper for evaluating AV stacks against adversarial NPCs.

    The ego vehicle is the **AV under test**, controlled by the caller
    (via ``step(av_action)``).  One NPC is designated *adversarial* and
    driven by a loaded PPO policy.  All remaining NPCs follow IDM.

    The wrapper intercepts the simulation loop so that the adversarial NPC's
    action is injected after IDM ``act()`` but before physics ``step()``.

    Parameters
    ----------
    adversarial_model_path : str
        Path to a stable-baselines3 PPO ``.zip`` checkpoint.  The policy
        must expect a 14D observation and produce a 2D action.
    av_policy : str
        How the AV (ego) behaves when ``step()`` is called:
        - ``"idm"``: ignore the passed action; ego uses built-in IDM.
        - ``"external"``: apply the action from ``step(av_action)`` directly.
    scenario_type : str
        Scenario type for the adversarial NPC's observation (``"rear_end"``
        or ``"cut_in"``).
    c_level : float
        Severity level encoded in the adversarial NPC's observation.
    lanes_count : int
        Number of highway lanes.
    vehicles_count : int
        Total NPC count (including the one that becomes adversarial).
    duration : int
        Maximum episode length in policy steps.
    render_mode : str
        ``"human"`` for pygame window, ``"none"`` for headless.
    """

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        adversarial_model_path: str,
        av_policy: str = "idm",
        scenario_type: str = "rear_end",
        c_level: float = 0.4,
        lanes_count: int = 3,
        vehicles_count: int = 15,
        duration: int = 200,
        render_mode: str = "none",
        safety_metrics_version: str = "v2",
    ) -> None:
        _try_import_highway_env()

        self._adversarial_model_path = adversarial_model_path
        self._av_policy = av_policy
        if av_policy not in {"idm", "mobil", "external"}:
            raise ValueError("Unknown defender controller")
        self._scenario_type = scenario_type
        self._c_level = c_level
        self._lanes_count = lanes_count
        self._vehicles_count = vehicles_count
        self._duration = duration
        self._render_mode = render_mode
        if safety_metrics_version not in {"v2", "legacy"}:
            raise ValueError("Unknown safety metric version")
        self._safety_metrics_version = safety_metrics_version

        # Build inner highway-env
        inner_env = self._make_inner_env()
        super().__init__(inner_env)

        # Override observation and action spaces to match AV's interface.
        # AV sees the same 14D obs that AdversarialHighwayEnv produces.
        self.observation_space = spaces.Box(
            low=-np.ones(14, dtype=np.float32),
            high=np.ones(14, dtype=np.float32),
            dtype=np.float32,
        )
        # AV action: 2D continuous [acceleration_norm, steering_norm]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Sub-components
        self._obs_adapter = NPCObservationAdapter(
            scenario_type=scenario_type,
            c_level=c_level,
            dt=1.0 / 10.0,  # policy_frequency = 10
        )
        self._action_executor = NPCActionExecutor()

        # Load adversarial PPO policy
        self._adversarial_policy = self._load_policy(adversarial_model_path)

        # Runtime state (set during reset)
        self._adversarial_npc = None  # the designated adversarial vehicle
        self._step_count: int = 0
        self._prev_av_speed: Optional[float] = None
        self._collision_occurred: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _make_inner_env(self) -> gym.Env:
        """Create the underlying highway-env environment."""
        config = {
            "lanes_count": self._lanes_count,
            "vehicles_count": self._vehicles_count,
            "duration": self._duration,
            "simulation_frequency": 10,
            "policy_frequency": 10,
            # Disable built-in rewards — evaluation uses external metrics
            "collision_reward": 0,
            "right_lane_reward": 0,
            "high_speed_reward": 0,
            "lane_change_reward": 0,
            "reward_speed_range": [0, 100],
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
        rm = self._render_mode if self._render_mode in ("human", "rgb_array") else None
        env = gym.make("highway-v0", render_mode=rm, config=config)
        return env

    def _load_policy(self, model_path: str):
        """Load a stable-baselines3 PPO policy from disk.

        Fail explicitly if the requested checkpoint cannot be loaded.
        A random fallback is not a valid evaluation of a trained method.
        """
        import os

        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        try:
            from stable_baselines3 import PPO

            model = PPO.load(model_path, device="cpu")
            logger.info(f"Loaded adversarial PPO policy from {model_path}")
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load PPO checkpoint: {model_path}") from exc

    # ------------------------------------------------------------------
    # Adversarial NPC designation
    # ------------------------------------------------------------------
    def _designate_adversarial_npc(self) -> bool:
        """Pick one IDM vehicle to become the adversarial NPC.

        Selection strategy depends on scenario type:
        - rear_end: NPC behind the AV in the same lane, or closest NPC.
        - cut_in:   NPC in an adjacent lane near the AV.

        Returns True if a valid NPC was designated.
        """
        road = self.unwrapped.road
        ego = self.unwrapped.vehicle  # AV under test
        npcs = [v for v in road.vehicles if v is not ego]

        if not npcs:
            return False

        ego_x, ego_y = ego.position
        ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

        if self._scenario_type == "rear_end":
            # Prefer an NPC *behind* the AV in the same lane
            candidates = []
            for v in npcs:
                vx, vy = v.position
                v_lane = v.lane_index[2] if hasattr(v, "lane_index") else -1
                lon_dist = ego_x - vx  # positive = NPC is behind
                if v_lane == ego_lane and 5.0 < lon_dist < 80.0:
                    candidates.append((v, lon_dist))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                self._adversarial_npc = candidates[0][0]
                return True

        elif self._scenario_type == "cut_in":
            # NPC in adjacent lane, close longitudinally
            candidates = []
            for v in npcs:
                vx, vy = v.position
                v_lane = v.lane_index[2] if hasattr(v, "lane_index") else -1
                lon_dist = abs(vx - ego_x)
                if abs(v_lane - ego_lane) == 1 and lon_dist < 50.0:
                    candidates.append((v, lon_dist))
            if candidates:
                candidates.sort(key=lambda x: x[1])
                self._adversarial_npc = candidates[0][0]
                return True

        # Fallback: pick the NPC closest to the AV
        best_v, best_dist = None, float("inf")
        for v in npcs:
            vx, vy = v.position
            d = math.sqrt((vx - ego_x) ** 2 + (vy - ego_y) ** 2)
            if d < best_dist:
                best_dist = d
                best_v = v
        if best_v is not None:
            self._adversarial_npc = best_v
            return True

        return False

    # ------------------------------------------------------------------
    # AV-centric observation (returned to the caller)
    # ------------------------------------------------------------------
    def _get_av_observation(self) -> np.ndarray:
        """Build a 14D observation from the AV's (ego's) perspective.

        Layout matches AdversarialHighwayEnv._get_observation() exactly,
        with the adversarial NPC serving as the "target" vehicle.
        """
        ego = self.unwrapped.vehicle
        ego_speed = float(ego.speed)

        # Ego acceleration
        if self._prev_av_speed is not None:
            ego_accel = (ego_speed - self._prev_av_speed) / 0.1
        else:
            ego_accel = 0.0
        ego_lane_idx = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

        # Target = adversarial NPC
        rel_lon_dist = _MAX_DISTANCE
        rel_lat_dist = 0.0
        rel_speed = 0.0
        target_accel = 0.0
        thw = _THW_INF
        ttc = _TTC_INF

        if self._adversarial_npc is not None:
            ego_x, ego_y = ego.position
            npc_x, npc_y = self._adversarial_npc.position
            npc_speed = float(self._adversarial_npc.speed)

            rel_lon_dist = npc_x - ego_x
            rel_lat_dist = npc_y - ego_y
            rel_speed = ego_speed - npc_speed

            target_accel = 0.0  # simplified; not tracked step-by-step here

            distance = math.sqrt(rel_lon_dist ** 2 + rel_lat_dist ** 2)
            if ego_speed > 0.1 and distance > 0.1:
                thw = distance / ego_speed
            if rel_speed > 0.01 and distance > 0.1:
                ttc = distance / rel_speed

        # Nearest other (not adversarial NPC)
        nearest_dist = _MAX_DISTANCE
        nearest_rel_speed = 0.0
        ego_x, ego_y = ego.position
        for v in self.unwrapped.road.vehicles:
            if v is ego or v is self._adversarial_npc:
                continue
            vx, vy = v.position
            d = math.sqrt((vx - ego_x) ** 2 + (vy - ego_y) ** 2)
            if d < nearest_dist:
                nearest_dist = d
                nearest_rel_speed = ego_speed - float(v.speed)

        # Scenario one-hot
        type_rear_end = 1.0 if self._scenario_type == "rear_end" else 0.0
        type_cut_in = 1.0 if self._scenario_type == "cut_in" else 0.0

        obs = np.array(
            [
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
                self._c_level,
                np.clip(nearest_dist / _MAX_DISTANCE, -1, 1),
                np.clip(nearest_rel_speed / _MAX_DELTA_V, -1, 1),
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment and designate one NPC as adversarial."""
        obs, info = self.env.reset(seed=seed, options=options)
        if self._av_policy in {"idm", "mobil"}:
            from highway_env.vehicle.behavior import IDMVehicle
            inner = self.unwrapped
            old = inner.vehicle
            replacement = IDMVehicle(road=inner.road, position=old.position.copy(),
                                     heading=old.heading, speed=old.speed,
                                     target_speed=old.speed)
            replacement.enable_lane_change = self._av_policy == "mobil"
            replacement.TIME_WANTED = 1.5
            inner.road.vehicles[inner.road.vehicles.index(old)] = replacement
            inner.vehicle = replacement

        self._step_count = 0
        self._prev_av_speed = None
        self._collision_occurred = False
        self._adversarial_npc = None
        self._obs_adapter.reset()

        # Designate one NPC as adversarial
        if not self._designate_adversarial_npc():
            raise RuntimeError("No adversarial NPC could be designated")

        av_obs = self._get_av_observation()
        eval_info = {
            "scenario_type": self._scenario_type,
            "c_level": self._c_level,
            "adversarial_npc_assigned": self._adversarial_npc is not None,
        }
        return av_obs, eval_info

    def step(
        self, av_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step.

        1. Build adversarial NPC observation and query loaded policy.
        2. Inject adversarial action into the simulation loop.
        3. Step the environment with the AV's action.
        4. Return AV-centric observation.

        Parameters
        ----------
        av_action : np.ndarray, shape (2,)
            Action for the AV (ego vehicle).  Ignored when av_policy="idm".
        """
        self._step_count += 1

        # --- 1. Get adversarial NPC action from loaded policy ---
        adv_action = None
        if self._adversarial_npc is not None:
            # Check NPC still exists on road
            road_vehicles = self.unwrapped.road.vehicles
            if self._adversarial_npc not in road_vehicles:
                logger.debug("Adversarial NPC despawned; attempting reassignment.")
                if not self._designate_adversarial_npc():
                    raise RuntimeError("Adversarial NPC disappeared during evaluation")

            if self._adversarial_npc is not None:
                npc_obs = self._obs_adapter.observe(
                    npc_vehicle=self._adversarial_npc,
                    av_vehicle=self.unwrapped.vehicle,
                    all_vehicles=self.unwrapped.road.vehicles,
                )

                if self._adversarial_policy is not None:
                    adv_action, _ = self._adversarial_policy.predict(
                        npc_obs, deterministic=True
                    )
                else:
                    raise RuntimeError("Adversarial policy is unavailable")

        if adv_action is None:
            raise RuntimeError("No adversarial action was produced")

        # --- 2. Step with modified simulation loop ---
        # We need to inject the adversarial action between road.act() and
        # road.step().  highway-env's _simulate does:
        #   for frame in range(frames):
        #       action_type.act(action)   # ego action
        #       road.act()                # ALL vehicles run IDM
        #       road.step(dt)             # physics integration
        #
        # We monkey-patch road.act() for this step to also override the
        # adversarial NPC's action dict after IDM runs.

        if adv_action is not None and self._adversarial_npc is not None:
            road = self.unwrapped.road
            _original_road_act = road.act

            captured_npc = self._adversarial_npc
            captured_action = adv_action
            executor = self._action_executor

            def _patched_road_act() -> None:
                # Let IDM run for all vehicles first
                _original_road_act()
                # Then override the adversarial NPC's action
                if captured_npc in road.vehicles:
                    executor.execute(captured_npc, captured_action)

            road.act = _patched_road_act  # type: ignore[assignment]

        # Determine ego action
        if self._av_policy in {"idm", "mobil"}:
            # reset() installs a real IDMVehicle. road.act() computes its
            # car-following action; MOBIL additionally enables lane changes.
            ego_action = np.zeros(2, dtype=np.float32)
        else:
            ego_action = np.clip(av_action, -1.0, 1.0).astype(np.float32)

        # Step inner environment
        try:
            inner_obs, inner_reward, inner_terminated, inner_truncated, inner_info = (
                self.env.step(ego_action)
            )
        finally:
            road.act = _original_road_act

        # --- 3. Check collision ---
        ego = self.unwrapped.vehicle
        if ego.crashed:
            self._collision_occurred = True

        # --- 4. Termination ---
        terminated = inner_terminated
        truncated = inner_truncated or (self._step_count >= self._duration)

        # --- 5. Build AV-centric observation and info ---
        av_obs = self._get_av_observation()
        self._prev_av_speed = float(ego.speed)

        # Build evaluation info
        eval_info: Dict[str, Any] = {
            "scenario_type": self._scenario_type,
            "c_level": self._c_level,
            "ego_speed": float(ego.speed),
            "collision": self._collision_occurred,
            "step_count": self._step_count,
            "adversarial_npc_alive": (
                self._adversarial_npc is not None
                and self._adversarial_npc in self.unwrapped.road.vehicles
            ),
        }

        # SSM between AV and adversarial NPC
        if self._adversarial_npc is not None:
            ego_x, ego_y = ego.position
            npc_x, npc_y = self._adversarial_npc.position
            distance = math.sqrt((npc_x - ego_x) ** 2 + (npc_y - ego_y) ** 2)
            closing_speed = float(ego.speed) - float(self._adversarial_npc.speed)
            if closing_speed > 0.01 and distance > 0.1:
                ttc = distance / closing_speed
            else:
                ttc = _TTC_INF
            if float(ego.speed) > 0.1 and distance > 0.1:
                thw = distance / float(ego.speed)
            else:
                thw = _THW_INF
            eval_info["ttc"] = ttc
            eval_info["thw"] = thw
            eval_info["distance_to_adversarial"] = distance

            # Keep policy observation/action mapping unchanged for old checkpoints.
            # Only reported safety measurements are corrected and versioned.
            av_state = VehicleState.from_vehicle(ego)
            npc_state = VehicleState.from_vehicle(self._adversarial_npc)
            pair_ttc = constant_velocity_ttc(av_state, npc_state)
            eval_info.update({
                "ttc_legacy": ttc, "thw_legacy": thw,
                "pair_ttc_v2": pair_ttc,
                "generator_headway_v2": following_headway(npc_state, av_state),
                "defender_headway_v2": following_headway(av_state, npc_state),
                "av_state_v2": av_state.__dict__,
                "npc_state_v2": npc_state.__dict__,
                "safety_metrics_version": self._safety_metrics_version,
            })
            if self._safety_metrics_version == "v2":
                eval_info["ttc"] = pair_ttc if pair_ttc is not None else _TTC_INF

        # Simple AV reward: negative for collision, small positive for survival
        reward = -10.0 if self._collision_occurred else 0.1

        return av_obs, reward, terminated, truncated, eval_info

    def close(self) -> None:
        """Clean up."""
        super().close()
        logger.info("AdversarialNPCEvalEnv closed.")
