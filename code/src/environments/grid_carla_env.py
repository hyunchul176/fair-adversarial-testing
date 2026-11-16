# -*- coding: utf-8 -*-
"""Grid evaluation environment for the AAAI measurement model (CARLA Town04).

This environment was written to produce the response jsonl of the AAAI measurement
model grid (AV × G × c × K). It inherits AdversarialCarlaEnv but runs two roles at once.

    ego        = test-taker AV (a DefensiveRL checkpoint chooses the action via an external call)
    target NPC = adversarial scenario generator (if G is fairtest_*, the adversarial agent's Phase 1 checkpoint
                 chooses the action from the NPC-view obs and controls the vehicle directly)

This is not a training environment. Neither the test-taker nor the adversary is trained.
Both checkpoints are pre-trained, and this environment collects the collisions, near misses
and SSMs that arise from the interaction of the two policies.

GENERATOR TYPES
    'baseline_idm'  : NPCs drive autonomously with TrafficManager IDM and PID. The target NPC
                      is not controlled separately. This is the baseline G of the grid.
    'fairtest_cutin'   : scenario_type is set to 'cut_in' and an agent checkpoint directly controls
                      an NPC in the adjacent lane.
    'fairtest_rearend' : scenario_type is set to 'rear_end' and an agent checkpoint directly controls
                      the leading NPC in the same lane.

OBSERVATION FOR ADVERSARIAL NPC
    The agent's Phase 1 checkpoint was trained on the 21D ego-centric obs from the ego's view. To
    generate obs from the NPC's view here, _get_observation must be called relative to the
    NPC vehicle. AdversarialCarlaEnv's _get_observation refers to self.ego_vehicle directly,
    so this environment temporarily swaps the ego_vehicle reference for the NPC, computes the
    obs and then restores the reference.

USAGE
    env = GridCarlaEnv(
        generator_type='fairtest_cutin',
        c_level=5,
        adv_policy=agent_phase1_model,  # stable_baselines3 PPO ckpt
        **adversarial_carla_env_kwargs,
    )
    obs, info = env.reset(seed=42)
    while not done:
        action, _ = av_policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.environments.adversarial_carla_env import AdversarialCarlaEnv

logger = logging.getLogger(__name__)


VALID_GENERATOR_TYPES = {"baseline_idm", "fairtest_cutin", "fairtest_rearend"}


class GridCarlaEnv(AdversarialCarlaEnv):
    """Two-agent grid evaluation environment for the AAAI measurement model.

    The ego is the test-taker AV and the target NPC is the adversarial scenario generator. Both
    policies are checkpoints trained beforehand elsewhere, and this environment collects the responses of
    one episode (collision or not, ttc, thw, ego speed, number of steps).
    """

    def __init__(
        self,
        generator_type: str,
        c_level: int,
        adv_policy: Any = None,
        **kwargs: Any,
    ):
        """Initialise GridCarlaEnv.

        Parameters
        ----------
        generator_type : str
            one of 'baseline_idm' | 'fairtest_cutin' | 'fairtest_rearend'.
        c_level : int
            severity level (0~10). Applied to _current_c_level on reset.
        adv_policy : stable_baselines3 model | None
            trained Phase 1 checkpoint of the adversarial agent. If generator_type is 'fairtest_*', this checkpoint
            chooses the target NPC's action at every step. Ignored for
            'baseline_idm'.
        **kwargs : Any
            arguments of the parent class AdversarialCarlaEnv (host, port, town, dt etc.).
        """
        if generator_type not in VALID_GENERATOR_TYPES:
            raise ValueError(
                f"Unknown generator_type '{generator_type}'. "
                f"Supported: {sorted(VALID_GENERATOR_TYPES)}"
            )
        # AdversarialCarlaEnv accepts only 'rear_end' | 'cut_in' as scenario_type.
        # generator_type → scenario_type mapping:
        if generator_type == "fairtest_cutin":
            kwargs.setdefault("scenario_type", "cut_in")
        elif generator_type == "fairtest_rearend":
            kwargs.setdefault("scenario_type", "rear_end")
        else:
            # baseline_idm: follows the parent class's default scenario_type (usually rear_end)
            kwargs.setdefault("scenario_type", "rear_end")

        super().__init__(**kwargs)
        self._generator_type = generator_type
        self._fixed_c_level = int(c_level)
        self._adv_policy = adv_policy
        self._adv_controlled_npc = None  # filled once the target NPC is determined
        self._adv_npc_initialized = False  # whether set_autopilot(False) has been called

        logger.info(
            "GridCarlaEnv initialized: generator=%s, c=%d, adv_policy=%s",
            generator_type, c_level,
            "loaded" if adv_policy is not None else "None",
        )

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset. After the parent reset, set c_level and the adversarial NPC state again."""
        obs, info = super().reset(*args, **kwargs)
        # force the severity level to the grid's c
        self._current_c_level = self._fixed_c_level
        # reset the adversarial NPC state
        self._adv_controlled_npc = None
        self._adv_npc_initialized = False
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Two-agent step.

        The ego is controlled by the test-taker's action, and the target NPC by the
        agent checkpoint or the TrafficManager, depending on generator_type.
        """
        # 1. apply the action to the test-taker (ego): parent class _apply_ego_action as is
        self._apply_ego_action(action)

        # 2. apply the action to the adversary (target NPC): only when generator_type is 'fairtest_*'
        if self._generator_type in ("fairtest_cutin", "fairtest_rearend"):
            self._apply_adversarial_npc_action()

        # 3. CARLA world tick + traffic tick
        self._world.tick()
        if self._traffic_module is not None:
            self._traffic_module.tick()

        # 4. clean up collided NPCs (same logic as the parent class step)
        self._sweep_crashed_npcs()

        # 5. update the ego Frenet position
        ego_loc = self.ego_vehicle.get_location()
        from src.environments.adversarial_carla_env import inertial_to_frenet_approx
        self._ego_s, _ = inertial_to_frenet_approx(
            ego_loc.x, ego_loc.y, self._global_csp, self._ego_s
        )

        # 6. check collision and off-road
        if len(self._ego_collision_history) > 0:
            self._collision_occurred = True

        ego_wp = self._map.get_waypoint(
            ego_loc, project_to_road=True,
            lane_type=self._carla.LaneType.Driving,
        )
        if ego_wp is not None:
            dist_to_road = ego_loc.distance(ego_wp.transform.location)
            self._ego_off_road = dist_to_road > self._lane_width * 1.5
        else:
            self._ego_off_road = True

        # 7. compute SSMs (ttc, thw, drac)
        ssm = self._compute_ssm()
        self._ttc_min_episode = min(self._ttc_min_episode, ssm["ttc"])

        # 8. Termination
        terminated, truncated = self._check_termination()

        # 9. Observation (test-taker's view)
        obs = self._get_observation(ssm=ssm)

        # 10. reward is not used in the grid (the test-taker is not trained). dummy 0.
        reward = 0.0
        reward_components = {"grid_eval": 0.0}

        # 11. Info: AdversarialCarlaEnv's _build_info as is
        agg_features = self._compute_aggressiveness_features(ssm)
        info = self._build_info(ssm, reward_components, agg_features)
        info["generator_type"] = self._generator_type
        info["c_level"] = self._current_c_level
        info["adv_npc_id"] = (
            self._adv_controlled_npc.id
            if self._adv_controlled_npc is not None else None
        )

        # 12. store the previous ego longitudinal velocity for the next step's acceleration
        ego_vel = self.ego_vehicle.get_velocity()
        ego_fwd = self.ego_vehicle.get_transform().get_forward_vector()
        self._prev_ego_vx_longitudinal = (
            ego_vel.x * ego_fwd.x + ego_vel.y * ego_fwd.y
        )

        self._step_count += 1

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Adversarial NPC control
    # ------------------------------------------------------------------
    def _apply_adversarial_npc_action(self) -> None:
        """Apply the action so that the agent's Phase 1 checkpoint controls the target NPC.

        1. When the target NPC is first identified, call set_autopilot(False) to
           release the TrafficManager's autonomous control.
        2. Generate the 21D obs from the NPC's view (temporary swap of the ego_vehicle reference).
        3. Choose the action with the agent checkpoint and call apply_control on the NPC.
        """
        if self._adv_policy is None:
            return

        # look up the target NPC again at every step (the NPC may disappear)
        target_npc, _ = self._find_target_vehicle()
        if target_npc is None or not target_npc.is_alive:
            self._adv_controlled_npc = None
            self._adv_npc_initialized = False
            return

        # release autonomous control on first identification
        if (self._adv_controlled_npc is None
                or self._adv_controlled_npc.id != target_npc.id):
            self._adv_controlled_npc = target_npc
            self._adv_npc_initialized = False

        if not self._adv_npc_initialized:
            try:
                target_npc.set_autopilot(False, self._tm.get_port())
                self._adv_npc_initialized = True
                logger.debug(
                    "Adversarial NPC %d autopilot disabled", target_npc.id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to disable autopilot for NPC %d: %s",
                    target_npc.id, exc,
                )

        # NPC-view obs: temporarily swap the ego_vehicle reference
        original_ego = self.ego_vehicle
        try:
            self.ego_vehicle = target_npc
            npc_obs = self._get_observation(ssm=None)
        finally:
            self.ego_vehicle = original_ego

        # the agent checkpoint chooses the action
        action, _ = self._adv_policy.predict(npc_obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        # apply VehicleControl to the NPC
        from src.environments.adversarial_carla_env import (
            _ACCEL_RANGE, _LANE_OFFSET_RANGE,
        )
        accel_cmd = float(action[0]) * _ACCEL_RANGE[1]
        # the NPC's lane_offset is not controlled by a PID here. Steering is
        # simplified instead (NPC vehicles have no EgoController instance).
        control = self._carla.VehicleControl()
        if accel_cmd >= 0:
            control.throttle = min(accel_cmd / _ACCEL_RANGE[1], 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel_cmd / _ACCEL_RANGE[1], 1.0)
        # simple lateral steering: action[1] (lane_offset_norm) applied proportionally as steer.
        # Precise lane tracking is not essential for the grid. Only the way the agent's adversarial intent
        # sets the position relative to the ego is kept.
        control.steer = float(np.clip(action[1] * 0.3, -1.0, 1.0))
        control.hand_brake = False
        control.manual_gear_shift = False

        try:
            target_npc.apply_control(control)
        except Exception as exc:
            logger.warning(
                "apply_control failed for adversarial NPC %d: %s",
                target_npc.id, exc,
            )

    def _sweep_crashed_npcs(self) -> None:
        """Clean up collided NPCs (the same logic as in AdversarialCarlaEnv.step, split out)."""
        import math
        self._npc_collision = False
        if self._traffic_module is None or self._step_count <= 40:
            return

        to_remove = []
        for i, actor_dict in enumerate(self._traffic_module.actors_batch):
            npc = actor_dict.get("Actor")
            if npc is None or not npc.is_alive:
                to_remove.append(i)
                continue
            npc_vel = npc.get_velocity()
            npc_speed = math.sqrt(npc_vel.x ** 2 + npc_vel.y ** 2)
            if npc_speed < 0.5:
                # but an adversarial NPC controlled by the agent checkpoint may stop for a
                # while, so it is kept and excluded from the collision cleanup.
                if (self._adv_controlled_npc is not None
                        and npc.id == self._adv_controlled_npc.id):
                    continue
                self._npc_collision = True
                try:
                    sensor = actor_dict.get("Sensor")
                    if sensor is not None:
                        try:
                            sensor.stop()
                            sensor.destroy()
                        except Exception:
                            pass
                    npc.destroy()
                except Exception:
                    pass
                to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self._traffic_module.actors_batch.pop(i)
