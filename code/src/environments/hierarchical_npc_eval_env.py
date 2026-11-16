"""Inject hierarchical generators against rule-based or learned defenders.

The generator remains highway-env's ego vehicle and receives the exact 22D
observation used in training. A learned defender replaces the selected
follower with a passive kinematic Vehicle whose physical action is set before
the same simulation tick. Rule-based defenders retain highway-env's IDM/MOBIL
vehicle implementation.
"""
from __future__ import annotations

import numpy as np

from src.environments.hierarchical_interaction_env import HierarchicalInteractionEnv
from src.environments.physical_defender_shield import shield_action
from src.evaluation.defender_observation import native_defender_observation
from src.evaluation.safety_metrics import VehicleState

ENV_VERSION = "hierarchical-npc-eval-v1"
DEFENDER_OBS_VERSION = "native-defender-25d-v1"


class HierarchicalGeneratorNPCEvalEnv(HierarchicalInteractionEnv):
    """Evaluate a generator against IDM, MOBIL/IDM, or a 25D learned policy."""

    DEFENDER_KINDS = {"idm", "mobil", "policy", "shielded_policy"}

    def __init__(self, *, defender_kind="idm", defender_policy=None, **kwargs):
        if defender_kind not in self.DEFENDER_KINDS:
            raise ValueError(f"Unknown defender kind: {defender_kind}")
        if defender_kind in {"policy", "shielded_policy"} and defender_policy is None:
            raise ValueError("A learned defender kind requires defender_policy")
        if defender_kind in {"idm", "mobil"} and defender_policy is not None:
            raise ValueError("Rule-based defender kinds do not accept defender_policy")
        self.defender_kind = defender_kind
        self.defender_policy = defender_policy
        self._defender_is_direct = defender_kind in {"policy", "shielded_policy"}
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self._defender_is_direct:
            from highway_env.vehicle.kinematics import Vehicle

            old_target = self.target
            target_index = self.inner.road.vehicles.index(old_target)
            self.target = Vehicle.create_from(old_target)
            self.inner.road.vehicles[target_index] = self.target
            self.target_id = target_index
        elif hasattr(self.target, "enable_lane_change"):
            self.target.enable_lane_change = self.defender_kind == "mobil"

        info.update(
            environment_version=ENV_VERSION,
            defender_kind=self.defender_kind,
            defender_observation_version=(DEFENDER_OBS_VERSION if self._defender_is_direct else None),
            defender_direct_control=self._defender_is_direct,
            target_id=self.target_id,
        )
        return self._observation(), info

    def _predict_defender_action(self, observation):
        prediction = self.defender_policy.predict(observation, deterministic=True)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid learned defender action")
        return action

    def _prepare_external_actors(self):
        if not self._defender_is_direct:
            return {
                "defender_kind": self.defender_kind,
                "defender_observation_version": None,
                "defender_raw_action": None,
                "defender_applied_action": None,
                "defender_shield_intervened": False,
            }

        observation = native_defender_observation(self.inner, self.target)
        raw_action = self._predict_defender_action(observation)
        applied_action = raw_action
        shield = {"intervened": False, "headway": None, "ttc": None, "brake_cap": None}
        if self.defender_kind == "shielded_policy":
            defender_state = VehicleState.from_vehicle(self.target)
            traffic = [
                VehicleState.from_vehicle(vehicle)
                for vehicle in self.inner.road.vehicles
                if vehicle is not self.target
            ]
            applied_action, shield = shield_action(defender_state, traffic, raw_action)

        physical_action = {
            "acceleration": 5.0 * float(applied_action[0]),
            "steering": 0.3 * float(applied_action[1]),
        }
        self.target.act(physical_action)
        return {
            "defender_kind": self.defender_kind,
            "defender_observation_version": DEFENDER_OBS_VERSION,
            "defender_observation": observation.tolist(),
            "defender_raw_action": raw_action.tolist(),
            "defender_applied_action": applied_action.tolist(),
            "defender_physical_action": physical_action,
            "defender_shield_intervened": shield["intervened"],
            "defender_shield_headway": shield["headway"],
            "defender_shield_ttc": shield["ttc"],
            "defender_shield_brake_cap": shield["brake_cap"],
        }
