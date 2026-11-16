"""Hierarchical cut-in generator with auditable low-level controllers.

The policy controls a longitudinal residual and merge intent.  A deterministic
gap controller and lane-centering controller execute the maneuver.  The merge
is forced after a bounded waiting period and failure to establish the intended
interaction by the deadline is terminal, preventing the no-action solution
observed in the v2 curriculum pilot.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from src.environments.physical_defender_env import body_on_straight_road
from src.environments.physical_defender_shield import shield_action
from src.environments.physical_interaction_env import PhysicalInteractionEnv
from src.evaluation.safety_metrics import VehicleState

ENV_VERSION = "hierarchical-interaction-v4"
OBS_VERSION = "hierarchical-generator-22d-v1"
MAX_LONGITUDINAL_ACCELERATION_MPS2 = 5.0


def longitudinal_controller(actual_lead, desired_lead, generator_speed, target_speed, residual, *, kp=0.08, kd=0.4, residual_weight=0.25):
    """Normalized acceleration for an interpretable lead-gap controller."""
    acceleration = kp * (desired_lead - actual_lead) + kd * (target_speed - generator_speed)
    normalized = acceleration / 5.0 + residual_weight * float(np.clip(residual, -1.0, 1.0))
    return float(np.clip(normalized, -1.0, 1.0))


def lane_controller(lateral_error, heading, *, preview_distance=30.0, heading_gain=0.6, steering_limit=0.3):
    """Preview steering toward a straight lane, normalized to highway-env input."""
    desired_heading = math.atan2(lateral_error, preview_distance)
    heading_error = math.atan2(math.sin(desired_heading - heading), math.cos(desired_heading - heading))
    steering_angle = heading_gain * heading_error
    return float(np.clip(steering_angle / steering_limit, -1.0, 1.0))


def apply_forward_stop_guard(action, speed, dt, *, max_acceleration=MAX_LONGITUDINAL_ACCELERATION_MPS2):
    """Clamp braking so the kinematic vehicle cannot cross through zero speed.

    highway-env's continuous action is normalized against +/-5 m/s^2 and its
    kinematic model permits negative speed.  This final actuator constraint
    represents a forward-only vehicle: braking may stop it, but cannot select
    reverse gear during the same policy step.
    """
    guarded = np.asarray(action, dtype=np.float32).copy()
    if guarded.shape != (2,) or not np.isfinite(guarded).all():
        raise ValueError("Invalid low-level action")
    if not np.isfinite(speed) or not np.isfinite(dt) or dt <= 0 or max_acceleration <= 0:
        raise ValueError("Invalid stop-guard dynamics")
    acceleration_floor = float(np.clip(-max(float(speed), 0.0) / (max_acceleration * dt), -1.0, 0.0))
    original_acceleration = float(guarded[0])
    guarded[0] = max(original_acceleration, acceleration_floor)
    return guarded, bool(guarded[0] > original_acceleration + 1e-9), acceleration_floor


class HierarchicalInteractionEnv(PhysicalInteractionEnv):
    def __init__(self, *, merge_force_step=60, interaction_deadline=140, residual_weight=0.25, initial_lead=25.0,
                 merge_contract="policy_or_deadline", shield_designated_target=True, **kwargs):
        scenario = kwargs.get("scenario_type", "cut_in")
        if scenario not in {None, "cut_in"}:
            raise ValueError("Hierarchical pilot supports cut_in only")
        kwargs["scenario_type"] = "cut_in"
        super().__init__(**kwargs)
        self.merge_force_step = int(merge_force_step)
        self.interaction_deadline = int(interaction_deadline)
        self.residual_weight = float(residual_weight)
        self.initial_lead = float(initial_lead)
        self.shield_designated_target = bool(shield_designated_target)
        if merge_contract not in {"policy_or_deadline", "ready_or_deadline"}:
            raise ValueError("Invalid merge contract")
        self.merge_contract = merge_contract
        if not 1 <= self.merge_force_step < self.interaction_deadline < self.driving._duration:
            raise ValueError("Invalid merge timing contract")
        if not 8.0 <= self.initial_lead <= 50.0:
            raise ValueError("initial_lead must provide a plausible non-overlapping cut-in gap")
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32)
        self._hier_step = 0
        self._merge_active = False
        self._merge_complete = False
        self._ever_valid = False
        self._valid_streak = 0

    def _observation(self):
        base = super()._observation()
        extra = np.asarray([
            float(getattr(self, "_merge_active", False)),
            min(getattr(self, "_hier_step", 0) / self.interaction_deadline, 1.0),
        ], dtype=np.float32)
        return np.concatenate([base, extra]).astype(np.float32)

    def _target_lane_error(self):
        generator = self.inner.vehicle
        longitudinal, _ = self._scenario_lane.local_coordinates(generator.position)
        center = self._scenario_lane.position(longitudinal, 0.0)
        return float(center[1] - generator.position[1])

    def _lead_geometry(self):
        generator, target = self.inner.vehicle, self.target
        target_speed = float(target.velocity[0])
        desired_lead = self.target_headway * max(target_speed, 15.0) + (generator.LENGTH + target.LENGTH) / 2
        actual_lead = float(generator.position[0] - target.position[0])
        return actual_lead, desired_lead, float(generator.velocity[0]), target_speed

    def reset(self, *, seed=None, options=None):
        self._hier_step = 0
        self._merge_active = False
        self._merge_complete = False
        self._ever_valid = False
        self._valid_streak = 0
        _, info = super().reset(seed=seed, options=options)
        generator = self.inner.vehicle
        original_actual_lead = float(generator.position[0] - self.target.position[0])
        # highway-env initializes traffic ahead of ego.  A cut-in experiment
        # needs an explicit adjacent follower, so construct that role rather
        # than asking the controller to overtake an unrelated lead vehicle.
        self.target.position = self.target.position.copy()
        self.target.position[0] = generator.position[0] - self.initial_lead
        self.target_id = self.inner.road.vehicles.index(self.target)
        if hasattr(self.target, "enable_lane_change"):
            self.target.enable_lane_change = False
        self._scenario_lane_index = self.target.lane_index
        self._scenario_lane = self.inner.road.network.get_lane(self._scenario_lane_index)
        self.previous_speed = None
        self.previous_potential = self._potential()
        actual, desired, _, _ = self._lead_geometry()
        self._previous_gap_error = abs(min(desired, 30.0) - actual)
        self._previous_lateral_error = abs(self._target_lane_error())
        info.update(
            environment_version=ENV_VERSION,
            observation_version=OBS_VERSION,
            merge_active=False,
            merge_complete=False,
            merge_contract=self.merge_contract,
            actual_lead=actual,
            desired_lead=desired,
            target_repositioned=True,
            target_original_actual_lead=original_actual_lead,
            target_initial_actual_lead=actual,
        )
        info.update(self._metrics())
        return self._observation(), info

    def _prepare_external_actors(self):
        """Apply actions for externally controlled traffic before one tick."""
        return {}

    def step(self, high_level_action):
        action = np.clip(np.asarray(high_level_action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid hierarchical action")

        old_generator_speed = float(self.inner.vehicle.speed)
        actual_lead, desired_lead, generator_speed, target_speed = self._lead_geometry()
        premerge_desired = min(desired_lead, 30.0)
        ready = actual_lead > 5.0 and abs(premerge_desired - actual_lead) <= 15.0
        requested = bool(action[1] > 0.2)
        forced = bool(self._hier_step >= self.merge_force_step and actual_lead > 5.0)
        contract_triggered = bool(ready and self.merge_contract == "ready_or_deadline")
        if (ready and requested) or contract_triggered or forced:
            self._merge_active = True

        controller_desired = desired_lead if self._merge_complete else premerge_desired
        acceleration = longitudinal_controller(
            actual_lead,
            controller_desired,
            generator_speed,
            target_speed,
            action[0],
            residual_weight=self.residual_weight,
        )
        lateral_error = self._target_lane_error()
        steering = lane_controller(lateral_error, float(self.inner.vehicle.heading)) if self._merge_active else 0.0
        controller_action = np.asarray([acceleration, steering], dtype=np.float32)
        ego_state = VehicleState.from_vehicle(self.inner.vehicle)
        traffic_states = [
            VehicleState.from_vehicle(vehicle)
            for vehicle in self.inner.road.vehicles
            if vehicle is not self.inner.vehicle
            and (self.shield_designated_target or vehicle is not self.target)
        ]
        applied_action, shield = shield_action(ego_state, traffic_states, controller_action)
        applied_action, stop_guard, stop_guard_floor = apply_forward_stop_guard(
            applied_action,
            old_generator_speed,
            1.0 / self.driving._policy_freq,
        )
        external_actor_info = self._prepare_external_actors()

        _, _, terminated, truncated, info = self.driving.step(applied_action)
        self._hier_step += 1
        target_onroad = body_on_straight_road(self.target, self.driving._lanes_count)
        target_collision = bool(self.target.crashed)
        physical_failure = bool(info["collision"] or not info["body_onroad"] or info["wrong_way"] or not target_onroad or target_collision)
        metrics = self._metrics()
        lateral_error_after = self._target_lane_error()
        actual_after, desired_after, _, _ = self._lead_geometry()
        gap_error = abs(desired_after - actual_after)
        lateral_progress = self._previous_lateral_error - abs(lateral_error_after)
        gap_progress = self._previous_gap_error - abs((desired_after if self._merge_complete else min(desired_after, 30.0)) - actual_after)
        self._previous_lateral_error = abs(lateral_error_after)
        self._previous_gap_error = abs((desired_after if self._merge_complete else min(desired_after, 30.0)) - actual_after)
        # Match PhysicalInteractionEnv: the returned observation uses the
        # speed immediately before this transition to expose actual acceleration.
        self.previous_speed = old_generator_speed

        same_lane = self.inner.vehicle.lane_index[2] == self._scenario_lane_index[2]
        if self._merge_active and same_lane and abs(lateral_error_after) < 0.5:
            self._merge_complete = True
        valid = bool(metrics["interaction_valid"] and metrics["follower_forward_speed"] >= 15.0)
        self._ever_valid |= valid
        self._valid_streak = self._valid_streak + 1 if valid else 0
        deadline_failure = bool(self._hier_step >= self.interaction_deadline and not self._ever_valid)
        failure = physical_failure or deadline_failure
        terminated = bool(terminated or failure)
        truncated = bool(truncated and not terminated)

        if failure:
            reward = -50.0 if physical_failure else -30.0
            parts = {"terminal_failure": reward, "gap_progress": 0.0, "lateral_progress": 0.0, "tracking": 0.0, "interaction": 0.0}
        else:
            tracking = 0.75 * math.exp(-abs(metrics["headway"] - self.target_headway) / 0.75) if valid else 0.0
            interaction = 0.25 if valid else 0.0
            parts = {
                "gap_progress": 0.08 * float(np.clip(gap_progress, -2.0, 2.0)),
                "lateral_progress": 0.4 * float(np.clip(lateral_progress, -0.5, 0.5)) if self._merge_active else 0.0,
                "readiness": 0.05 * math.exp(-abs(min(desired_after, 30.0) - actual_after) / 15.0),
                "tracking": tracking,
                "interaction": interaction,
                "speed": 0.02 * float(np.clip(info["forward_speed"] / 30.0, 0.0, 1.0)),
                "residual_cost": -0.01 * float(action[0]) ** 2,
                "terminal_failure": 0.0,
            }
            reward = float(sum(parts.values()))

        if deadline_failure:
            info["termination_reason"] = "interaction_deadline"
        elif target_collision and not info["collision"]:
            info["termination_reason"] = "target_collision"
        elif not target_onroad and info["body_onroad"]:
            info["termination_reason"] = "target_offroad"
        info.update(
            environment_version=ENV_VERSION,
            observation_version=OBS_VERSION,
            scenario_requested="cut_in",
            scenario_actual="cut_in",
            target_id=self.target_id,
            c_level=self.c,
            target_headway=self.target_headway,
            target_onroad=target_onroad,
            target_collision=target_collision,
            merge_requested=requested,
            merge_contract=self.merge_contract,
            merge_contract_triggered=contract_triggered,
            merge_forced=forced,
            merge_active=self._merge_active,
            merge_complete=self._merge_complete,
            interaction_ever_valid=self._ever_valid,
            valid_streak=self._valid_streak,
            interaction_deadline_failure=deadline_failure,
            actual_lead=actual_after,
            desired_lead=desired_after,
            gap_error=gap_error,
            lateral_error=lateral_error_after,
            high_level_action=action.tolist(),
            controller_action=controller_action.tolist(),
            applied_action=applied_action.tolist(),
            shield_intervened=shield["intervened"],
            shield_designated_target=self.shield_designated_target,
            shield_headway=shield["headway"],
            shield_ttc=shield["ttc"],
            shield_brake_cap=shield["brake_cap"],
            low_speed_stop_guard_intervened=stop_guard,
            low_speed_stop_guard_floor=stop_guard_floor,
            reward_components=parts,
            **metrics,
        )
        info.update(external_actor_info)
        return self._observation(), reward, terminated, truncated, info
