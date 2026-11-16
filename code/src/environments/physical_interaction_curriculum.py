"""Scenario-specific curriculum environment for physical interaction v2.

The main change is dense directional credit for cut-in lane acquisition.  V1
gave an invalid interaction almost constant reward, while its potential change
was too small to teach steering.  This version remains collision terminal and
does not relax the physical validity contract.
"""
from __future__ import annotations

import math

import numpy as np

from src.environments.physical_defender_env import body_on_straight_road
from src.environments.physical_interaction_env import PhysicalInteractionEnv, physical_interaction_reward

ENV_VERSION = "physical-interaction-curriculum-v2"


def cutin_curriculum_reward(
    metrics,
    target_headway,
    forward_speed,
    action,
    lateral_progress,
    lead_progress,
    failure,
):
    if failure:
        return -50.0, {
            "terminal_failure": -50.0,
            "lane_progress": 0.0,
            "lead_progress": 0.0,
            "tracking": 0.0,
            "interaction": 0.0,
        }
    valid = metrics["interaction_valid"] and metrics["follower_forward_speed"] >= 15.0
    lane_term = 1.0 * float(np.clip(lateral_progress, -0.5, 0.5))
    lead_term = 0.25 * float(np.clip(lead_progress / 5.0, -0.2, 0.2))
    tracking = 0.5 * math.exp(-abs(metrics["headway"] - target_headway) / 0.75) if valid else -0.02
    interaction = 0.25 if valid else 0.0
    speed = 0.03 * float(np.clip(forward_speed / 30.0, 0.0, 1.0))
    ttc = metrics["ttc"]
    risk = -1.0 * max(0.0, 1.0 - ttc / 1.0) if ttc is not None else 0.0
    steering = -0.005 * float(action[1]) ** 2
    parts = {
        "lane_progress": lane_term,
        "lead_progress": lead_term,
        "tracking": tracking,
        "interaction": interaction,
        "speed": speed,
        "risk": risk,
        "steering": steering,
        "terminal_failure": 0.0,
    }
    return float(sum(parts.values())), parts


class PhysicalInteractionCurriculumEnv(PhysicalInteractionEnv):
    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._previous_lateral_error, self._previous_lead_error = self._curriculum_errors()
        info["environment_version"] = ENV_VERSION
        return observation, info

    def _curriculum_errors(self):
        generator, target = self.inner.vehicle, self.target
        lateral_error = abs(float(generator.position[1] - target.position[1]))
        desired = self.target_headway * max(float(target.velocity[0]), 15.0) + (generator.LENGTH + target.LENGTH) / 2
        actual = float(generator.position[0] - target.position[0])
        return lateral_error, abs(actual - desired)

    def step(self, action):
        if self.target is None or self.target not in self.inner.road.vehicles:
            raise RuntimeError("Designated target is unavailable")
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid action")
        old_speed = float(self.inner.vehicle.speed)
        _, _, terminated, truncated, info = self.driving.step(action)
        target_onroad = body_on_straight_road(self.target, self.driving._lanes_count)
        target_collision = bool(self.target.crashed)
        failure = info["collision"] or not info["body_onroad"] or info["wrong_way"] or not target_onroad or target_collision
        terminated = bool(terminated or failure)
        truncated = bool(truncated and not terminated)
        metrics = self._metrics()
        lateral_error, lead_error = self._curriculum_errors()
        lateral_progress = self._previous_lateral_error - lateral_error
        lead_progress = self._previous_lead_error - lead_error
        if self.scenario == "cut_in":
            reward, parts = cutin_curriculum_reward(
                metrics,
                self.target_headway,
                info["forward_speed"],
                action,
                lateral_progress,
                lead_progress,
                failure,
            )
        else:
            potential = 0.0 if terminated or truncated else self._potential()
            reward, parts = physical_interaction_reward(
                metrics,
                self.target_headway,
                info["forward_speed"],
                action,
                0.99 * potential - self.previous_potential,
                failure,
            )
            self.previous_potential = potential
        self._previous_lateral_error = lateral_error
        self._previous_lead_error = lead_error
        self.previous_speed = old_speed
        info.update(
            environment_version=ENV_VERSION,
            observation_version="shared-generator-20d-v1",
            scenario_requested=self.scenario,
            scenario_actual=self.scenario,
            target_id=self.target_id,
            c_level=self.c,
            target_headway=self.target_headway,
            target_onroad=target_onroad,
            target_collision=target_collision,
            lateral_error=lateral_error,
            lead_error=lead_error,
            lateral_progress=lateral_progress,
            lead_progress=lead_progress,
            reward_components=parts,
            **metrics,
        )
        if target_collision and not info["collision"]:
            info["termination_reason"] = "target_collision"
        elif not target_onroad and info["body_onroad"]:
            info["termination_reason"] = "target_offroad"
        return self._observation(), reward, terminated, truncated, info
