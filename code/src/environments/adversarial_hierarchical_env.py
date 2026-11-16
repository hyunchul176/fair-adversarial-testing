"""Risk-seeking hierarchical generator with a background-only safety shield."""
from __future__ import annotations

import numpy as np

from src.environments.hierarchical_npc_eval_env import HierarchicalGeneratorNPCEvalEnv
from src.evaluation.risk_primitive_policy import compose_primitive_action, ttc_regulator_action

ENV_VERSION = "adversarial-hierarchical-generator-v4"


def closing_risk_signals(ttc, headway, closing_speed, follower_speed, *,
                         ttc_horizon, headway_threshold, closing_speed_scale):
    """Bounded dense risk signals for a cut-in leader and its follower."""
    closing_speed = max(0.0, float(closing_speed))
    closing_pressure = min(closing_speed / closing_speed_scale, 1.0)
    headway_pressure = (0.0 if headway is None else
                        max(0.0, 1.0 - float(headway) / headway_threshold))
    effective_ttc = None if ttc is None else float(ttc)
    if effective_ttc is None and headway is not None and closing_speed > 1e-6:
        gap = max(0.0, float(headway) * max(float(follower_speed), 0.0))
        effective_ttc = gap / closing_speed
    # A hard cutoff at ``ttc_horizon`` made the rare-event objective sparse:
    # trajectories with TTC 6.1 s and 60 s received the same TTC signal.  The
    # exponential potential remains bounded while providing a smooth approach
    # gradient from the far field.
    ttc_pressure = (0.0 if effective_ttc is None else float(np.clip(
        np.exp(-max(0.0, effective_ttc) / ttc_horizon), 0.0, 1.0)))
    risk_potential = float(np.clip(
        0.70 * ttc_pressure + 0.20 * headway_pressure + 0.10 * closing_pressure,
        0.0, 1.0,
    ))
    return {
        "closing_speed": closing_speed,
        "closing_pressure": closing_pressure,
        "effective_ttc": effective_ttc,
        "ttc_pressure": ttc_pressure,
        "headway_pressure": headway_pressure,
        "risk_potential": risk_potential,
    }


def first_ttc_event_reward(effective_ttc, *, threshold, bonus, already_awarded):
    """Return a one-shot reward when a rollout first crosses the TTC gate."""
    event = effective_ttc is not None and float(effective_ttc) <= float(threshold)
    return (float(bonus) if event and not already_awarded else 0.0,
            bool(already_awarded or event))


class AdversarialHierarchicalGeneratorEnv(HierarchicalGeneratorNPCEvalEnv):
    """Train risk toward the designated defender while protecting other actors."""

    def __init__(self, *, risk_ttc_threshold=6.0, risk_headway_threshold=1.5,
                 closing_speed_scale=6.0, pair_collision_bonus=8.0,
                 ttc_event_threshold=1.5, ttc_event_bonus=6.0,
                 action_anchor=None, ttc_action_anchor=None, action_residual_scale=1.0,
                 pair_collision_reward=None,
                 background_failure_penalty=12.0, offroad_penalty=12.0, **kwargs):
        if kwargs.get("shield_designated_target", False):
            raise ValueError("Adversarial training requires a background-only generator shield")
        kwargs["shield_designated_target"] = False
        super().__init__(**kwargs)
        self.risk_ttc_threshold = float(risk_ttc_threshold)
        self.risk_headway_threshold = float(risk_headway_threshold)
        self.closing_speed_scale = float(closing_speed_scale)
        self.pair_collision_bonus = float(pair_collision_bonus)
        self.ttc_event_threshold = float(ttc_event_threshold)
        self.ttc_event_bonus = float(ttc_event_bonus)
        self.action_anchor = (None if action_anchor is None else
                              np.asarray(action_anchor, dtype=np.float32))
        self.ttc_action_anchor = None if ttc_action_anchor is None else dict(ttc_action_anchor)
        self.action_residual_scale = float(action_residual_scale)
        self.pair_collision_reward = float(
            pair_collision_bonus if pair_collision_reward is None else pair_collision_reward
        )
        self.background_failure_penalty = float(background_failure_penalty)
        self.offroad_penalty = float(offroad_penalty)
        if min(self.risk_ttc_threshold, self.risk_headway_threshold, self.closing_speed_scale,
               self.pair_collision_bonus, self.ttc_event_threshold, self.ttc_event_bonus,
               self.background_failure_penalty, self.offroad_penalty) <= 0:
            raise ValueError("Adversarial reward parameters must be positive")
        if self.action_anchor is not None:
            compose_primitive_action(np.zeros(2, dtype=np.float32), self.action_anchor,
                                     self.action_residual_scale)
        if self.action_anchor is not None and self.ttc_action_anchor is not None:
            raise ValueError("Choose either a constant or TTC action anchor")
        if self.ttc_action_anchor is not None:
            ttc_regulator_action(np.zeros(22, dtype=np.float32), **self.ttc_action_anchor)
            compose_primitive_action(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32),
                                     self.action_residual_scale)
        if not np.isfinite(self.pair_collision_reward):
            raise ValueError("pair_collision_reward must be finite")

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._previous_risk_potential = 0.0
        self._ttc_event_awarded = False
        self._anchor_observation = np.asarray(observation, dtype=np.float32).copy()
        info["environment_version"] = ENV_VERSION
        return observation, info

    def step(self, high_level_action):
        policy_action = np.clip(np.asarray(high_level_action, dtype=np.float32), -1.0, 1.0)
        if self.ttc_action_anchor is not None:
            anchor_action = ttc_regulator_action(
                self._anchor_observation, **self.ttc_action_anchor
            )
        else:
            anchor_action = self.action_anchor
        executed_action = (policy_action if anchor_action is None else
                           compose_primitive_action(policy_action, anchor_action,
                                                    self.action_residual_scale))
        observation, _, terminated, truncated, info = super().step(executed_action)
        self._anchor_observation = np.asarray(observation, dtype=np.float32).copy()
        generator_collision = bool(info.get("collision"))
        defender_collision = bool(info.get("target_collision"))
        pair_collision = generator_collision and defender_collision
        background_collision = generator_collision != defender_collision
        offroad = bool(not info.get("body_onroad", True) or not info.get("target_onroad", True)
                       or info.get("wrong_way", False))

        merged = bool(info.get("merge_complete", False))
        closing_speed = (float(self.target.velocity[0]) - float(self.inner.vehicle.velocity[0])) if merged else 0.0
        signals = closing_risk_signals(
            info.get("ttc"), info.get("headway"), closing_speed,
            info.get("follower_forward_speed", 0.0),
            ttc_horizon=self.risk_ttc_threshold,
            headway_threshold=self.risk_headway_threshold,
            closing_speed_scale=self.closing_speed_scale,
        )
        risk_progress = signals["risk_potential"] - self._previous_risk_potential
        self._previous_risk_potential = signals["risk_potential"]
        ttc_event_reward, self._ttc_event_awarded = first_ttc_event_reward(
            signals["effective_ttc"], threshold=self.ttc_event_threshold,
            bonus=self.ttc_event_bonus, already_awarded=self._ttc_event_awarded,
        ) if merged else (0.0, self._ttc_event_awarded)
        action = policy_action

        components = {
            "risk_state": 2.0 * signals["risk_potential"],
            "risk_progress": 4.0 * float(np.clip(risk_progress, -0.25, 0.25)),
            "closing": 0.5 * signals["closing_pressure"] if merged else 0.0,
            "pair_collision": self.pair_collision_reward if pair_collision else 0.0,
            "ttc_event": ttc_event_reward,
            "merge": 0.10 if info.get("merge_complete", False) else 0.0,
            "interaction": 0.10 if info.get("interaction_valid", False) else 0.0,
            "background_failure": -self.background_failure_penalty if background_collision else 0.0,
            "offroad": -self.offroad_penalty if offroad else 0.0,
            "shield_use": -0.02 if info.get("shield_intervened", False) else 0.0,
            "action_cost": -0.002 * float(np.square(action).sum()),
        }
        reward = float(sum(components.values()))
        info.update(
            environment_version=ENV_VERSION,
            pair_collision=pair_collision,
            background_collision=background_collision,
            risk_pressure=signals["risk_potential"],
            risk_progress=risk_progress,
            ttc_event_rewarded=bool(ttc_event_reward),
            policy_residual_action=(policy_action.tolist() if anchor_action is not None else None),
            anchor_high_level_action=(anchor_action.tolist() if anchor_action is not None else None),
            executed_high_level_action=executed_action.tolist(),
            **signals,
            adversarial_reward_components=components,
        )
        return observation, reward, terminated, truncated, info
