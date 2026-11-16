"""Transparent fixed-action reference for adversarial-generator experiments."""
from __future__ import annotations

import numpy as np


def compose_primitive_action(residual, anchor, scale):
    """Compose a bounded state-dependent residual around a fixed primitive."""
    residual = np.asarray(residual, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    scale = float(scale)
    if residual.shape[-1:] != (2,) or anchor.shape != (2,):
        raise ValueError("Residual and anchor actions must end in two values")
    if not np.isfinite(residual).all() or not np.isfinite(anchor).all() or not 0.0 < scale <= 1.0:
        raise ValueError("Residual composition inputs must be finite and scale in (0, 1]")
    return np.clip(anchor + scale * residual, -1.0, 1.0)


def ttc_regulator_action(observation, target_ttc, gain, *, pre_acceleration=0.0,
                         recovery_floor=0.4, recovery_action=1.0,
                         closing_speed_floor=0.05):
    """Compute a merge action that regulates TTC only while the follower closes.

    Observation index 8 is target-minus-generator longitudinal velocity scaled
    by 30 m/s.  In a cut-in, the target is the follower after the merge, so a
    positive value is the condition under which braking can reduce TTC.
    """
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape[-1:] != (22,) or not np.isfinite(observation).all():
        raise ValueError("TTC regulator requires finite registered 22D observations")
    target_ttc = float(target_ttc)
    gain = float(gain)
    pre_acceleration = float(pre_acceleration)
    recovery_floor = float(recovery_floor)
    recovery_action = float(recovery_action)
    closing_speed_floor = float(closing_speed_floor)
    if not (0.5 <= target_ttc <= 5.0 and 0.0 < gain <= 2.0
            and -1.0 <= pre_acceleration <= 1.0
            and 0.0 < recovery_floor < target_ttc
            and -1.0 <= recovery_action <= 1.0
            and 0.0 <= closing_speed_floor <= 5.0):
        raise ValueError("Invalid TTC regulator parameters")
    scalar = observation.ndim == 1
    batch = observation[None, :] if scalar else observation
    interaction_valid = batch[:, 17] > 0.5
    closing_speed = batch[:, 8] * 30.0
    closing = closing_speed > closing_speed_floor
    observed_ttc = np.clip(batch[:, 13], 0.0, 1.0) * 10.0
    regulated = np.clip(gain * (target_ttc - observed_ttc), -1.0, 1.0)
    regulated = np.where(
        observed_ttc < recovery_floor, recovery_action, regulated
    )
    acceleration = np.where(
        interaction_valid & closing, regulated, pre_acceleration
    ).astype(np.float32)
    actions = np.stack([acceleration, np.ones_like(acceleration)], axis=1)
    return actions[0] if scalar else actions


class ConstantPolicy:
    """Expose a fixed maneuver through the Stable-Baselines predict contract."""

    def __init__(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("risk_primitive_action must contain two finite values")
        self.action = np.clip(action, -1.0, 1.0)

    def predict(self, observation, deterministic=True):
        del deterministic
        observation = np.asarray(observation)
        if observation.ndim <= 1:
            return self.action.copy(), None
        return np.repeat(self.action[None, :], observation.shape[0], axis=0), None


class PrimitiveResidualPolicy:
    """Compose a learned policy output with a fixed primitive at inference."""

    def __init__(self, policy, anchor, scale):
        self.policy = policy
        self.anchor = ConstantPolicy(anchor).action
        self.scale = float(scale)

    def predict(self, observation, deterministic=True):
        prediction = self.policy.predict(observation, deterministic=deterministic)
        residual = prediction[0] if isinstance(prediction, tuple) else prediction
        state = prediction[1] if isinstance(prediction, tuple) and len(prediction) > 1 else None
        return compose_primitive_action(residual, self.anchor, self.scale), state


class TTCResidualPolicy:
    """Compose a learned residual with the closed-loop TTC primitive."""

    def __init__(self, policy, regulator, scale):
        self.policy = policy
        self.regulator = dict(regulator)
        self.scale = float(scale)
        ttc_regulator_action(np.zeros(22, dtype=np.float32), **self.regulator)

    def predict(self, observation, deterministic=True):
        prediction = self.policy.predict(observation, deterministic=deterministic)
        residual = prediction[0] if isinstance(prediction, tuple) else prediction
        state = prediction[1] if isinstance(prediction, tuple) and len(prediction) > 1 else None
        anchor = ttc_regulator_action(observation, **self.regulator)
        return compose_primitive_action(residual, anchor, self.scale), state


class TemporalBrakePolicy:
    """Merge immediately, then switch from a pre-action to a braking action."""

    def __init__(self, pre_acceleration, brake_start_step, *,
                 post_acceleration=-1.0, interaction_deadline=140):
        self.pre_acceleration = float(pre_acceleration)
        self.post_acceleration = float(post_acceleration)
        self.brake_start_step = int(brake_start_step)
        self.interaction_deadline = int(interaction_deadline)
        if not (-1.0 <= self.pre_acceleration <= 1.0
                and -1.0 <= self.post_acceleration <= 1.0
                and 0 <= self.brake_start_step < self.interaction_deadline):
            raise ValueError("Invalid temporal brake primitive")

    def predict(self, observation, deterministic=True):
        del deterministic
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1:] != (22,):
            raise ValueError("Temporal brake primitive requires the registered 22D observation")
        scalar = observation.ndim == 1
        batch = observation[None, :] if scalar else observation
        steps = batch[:, 21] * self.interaction_deadline
        acceleration = np.where(
            steps < self.brake_start_step,
            self.pre_acceleration,
            self.post_acceleration,
        ).astype(np.float32)
        actions = np.stack([acceleration, np.ones_like(acceleration)], axis=1)
        return (actions[0] if scalar else actions), None


class ObservationTriggeredBrakePolicy:
    """Brake after an auditable interaction, headway, or TTC trigger."""

    def __init__(self, trigger, *, pre_acceleration=0.0, post_acceleration=-1.0,
                 threshold=None):
        if trigger not in {"interaction_valid", "headway", "ttc"}:
            raise ValueError("Unknown observation trigger")
        self.trigger = trigger
        self.pre_acceleration = float(pre_acceleration)
        self.post_acceleration = float(post_acceleration)
        self.threshold = None if threshold is None else float(threshold)
        if not (-1.0 <= self.pre_acceleration <= 1.0
                and -1.0 <= self.post_acceleration <= 1.0):
            raise ValueError("Invalid acceleration")
        if trigger == "interaction_valid" and threshold is not None:
            raise ValueError("interaction_valid does not use a threshold")
        if trigger in {"headway", "ttc"} and (self.threshold is None or self.threshold <= 0.0):
            raise ValueError("headway and ttc triggers require a positive threshold")

    def predict(self, observation, deterministic=True):
        del deterministic
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape[-1:] != (22,):
            raise ValueError("Observation-triggered primitive requires the registered 22D observation")
        scalar = observation.ndim == 1
        batch = observation[None, :] if scalar else observation
        interaction_valid = batch[:, 17] > 0.5
        if self.trigger == "interaction_valid":
            triggered = interaction_valid
        elif self.trigger == "headway":
            triggered = interaction_valid & (batch[:, 12] * 10.0 <= self.threshold)
        else:
            triggered = interaction_valid & (batch[:, 13] * 10.0 <= self.threshold)
        acceleration = np.where(
            triggered, self.post_acceleration, self.pre_acceleration
        ).astype(np.float32)
        actions = np.stack([acceleration, np.ones_like(acceleration)], axis=1)
        return (actions[0] if scalar else actions), None


class TTCRegulatorPolicy:
    """Closed-loop residual controller that regulates TTC after interaction."""

    def __init__(self, target_ttc, gain, *, pre_acceleration=0.0,
                 recovery_floor=0.4, recovery_action=1.0,
                 closing_speed_floor=0.05):
        self.target_ttc = float(target_ttc)
        self.gain = float(gain)
        self.pre_acceleration = float(pre_acceleration)
        self.recovery_floor = float(recovery_floor)
        self.recovery_action = float(recovery_action)
        self.closing_speed_floor = float(closing_speed_floor)
        self.parameters = {
            "target_ttc": self.target_ttc,
            "gain": self.gain,
            "pre_acceleration": self.pre_acceleration,
            "recovery_floor": self.recovery_floor,
            "recovery_action": self.recovery_action,
            "closing_speed_floor": self.closing_speed_floor,
        }
        ttc_regulator_action(np.zeros(22, dtype=np.float32), **self.parameters)

    def predict(self, observation, deterministic=True):
        del deterministic
        return ttc_regulator_action(observation, **self.parameters), None
