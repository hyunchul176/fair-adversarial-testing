"""
THW-based adversarial reward module for scenario generation.

Extracted from AdversarialCarlaEnv._compute_reward / _compute_r_type so that
the same reward logic can be reused in highway-env or any other driving
environment without importing CARLA.

Research motivation:
    TTC = distance / relative_speed requires constant approach (delta_v > 0)
    to be finite, making collision structurally inevitable.
    THW = distance / ego_speed can be maintained at any delta_v, including
    delta_v ~ 0 (speed matching), enabling stable car-following.
    Severity mapping: THW_target = 3.0 - 2.5 * c_level
        c=0.0 -> THW=3.0s  (conservative)
        c=0.5 -> THW=1.75s (moderate -- HighD THW median is 1.71s)
        c=0.8 -> THW=1.0s  (aggressive -- HighD aggressive THW < 1.0s)
        c=1.0 -> THW=0.5s  (extreme)

Reference: adversarial_carla_env.py (CARLA implementation, validated in Phase 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Module-level constants (mirrored from adversarial_carla_env.py)
# ---------------------------------------------------------------------------
_DEFAULT_MAX_DISTANCE: float = 100.0   # metres
_DEFAULT_MAX_THW: float = 10.0         # seconds
_DEFAULT_MIN_REWARD_SPEED: float = 20.0  # m/s


@dataclass
class THWReward:
    """
    THW-based adversarial reward for scenario generation.

    R_base = w_type * R_type + w_sev * R_severity + w_ssm * R_ssm
             + r_near_miss + r_collision

    Validated in CARLA, now used in highway-env.

    Parameters
    ----------
    w_type : float
        Weight for scenario-type achievement reward.
    w_severity : float
        Weight for THW target-tracking reward.
    w_ssm : float
        Weight for surrogate-safety-measure penalty.
    min_reward_speed : float
        Speed (m/s) at which the positive severity reward reaches full scale.
    max_distance : float
        Distance (m) beyond which the ego is considered "not interacting".
    max_thw : float
        THW (s) above which we treat the ego as too far (fallback penalty).
    """

    w_type: float = 1.0
    w_severity: float = 3.0
    w_ssm: float = 1.5
    min_reward_speed: float = _DEFAULT_MIN_REWARD_SPEED
    max_distance: float = _DEFAULT_MAX_DISTANCE
    max_thw: float = _DEFAULT_MAX_THW

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def compute(
        self,
        ego_speed: float,
        thw_actual: float,
        thw_target: float,
        closing_speed: float,
        distance: float,
        ttc: float,
        drac: float,
        collision: bool,
        off_road: bool,
        scenario_type: str,
        c_level: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute the full base adversarial reward.

        Parameters
        ----------
        ego_speed : float
            Ego longitudinal speed (m/s).
        thw_actual : float
            Measured time headway to leading vehicle (s).
        thw_target : float
            Desired THW derived from c_level: ``3.0 - 2.5 * c_level``.
        closing_speed : float
            Rate of gap closure to lead vehicle (m/s, positive = closing).
        distance : float
            Euclidean bumper-to-bumper distance to lead vehicle (m).
        ttc : float
            Time-to-collision (s).  Use a large sentinel (e.g. 999) when
            the vehicles are diverging.
        drac : float
            Deceleration Rate to Avoid Collision (m/s^2).
        collision : bool
            Whether the ego collided this step.
        off_road : bool
            Whether the ego left the road surface.
        scenario_type : str
            ``"rear_end"`` or ``"cut_in"``.
        c_level : float
            Severity conditioning level in [0, 1].

        Returns
        -------
        r_total : float
            Scalar reward.
        components : dict
            Breakdown: r_type, r_severity, r_ssm, r_near_miss, r_collision,
            r_total, thw_target.
        """
        r_type = self._compute_r_type(
            distance=distance,
            ttc=ttc,
            thw_actual=thw_actual,
            thw_target=thw_target,
            collision=collision,
            scenario_type=scenario_type,
        )

        r_severity = self._compute_r_severity(
            ego_speed=ego_speed,
            thw_actual=thw_actual,
            thw_target=thw_target,
            closing_speed=closing_speed,
            distance=distance,
        )

        r_ssm = self._compute_r_ssm(
            ego_speed=ego_speed,
            ttc=ttc,
            drac=drac,
            collision=collision,
            off_road=off_road,
        )

        r_near_miss = self._compute_r_near_miss(
            ego_speed=ego_speed,
            thw_actual=thw_actual,
            thw_target=thw_target,
            collision=collision,
        )

        r_collision = -3.0 if collision else 0.0

        r_total = (
            self.w_type * r_type
            + self.w_severity * r_severity
            + self.w_ssm * r_ssm
            + r_near_miss
            + r_collision
        )

        components = {
            "r_type": r_type,
            "r_severity": r_severity,
            "r_ssm": r_ssm,
            "r_near_miss": r_near_miss,
            "r_collision": r_collision,
            "r_total": r_total,
            "thw_target": thw_target,
        }
        return r_total, components

    # ------------------------------------------------------------------ #
    # THW target-tracking severity reward                                 #
    # ------------------------------------------------------------------ #
    def _compute_r_severity(
        self,
        ego_speed: float,
        thw_actual: float,
        thw_target: float,
        closing_speed: float,
        distance: float,
    ) -> float:
        """
        R_severity: how well the ego tracks the commanded THW.

        Speed gating: positive rewards are scaled by
        ``min(ego_speed / min_reward_speed, 1.0)``; negative rewards are
        **not** scaled (penalties always apply at full strength).
        """
        speed_factor = min(ego_speed / self.min_reward_speed, 1.0)

        if ego_speed > 1.0 and distance < self.max_distance:
            proximity_bonus = 0.5 * max(0.0, 1.0 - distance / self.max_distance)

            if thw_actual < self.max_thw:
                thw_error = min(
                    abs(thw_actual - thw_target) / max(thw_target, 0.3), 1.0
                )
                base_severity = -thw_error + proximity_bonus

                # Bonus: close to target THW
                if abs(thw_actual - thw_target) < 0.3:
                    base_severity += 0.5

                # Bonus: speed-matched and reasonably close
                if abs(closing_speed) < 1.0 and thw_actual < thw_target * 2.0:
                    base_severity += 0.3

                # Speed gate: scale positive, keep negative
                if base_severity > 0:
                    return base_severity * speed_factor
                return base_severity
            else:
                # THW too large -- not really following
                return -1.0 + proximity_bonus
        else:
            # Stopped or too far away
            return -1.0

    # ------------------------------------------------------------------ #
    # Surrogate safety measure penalties                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_r_ssm(
        ego_speed: float,
        ttc: float,
        drac: float,
        collision: bool,
        off_road: bool,
    ) -> float:
        """
        R_ssm: physical-plausibility penalties.

        - TTC < 0.3s:   -3.0  (unavoidable crash)
        - DRAC > 8.0:   -1.0  (unrealistic deceleration required)
        - Collision:     -2.0  (unrealistic outcome on highway)
        - Off-road:      -2.0
        - Speed < 15 m/s: up to -2.0 linear penalty (highway realism)
        """
        r_ssm = 0.0

        if ttc < 0.3:
            r_ssm -= 3.0

        if drac > 8.0:
            r_ssm -= 1.0

        if collision:
            r_ssm -= 2.0

        if off_road:
            r_ssm -= 2.0

        if ego_speed < 15.0:
            r_ssm -= 2.0 * (1.0 - ego_speed / 15.0)

        return r_ssm

    # ------------------------------------------------------------------ #
    # Near-miss bonus                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_r_near_miss(
        ego_speed: float,
        thw_actual: float,
        thw_target: float,
        collision: bool,
    ) -> float:
        """
        +0.3 for stable close following within 120% of target THW,
        above THW 0.3 s, no collision, and moving.
        """
        if (
            not collision
            and ego_speed > 1.0
            and thw_actual < thw_target * 1.2
            and thw_actual > 0.3
        ):
            return 0.3
        return 0.0

    # ------------------------------------------------------------------ #
    # Scenario-type achievement reward                                    #
    # ------------------------------------------------------------------ #
    def _compute_r_type(
        self,
        distance: float,
        ttc: float,
        thw_actual: float,
        thw_target: float,
        collision: bool,
        scenario_type: str,
    ) -> float:
        """
        R_type: scenario-type achievement.

        Collision is PUNISHED (-0.5). Sustained close interaction is
        rewarded. Near-misses (TTC < 1.0, no collision) are valuable.

        For ``rear_end``:
            - Collision:      -0.5
            - Close following (dist < 30, TTC > 1.0): proportional to closeness
            - Near-miss (TTC < 1.0, no collision): 0.5
            - In range (< max_distance) but far: 0.0
            - Out of range:  -0.5

        For ``cut_in`` (simplified -- no lane query in highway-env):
            - Collision:      -0.5
            - Close (dist < 15):  proportional to closeness
            - Medium (dist < 30): 0.3 * proportional
            - Far:            0.0 / -0.5
        """
        if collision:
            return -0.5

        if scenario_type == "rear_end":
            if distance < 30.0 and ttc > 1.0:
                return max(0.0, min((30.0 - distance) / 30.0, 1.0))
            elif 0.0 < ttc < 1.0:
                return 0.5
            elif distance < self.max_distance:
                return 0.0
            return -0.5

        elif scenario_type == "cut_in":
            # Simplified cut-in: lateral proximity handled externally.
            # Here we only check longitudinal distance.
            if distance < 15.0:
                return max(0.0, min((15.0 - distance) / 15.0, 1.0))
            elif distance < 30.0:
                return 0.3 * max(0.0, min((30.0 - distance) / 30.0, 1.0))
            elif distance < self.max_distance:
                return 0.0
            return -0.5

        return 0.0

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def thw_target_from_c_level(c_level: float) -> float:
        """Map severity conditioning level to target THW (seconds)."""
        return 3.0 - 2.5 * c_level
