"""
Method C: Rule-Based Constraint Baseline (AuthSim-style)

PURPOSE:
    Implements an AuthSim-inspired 3-layer safety region reward for adversarial
    scenario generation.  This serves as a baseline that uses hard-coded
    distance thresholds instead of:
      - THW-based severity tracking (Method A / our proposed method)
      - Behavioral realism constraint via Hellinger distance (our method)

DESIGN:
    The reward function partitions ego-to-target distance into three regions:

        distance < 10 m  -->  DANGER zone   (penalized: too close / collision-prone)
        10 <= dist < 30 m -> BOUNDARY zone  (rewarded: this is the "sweet spot")
        distance >= 30 m  -->  SAFETY zone  (penalized: too far, not adversarial)

    Collisions are PENALIZED (AuthSim also penalizes unrealistic crashes).
    There is NO THW tracking, NO c_level conditioning, NO Hellinger constraint.

    This is a simplified abstraction of AuthSim's multi-layer safety evaluation
    (AuthSim uses longitudinal + lateral envelopes).  The key limitation — and
    the point of including this baseline — is that pure distance rules cannot
    capture behavioral realism (aggressiveness distribution matching) the way
    our proposed constraint does.

ARCHITECTURE:
    Inherits from AdversarialHighwayEnv, overriding ONLY _compute_reward().
    Everything else (observation space, action space, SSM computation,
    aggressiveness feature extraction, termination logic) is inherited.

REFERENCE:
    - AuthSim (T-ITS 2025): Multi-layer safety region for autonomous testing
    - Section IV-C baseline comparison in our paper
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from src.environments.adversarial_highway_env import AdversarialHighwayEnv

logger = logging.getLogger(__name__)


class RuleBasedConstraintEnv(AdversarialHighwayEnv):
    """
    Method C: Rule-based constraint baseline (AuthSim-style).

    3-layer safety region with distance-based rules.  No THW tracking,
    no c_level conditioning, no behavioral constraint.

    The boundary region (10-30 m) is the rewarded "sweet spot" — this
    mirrors AuthSim's approach of defining acceptable interaction
    distances via engineered thresholds rather than learned constraints.

    Inherits the full AdversarialHighwayEnv (observation, action, SSM,
    aggressiveness features, termination) and only replaces the reward.
    """

    # Distance thresholds (meters) defining the 3-layer safety region
    DANGER_THRESHOLD: float = 10.0   # closer than this = danger
    BOUNDARY_THRESHOLD: float = 30.0  # between danger and this = target zone
    # Beyond boundary = safety region (too far)

    # Reward magnitudes
    R_DANGER: float = -5.0
    R_BOUNDARY: float = 2.0
    R_SAFETY: float = -1.0
    R_COLLISION: float = -10.0
    R_SLOW_PENALTY: float = -1.0
    SPEED_THRESHOLD: float = 15.0  # m/s — minimum acceptable highway speed

    def _compute_reward(self, ssm: dict) -> Tuple[float, dict]:
        """
        Compute rule-based 3-layer distance reward (AuthSim-style).

        Unlike the base class THW-based reward:
        - No scenario-type-specific reward (R_type)
        - No severity tracking (R_severity via THW)
        - No SSM plausibility penalties (R_ssm)
        - No near-miss bonus
        - No behavioral constraint term

        Instead: simple distance-region classification + collision penalty.

        Returns:
            (total_reward, reward_components_dict)
        """
        distance = ssm["distance"]
        r = 0.0

        # ---- 3-layer safety region (AuthSim simplified) ----
        if distance < self.DANGER_THRESHOLD:
            # Danger region: too close, high collision risk
            r_distance = self.R_DANGER
        elif distance < self.BOUNDARY_THRESHOLD:
            # Boundary region (target): adversarial but controlled
            r_distance = self.R_BOUNDARY
        else:
            # Safety region: too far, not adversarial enough
            r_distance = self.R_SAFETY

        r += r_distance

        # ---- Collision penalty (AuthSim penalizes unrealistic collisions) ----
        r_collision = 0.0
        if self._collision_occurred:
            r_collision = self.R_COLLISION
            r += r_collision

        # ---- Speed maintenance ----
        ego = self._inner_env.unwrapped.vehicle
        ego_speed = float(ego.speed)

        r_speed = 0.0
        if ego_speed < self.SPEED_THRESHOLD:
            r_speed = self.R_SLOW_PENALTY
            r += r_speed

        components = {
            "r_distance": r_distance,
            "r_collision": r_collision,
            "r_speed": r_speed,
            "r_total": r,
            # Provide zero values for standard component keys so that
            # callbacks expecting these fields don't break
            "r_type": 0.0,
            "r_severity": 0.0,
            "r_ssm": 0.0,
            "r_near_miss": 0.0,
            "r_base": r,
            "r_constraint": 0.0,
            "lambda_val": 0.0,
            "thw_target": 0.0,  # not used in Method C
        }

        return r, components
