"""
Method B: Naive Adversarial RL Baseline Environment

PURPOSE:
    Baseline for comparison experiments. Uses a simple collision/proximity
    reward WITHOUT THW tracking, c_level control, or near-miss bonuses.
    Collision is REWARDED (+10), not penalized — the agent maximizes crashes.

RESEARCH CONTEXT:
    This represents the "naive" adversarial approach commonly seen in early
    adversarial testing literature: maximize collision rate without behavioral
    realism constraints or nuanced scenario control. By comparing against
    Method A (our full reward) and Method C (constrained), we demonstrate
    that naive reward shaping produces unrealistic, overly aggressive behavior.

KEY DIFFERENCES FROM METHOD A:
    - NO THW target tracking (no c_level-based severity control)
    - NO near-miss bonus (no incentive for sustained dangerous interactions)
    - NO speed matching bonus
    - Collision is REWARDED (+10), not penalized (-3)
    - c_level is always 0 (observation slot present but fixed)
    - Simple linear proximity reward: closer = better
    - Speed penalty only (no speed gating of severity)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from src.environments.adversarial_highway_env import AdversarialHighwayEnv

logger = logging.getLogger(__name__)


class NaiveAdversarialEnv(AdversarialHighwayEnv):
    """
    Method B: Naive Adversarial RL baseline.

    Simple collision/proximity reward without THW tracking or c_level control.
    Inherits observation space, action space, SSM computation, and episode
    management from AdversarialHighwayEnv. Only the reward function is
    overridden.
    """

    def __init__(self, **kwargs):
        # Force c_level=0 and disable constraint for naive baseline
        kwargs["c_level"] = 0.0
        kwargs["constrained"] = False
        super().__init__(**kwargs)
        logger.info(
            "NaiveAdversarialEnv initialized (Method B baseline). "
            "Collision rewarded, no THW tracking, no c_level control."
        )

    def _compute_reward(self, ssm: dict) -> Tuple[float, dict]:
        """
        Naive adversarial reward: proximity + collision bonus + speed penalty.

        R = proximity_reward + collision_bonus + speed_penalty

        No THW tracking, no c_level conditioning, no near-miss bonus,
        no speed matching. Collision is rewarded, not penalized.
        """
        ego = self._inner_env.unwrapped.vehicle
        ego_speed = float(ego.speed)
        distance = ssm["distance"]

        # Proximity: closer = better (linear, [0, 1])
        r_proximity = max(0.0, 1.0 - distance / 50.0)

        # Collision bonus: +10 for crashing
        r_collision = 10.0 if self._collision_occurred else 0.0

        # Speed maintenance: penalize slow driving on highway
        r_speed = -1.0 if ego_speed < 15.0 else 0.0

        r_total = r_proximity + r_collision + r_speed

        components = {
            "r_proximity": r_proximity,
            "r_collision": r_collision,
            "r_speed": r_speed,
            "r_total": r_total,
        }

        return r_total, components
