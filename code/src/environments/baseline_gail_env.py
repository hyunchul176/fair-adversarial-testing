"""
Method D: GAIL-Style Naturalness Reward Baseline Environment

PURPOSE:
    Baseline for comparison experiments. Uses a pretrained discriminator to add
    a naturalness bonus to the base adversarial reward. The discriminator was
    trained to distinguish "expert" (sim-NDD normal driving) features from
    "non-expert" (random policy) features.

RESEARCH CONTEXT:
    This represents the GAIL (Generative Adversarial Imitation Learning) approach
    to behavioral realism, commonly used in:
    - Hao et al. (T-IV 2024): GAIL-based naturalistic adversarial driving
    - TrafficGen (NeurIPS 2022): GAN-based traffic generation

    Key differences from our proposed method (Method A + Hellinger constraint):
    - Reward shaping, NOT constraint: discriminator bonus is added to reward,
      not enforced as a hard constraint via Lagrangian.
    - No independent severity control: cannot separately tune realism vs danger.
    - Fixed discriminator: pretrained and frozen during RL (not adversarial
      co-training as in true GAIL).
    - No distributional guarantee: D(s) measures point-wise "expertness", not
      distributional distance.

KEY DESIGN:
    R_total = R_base + beta * r_disc
    where:
        R_base = collision bonus + proximity reward + speed penalty
                 (same as Method B / NaiveAdversarialEnv)
        r_disc = -log(1 - D(agg_features))  (GAIL reward)
        beta = 1.0 (equal weight between adversarial and naturalness)
        D = pretrained discriminator, outputs P(expert | features)

    The GAIL reward r_disc is high when the discriminator classifies the agent's
    current behavior as "expert-like" (naturalistic). This encourages the agent
    to behave naturalistically while also being adversarial.

ARCHITECTURE:
    Inherits from AdversarialHighwayEnv, overriding ONLY _compute_reward()
    and adding discriminator loading. Everything else (observation, action,
    SSM, aggressiveness features, termination) is inherited.

IMPORTANT:
    - Discriminator is PRETRAINED and FIXED during RL training
    - beta=1.0 gives equal weight to base reward and naturalness
    - Base reward matches Method B (collision bonus + proximity)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.environments.adversarial_highway_env import AdversarialHighwayEnv

logger = logging.getLogger(__name__)


class Discriminator(nn.Module):
    """
    GAIL discriminator: MLP [64, 64] with sigmoid output.

    Must match the architecture used in scripts/train_gail_discriminator.py.
    """

    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GAILStyleEnv(AdversarialHighwayEnv):
    """
    Method D: GAIL-style naturalness reward baseline.

    Uses a pretrained discriminator to add a naturalness bonus to the
    base adversarial reward. The discriminator outputs P(expert | features),
    and the GAIL reward is r_disc = -log(1 - D(features)).

    High D(features) -> high r_disc -> behavior looks naturalistic.
    Low D(features) -> low r_disc -> behavior looks unnatural.

    The base reward is the same as Method B (NaiveAdversarialEnv):
    proximity + collision bonus + speed penalty. This isolates the effect
    of the GAIL naturalness signal compared to our Hellinger constraint.
    """

    def __init__(
        self,
        *args,
        beta: float = 1.0,
        disc_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            beta: Weight for the discriminator reward (default 1.0).
            disc_path: Path to pretrained discriminator weights.
                       Defaults to models/gail_discriminator.pth.
            scaler_path: Path to fitted StandardScaler.
                         Defaults to models/gail_scaler.pkl.
            *args, **kwargs: Passed to AdversarialHighwayEnv.
        """
        # Force c_level=0 and disable constraint for GAIL baseline
        kwargs["c_level"] = 0.0
        kwargs["constrained"] = False
        super().__init__(*args, **kwargs)

        self.beta = beta
        self._disc: Optional[Discriminator] = None
        self._gail_scaler = None

        # Resolve paths
        project_root = Path(__file__).resolve().parent.parent.parent
        self._disc_path = disc_path or str(
            project_root / "models" / "gail_discriminator.pth"
        )
        self._scaler_path = scaler_path or str(
            project_root / "models" / "gail_scaler.pkl"
        )

        self._load_discriminator()

        logger.info(
            f"GAILStyleEnv initialized (Method D baseline). "
            f"beta={self.beta}, disc_path={self._disc_path}"
        )

    def _load_discriminator(self):
        """Load pretrained discriminator and scaler from disk."""
        if not os.path.exists(self._disc_path):
            logger.error(
                f"GAIL discriminator not found at {self._disc_path}. "
                "Run scripts/train_gail_discriminator.py first."
            )
            raise FileNotFoundError(
                f"GAIL discriminator not found: {self._disc_path}"
            )

        if not os.path.exists(self._scaler_path):
            logger.error(
                f"GAIL scaler not found at {self._scaler_path}. "
                "Run scripts/train_gail_discriminator.py first."
            )
            raise FileNotFoundError(
                f"GAIL scaler not found: {self._scaler_path}"
            )

        # Load discriminator
        self._disc = Discriminator(input_dim=6)
        self._disc.load_state_dict(
            torch.load(self._disc_path, map_location="cpu", weights_only=True)
        )
        self._disc.eval()
        logger.info(f"GAIL discriminator loaded from {self._disc_path}")

        # Load scaler
        import joblib

        self._gail_scaler = joblib.load(self._scaler_path)
        logger.info(f"GAIL scaler loaded from {self._scaler_path}")

    def _compute_reward(self, ssm: dict) -> Tuple[float, dict]:
        """
        Compute GAIL-style reward: base adversarial + discriminator bonus.

        R_total = R_base + beta * r_disc

        Base reward (same as Method B):
            r_proximity = max(0, 1 - distance/50)
            r_collision = +10 if collision else 0
            r_speed = -1 if ego_speed < 15 else 0

        GAIL reward:
            r_disc = -log(max(1 - D(features), 1e-6))
            where D = pretrained discriminator
        """
        ego = self._inner_env.unwrapped.vehicle
        ego_speed = float(ego.speed)
        distance = ssm["distance"]

        # ----- Base reward (same as Method B / NaiveAdversarialEnv) -----
        # Proximity: closer = better (linear, [0, 1])
        r_proximity = max(0.0, 1.0 - distance / 50.0)

        # Collision bonus: +10 for crashing
        r_collision = 10.0 if self._collision_occurred else 0.0

        # Speed maintenance: penalize slow driving on highway
        r_speed = -1.0 if ego_speed < 15.0 else 0.0

        r_base = r_proximity + r_collision + r_speed

        # ----- GAIL discriminator reward -----
        r_disc = 0.0
        if self._disc is not None and self._gail_scaler is not None:
            agg_features = self._compute_aggressiveness_features(ssm)
            scaled = self._gail_scaler.transform(
                agg_features.reshape(1, -1)
            )
            with torch.no_grad():
                d_out = self._disc(torch.FloatTensor(scaled)).item()
            # GAIL reward: -log(1 - D(s))
            # High D(s) = expert-like -> high reward
            r_disc = -np.log(max(1.0 - d_out, 1e-6))

        r_total = r_base + self.beta * r_disc

        components = {
            "r_proximity": r_proximity,
            "r_collision": r_collision,
            "r_speed": r_speed,
            "r_base": r_base,
            "r_discriminator": r_disc,
            "r_total": r_total,
            # Standard keys for callback compatibility
            "r_type": 0.0,
            "r_severity": 0.0,
            "r_ssm": 0.0,
            "r_near_miss": 0.0,
            "r_constraint": 0.0,
            "lambda_val": 0.0,
            "thw_target": 0.0,
        }

        return r_total, components
