"""
Method D' Full GAIL environment with co-trained discriminator.

Key differences from baseline_gail_env.py (Method D):
- Discriminator updated online during training (not frozen)
- Reward = log D(s) - log(1-D(s)) (proper GAIL, not -log(1-D) bonus)
- Expert = HighD aggressive moments (2D: jerk_x, a_x)
- Discriminator held externally via set_discriminator() so trainer can update it

The discriminator is a shared object: one instance per training run, held by the
trainer and plugged into the env. The env reads D(s) at each step and returns
the GAIL reward in place of the adversarial reward.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.environments.adversarial_highway_env import AdversarialHighwayEnv

logger = logging.getLogger(__name__)


class GAILDiscriminator(nn.Module):
    """State-only discriminator: MLP over behavioral features."""

    def __init__(self, input_dim: int = 2, hidden: list = [128, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit (unsigmoided)."""
        return self.net(x)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid(logit) = P(expert | x)."""
        return torch.sigmoid(self.forward(x))


class GAILFullEnv(AdversarialHighwayEnv):
    """
    Adversarial env with GAIL reward replacing the base adversarial reward.

    The discriminator must be injected via `set_discriminator(disc)` before
    rollouts begin. If unset, reward is 0 and a warning is logged.
    """

    def __init__(self, *args, gail_reward_clamp: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._gail_disc: Optional[GAILDiscriminator] = None
        self._gail_scaler_mean: Optional[np.ndarray] = None
        self._gail_scaler_std: Optional[np.ndarray] = None
        self._gail_clamp = gail_reward_clamp
        self._last_disc_prob: float = 0.5
        self._gail_feat_buffer: list = []   # stores recent 2D features

    def get_gail_feat_buffer(self) -> np.ndarray:
        """Return current feature buffer as (N, 2) array, then clear."""
        if not self._gail_feat_buffer:
            return np.empty((0, 2), dtype=np.float32)
        arr = np.array(self._gail_feat_buffer, dtype=np.float32)
        return arr

    def clear_gail_feat_buffer(self):
        self._gail_feat_buffer = []

    def set_discriminator(self, disc: GAILDiscriminator,
                          scaler_mean: np.ndarray, scaler_std: np.ndarray):
        self._gail_disc = disc
        self._gail_scaler_mean = scaler_mean
        self._gail_scaler_std = scaler_std

    def _extract_disc_features(self, ssm: dict) -> np.ndarray:
        """Extract 2D behavioral features: [jerk_x, a_x]."""
        agg = self._compute_aggressiveness_features(ssm)  # 6D
        return np.array([agg[0], agg[1]], dtype=np.float32)  # jerk, a_x

    def _compute_reward(self, ssm: dict) -> Tuple[float, dict]:
        """Override base reward with GAIL reward."""
        feat_2d = self._extract_disc_features(ssm)

        if self._gail_disc is None:
            return 0.0, {"gail_reward": 0.0, "disc_prob": 0.5}

        if self._gail_scaler_mean is not None:
            feat_scaled = (feat_2d - self._gail_scaler_mean) / (self._gail_scaler_std + 1e-8)
        else:
            feat_scaled = feat_2d

        with torch.no_grad():
            x = torch.as_tensor(feat_scaled, dtype=torch.float32).unsqueeze(0)
            logit = self._gail_disc.forward(x).item()
        # GAIL reward = log D(s) - log(1 - D(s)) = logit (directly)
        # Equivalent: log-odds. Large positive => looks expert-like.
        reward = float(np.clip(logit, -self._gail_clamp, self._gail_clamp))

        # Collision penalty: still negative to discourage outright crash
        if ssm.get("collision", False):
            reward -= 5.0

        disc_prob = 1.0 / (1.0 + np.exp(-logit))
        self._last_disc_prob = disc_prob

        # Store feature in buffer for discriminator training
        self._gail_feat_buffer.append([float(feat_2d[0]), float(feat_2d[1])])
        if len(self._gail_feat_buffer) > 20000:
            # Ring buffer trim
            self._gail_feat_buffer = self._gail_feat_buffer[-10000:]

        return reward, {
            "gail_reward": reward,
            "disc_prob": float(disc_prob),
            "disc_logit": float(logit),
            "feat_jerk": float(feat_2d[0]),
            "feat_ax": float(feat_2d[1]),
        }
