"""
Aggressiveness Calculator — Real-time Feature Computation During RL Training

PURPOSE: During adversarial RL training, compute aggressiveness features at each
timestep and maintain a sliding window to estimate the agent's current behavioral
distribution for comparison against the NDD reference (GMM).

RESEARCH CONTEXT:
- AggFollower (Chen et al., T-IV 2024) introduced moment-level aggressiveness tokens
  for car-following prediction. We adapt this concept for adversarial RL constraint.
- The calculator bridges the gap between NDD statistics (offline) and RL training (online).

CRITICAL DESIGN DECISIONS:
1. Window size W=50 (10s at 5Hz): too small → unstable estimation, too large → sluggish
2. Use GMM log-likelihood as continuous signal (not discretized tokens like AggFollower)
3. Hellinger distance (not KL) for constraint measurement
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional, Tuple
import logging
from src.evaluation.realism_metrics import marginal_hellinger_scaled, marginal_hellinger_from_features

logger = logging.getLogger(__name__)


class AggressivenessCalculator:
    """
    Compute real-time aggressiveness features and distributional distance.
    
    Used during RL training to:
    1. Extract 6D aggressiveness features from agent actions
    2. Maintain sliding window of recent features
    3. Compute Hellinger distance to NDD reference distribution
    4. Provide GMM log-likelihood as continuous reward signal
    """
    
    FEATURE_NAMES = ["jerk_x", "a_x", "delta_v", "delta_d", "thw", "ttc"]
    
    def __init__(
        self,
        gmm_model,        # sklearn GaussianMixture, pre-fitted on HighD
        scaler,           # sklearn StandardScaler, pre-fitted on HighD
        window_size: int = 50,   # 50 steps = 10s at 5Hz
        epsilon: float = 0.01,   # numerical stability for TTC
        # 2D behavioral constraint (optional)
        behavioral_gmm=None,      # sklearn GaussianMixture fitted on 2D (jerk_x, a_x)
        behavioral_scaler=None,   # sklearn StandardScaler fitted on 2D (jerk_x, a_x)
    ):
        self.gmm = gmm_model
        self.scaler = scaler
        self.window_size = window_size
        self.epsilon = epsilon

        self.feature_buffer = deque(maxlen=window_size)
        self.prev_ax: Optional[float] = None

        # Pre-sample reference data from GMM for efficient Hellinger computation
        self._ref_samples_scaled = self.gmm.sample(1000)[0]

        # 2D behavioral constraint: constrains only HOW the agent drives
        # (jerk_x, a_x) rather than WHERE it drives (spacing, relative speed).
        # This avoids conflict between the adversarial objective (close following,
        # high relative speed) and the realism constraint.
        self._behavioral_scaler = behavioral_scaler
        self._behavioral_ref_samples: Optional[np.ndarray] = None
        if behavioral_gmm is not None and behavioral_scaler is not None:
            self._behavioral_ref_samples = behavioral_gmm.sample(1000)[0]
            logger.info(
                "Behavioral 2D constraint initialized (jerk_x, a_x only)"
            )
    
    def reset(self):
        """Reset at the start of each episode."""
        self.feature_buffer.clear()
        self.prev_ax = None
    
    def compute_features(
        self,
        agent_ax: float,          # agent's longitudinal acceleration
        agent_vx: float,          # agent's longitudinal velocity
        preceding_vx: float,     # preceding vehicle's velocity
        spacing: float,           # distance headway to preceding vehicle
        dt: float = 0.2,         # time step (5Hz)
    ) -> np.ndarray:
        """
        Compute 6D aggressiveness feature vector for current timestep.
        
        Returns:
            feature: np.ndarray of shape (6,)
        """
        # Jerk: rate of acceleration change
        jerk_x = (agent_ax - self.prev_ax) / dt if self.prev_ax is not None else 0.0
        self.prev_ax = agent_ax
        
        # Relative speed (positive = approaching)
        delta_v = agent_vx - preceding_vx
        
        # Distance headway
        delta_d = max(spacing, 0.1)  # avoid zero
        
        # Time headway
        thw = delta_d / max(agent_vx, 0.1)
        
        # Time to collision (only meaningful when approaching)
        if delta_v > self.epsilon:
            ttc = delta_d / delta_v
        else:
            ttc = 999.0  # not approaching → effectively infinite TTC
        
        feature = np.array([jerk_x, agent_ax, delta_v, delta_d, thw, ttc])
        self.feature_buffer.append(feature)
        
        return feature
    
    def get_gmm_log_likelihood(self, feature: Optional[np.ndarray] = None) -> float:
        """
        Compute log-likelihood of a single feature under the NDD GMM.
        Can be used as a per-step reward signal.
        
        Higher = more consistent with NDD aggressive moments.
        """
        if feature is None:
            if len(self.feature_buffer) == 0:
                return 0.0
            feature = self.feature_buffer[-1]
        
        feature_scaled = self.scaler.transform(feature.reshape(1, -1))
        return float(self.gmm.score_samples(feature_scaled)[0])
    
    def compute_hellinger_distance(self) -> float:
        """
        Compute Hellinger distance between agent's recent behavior
        and NDD aggressive moment distribution.

        This is the core constraint metric:
        D_Hellinger(P_agent_agg, P_NDD_agg) ≤ ε

        Uses per-feature 1D histogram comparison, which is statistically valid
        even with ~50 samples (unlike fitting a 6D distribution). The per-feature
        Hellinger distances are averaged for the overall metric.

        Returns:
            hellinger: float in [0, 1], 0 = identical distributions
        """
        if len(self.feature_buffer) < self.window_size // 2:
            return 0.0  # Not enough data for reliable estimation

        features = np.array(self.feature_buffer)
        features_scaled = self.scaler.transform(features)

        # Compute per-feature Hellinger distance using pre-sampled reference
        distances = []
        for i in range(features_scaled.shape[1]):
            h = self._histogram_hellinger_1d(
                features_scaled[:, i],
                self._ref_samples_scaled[:, i],
            )
            distances.append(h)

        return float(np.mean(distances))

    def compute_behavioral_hellinger(self) -> float:
        """
        Compute Hellinger distance using only behavioral features (jerk_x, a_x).

        Unlike the full 6D Hellinger, this only constrains HOW the agent drives
        (acceleration patterns), not WHERE it drives (spacing, relative speed).
        This avoids the fundamental conflict where the adversarial objective
        pushes for close following / high relative speed while the 6D realism
        constraint penalizes exactly those features.

        Requires behavioral_gmm and behavioral_scaler to be provided at init.

        Returns:
            hellinger: float in [0, 1], 0 = identical distributions.
                       Returns 0.0 if insufficient data or no 2D model loaded.
        """
        if len(self.feature_buffer) < self.window_size // 2:
            return 0.0

        if self._behavioral_ref_samples is None:
            return 0.0

        features = np.array(self.feature_buffer)
        # Extract behavioral features only: columns 0 (jerk_x) and 1 (a_x)
        behavioral = features[:, :2]

        behavioral_scaled = self._behavioral_scaler.transform(behavioral)

        distances = []
        for i in range(2):
            h = self._histogram_hellinger_1d(
                behavioral_scaled[:, i],
                self._behavioral_ref_samples[:, i],
            )
            distances.append(h)

        return float(np.mean(distances))

    def get_per_feature_distances(self) -> dict:
        """
        Compute per-feature distributional distances for detailed analysis.
        Used in ablation studies and paper figures.
        """
        if len(self.feature_buffer) < self.window_size // 2:
            return {name: 0.0 for name in self.FEATURE_NAMES}

        features = np.array(self.feature_buffer)
        features_scaled = self.scaler.transform(features)
        distances = {}

        for i, name in enumerate(self.FEATURE_NAMES):
            distances[name] = self._histogram_hellinger_1d(
                features_scaled[:, i],
                self._ref_samples_scaled[:, i],
            )

        return distances

    def _histogram_hellinger_1d(
        self, p_samples: np.ndarray, q_samples: np.ndarray, n_bins: int = 30
    ) -> float:
        """Reference-fixed histogram distance with explicit overflow bins."""
        return marginal_hellinger_scaled(p_samples, q_samples, n_bins)

    @staticmethod
    def compute_hellinger_from_features(
        features: np.ndarray,
        scaler,
        ref_samples_scaled: np.ndarray,
        n_bins: int = 30,
    ) -> float:
        """
        Compute Hellinger distance from a full feature array (no window limit).

        For EVALUATION ONLY. Uses all provided features instead of a sliding window.

        Args:
            features: (N, D) array of raw features
            scaler: fitted StandardScaler
            ref_samples_scaled: (M, D) pre-scaled reference samples from GMM
            n_bins: histogram bins for 1D Hellinger

        Returns:
            Mean per-dimension Hellinger distance
        """
        return marginal_hellinger_from_features(features, scaler, ref_samples_scaled, n_bins)

    @staticmethod
    def compute_behavioral_hellinger_from_features(
        features: np.ndarray,
        behavioral_scaler,
        behavioral_ref_samples_scaled: np.ndarray,
        n_bins: int = 30,
    ) -> float:
        """2D behavioral Hellinger from full feature array (jerk_x, a_x only)."""
        behavioral = features[:, :2]  # jerk_x, a_x
        return AggressivenessCalculator.compute_hellinger_from_features(
            behavioral, behavioral_scaler, behavioral_ref_samples_scaled, n_bins
        )
