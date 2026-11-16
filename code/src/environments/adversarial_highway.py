"""
Adversarial Highway Environment Wrapper

PURPOSE: Wrap highway-env to support:
1. Goal-conditioned adversarial training (scenario_type + c_level)
2. BV (Background Vehicle) control by RL agent
3. Real-time SSM computation (TTC, DRAC)
4. Aggressiveness feature extraction at each step

RESEARCH CONTEXT:
- AuthSim (T-ITS 2025) used highway-env with 3 lanes — we follow the same setup
- Hao-NatPrior (T-IV 2024) also used highway-env for adversarial RL
- highway-env is acceptable for AAP publication (Zhu et al., AAP 2025 did so)

OBSERVATION SPACE:
[ego_kin(6) | bv1_kin(6) | ... | bv4_kin(6) | type_embed(16) | c_level(1)] = 47D

ACTION SPACE:
Continuous [acceleration, steering] for the adversarial BV
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class AdversarialHighwayEnv(gym.Wrapper):
    """
    Wraps highway-env for adversarial scenario generation.
    
    Key modifications from standard highway-env:
    1. RL controls one BV (adversarial agent), not the ego vehicle
    2. Ego vehicle uses default IDM/MOBIL policy
    3. Observation includes scenario_type embedding and c_level
    4. Reward is adversarial (R_type + R_severity + R_ssm)
    5. Info dict includes SSM values and aggressiveness features
    """
    
    def __init__(
        self,
        scenario_type: Optional[str] = None,   # None = sample randomly
        c_level: Optional[float] = None,        # None = sample uniformly [0, 1]
        config: Optional[Dict] = None,
    ):
        # Create base highway-env
        env_config = self._get_env_config(config)
        env = gym.make("highway-v0", config=env_config)
        super().__init__(env)
        
        self.scenario_type = scenario_type
        self.c_level = c_level
        self._current_type = None
        self._current_c_level = None
        
        # Type embedding (simple one-hot for now, upgrade to learned embedding in agent)
        self.type_map = {"rear_end": 0, "cut_in": 1}
    
    def _get_env_config(self, config: Optional[Dict] = None) -> Dict:
        """Get highway-env configuration."""
        default = {
            "lanes_count": 3,
            "vehicles_count": 5,
            "duration": 40,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "normalize": False,
            },
            "action": {
                "type": "ContinuousAction",
            },
            # Ego uses default policy (IDM/MOBIL)
            "controlled_vehicles": 1,
        }
        if config:
            default.update(config)
        return default
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment with new scenario_type and c_level.
        
        Returns augmented observation: [kinematics | type_embed | c_level]
        """
        obs, info = self.env.reset(**kwargs)
        
        # Sample or use fixed scenario parameters
        if self.scenario_type is None:
            self._current_type = np.random.choice(["rear_end", "cut_in"])
        else:
            self._current_type = self.scenario_type
        
        if self.c_level is None:
            self._current_c_level = np.random.uniform(0.0, 1.0)
        else:
            self._current_c_level = self.c_level
        
        # Augment observation
        augmented_obs = self._augment_observation(obs)
        
        info["scenario_type"] = self._current_type
        info["c_level"] = self._current_c_level
        
        return augmented_obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step with adversarial action.
        
        The RL agent controls one BV. Other BVs follow IDM/MOBIL.
        
        Returns:
            obs: augmented observation
            reward: adversarial reward (R_base, NOT constrained — constraint added by agent)
            terminated: episode done (collision or success)
            truncated: episode timeout
            info: includes SSM values, aggressiveness features, collision info
        """
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Compute SSM values
        ssm = self._compute_ssm(info)
        info.update(ssm)
        
        # Compute aggressiveness features for constraint
        agg_features = self._compute_aggressiveness_features(info)
        info["aggressiveness_features"] = agg_features
        
        # Compute adversarial reward
        reward = self._compute_reward(info)
        info["reward_components"] = self._last_reward_components
        
        # Augment observation
        augmented_obs = self._augment_observation(obs)
        
        return augmented_obs, reward, terminated, truncated, info
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """Add scenario_type embedding and c_level to observation."""
        # Type one-hot (2D) — will be converted to 16D embedding by the policy network
        type_idx = self.type_map[self._current_type]
        type_onehot = np.zeros(2, dtype=np.float32)
        type_onehot[type_idx] = 1.0
        
        # c_level scalar
        c_level_arr = np.array([self._current_c_level], dtype=np.float32)
        
        # Flatten obs if needed
        obs_flat = obs.flatten().astype(np.float32)
        
        return np.concatenate([obs_flat, type_onehot, c_level_arr])
    
    def _compute_ssm(self, info: Dict) -> Dict:
        """
        Compute Surrogate Safety Measures (TTC, DRAC) from current state.
        
        TTC = distance / relative_speed (when approaching)
        DRAC = (relative_speed)^2 / (2 * distance)
        """
        # TODO: Extract from highway-env vehicle states
        # Need: adversarial BV position/velocity, ego position/velocity
        
        ttc = 999.0  # placeholder
        drac = 0.0   # placeholder
        
        return {
            "ttc": ttc,
            "drac": drac,
            "ttc_min_so_far": min(getattr(self, "_ttc_min", 999), ttc),
        }
    
    def _compute_aggressiveness_features(self, info: Dict) -> np.ndarray:
        """
        Extract 6D aggressiveness feature vector for current timestep.
        Used by AggressivenessCalculator for constraint computation.
        """
        # TODO: Extract from highway-env state
        return np.zeros(6, dtype=np.float32)  # placeholder
    
    def _compute_reward(self, info: Dict) -> float:
        """
        Compute base adversarial reward R_base.
        
        NOTE: This is R_base ONLY. The constraint term (-λ * (D-ε))
        is added by ConstrainedAdversarialAgent, NOT here.
        This separation is critical for the constrained RL formulation.
        """
        # TODO: Implement using BaseReward class
        self._last_reward_components = {}
        return 0.0  # placeholder
