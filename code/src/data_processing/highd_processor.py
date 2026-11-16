"""
HighD Data Processing Module — Phase 0

PURPOSE: Extract 6D aggressiveness features from HighD naturalistic driving data,
identify aggressive moments, and fit a GMM distribution.

This module is the foundation of the entire research. The GMM fitted here becomes
the behavioral realism constraint in Phase 2's Constrained RL.

RESEARCH CONTEXT: 
- HAD-Gen (AAP 2025) found only 7/10,080 aggressive DRIVERS (0.07%)
- But aggressive MOMENTS (frames with aggressive features) are 1-5% of all frames
- This moment-level perspective resolves the "Curse of Rarity" problem

KEY OUTPUT:
- gmm_model.pkl: fitted GMM on aggressive moment features
- scaler.pkl: StandardScaler for feature normalization
- feature_statistics.json: HighD feature distributions for reference
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AggressivenessConfig:
    """Configuration for aggressive moment identification."""
    ttc_threshold: float = 2.0        # seconds (Wang et al. 2021 "critical")
    thw_threshold: float = 1.0        # seconds (Xue et al. 2019 "high-risk")
    accel_percentile: float = 85      # percentile for |a_x|
    jerk_percentile: float = 85       # percentile for |jerk_x|
    mode: str = "any"                 # "any" = OR logic


class HighDProcessor:
    """
    Process HighD dataset to extract aggressiveness features.
    
    HighD data format:
    - XX_tracks.csv: frame-by-frame vehicle trajectories
    - XX_tracksMeta.csv: per-vehicle metadata
    - XX_recordingMeta.csv: per-recording metadata
    
    Key columns in tracks.csv:
    - frame, id, x, y, xVelocity, yVelocity, xAcceleration, yAcceleration
    - width, height, dhw (distance headway), thw, ttc
    - precedingId, followingId, laneId
    """
    
    FEATURE_NAMES = ["jerk_x", "a_x", "delta_v", "delta_d", "thw", "ttc"]
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
    
    def _validate_data_dir(self):
        """Check that HighD data files exist.

        Validates that at least recording 01's files are present, since all
        60 recordings follow the same naming convention. If 01 exists, the
        directory structure is correct.
        """
        required_files = [
            self.data_dir / "01_tracks.csv",
            self.data_dir / "01_tracksMeta.csv",
            self.data_dir / "01_recordingMeta.csv",
        ]
        missing = [f for f in required_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"HighD data directory validation failed. Missing files: "
                f"{[str(f) for f in missing]}. "
                f"Expected HighD data in: {self.data_dir}"
            )
        # Count available recordings
        track_files = sorted(self.data_dir.glob("*_tracks.csv"))
        self.num_recordings = len(track_files)
        logger.info(
            f"HighD data directory validated: {self.data_dir} "
            f"({self.num_recordings} recordings found)"
        )
    
    def load_recording(self, recording_id: int) -> pd.DataFrame:
        """Load a single HighD recording's track data.

        Each recording is stored as {id:02d}_tracks.csv with 25Hz trajectory
        data for all vehicles in that recording segment. Files range from ~100K
        to 2M+ rows depending on traffic density and recording duration.

        Args:
            recording_id: Recording number (1-60).

        Returns:
            DataFrame with all track data for the recording.

        Raises:
            FileNotFoundError: If the recording file does not exist.
        """
        file_path = self.data_dir / f"{recording_id:02d}_tracks.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Recording file not found: {file_path}")
        logger.info(f"Loading recording {recording_id:02d} from {file_path}")
        tracks = pd.read_csv(file_path)
        logger.info(
            f"Recording {recording_id:02d}: {len(tracks):,} rows, "
            f"{tracks['id'].nunique()} vehicles, "
            f"frames {tracks['frame'].min()}-{tracks['frame'].max()}"
        )
        return tracks
    
    def compute_aggressiveness_features(
        self, tracks: pd.DataFrame, dt: float = 0.04
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 6D aggressiveness feature vector for each frame.

        Features (order matches FEATURE_NAMES):
        1. jerk_x: d(xAcceleration)/dt -- acceleration change rate
        2. a_x: xAcceleration -- instantaneous acceleration
        3. delta_v: xVelocity - precedingXVelocity -- relative speed to preceding
        4. delta_d: dhw -- distance headway (spacing) from HighD column
        5. thw: pre-computed thw from HighD (negative/zero set to 999)
        6. ttc: pre-computed ttc from HighD (negative/zero set to 999)

        Only frames where precedingId != 0 are kept. When precedingId == 0,
        HighD writes sentinel values (dhw=0, thw=0, ttc=0) that would
        contaminate the distribution.

        Jerk is computed as grouped diff of xAcceleration within each vehicle
        trajectory (group by id, use diff / dt). The first frame of each
        vehicle's trajectory has no jerk and is dropped.

        Args:
            tracks: HighD tracks DataFrame for one recording.
            dt: time step (1/25 Hz = 0.04s for HighD).

        Returns:
            Tuple of:
                features: np.ndarray of shape (N_valid_frames, 6)
                valid_indices: np.ndarray of original DataFrame row indices
        """
        # Step 1: Filter to frames with a preceding vehicle
        has_preceding = tracks["precedingId"] != 0
        valid = tracks.loc[has_preceding].copy()

        if len(valid) == 0:
            logger.warning("No frames with preceding vehicle found")
            return np.empty((0, 6)), np.empty(0, dtype=int)

        # Step 2: Compute jerk_x via grouped differentiation
        # Sort by (id, frame) to ensure correct temporal order within each vehicle
        valid = valid.sort_values(["id", "frame"])
        valid["jerk_x"] = valid.groupby("id")["xAcceleration"].diff() / dt

        # Drop frames where jerk is NaN (first frame of each vehicle)
        valid = valid.dropna(subset=["jerk_x"])

        if len(valid) == 0:
            logger.warning("No valid frames after jerk computation")
            return np.empty((0, 6)), np.empty(0, dtype=int)

        # Step 3: Extract / compute each feature
        # a_x: direct column
        a_x = valid["xAcceleration"].values

        # jerk_x: just computed above
        jerk_x = valid["jerk_x"].values

        # delta_v: xVelocity - precedingXVelocity
        # Sign convention: positive delta_v means ego is faster than preceding
        # (approaching). Works for both driving directions because preceding
        # is in the same direction, so sign convention is consistent.
        delta_v = valid["xVelocity"].values - valid["precedingXVelocity"].values

        # delta_d: dhw (distance headway), direct column
        delta_d = valid["dhw"].values

        # thw: use pre-computed HighD column.
        # HighD thw can be 0 in edge cases even with precedingId != 0 (e.g.,
        # very slow vehicle). Cap negatives/zeros at 999 for downstream safety.
        thw = valid["thw"].values.copy()
        thw[thw <= 0] = 999.0

        # ttc: use pre-computed HighD column.
        # Negative TTC means vehicles are diverging. Only positive TTC < threshold
        # is meaningful for aggressiveness. Set non-positive to 999.
        ttc = valid["ttc"].values.copy()
        ttc[ttc <= 0] = 999.0

        # Step 4: Stack into (N, 6) feature array
        features = np.column_stack([jerk_x, a_x, delta_v, delta_d, thw, ttc])

        valid_indices = valid.index.values

        logger.debug(
            f"Computed features: {features.shape[0]:,} valid frames "
            f"from {has_preceding.sum():,} frames with preceding vehicle"
        )

        return features, valid_indices
    
    def identify_aggressive_moments(
        self,
        features: np.ndarray,
        config: AggressivenessConfig,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Identify frames that are 'aggressive moments'.

        A frame is aggressive if ANY of (OR logic, config.mode == "any"):
        - 0 < TTC < ttc_threshold (default 2.0s) -- approaching & dangerous
        - 0 < THW < thw_threshold (default 1.0s) -- dangerously close following
        - |a_x| > accel_percentile of distribution -- extreme acceleration
        - |jerk_x| > jerk_percentile of distribution -- abrupt maneuver

        Feature column mapping (matches FEATURE_NAMES):
          0: jerk_x, 1: a_x, 2: delta_v, 3: delta_d, 4: thw, 5: ttc

        The thw and ttc columns have already been sanitized in
        compute_aggressiveness_features (non-positive values set to 999),
        so we only need to check < threshold. Values of 999 will never
        trigger the threshold.

        Args:
            features: (N, 6) array from compute_aggressiveness_features.
            config: AggressivenessConfig with thresholds and percentiles.

        Returns:
            Tuple of:
                mask: boolean array of shape (N,), True for aggressive moments
                per_criterion_counts: dict with count for each criterion
        """
        n = features.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool), {
                "ttc": 0, "thw": 0, "accel": 0, "jerk": 0
            }

        jerk_x = features[:, 0]
        a_x = features[:, 1]
        thw = features[:, 4]
        ttc = features[:, 5]

        # Criterion 1: TTC < threshold (only positive/valid TTC values matter;
        # 999 sentinels will never satisfy < 2.0)
        ttc_mask = ttc < config.ttc_threshold

        # Criterion 2: THW < threshold
        thw_mask = thw < config.thw_threshold

        # Criterion 3: |a_x| > percentile threshold
        accel_threshold = np.percentile(np.abs(a_x), config.accel_percentile)
        accel_mask = np.abs(a_x) > accel_threshold

        # Criterion 4: |jerk_x| > percentile threshold
        jerk_threshold = np.percentile(np.abs(jerk_x), config.jerk_percentile)
        jerk_mask = np.abs(jerk_x) > jerk_threshold

        # Combine with OR logic (any criterion triggers aggressive label)
        if config.mode == "any":
            mask = ttc_mask | thw_mask | accel_mask | jerk_mask
        else:
            # "all" mode would require all criteria (much stricter)
            mask = ttc_mask & thw_mask & accel_mask & jerk_mask

        per_criterion_counts = {
            "ttc": int(ttc_mask.sum()),
            "thw": int(thw_mask.sum()),
            "accel": int(accel_mask.sum()),
            "jerk": int(jerk_mask.sum()),
        }

        logger.info(
            f"Aggressive moment identification: "
            f"TTC<{config.ttc_threshold}s={per_criterion_counts['ttc']:,}, "
            f"THW<{config.thw_threshold}s={per_criterion_counts['thw']:,}, "
            f"|a_x|>P{config.accel_percentile}({accel_threshold:.3f})="
            f"{per_criterion_counts['accel']:,}, "
            f"|jerk_x|>P{config.jerk_percentile}({jerk_threshold:.3f})="
            f"{per_criterion_counts['jerk']:,}, "
            f"combined ({config.mode})={mask.sum():,}/{n:,} "
            f"({mask.sum()/n*100:.2f}%)"
        )

        return mask, per_criterion_counts
    
    def fit_gmm(
        self,
        aggressive_features: np.ndarray,
        k_range: Tuple[int, int] = (2, 12),
        max_samples_for_bic: int = 500_000,
    ) -> Tuple:  # Returns (GaussianMixture, StandardScaler, int)
        """
        Fit GMM on aggressive moment features.

        Why GMM and not GAIL or kernel density:
        1. GMM gives continuous log-likelihood usable as a direct RL signal
        2. No online estimation needed -- training stability in Phase 2
        3. Interpretable components (e.g., "low-speed tailgating" cluster)
        4. Differentiates us from Hao et al.'s GAIL approach (T-IV 2024)

        Steps:
        1. Z-score normalize features with StandardScaler (fit on ALL data)
        2. Search optimal K in [k_min, k_max) via BIC on subsample (for speed)
        3. Fit final GMM with optimal K on subsample using full covariance

        Subsampling rationale: GMM with 11M samples is computationally
        prohibitive. 500K random samples preserve the distribution shape
        while reducing fit time from hours to minutes.

        All GMM fits use random_state=42 for reproducibility as required
        by the research protocol.

        Args:
            aggressive_features: (N_agg, 6) array of aggressive moment features.
            k_range: (min_k, max_k) range for BIC search (max exclusive).
            max_samples_for_bic: subsample size for GMM fitting (default 500K).

        Returns:
            Tuple of (gmm_model, scaler, optimal_k).
        """
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        n_samples = aggressive_features.shape[0]
        logger.info(
            f"Fitting GMM on {n_samples:,} aggressive moment features "
            f"(K range: {k_range[0]}-{k_range[1]-1})"
        )

        # Step 1: Z-score normalization (fit on ALL data for accurate scaling)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(aggressive_features)

        # Subsample for GMM fitting if dataset is very large
        if n_samples > max_samples_for_bic:
            rng = np.random.RandomState(42)
            subsample_idx = rng.choice(n_samples, max_samples_for_bic, replace=False)
            features_for_gmm = features_scaled[subsample_idx]
            logger.info(
                f"Subsampled {max_samples_for_bic:,} from {n_samples:,} "
                f"for GMM fitting (preserves distribution, speeds up BIC search)"
            )
        else:
            features_for_gmm = features_scaled

        # Step 2: BIC search for optimal K
        self._bic_scores: Dict[int, float] = {}
        for k in range(k_range[0], k_range[1]):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                max_iter=200,
                n_init=10,
                random_state=42,
            )
            gmm.fit(features_for_gmm)
            self._bic_scores[k] = gmm.bic(features_for_gmm)
            logger.info(f"  K={k}: BIC={self._bic_scores[k]:.2f}")

        optimal_k = min(self._bic_scores, key=self._bic_scores.get)
        logger.info(
            f"Optimal K={optimal_k} (BIC={self._bic_scores[optimal_k]:.2f})"
        )

        # Step 3: Fit final GMM with optimal K
        gmm_final = GaussianMixture(
            n_components=optimal_k,
            covariance_type="full",
            max_iter=200,
            n_init=10,
            random_state=42,
        )
        gmm_final.fit(features_for_gmm)

        logger.info(
            f"GMM fitted: {optimal_k} components, "
            f"converged={gmm_final.converged_}, "
            f"n_iter={gmm_final.n_iter_}"
        )

        # Log component weights for interpretability
        for i, weight in enumerate(gmm_final.weights_):
            logger.info(
                f"  Component {i}: weight={weight:.4f}, "
                f"mean={gmm_final.means_[i].round(3)}"
            )

        return gmm_final, scaler, optimal_k
    
    def _process_single_recording(self, rec_id: int) -> Tuple[int, Optional[np.ndarray]]:
        """Process a single recording, returning (total_frames, features_or_None)."""
        try:
            tracks = self.load_recording(rec_id)
            total = len(tracks)
            features, _ = self.compute_aggressiveness_features(tracks)
            if features.shape[0] > 0:
                return total, features
            else:
                logger.warning(f"Recording {rec_id:02d}: no valid features extracted")
                return total, None
        except FileNotFoundError:
            logger.warning(f"Recording {rec_id:02d}: file not found, skipping")
            return 0, None
        except Exception as e:
            logger.error(f"Recording {rec_id:02d}: error: {e}", exc_info=True)
            return 0, None

    def process_all_recordings(self, config: AggressivenessConfig, n_jobs: int = -1) -> Dict:
        """
        Process all 60 HighD recordings in parallel and return aggregate results.

        Uses joblib to parallelize recording loading and feature extraction
        across all available CPU cores. Only aggressive moment identification
        and GMM fitting are done sequentially on the combined data.

        Args:
            config: AggressivenessConfig controlling thresholds and percentiles.
            n_jobs: Number of parallel workers. -1 = all cores (default).

        Returns:
            Dict with keys:
                total_frames, valid_frames, aggressive_frames, aggressive_ratio,
                feature_statistics, all_features, aggressive_features,
                gmm, scaler, optimal_k, per_criterion_counts, bic_scores,
                per_recording_stats
        """
        from joblib import Parallel, delayed

        rec_ids = list(range(1, self.num_recordings + 1))
        logger.info(f"Processing {len(rec_ids)} recordings in parallel (n_jobs={n_jobs})")

        # Parallel feature extraction across all recordings
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self._process_single_recording)(rec_id) for rec_id in rec_ids
        )

        all_features_list: List[np.ndarray] = []
        total_frames = 0
        per_recording_stats = []

        for rec_id, (n_frames, features) in zip(rec_ids, results_list):
            total_frames += n_frames
            if features is not None:
                all_features_list.append(features)
                per_recording_stats.append({
                    "recording_id": rec_id,
                    "total_frames": n_frames,
                    "valid_frames": features.shape[0],
                })
            else:
                per_recording_stats.append({
                    "recording_id": rec_id,
                    "total_frames": n_frames,
                    "valid_frames": 0,
                })

        if not all_features_list:
            raise RuntimeError(
                "No valid features extracted from any recording. "
                "Check data directory and file format."
            )

        # Concatenate all features across recordings
        all_features = np.vstack(all_features_list)
        valid_frames = all_features.shape[0]

        logger.info(
            f"Combined features: {valid_frames:,} valid frames "
            f"from {total_frames:,} total frames "
            f"({valid_frames / total_frames * 100:.1f}% have preceding vehicle + valid jerk)"
        )

        # Identify aggressive moments on the combined dataset
        aggressive_mask, per_criterion_counts = self.identify_aggressive_moments(
            all_features, config
        )
        aggressive_features = all_features[aggressive_mask]
        aggressive_frames = int(aggressive_mask.sum())
        aggressive_ratio = aggressive_frames / valid_frames if valid_frames > 0 else 0.0

        logger.info(
            f"Aggressive moments: {aggressive_frames:,} / {valid_frames:,} "
            f"({aggressive_ratio:.4%})"
        )

        # Compute per-feature statistics
        feature_statistics = {}
        percentile_points = [5, 25, 50, 75, 85, 95, 99]
        for i, name in enumerate(self.FEATURE_NAMES):
            col = all_features[:, i]
            feature_statistics[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "percentiles": {
                    str(p): float(np.percentile(col, p))
                    for p in percentile_points
                },
            }

        # Fit GMM on aggressive features
        if aggressive_frames < 100:
            logger.warning(
                f"Only {aggressive_frames} aggressive frames -- too few for "
                f"reliable GMM fitting. Returning results without GMM."
            )
            return {
                "total_frames": total_frames,
                "valid_frames": valid_frames,
                "aggressive_frames": aggressive_frames,
                "aggressive_ratio": aggressive_ratio,
                "feature_statistics": feature_statistics,
                "all_features": all_features,
                "aggressive_features": aggressive_features,
                "gmm": None,
                "scaler": None,
                "optimal_k": None,
                "per_criterion_counts": per_criterion_counts,
            }

        gmm, scaler, optimal_k = self.fit_gmm(aggressive_features)

        # Compute per-recording aggressive ratios (using global thresholds)
        offset = 0
        for stat in per_recording_stats:
            n = stat["valid_frames"]
            if n > 0:
                rec_mask = aggressive_mask[offset:offset + n]
                stat["aggressive_frames"] = int(rec_mask.sum())
                stat["aggressive_ratio"] = stat["aggressive_frames"] / n
            else:
                stat["aggressive_frames"] = 0
                stat["aggressive_ratio"] = 0.0
            offset += n

        return {
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "aggressive_frames": aggressive_frames,
            "aggressive_ratio": aggressive_ratio,
            "feature_statistics": feature_statistics,
            "all_features": all_features,
            "aggressive_features": aggressive_features,
            "gmm": gmm,
            "scaler": scaler,
            "optimal_k": optimal_k,
            "per_criterion_counts": per_criterion_counts,
            "bic_scores": getattr(self, '_bic_scores', {}),
            "per_recording_stats": per_recording_stats,
        }


def run_go_no_go_0(results: Dict) -> str:
    """
    GO/NO-GO #0 Decision: Is HighD data sufficient?
    
    GO: aggressive_ratio >= 0.5% (≥188K frames)
    CONDITIONAL GO: 0.1% - 0.5% → relax criteria or add NGSIM
    NO-GO: < 0.1% → fundamental direction reconsideration
    """
    ratio = results["aggressive_ratio"]
    
    if ratio >= 0.005:
        return "GO"
    elif ratio >= 0.001:
        return "CONDITIONAL_GO"
    else:
        return "NO_GO"
