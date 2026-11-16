"""Leakage-safe response-conditioned candidate outcome surfaces."""
from __future__ import annotations

from collections import defaultdict
import itertools

import numpy as np


FAMILIES = ("policy_456", "shielded_789")


def response_features(row: dict) -> np.ndarray:
    probe = row.get("response_probe", {})
    if not probe.get("valid") or not isinstance(probe.get("summary"), dict):
        raise ValueError("response-conditioned rows require a valid probe")
    values = np.asarray(probe["summary"]["features"], dtype=np.float32)
    if values.shape != (18,) or not np.isfinite(values).all():
        raise ValueError("response feature vector must contain 18 finite values")
    return values


def row_key(row: dict) -> tuple[int, str, int]:
    return int(row["training_seed"]), str(row["defender_id"]), int(row["seed"])


def joint_checkpoint_scene_folds(rows: list[dict], scene_fold_count: int = 5,
                                 split_seed: int = 20260919) -> list[dict]:
    """Cross held-out checkpoints with disjoint scene folds.

    Every test row is predicted exactly once.  Its checkpoint and every row from
    its scene are both absent from the corresponding training partition.
    """
    checkpoints = sorted({int(row["training_seed"]) for row in rows})
    scenes = np.asarray(sorted({int(row["seed"]) for row in rows}), dtype=np.int64)
    if len(checkpoints) < 2 or len(scenes) < scene_fold_count:
        raise ValueError("insufficient checkpoints or scenes for joint folds")
    rng = np.random.default_rng(split_seed)
    rng.shuffle(scenes)
    scene_folds = [set(int(value) for value in fold)
                   for fold in np.array_split(scenes, scene_fold_count)]
    result = []
    for checkpoint, scene_index in itertools.product(checkpoints, range(scene_fold_count)):
        heldout_scenes = scene_folds[scene_index]
        train = [row for row in rows
                 if int(row["training_seed"]) != checkpoint
                 and int(row["seed"]) not in heldout_scenes]
        test = [row for row in rows
                if int(row["training_seed"]) == checkpoint
                and int(row["seed"]) in heldout_scenes]
        if not train or not test:
            raise ValueError("empty joint checkpoint/scene fold")
        result.append({
            "heldout_checkpoint": checkpoint,
            "heldout_scenes": sorted(heldout_scenes),
            "train": train,
            "test": test,
        })
    test_keys = [row_key(row) for fold in result for row in fold["test"]]
    if len(test_keys) != len(rows) or len(set(test_keys)) != len(rows):
        raise ValueError("joint folds must cover each row exactly once")
    return result


def _family_vector(row: dict) -> np.ndarray:
    result = np.zeros(len(FAMILIES), dtype=np.float32)
    result[FAMILIES.index(str(row["defender_id"]))] = 1.0
    return result


def _row_vector(row: dict, include_response: bool) -> np.ndarray:
    parts = [np.asarray(row["features"], dtype=np.float32)]
    if include_response:
        parts.append(response_features(row))
    parts.append(_family_vector(row))
    return np.concatenate(parts)


def _candidate_parameters(protocol: dict) -> list[np.ndarray]:
    grid = protocol["candidate_grid"]
    return [np.asarray([float(ttc) / 1.6, float(gain) / 0.8,
                        float(authority) / 4.0], dtype=np.float32)
            for ttc, gain, authority in itertools.product(
                grid["target_ttc"], grid["speed_gain"], grid["authority"])]


def _expanded(rows: list[dict], protocol: dict, include_response: bool):
    parameters = _candidate_parameters(protocol)
    x, safe, normalized, responsive = [], [], [], []
    for row in rows:
        base = _row_vector(row, include_response)
        if len(row["candidates"]) != len(parameters):
            raise ValueError("candidate grid and row candidate count differ")
        for candidate, parameter in zip(row["candidates"], parameters):
            x.append(np.concatenate([base, parameter]))
            safe.append(bool(candidate["admissible"]))
            responsive.append(bool(row["responsive"]))
            normalized.append(
                float(candidate["paired_risk_delta"]) / float(row["envelope"])
                if row["responsive"] else 0.0
            )
    return (np.stack(x), np.asarray(safe, dtype=np.int64),
            np.asarray(normalized, dtype=np.float32), np.asarray(responsive, dtype=bool))


def _oracle_candidate(row: dict, level: float) -> int:
    if not row["responsive"] or float(row["envelope"]) <= 0.0:
        return 0
    safe = [candidate for candidate in row["candidates"] if candidate["admissible"]]
    if not safe:
        return 0
    return min((abs(float(candidate["paired_risk_delta"]) / float(row["envelope"])
                    - level), int(candidate["index"])) for candidate in safe)[1]


def _prediction(row: dict, level: float, score: float, candidate: int,
                method: str) -> dict:
    return {
        "training_seed": int(row["training_seed"]),
        "seed": int(row["seed"]),
        "defender_id": str(row["defender_id"]),
        "level": float(level),
        "feasibility_score": float(score),
        "predict_feasible": bool(score >= 0.5),
        "candidate_index": int(candidate),
        "method": method,
    }


def predict_nearest_response(train_rows: list[dict], test_rows: list[dict],
                             protocol: dict) -> list[dict]:
    levels = [float(value) for value in protocol["risk_levels"]]
    train_matrix = np.stack([_row_vector(row, True) for row in train_rows])
    mean, scale = train_matrix.mean(axis=0), train_matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    by_family = defaultdict(list)
    for row in train_rows:
        by_family[str(row["defender_id"])].append(row)
    predictions = []
    for row in test_rows:
        vector = (_row_vector(row, True) - mean) / scale
        neighbor = min(by_family[str(row["defender_id"])], key=lambda item: (
            float(np.linalg.norm((_row_vector(item, True) - mean) / scale - vector)),
            int(item["training_seed"]), int(item["seed"])))
        score = float(neighbor["responsive"])
        for level in levels:
            predictions.append(_prediction(
                row, level, score, _oracle_candidate(neighbor, level),
                "nearest_response"))
    return predictions


def predict_learned_surface(train_rows: list[dict], test_rows: list[dict],
                            protocol: dict, method: str) -> list[dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    if method not in {"response_conditioned", "robust_lower_tail"}:
        raise ValueError(f"unknown learned surface: {method}")
    train_scene_x = np.stack([_row_vector(row, True) for row in train_rows])
    train_scene_y = np.asarray([row["responsive"] for row in train_rows], dtype=np.int64)
    feasibility = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=10, l2_regularization=0.1, random_state=2700)
    positives = max(1, int(train_scene_y.sum()))
    negatives = max(1, int(len(train_scene_y) - train_scene_y.sum()))
    weights = np.where(train_scene_y == 1, len(train_scene_y) / (2 * positives),
                       len(train_scene_y) / (2 * negatives))
    feasibility.fit(train_scene_x, train_scene_y, sample_weight=weights)

    train_x, train_safe, train_normalized, train_responsive = _expanded(
        train_rows, protocol, True)
    safety = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=10, l2_regularization=0.1, random_state=2800)
    positives = max(1, int(train_safe.sum()))
    negatives = max(1, int(len(train_safe) - train_safe.sum()))
    weights = np.where(train_safe == 1, len(train_safe) / (2 * positives),
                       len(train_safe) / (2 * negatives))
    safety.fit(train_x, train_safe, sample_weight=weights)
    risk_args = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                     min_samples_leaf=10, l2_regularization=0.1, random_state=2900)
    if method == "robust_lower_tail":
        risk_args.update(loss="quantile", quantile=0.2)
    risk = HistGradientBoostingRegressor(**risk_args)
    calibration = (train_safe == 1) & train_responsive
    if calibration.sum() < 2:
        raise ValueError("insufficient safe responsive candidates")
    risk.fit(train_x[calibration], train_normalized[calibration])

    test_scene_x = np.stack([_row_vector(row, True) for row in test_rows])
    feasibility_probability = feasibility.predict_proba(test_scene_x)[:, 1]
    test_x, _, _, _ = _expanded(test_rows, protocol, True)
    safe_probability = safety.predict_proba(test_x)[:, 1]
    predicted_risk = np.maximum(0.0, risk.predict(test_x))
    candidate_count = len(_candidate_parameters(protocol))
    safety_threshold = 0.8 if method == "robust_lower_tail" else 0.5
    levels = [float(value) for value in protocol["risk_levels"]]
    predictions, cursor = [], 0
    for row, score in zip(test_rows, feasibility_probability):
        candidate_safe = safe_probability[cursor:cursor + candidate_count]
        candidate_risk = predicted_risk[cursor:cursor + candidate_count]
        cursor += candidate_count
        safe_indices = np.flatnonzero(candidate_safe >= safety_threshold)
        if len(safe_indices) == 0:
            safe_indices = np.asarray([int(np.argmax(candidate_safe))])
        for level in levels:
            candidate = min(safe_indices, key=lambda index: (
                abs(float(candidate_risk[index]) - level), int(index)))
            predictions.append(_prediction(row, level, float(score), candidate, method))
    return predictions


def predict_frozen_family_surface(training_rows: list[dict], test_rows: list[dict],
                                  protocol: dict) -> list[dict]:
    """Reproduce the historical static family-only surface without new-row fitting."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    raw = [row for row in training_rows if row["defender_id"] == "policy_456"]
    feasibility = make_pipeline(
        StandardScaler(),
        SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True,
            random_state=1700))
    feasibility.fit(np.asarray([row["features"] for row in raw], dtype=np.float32),
                    np.asarray([row["responsive"] for row in raw], dtype=np.int64))
    parameters = _candidate_parameters(protocol)
    x, safe, normalized, responsive = [], [], [], []
    for row in training_rows:
        family = np.zeros(4, dtype=np.float32)
        historical = ("idm", "mobil", "policy_456", "shielded_789")
        family[historical.index(row["defender_id"])] = 1.0
        base = np.concatenate([np.asarray(row["features"], dtype=np.float32), family])
        for candidate, parameter in zip(row["candidates"], parameters):
            x.append(np.concatenate([base, parameter]))
            safe.append(bool(candidate["admissible"]))
            responsive.append(bool(row["responsive"]))
            normalized.append(float(candidate["paired_risk_delta"]) / float(row["envelope"])
                              if row["responsive"] else 0.0)
    x, safe = np.stack(x), np.asarray(safe, dtype=np.int64)
    normalized, responsive = np.asarray(normalized), np.asarray(responsive, dtype=bool)
    safety = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=10, l2_regularization=0.1, random_state=1800)
    positives, negatives = max(1, int(safe.sum())), max(1, int(len(safe) - safe.sum()))
    safety.fit(x, safe, sample_weight=np.where(
        safe == 1, len(safe) / (2 * positives), len(safe) / (2 * negatives)))
    risk = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
        min_samples_leaf=10, l2_regularization=0.1, random_state=1900)
    calibration = (safe == 1) & responsive
    risk.fit(x[calibration], normalized[calibration])

    raw_test = [row for row in test_rows if row["defender_id"] == "policy_456"]
    raw_probability = feasibility.predict_proba(
        np.asarray([row["features"] for row in raw_test], dtype=np.float32))[:, 1]
    by_key = {(int(row["training_seed"]), int(row["seed"])): float(value)
              for row, value in zip(raw_test, raw_probability)}
    test_x = []
    historical = ("idm", "mobil", "policy_456", "shielded_789")
    for row in test_rows:
        family = np.zeros(4, dtype=np.float32)
        family[historical.index(row["defender_id"])] = 1.0
        base = np.concatenate([np.asarray(row["features"], dtype=np.float32), family])
        test_x.extend(np.concatenate([base, parameter]) for parameter in parameters)
    safe_probability = safety.predict_proba(np.stack(test_x))[:, 1]
    predicted_risk = np.maximum(0.0, risk.predict(np.stack(test_x)))
    levels = [float(value) for value in protocol["risk_levels"]]
    predictions, cursor = [], 0
    for row in test_rows:
        candidate_safe = safe_probability[cursor:cursor + len(parameters)]
        candidate_risk = predicted_risk[cursor:cursor + len(parameters)]
        cursor += len(parameters)
        safe_indices = np.flatnonzero(candidate_safe >= 0.5)
        if len(safe_indices) == 0:
            safe_indices = np.asarray([int(np.argmax(candidate_safe))])
        score = by_key[(int(row["training_seed"]), int(row["seed"]))]
        for level in levels:
            candidate = min(safe_indices, key=lambda index: (
                abs(float(candidate_risk[index]) - level), int(index)))
            predictions.append(_prediction(
                row, level, score, candidate, "frozen_family_only"))
    return predictions


def score_predictions(rows: list[dict], predictions: list[dict], tolerance: float) -> dict:
    lookup = {row_key(row): row for row in rows}
    scored = []
    for prediction in predictions:
        key = (int(prediction["training_seed"]), str(prediction["defender_id"]),
               int(prediction["seed"]))
        row = lookup[key]
        responsive, execute = bool(row["responsive"]), bool(prediction["predict_feasible"])
        error = None
        if execute and responsive:
            candidate = row["candidates"][int(prediction["candidate_index"])]
            if candidate["admissible"]:
                error = abs(float(candidate["paired_risk_delta"]) / float(row["envelope"])
                            - float(prediction["level"]))
        within = error is not None and error <= tolerance
        success = (not responsive and not execute) or (responsive and execute and within)
        scored.append({**prediction, "responsive": responsive, "normalized_error": error,
                       "within_tolerance": within, "decision_success": success})

    def aggregate(items: list[dict]) -> dict:
        responsive = [row for row in items if row["responsive"]]
        nonresponsive = [row for row in items if not row["responsive"]]
        executed = [row for row in responsive if row["normalized_error"] is not None]
        return {
            "examples": len(items),
            "responsive_examples": len(responsive),
            "decision_success_rate": float(np.mean([row["decision_success"] for row in items])),
            "responsive_end_to_end_coverage": (
                float(np.mean([row["within_tolerance"] for row in responsive]))
                if responsive else None),
            "executed_responsive_mae": (
                float(np.mean([row["normalized_error"] for row in executed]))
                if executed else None),
            "false_execution_rate": (
                float(np.mean([row["predict_feasible"] for row in nonresponsive]))
                if nonresponsive else None),
        }

    result = {}
    for method in sorted({row["method"] for row in scored}):
        selected = [row for row in scored if row["method"] == method]
        result[method] = {
            "overall": aggregate(selected),
            "per_family": {family: aggregate(
                [row for row in selected if row["defender_id"] == family])
                for family in FAMILIES},
            "per_checkpoint": {str(seed): aggregate(
                [row for row in selected if int(row["training_seed"]) == seed])
                for seed in sorted({int(row["training_seed"]) for row in selected})},
        }
    return result
