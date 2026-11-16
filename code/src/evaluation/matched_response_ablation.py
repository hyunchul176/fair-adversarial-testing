"""Matched development diagnostics; inference features never access outcome labels."""
from __future__ import annotations

import copy
from collections import defaultdict
import numpy as np

from .response_conditioned_surface import (
    FAMILIES, _candidate_parameters, _expanded, _prediction, response_features, row_key,
)

ARMS = {
    "scene_only": ("scene", "scene"),
    "response_feasibility_only": ("response", "scene"),
    "response_surface_only": ("scene", "response"),
    "response_full": ("response", "response"),
    "shuffled_response": ("shuffled", "shuffled"),
}
PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
              min_samples_leaf=10, l2_regularization=0.1)


def scene_matrix(rows, use_response):
    """Constant zero placeholders keep dimensions equal across ablation arms."""
    result = []
    for row in rows:
        scene = np.asarray(row["features"], dtype=np.float32)
        if scene.shape != (36,) or not np.isfinite(scene).all():
            raise ValueError("Expected 36 finite scene features")
        family = np.zeros(2, dtype=np.float32)
        family[FAMILIES.index(row["defender_id"])] = 1
        response = response_features(row) if use_response else np.zeros(18, dtype=np.float32)
        result.append(np.concatenate([scene, response, family]))
    return np.stack(result)


def candidate_matrix(rows, protocol, use_response):
    parameters = _candidate_parameters(protocol)
    return np.stack([np.concatenate([base, p])
                     for base in scene_matrix(rows, use_response) for p in parameters])


def shuffled_training_rows(rows, seed):
    """Permute entire probe vectors within family, training partition only."""
    shuffled = copy.deepcopy(rows)
    rng = np.random.default_rng(seed)
    for family in FAMILIES:
        indices = [i for i, row in enumerate(rows) if row["defender_id"] == family]
        for target, source in zip(indices, rng.permutation(indices)):
            shuffled[target]["response_probe"] = copy.deepcopy(rows[int(source)]["response_probe"])
    return shuffled


def balanced_weights(y):
    positives, negatives = max(1, int(y.sum())), max(1, len(y) - int(y.sum()))
    return np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))


def predict_fold(train, test, protocol, shuffle_seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    # Outcome access is restricted to the training partition here.
    _, safe, target, responsive = _expanded(train, protocol, True)
    scene_y = np.asarray([r["responsive"] for r in train], dtype=np.int64)
    calibration = (safe == 1) & responsive
    if calibration.sum() < 2 or len(set(scene_y)) != 2 or len(set(safe)) != 2:
        raise ValueError("Insufficient training classes/calibration outcomes")
    cache = {}
    for mode in ("scene", "response", "shuffled"):
        inputs = shuffled_training_rows(train, shuffle_seed) if mode == "shuffled" else train
        use_response = mode != "scene"
        x = candidate_matrix(inputs, protocol, use_response)
        feasibility = HistGradientBoostingClassifier(**PARAMS, random_state=2700)
        safety = HistGradientBoostingClassifier(**PARAMS, random_state=2800)
        risk = HistGradientBoostingRegressor(**PARAMS, random_state=2900)
        feasibility.fit(scene_matrix(inputs, use_response), scene_y,
                        sample_weight=balanced_weights(scene_y))
        safety.fit(x, safe, sample_weight=balanced_weights(safe))
        risk.fit(x[calibration], target[calibration])
        test_x = candidate_matrix(test, protocol, use_response)
        count = len(_candidate_parameters(protocol))
        cache[mode] = (
            feasibility.predict_proba(scene_matrix(test, use_response))[:, 1],
            safety.predict_proba(test_x)[:, 1].reshape(len(test), count),
            np.maximum(0, risk.predict(test_x)).reshape(len(test), count),
        )
    predictions = []
    for arm, (feasibility_mode, surface_mode) in ARMS.items():
        for i, row in enumerate(test):
            score = cache[feasibility_mode][0][i]
            safe_p, risk_p = cache[surface_mode][1][i], cache[surface_mode][2][i]
            indices = np.flatnonzero(safe_p >= .5)
            fallback = len(indices) == 0
            if fallback:
                indices = np.asarray([int(np.argmax(safe_p))])
            for level in protocol["risk_levels"]:
                index = int(min(indices, key=lambda j: (abs(float(risk_p[j]) - level), int(j))))
                predictions.append({**_prediction(row, level, float(score), index, arm),
                    "selected_predicted_risk": float(risk_p[index]),
                    "selected_admissibility_probability": float(safe_p[index]),
                    "no_predicted_admissible_fallback": fallback})
    surfaces = []
    for mode, (_, safety, risk) in cache.items():
        for i, row in enumerate(test):
            surfaces.append({"training_seed": row_key(row)[0], "defender_id": row_key(row)[1],
                "seed": row_key(row)[2], "mode": mode,
                "admissibility_probability": safety[i].tolist(), "predicted_risk": risk[i].tolist()})
    return predictions, surfaces


def score_details(rows, predictions, tolerance):
    lookup = {row_key(r): r for r in rows}
    details = []
    for prediction in predictions:
        row = lookup[row_key(prediction)]
        responsive, execute = bool(row["responsive"]), bool(prediction["predict_feasible"])
        candidate = row["candidates"][prediction["candidate_index"]]
        admissible = bool(candidate["admissible"])
        error = (abs(float(candidate["paired_risk_delta"]) / float(row["envelope"])
                     - prediction["level"]) if responsive and admissible else None)
        oracle_errors = [abs(float(c["paired_risk_delta"]) / float(row["envelope"])
                             - prediction["level"]) for c in row["candidates"]
                         if responsive and c["admissible"]] if responsive else []
        oracle_error = min(oracle_errors) if oracle_errors else None
        success = bool((not responsive and not execute) or
                       (responsive and execute and error is not None and error <= tolerance))
        # Bounded diagnostic loss: abstention/inadmissibility on responsive rows = 1;
        # false execution on nonresponsive rows = 1. Not a physical safety metric.
        loss = (min(1., error) if execute and error is not None else 1.) if responsive else float(execute)
        details.append({**prediction, "responsive": responsive, "selected_admissible": admissible,
            "selected_error_if_executed": error, "decision_success": success,
            "unsafe_execution": execute and not admissible, "bounded_decision_loss": loss,
            "oracle_error": oracle_error, "oracle_reachable": oracle_error is not None and oracle_error <= tolerance})
    return details


def diagnostic_summary(details):
    def summarize(items):
        responsive = [r for r in items if r["responsive"]]
        executed = [r for r in items if r["predict_feasible"]]
        reachable = [r for r in responsive if r["oracle_reachable"]]
        return {
            "examples": len(items), "executions": len(executed),
            "unsafe_executions": sum(r["unsafe_execution"] for r in items),
            "unsafe_rate_per_execution": float(np.mean([r["unsafe_execution"] for r in executed])) if executed else None,
            "bounded_decision_loss": float(np.mean([r["bounded_decision_loss"] for r in items])),
            "responsive_abstentions": sum(not r["predict_feasible"] for r in responsive),
            "oracle_reachable_responsive": len(reachable),
            "oracle_reachable_misses": sum(not r["decision_success"] for r in reachable),
        }
    result = {}
    for arm in ARMS:
        selected = [r for r in details if r["method"] == arm]
        result[arm] = {"overall": summarize(selected), "per_family": {f: summarize([
            r for r in selected if r["defender_id"] == f]) for f in FAMILIES},
            "per_family_level": {f"{f}/{level}": summarize([r for r in selected
                if r["defender_id"] == f and r["level"] == level])
                for f in FAMILIES for level in sorted({r["level"] for r in selected})}}
    return result


def paired_differences(details):
    """Descriptive paired differences, preserving scene/checkpoint dependence."""
    baseline = {(row_key(r), r["level"]): r for r in details if r["method"] == "scene_only"}
    result = {}
    for arm in ARMS:
        groups = defaultdict(list)
        for r in details:
            if r["method"] != arm:
                continue
            delta = int(r["decision_success"]) - int(baseline[row_key(r), r["level"]]["decision_success"])
            for key in ("overall", f"family/{r['defender_id']}", f"checkpoint/{r['training_seed']}", f"scene/{r['seed']}"):
                groups[key].append(delta)
        result[arm] = {key: float(np.mean(values)) for key, values in groups.items()}
    return result
