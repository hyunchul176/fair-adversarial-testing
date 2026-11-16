"""Exploratory target-aware candidate selection with explicit rejection costs."""
from __future__ import annotations

import copy
import numpy as np

from .matched_response_ablation import PARAMS, scene_matrix, candidate_matrix, balanced_weights
from .response_conditioned_surface import (
    _expanded, _candidate_parameters, _prediction, predict_learned_surface,
)

ARMS = ("loss_scene", "loss_response", "success_scene", "success_response",
        "point_within_scene_shuffle")


def target_labels(rows, levels, tolerance):
    """Responsive training rows only; unsafe candidates must remain in the loss."""
    losses, successes = [], []
    for row in rows:
        if not row["responsive"] or float(row["envelope"]) <= 0:
            raise ValueError("Target labels require a positive responsive envelope")
        for candidate in row["candidates"]:
            error = np.abs(float(candidate["paired_risk_delta"]) / float(row["envelope"])
                           - np.asarray(levels))
            if candidate["admissible"]:
                losses.extend(np.minimum(1., error).tolist())
                successes.extend((error <= tolerance).astype(int).tolist())
            else:
                losses.extend([1.] * len(levels))
                successes.extend([0] * len(levels))
    return np.asarray(losses), np.asarray(successes)


def target_matrix(rows, protocol, use_response):
    base = candidate_matrix(rows, protocol, use_response)
    levels = protocol["risk_levels"]
    return np.column_stack([np.repeat(base, len(levels), axis=0),
                            np.tile(levels, len(base))])


def within_scene_shuffle(rows, seed):
    """Permutation confined to training scene+family groups, without replacement."""
    result = copy.deepcopy(rows)
    rng = np.random.default_rng(seed)
    groups = sorted({(r["seed"], r["defender_id"]) for r in rows})
    for scene, family in groups:
        indices = [i for i,r in enumerate(rows) if (r["seed"], r["defender_id"]) == (scene, family)]
        for target, source in zip(indices, rng.permutation(indices)):
            result[target]["response_probe"] = copy.deepcopy(rows[int(source)]["response_probe"])
    return result


def predict_fold(train, test, protocol, shuffle_seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    feasibility = HistGradientBoostingClassifier(**PARAMS, random_state=2700)
    y = np.asarray([r["responsive"] for r in train], dtype=np.int64)
    feasibility.fit(scene_matrix(train, True), y, sample_weight=balanced_weights(y))
    feasibility_p = feasibility.predict_proba(scene_matrix(test, True))[:,1]
    responsive_train = [r for r in train if r["responsive"]]
    losses, successes = target_labels(responsive_train, protocol["risk_levels"],
                                      protocol["calibration_tolerance"])
    predictions = []
    n_candidates, n_levels = len(_candidate_parameters(protocol)), len(protocol["risk_levels"])
    for mode in ("scene", "response"):
        include = mode == "response"
        x = target_matrix(responsive_train, protocol, include)
        test_x = target_matrix(test, protocol, include)
        loss_model = HistGradientBoostingRegressor(**PARAMS, random_state=3900, early_stopping=False)
        success_model = HistGradientBoostingClassifier(**PARAMS, random_state=3800, early_stopping=False)
        loss_model.fit(x, losses)
        success_model.fit(x, successes)
        loss_p = np.clip(loss_model.predict(test_x), 0., 1.).reshape(len(test), n_candidates, n_levels)
        success_p = success_model.predict_proba(test_x)[:,1].reshape(len(test), n_candidates, n_levels)
        for i, row in enumerate(test):
            for j, level in enumerate(protocol["risk_levels"]):
                for objective in ("loss", "success"):
                    if objective == "loss":
                        index = min(range(n_candidates), key=lambda c: (float(loss_p[i,c,j]), c))
                    else:
                        index = min(range(n_candidates), key=lambda c: (-float(success_p[i,c,j]), float(loss_p[i,c,j]), c))
                    predictions.append({**_prediction(row, level, feasibility_p[i], index, objective+"_"+mode),
                        "selected_expected_bounded_loss": float(loss_p[i,index,j]),
                        "selected_target_success_probability": float(success_p[i,index,j])})
    permuted = within_scene_shuffle(train, shuffle_seed)
    # Same original v3 model, including feasibility; test labels never used in fit.
    shuffled = predict_learned_surface(permuted, test, protocol, "response_conditioned")
    for p in shuffled:
        p["method"] = "point_within_scene_shuffle"
    predictions.extend(shuffled)
    return predictions
