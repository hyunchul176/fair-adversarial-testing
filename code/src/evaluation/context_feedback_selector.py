"""Leakage-safe labels, splits, baselines, and metrics for context selection."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


DEFENDERS = ('idm', 'mobil', 'policy_456', 'shielded_789')
ABSTAIN = 18


def oracle_candidate(row, level):
    """Return the in-grid candidate with the smallest normalized-risk error."""
    if not row['responsive']:
        return ABSTAIN
    envelope = float(row['envelope'])
    safe = [candidate for candidate in row['candidates'] if candidate['admissible']]
    return min(
        (abs(float(candidate['paired_risk_delta']) / envelope - float(level)),
         int(candidate['index']))
        for candidate in safe
    )[1]


def grouped_folds(rows, fold_count=5, split_seed=20260917):
    """Split by scene seed so four defender rows never cross a fold boundary."""
    seeds = np.asarray(sorted({int(row['seed']) for row in rows}), dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    shuffled = seeds.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, fold_count)
    return [sorted(int(seed) for seed in fold) for fold in folds]


def fit_standardizer(rows, seeds):
    selected = [np.asarray(row['features'], dtype=np.float32)
                for row in rows if int(row['seed']) in set(seeds)]
    matrix = np.stack(selected)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def fixed_candidate(rows, level, defender_id=None):
    """Select one training-only candidate, penalizing unsafe executions."""
    responsive = [row for row in rows if row['responsive'] and
                  (defender_id is None or row['defender_id'] == defender_id)]
    if not responsive:
        return 0
    losses = []
    for index in range(18):
        values = []
        for row in responsive:
            candidate = row['candidates'][index]
            if not candidate['admissible']:
                values.append(1.0)
            else:
                values.append(abs(float(candidate['paired_risk_delta']) /
                                  float(row['envelope']) - float(level)))
        losses.append((float(np.mean(values)), index))
    return min(losses)[1]


def baseline_predictions(train_rows, test_rows, levels, kind):
    """Build predictions without consulting outcomes from test scenes."""
    predictions = []
    global_prevalence = float(np.mean([row['responsive'] for row in train_rows]))
    defender_prevalence = {
        defender: float(np.mean([row['responsive'] for row in train_rows
                                 if row['defender_id'] == defender]))
        for defender in DEFENDERS
    }
    global_candidates = {level: fixed_candidate(train_rows, level) for level in levels}
    defender_candidates = {
        (defender, level): fixed_candidate(train_rows, level, defender)
        for defender in DEFENDERS for level in levels
    }
    mean, scale = fit_standardizer(train_rows, {int(row['seed']) for row in train_rows})
    by_defender = defaultdict(list)
    for row in train_rows:
        by_defender[row['defender_id']].append(row)

    for row in test_rows:
        defender = row['defender_id']
        for level in levels:
            if kind == 'universal':
                score = global_prevalence
                candidate = global_candidates[level]
            elif kind == 'defender_only':
                score = defender_prevalence[defender]
                candidate = defender_candidates[(defender, level)]
            elif kind == 'nearest_context':
                feature = (np.asarray(row['features'], dtype=np.float32) - mean) / scale
                neighbor = min(
                    by_defender[defender],
                    key=lambda item: (float(np.linalg.norm(
                        (np.asarray(item['features'], dtype=np.float32) - mean) / scale - feature)),
                                      int(item['seed'])))
                score = float(neighbor['responsive'])
                candidate = oracle_candidate(neighbor, level)
                if candidate == ABSTAIN:
                    candidate = 0
            else:
                raise ValueError(f'Unknown baseline: {kind}')
            feasible = score >= 0.5
            predictions.append({
                'seed': int(row['seed']), 'defender_id': defender, 'level': float(level),
                'feasibility_score': score, 'predict_feasible': feasible,
                'candidate_index': int(candidate), 'method': kind,
            })
    return predictions


def score_predictions(rows, predictions, tolerance=0.15):
    row_lookup = {(int(row['seed']), row['defender_id']): row for row in rows}
    scored = []
    for prediction in predictions:
        row = row_lookup[(int(prediction['seed']), prediction['defender_id'])]
        responsive = bool(row['responsive'])
        execute = bool(prediction['predict_feasible'])
        admissible = False
        error = None
        within = False
        if execute and responsive:
            candidate = row['candidates'][int(prediction['candidate_index'])]
            admissible = bool(candidate['admissible'])
            if admissible:
                error = abs(float(candidate['paired_risk_delta']) / float(row['envelope'])
                            - float(prediction['level']))
                within = error <= tolerance
        decision_success = (not responsive and not execute) or (responsive and execute and within)
        scored.append({**prediction, 'responsive': responsive, 'executed_admissible': admissible,
                       'normalized_error': error, 'within_tolerance': within,
                       'decision_success': decision_success})

    def aggregate(items):
        labels = [item['responsive'] for item in items]
        predicted = [item['predict_feasible'] for item in items]
        positives = sum(labels)
        negatives = len(labels) - positives
        sensitivity = (sum(label and pred for label, pred in zip(labels, predicted)) / positives
                       if positives else None)
        specificity = (sum((not label) and (not pred) for label, pred in zip(labels, predicted)) /
                       negatives if negatives else None)
        balanced = ((sensitivity + specificity) / 2
                    if sensitivity is not None and specificity is not None else None)
        responsive_items = [item for item in items if item['responsive']]
        penalty_errors = [item['normalized_error'] if item['normalized_error'] is not None else 1.0
                          for item in responsive_items]
        executed = [item for item in responsive_items if item['normalized_error'] is not None]
        return {
            'examples': len(items), 'responsive_examples': len(responsive_items),
            'feasibility_sensitivity': sensitivity, 'feasibility_specificity': specificity,
            'feasibility_balanced_accuracy': balanced,
            'responsive_end_to_end_mae': float(np.mean(penalty_errors)) if penalty_errors else None,
            'responsive_end_to_end_coverage': (
                sum(item['within_tolerance'] for item in responsive_items) / len(responsive_items)
                if responsive_items else None),
            'executed_responsive_mae': (
                float(np.mean([item['normalized_error'] for item in executed])) if executed else None),
            'decision_success_rate': (
                float(np.mean([item['decision_success'] for item in items])) if items else None),
            'false_execution_rate': (
                sum((not item['responsive']) and item['predict_feasible'] for item in items) / negatives
                if negatives else None),
        }

    methods = sorted({item['method'] for item in scored})
    metrics = {}
    for method in methods:
        method_items = [item for item in scored if item['method'] == method]
        metrics[method] = {'overall': aggregate(method_items), 'per_defender': {
            defender: aggregate([item for item in method_items if item['defender_id'] == defender])
            for defender in DEFENDERS
        }}
    return scored, metrics
