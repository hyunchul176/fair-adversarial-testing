"""Separate numerical-stability revision; frozen v1 and confirmation stay intact."""
import math
import numpy as np
from .matched_response_ablation import scene_matrix

TIE_ATOL = 1e-12


def stable_extreme(values, indices, maximize=False):
    """Smallest index within an absolute 1e-12 tie, independent of BLAS order."""
    indices = list(indices)
    best = (max if maximize else min)(float(values[i]) for i in indices)
    return min(i for i in indices if abs(float(values[i]) - best) <= TIE_ATOL)


def weighted(weights, values):
    return math.fsum(float(w) * float(v) for w, v in zip(weights, values))


def posterior(log_weights):
    largest = max(log_weights)
    exp = [math.exp(float(v) - float(largest)) for v in log_weights]
    total = math.fsum(exp)
    return [v / total for v in exp]


def adapt(train, observed, query, protocol, include_response, strategy, budget, random_seed):
    selected = [r for r in train if r['defender_id'] == observed['defender_id']]
    if not selected:
        raise ValueError('Source prior is empty')
    matrix = scene_matrix(selected, include_response).astype(np.float64)
    test = scene_matrix([observed], include_response)[0].astype(np.float64)
    std = matrix.std(axis=0)
    active = std > 1e-6
    distances = [math.fsum(float(v) ** 2 for v in (row[active] - test[active]) / std[active]) / int(active.sum())
                 if active.any() else 0. for row in matrix]
    # Training rows are in fixed checkpoint/scene order; rounded distances define ties.
    order = sorted(range(len(selected)), key=lambda i: (round(distances[i], 12), i))[:protocol['neighbors']]
    selected = [selected[i] for i in order]
    bandwidth = max(1e-6, float(np.median([distances[i] for i in order])))
    logs = [-distances[i] / (2 * bandwidth) for i in order]
    deltas = [[float(c['paired_risk_delta']) for c in row['candidates']] for row in selected]
    safe = [[bool(c['admissible']) for c in row['candidates']] for row in selected]
    count = len(deltas[0])
    permutation = np.random.default_rng(random_seed).permutation(count).tolist()
    queried = []
    for _ in range(budget):
        weights = posterior(logs)
        remaining = [i for i in range(count) if i not in queried]
        if strategy == 'fixed':
            index = next(i for i in protocol['fixed_query_order'] if i in remaining)
        elif strategy == 'random':
            index = next(i for i in permutation if i in remaining)
        elif strategy == 'disagreement':
            scores = []
            for i in range(count):
                mean = weighted(weights, [row[i] for row in deltas])
                variance = weighted(weights, [(row[i] - mean) ** 2 for row in deltas])
                prob = min(1., max(0., weighted(weights, [row[i] for row in safe])))
                scores.append(variance + prob * (1 - prob))
            index = stable_extreme(scores, remaining, maximize=True)
        else:
            raise ValueError('Unknown query strategy')
        outcome = query(index)
        for h in range(len(selected)):
            logs[h] += -.5 * ((deltas[h][index] - outcome['paired_risk_delta']) / protocol['risk_likelihood_sigma']) ** 2
            match = protocol['admissibility_match_likelihood']
            logs[h] += math.log(match if safe[h][index] == outcome['admissible'] else 1 - match)
            deltas[h][index] = float(outcome['paired_risk_delta'])
            safe[h][index] = bool(outcome['admissible'])
        queried.append(index)
    weights = posterior(logs)
    envelopes = [max((value for value, ok in zip(row, mask) if ok), default=-math.inf)
                 for row, mask in zip(deltas, safe)]
    responsive = [v >= protocol['responsiveness_threshold'] for v in envelopes]
    probability = min(1., max(0., weighted(weights, responsive)))
    indices, losses = [], []
    for level in protocol['risk_levels']:
        expected = [weighted(weights, [min(1., abs(deltas[h][i] / envelopes[h] - level))
                    if safe[h][i] and responsive[h] else 1. for h in range(len(selected))])
                    for i in range(count)]
        index = stable_extreme(expected, range(count))
        indices.append(index)
        losses.append(expected[index])
    return {'feasibility_score': probability, 'candidate_indices': indices,
            'expected_losses': losses, 'queried_indices': queried,
            'posterior_effective_hypotheses': 1 / math.fsum(w*w for w in weights)}
