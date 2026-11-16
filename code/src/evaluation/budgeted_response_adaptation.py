"""Budgeted cached-simulator queries with no access to held-out envelopes."""
from __future__ import annotations
import numpy as np
from .matched_response_ablation import scene_matrix
from .response_conditioned_surface import _prediction


class CandidateQueries:
    """Evaluator-owned query interface; only paid candidate outcomes are revealed."""
    def __init__(self, candidates, budget):
        self.__outcomes = [(bool(c['admissible']),float(c['paired_risk_delta'])) for c in candidates]
        self.budget = int(budget)
        self.trace = []

    def query(self, index):
        if len(self.trace) >= self.budget:
            raise ValueError('Query budget exceeded')
        if not 0 <= index < len(self.__outcomes) or any(t['index']==index for t in self.trace):
            raise ValueError('Invalid or duplicate query')
        safe, delta = self.__outcomes[index]
        result = {'index':int(index),'admissible':safe,'paired_risk_delta':delta}
        self.trace.append(result)
        return dict(result)


def posterior_weights(log_weights):
    values = np.exp(log_weights - np.max(log_weights))
    return values / values.sum()


def select_from_hypotheses(deltas, admissible, weights, levels, threshold):
    envelopes = np.max(np.where(admissible,deltas,-np.inf),axis=1)
    responsive = envelopes >= threshold
    probability = float(weights @ responsive)
    normalized = np.divide(deltas,envelopes[:,None],out=np.zeros_like(deltas),
                           where=responsive[:,None])
    indices, losses = [], []
    for level in levels:
        loss = np.where(admissible & responsive[:,None],np.minimum(1.,np.abs(normalized-level)),1.)
        expected = weights @ loss
        index = int(np.argmin(expected))
        indices.append(index); losses.append(float(expected[index]))
    return probability, indices, losses


def adapt(train, observed, query, protocol, include_response, strategy, budget, random_seed):
    """Only train outcomes, observable test features and budgeted query API enter."""
    family = observed['defender_id']
    selected = [r for r in train if r['defender_id']==family]
    matrix = scene_matrix(selected,include_response).astype(np.float64)
    test = scene_matrix([observed],include_response)[0].astype(np.float64)
    std = matrix.std(axis=0)
    active = std > 1e-6
    distances = np.mean(((matrix[:,active]-test[active])/std[active])**2,axis=1) if active.any() else np.zeros(len(matrix))
    order = np.argsort(distances,kind='stable')[:min(protocol['neighbors'],len(selected))]
    selected = [selected[int(i)] for i in order]
    distance = distances[order]
    bandwidth = max(1e-6,float(np.median(distance)))
    log_weights = -distance/(2*bandwidth)
    deltas = np.asarray([[c['paired_risk_delta'] for c in r['candidates']] for r in selected],dtype=float)
    safe = np.asarray([[c['admissible'] for c in r['candidates']] for r in selected],dtype=bool)
    count = deltas.shape[1]
    permutation = np.random.default_rng(random_seed).permutation(count).tolist()
    fixed = protocol['fixed_query_order']
    queried = []
    for step in range(budget):
        weights = posterior_weights(log_weights)
        remaining = [i for i in range(count) if i not in queried]
        if strategy == 'fixed':
            index = next(i for i in fixed if i in remaining)
        elif strategy == 'random':
            index = next(i for i in permutation if i in remaining)
        elif strategy == 'disagreement':
            mean = weights @ deltas
            risk_variance = weights @ (deltas-mean)**2
            safe_probability = weights @ safe
            disagreement = risk_variance + safe_probability*(1-safe_probability)
            index = min(remaining,key=lambda i:(-float(disagreement[i]),i))
        else:
            raise ValueError('Unknown query strategy')
        observation = query(index)
        actual_safe, actual_delta = bool(observation['admissible']),float(observation['paired_risk_delta'])
        log_weights += -.5*((deltas[:,index]-actual_delta)/protocol['risk_likelihood_sigma'])**2
        log_weights += np.log(np.where(safe[:,index]==actual_safe,protocol['admissibility_match_likelihood'],
                                      1-protocol['admissibility_match_likelihood']))
        # Observed outcomes override hypotheses only for this paid candidate.
        deltas[:,index] = actual_delta; safe[:,index] = actual_safe
        queried.append(int(index))
    weights = posterior_weights(log_weights)
    probability, indices, losses = select_from_hypotheses(deltas,safe,weights,
                        protocol['risk_levels'],protocol['responsiveness_threshold'])
    return {'feasibility_score':probability,'candidate_indices':indices,'expected_losses':losses,
            'queried_indices':queried,'posterior_effective_hypotheses':float(1/(weights@weights))}


def arms(protocol):
    return [(f'{mode}_{strategy}_q{budget}',mode,strategy,budget)
            for mode in ('scene','response')
            for budget in protocol['query_budgets']
            for strategy in (('fixed',) if budget in (0,18) else ('fixed','random','disagreement'))]


def predict_fold(train,test,protocol):
    predictions, traces = [], []
    for row in test:
        # Do not pass held-out labels, envelopes, candidate lists or identity to adapt.
        observed = {k:row[k] for k in ('features','response_probe','defender_id')}
        random_seed = protocol['random_seed'] + int(row['seed'])
        for name,mode,strategy,budget in arms(protocol):
            interface = CandidateQueries(row['candidates'],budget)
            result = adapt(train,observed,interface.query,protocol,mode=='response',strategy,budget,random_seed)
            if len(interface.trace)!=budget:
                raise ValueError('Incomplete query budget')
            final_indices = {i for i in result['candidate_indices'] if result['feasibility_score']>=.5}
            queried = set(result['queried_indices'])
            trace = {'training_seed':int(row['training_seed']),'seed':int(row['seed']),
                     'defender_id':row['defender_id'],'method':name,'queries':interface.trace,
                     'query_count':budget,'nominal_probe_ticks':20 if mode=='response' else 0,
                     'query_inadmissible_count':sum(not q['admissible'] for q in interface.trace),
                     'unique_candidate_evaluations_for_four_targets':len(queried|final_indices),
                     'paired_rollout_upper_bound_for_four_targets':3*len(queried|final_indices),
                     'posterior_effective_hypotheses':result['posterior_effective_hypotheses']}
            traces.append(trace)
            for level,index,loss in zip(protocol['risk_levels'],result['candidate_indices'],result['expected_losses']):
                predictions.append({**_prediction(row,level,result['feasibility_score'],index,name),
                    'expected_bounded_loss':loss,'query_budget':budget,
                    'selected_candidate_previously_queried':index in queried})
    return predictions,traces
