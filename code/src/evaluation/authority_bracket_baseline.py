"""Paid-outcome-only baselines; no simulator, training prior or oracle access."""
import math
import random


def maximin(remaining, queried):
    return min(remaining, key=lambda i: (-min(abs(i-j) for j in queried), i))


def next_adaptive(observed, count, levels, threshold):
    remaining = [i for i in range(count) if i not in observed]
    safe = {i: o['paired_risk_delta'] for i,o in observed.items() if o['admissible']}
    envelope = max(safe.values(), default=0.)
    if envelope >= threshold:
        # Largest paid-set target error first; ties go to the smaller target.
        targets = sorted(levels, key=lambda level: ( -min(abs(v/envelope-level) for v in safe.values()), level))
        for level in targets:
            target = level * envelope
            proposals = []
            for low in sorted(safe):
                for high in sorted(safe):
                    interior = [i for i in remaining if low < i < high]
                    if not interior or abs(safe[high]-safe[low]) <= 1e-12:
                        continue
                    if not min(safe[low],safe[high]) <= target <= max(safe[low],safe[high]):
                        continue
                    interpolated = low+(target-safe[low])*(high-low)/(safe[high]-safe[low])
                    index = min(interior, key=lambda i:(abs(i-interpolated),i))
                    proposals.append((high-low,abs(index-interpolated),low,high,index))
            if proposals:
                return min(proposals)[-1]
    return maximin(remaining, observed)


def select(query, config, strategy, seed):
    count=config['candidate_count']
    budget=config['query_budget']
    if not 2 <= budget <= count:
        raise ValueError('Invalid budget')
    observed={}
    order=[0,count-1]
    rng=random.Random(seed)
    random_order=rng.sample(list(range(1,count-1)),count-2)
    for step in range(budget):
        if step < 2:
            index=order[step]
        elif strategy=='adaptive':
            index=next_adaptive(observed,count,config['risk_levels'],config['responsiveness_threshold'])
        elif strategy=='fixed':
            index=next(i for i in config['fixed_order'] if i not in observed)
        elif strategy=='random':
            index=next(i for i in random_order if i not in observed)
        else:
            raise ValueError('Unknown strategy')
        outcome=query(index)
        if index in observed or outcome['index']!=index or not math.isfinite(outcome['paired_risk_delta']):
            raise ValueError('Invalid paid observation')
        observed[index]=dict(outcome)
    safe={i:o['paired_risk_delta'] for i,o in observed.items() if o['admissible']}
    envelope=max(safe.values(),default=0.)
    execute=envelope>=config['responsiveness_threshold']
    indices=[min(safe,key=lambda i:(abs(safe[i]/envelope-level),i)) for level in config['risk_levels']] if execute else [None]*len(config['risk_levels'])
    return {'queried_indices':list(observed),'paid_envelope':envelope,'execute':execute,
            'candidate_indices':indices,'paid_outcomes':list(observed.values())}
