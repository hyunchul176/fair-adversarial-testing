"""Finite-table information bounds, without simulation or policy training.

Rows are response conditions sharing the same initial information. Columns must
be the SAME available plans in every row. A row-wise oracle uses privileged
response identity; its gain is only headroom, never deployable feedback benefit.
For a common-prefix experiment, columns may instead identify suffixes. Callers
must independently prove shared prefixes, causal observation availability and
consistent terminal scoring. This module cannot certify those facts.
"""
import math


def finite_information_value(utilities, *, probabilities=None, observations=None):
    """Return exact expected-utility optima under three information partitions.

    `utilities` maps nonempty string response IDs to identical sets of string
    plan IDs and finite utilities. Uniform response probabilities are the default.
    `observations` optionally maps each response to an observed-equivalence label.
    Labels supplied by a caller are NOT proof that response identity is observable.
    Ties select the lexicographically smallest plan; no performance tolerance is
    used to enlarge tie sets or manufacture a positive information advantage.
    """
    if not isinstance(utilities, dict) or not utilities:
        raise ValueError('Nonempty response/plan utility table required')
    states = sorted(utilities) if all(isinstance(x, str) and x for x in utilities) else []
    if not states:
        raise ValueError('Response IDs must be nonempty strings')
    first = utilities[states[0]]
    if not isinstance(first, dict) or not first or not all(isinstance(x, str) and x for x in first):
        raise ValueError('Plan IDs must be nonempty strings')
    plans = sorted(first)
    for state in states:
        row = utilities[state]
        if not isinstance(row, dict) or set(row) != set(plans):
            raise ValueError('Every response requires exactly the same plans')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in row.values()):
            raise ValueError('Utilities must be finite numbers; missing outcomes cannot be dropped')
    if probabilities is None:
        probabilities = {state: 1 / len(states) for state in states}
    if not isinstance(probabilities, dict) or set(probabilities) != set(states):
        raise ValueError('One probability per response required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in probabilities.values()) or not math.isclose(math.fsum(probabilities.values()), 1., abs_tol=1e-12, rel_tol=0):
        raise ValueError('Strictly positive probabilities must sum to one')
    if observations is None:
        observations = dict.fromkeys(states, 'no_additional_information')
    if (not isinstance(observations, dict) or set(observations) != set(states)
            or not all(isinstance(v, str) and v for v in observations.values())):
        raise ValueError('One nonempty observation-equivalence label per response required')

    def best(members):
        values = {plan: math.fsum(probabilities[state] * utilities[state][plan] for state in members)
                  for plan in plans}
        optimum = max(values.values())
        return dict(value=optimum, plan=next(plan for plan in plans if values[plan] == optimum),
                    tied_plans=[plan for plan in plans if values[plan] == optimum])

    common = best(states)
    groups = {label: [state for state in states if observations[state] == label]
              for label in sorted(set(observations.values()))}
    conditioned = {label: best(members) for label, members in groups.items()}
    oracle = {state: best([state]) for state in states}
    observed_value = math.fsum(x['value'] for x in conditioned.values())
    oracle_value = math.fsum(x['value'] for x in oracle.values())
    common_optima = sorted(set.intersection(*(set(x['tied_plans']) for x in oracle.values())))
    return dict(common=common, observation_groups=groups, observation_choices=conditioned,
                observation_value=observed_value, oracle_choices=oracle, oracle_value=oracle_value,
                observation_advantage=observed_value-common['value'],
                privileged_oracle_advantage=oracle_value-common['value'],
                shared_rowwise_optimal_plans=common_optima,
                interpretation='Finite-table information bounds, not a trained policy, held-out estimate or causal effect')
