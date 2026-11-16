"""Run the frozen adaptive rule with paid live queries and accounted final evaluations."""
from .budgeted_response_adaptation import adapt
from .live_candidate_queries import LiveCandidateQueries


def select_live(train, observed, evaluator, protocol, include_response, strategy,
                random_seed, initial_hash, candidate_count=18, budget=4, selector=adapt):
    # Never forward target identity, candidate outcomes or envelope to the selector.
    allowed = {k: observed[k] for k in ('features', 'response_probe', 'defender_id')}
    query = LiveCandidateQueries(evaluator, candidate_count, budget, initial_hash)
    selection = selector(train, allowed, query.query, protocol, include_response,
                      strategy, budget, random_seed)
    if len(query.trace) != budget:
        raise ValueError('Incomplete live query budget')
    # Final evaluation occurs only after every decision is fixed. A unique final
    # candidate already queried is reused; a new final candidate is separately paid.
    execution = LiveCandidateQueries(evaluator, candidate_count, candidate_count, initial_hash)
    outcomes = {}
    if selection['feasibility_score'] >= .5:
        for index in sorted(set(selection['candidate_indices'])):
            outcomes[index] = (query.cached_outcome(index)
                               if index in selection['queried_indices']
                               else execution.query(index))
    return {'selection': selection, 'query_trace': query.trace,
            'query_outcomes': [query.cached_outcome(i) for i in selection['queried_indices']],
            'query_cost': query.cost_summary(), 'final_outcomes': outcomes,
            'final_trace': execution.trace, 'final_cost': execution.cost_summary()}
