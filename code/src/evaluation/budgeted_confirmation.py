"""Frozen confirmation scoring; no method selection on confirmation outcomes."""
from __future__ import annotations
import numpy as np
from .response_conditioned_surface import FAMILIES, row_key, score_predictions
from .matched_response_ablation import score_details

PRIMARY = 'response_disagreement_q4'
COMPARATORS = ('response_fixed_q4','response_random_q4','scene_disagreement_q4')
METHODS = tuple(f'{mode}_{strategy}_q4' for mode in ('scene','response')
                for strategy in ('fixed','random','disagreement')) + ('point_response_q0',)


def validate_partition(train, test, expected_seeds, expected_scenes):
    if {r['training_seed'] for r in train} & {r['training_seed'] for r in test}:
        raise ValueError('Checkpoint leakage')
    if {r['seed'] for r in train} & {r['seed'] for r in test}:
        raise ValueError('Scene leakage')
    expected={(s,f,t) for s in expected_seeds for f in FAMILIES for t in expected_scenes}
    keys=[row_key(r) for r in test]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('Incomplete, duplicate or unexpected confirmation rows')
    for r in test:
        probe=r['response_probe']
        if not probe['valid'] or probe['initial_hash']!=r['initial_hash']:
            raise ValueError('Probe invalid or initial-world mismatch')
        if len(r['features'])!=36 or len(probe['summary']['features'])!=18:
            raise ValueError('Feature dimensions changed')
        if not np.isfinite(r['features']+probe['summary']['features']).all():
            raise ValueError('Nonfinite features')
        if [c['index'] for c in r['candidates']]!=list(range(18)):
            raise ValueError('Candidate grid ordering changed')


def summarize(rows,predictions,traces,tolerance):
    expected={(m,row_key(r),float(v)) for m in METHODS for r in rows for v in (0.,.33,.67,1.)}
    keys=[(p['method'],row_key(p),p['level']) for p in predictions]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('Incomplete or duplicate predictions')
    expected_traces={(m,row_key(r)) for m in METHODS for r in rows}
    trace_keys=[(t['method'],row_key(t)) for t in traces]
    if len(trace_keys)!=len(expected_traces) or set(trace_keys)!=expected_traces:
        raise ValueError('Incomplete or duplicate query-cost traces')
    for t in traces:
        budget=0 if t['method']=='point_response_q0' else 4
        if t['query_count']!=budget or len(t['queries'])!=budget or len({q['index'] for q in t['queries']})!=budget:
            raise ValueError('Query budget mismatch')
    details=score_details(rows,predictions,tolerance)
    metrics=score_predictions(rows,predictions,tolerance)
    def diag(ps,ts):
        executed=[p for p in ps if p['predict_feasible']]
        queries=sum(t['query_count'] for t in ts)
        unsafe_queries=sum(t['query_inadmissible_count'] for t in ts)
        return {'examples':len(ps),'executions':len(executed),
            'inadmissible_executions':sum(p['unsafe_execution'] for p in ps),
            'inadmissible_rate_per_execution':float(np.mean([p['unsafe_execution'] for p in executed])) if executed else None,
            'bounded_decision_loss':float(np.mean([p['bounded_decision_loss'] for p in ps])),
            'query_count':queries,'inadmissible_queries':unsafe_queries,
            'inadmissible_query_rate':unsafe_queries/queries if queries else None,
            'mean_candidate_evaluations_four_targets':float(np.mean([t['unique_candidate_evaluations_for_four_targets'] for t in ts])),
            'mean_paired_rollout_upper_bound_four_targets':float(np.mean([t['paired_rollout_upper_bound_for_four_targets'] for t in ts])),
            'nominal_probe_ticks':ts[0]['nominal_probe_ticks']}
    diagnostics={}
    for m in METHODS:
        ps=[p for p in details if p['method']==m];ts=[t for t in traces if t['method']==m]
        diagnostics[m]={'overall':diag(ps,ts),'per_family':{f:diag([p for p in ps if p['defender_id']==f],[t for t in ts if t['defender_id']==f]) for f in FAMILIES}}
    comparisons={}
    for m in COMPARATORS:
        comparisons[m]={}
        for f in FAMILIES:
            p,b=metrics[PRIMARY]['per_family'][f],metrics[m]['per_family'][f]
            comparisons[m][f]={k:(p[k]-b[k] if p[k] is not None and b[k] is not None else None)
                               for k in ('decision_success_rate','responsive_end_to_end_coverage','executed_responsive_mae')}
    return {'metrics':metrics,'diagnostics':diagnostics,'primary_minus_comparator':comparisons}


def confirmation_decision(summary, absolute_gate, config):
    deltas=summary['primary_minus_comparator']
    raw_gain=all(deltas[m]['policy_456']['responsive_end_to_end_coverage'] is not None
                 and deltas[m]['policy_456']['responsive_end_to_end_coverage']>0 for m in COMPARATORS)
    point_margins=all(deltas[m][f]['responsive_end_to_end_coverage'] is not None
                      and deltas[m][f]['executed_responsive_mae'] is not None
                      and deltas[m][f]['responsive_end_to_end_coverage']>=-config['coverage_margin']
                      and deltas[m][f]['executed_responsive_mae']<=config['mae_margin']
                      for m in COMPARATORS for f in FAMILIES)
    return {'absolute_gate':absolute_gate,'raw_coverage_exceeds_all_three_comparators':raw_gain,
            'within_prespecified_point_margins':point_margins,
            'budgeted_method_confirmation_gate_passed':bool(absolute_gate['passed'] and raw_gain and point_margins),
            'statistical_superiority_established':False,
            'interpretation':'Prespecified replication gate; point margins are not an inferential noninferiority test.'}
