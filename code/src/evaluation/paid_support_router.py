"""Development baseline routing on four paid observations, never controller labels."""
import math
from .axis_bracket_query_baseline import select as base_select
from .multidomain_query_baseline import same_selection as base_same_selection


def route(candidates, paid, config):
    safe={i:o['paired_risk_delta'] for i,o in paid.items() if o['admissible']}
    envelope=max(safe.values(),default=0.)
    if envelope < config['responsiveness_threshold']:
        return {'strategy':'gp_straddle','reason':'no_observed_responsive_envelope','level':None}
    level=min(config['risk_levels'],key=lambda l:(-min(abs(v/envelope-l) for v in safe.values()),l))
    error=min(abs(v/envelope-level) for v in safe.values())
    if error <= config['tolerance']:
        return {'strategy':'fixed','reason':'all_paid_targets_within_tolerance','level':level}
    target=level*envelope
    keys=('target_ttc','speed_gain','authority')
    for low in sorted(safe):
        for high in sorted(safe):
            if high <= low or abs(safe[high]-safe[low]) <= 1e-12:
                continue
            a,b=candidates[low],candidates[high]
            axes=[k for k in keys if abs(a[k]-b[k])>1e-12]
            if len(axes)!=1 or not min(safe[low],safe[high])<=target<=max(safe[low],safe[high]):
                continue
            axis=axes[0]
            if any(i not in paid and all(abs(c[k]-a[k])<=1e-12 for k in keys if k!=axis)
                   and min(a[axis],b[axis])<c[axis]<max(a[axis],b[axis])
                   for i,c in enumerate(candidates)):
                return {'strategy':'axis_bracket','reason':'worst_target_has_paid_axis_support','level':level}
    return {'strategy':'gp_straddle','reason':'worst_target_lacks_paid_axis_support','level':level}


def select(query,config,strategy,seed,predict=None):
    if strategy!='paid_support_router':
        return base_select(query,config,strategy,seed,predict=predict)
    paid={}
    for i in config['anchors']:
        outcome=query(i)
        if i in paid or outcome['index']!=i or not math.isfinite(outcome['paired_risk_delta']):
            raise ValueError('Invalid anchor observation')
        paid[i]=dict(outcome)
    decision=route(config['candidates'],paid,config)
    # The common anchors are charged once; the base selector sees their cache.
    def once(i):
        if i not in paid:
            paid[i]=query(i)
        return paid[i]
    result=base_select(once,config,decision['strategy'],seed,predict=predict)
    result['routing']=decision
    return result


def same_selection(a,b):
    return base_same_selection(a,b) and a.get('routing')==b.get('routing')


def replay(query,config,strategy,seed,trace):
    records=iter(trace)
    def cached(x,observed,cfg):
        row=next(records)
        assert row['paid_indices']==list(observed) and row['step']==len(observed)
        return {k:v for k,v in row.items() if k not in ('step','paid_indices')}
    result=select(query,config,strategy,seed,predict=cached)
    assert next(records,None) is None
    return result
