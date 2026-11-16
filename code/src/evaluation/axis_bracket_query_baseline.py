"""Paid response interpolation along existing physical parameter axes; no novelty claim."""
import math
import random
from .multidomain_query_baseline import coordinates,farthest,posterior,choose_gp,same_selection


def next_bracket(candidates,observed,config):
    keys=('target_ttc','speed_gain','authority');scales=(.8,.4,2.)
    remaining=[i for i in range(len(candidates)) if i not in observed]
    safe={i:o['paired_risk_delta'] for i,o in observed.items() if o['admissible']}
    envelope=max(safe.values(),default=0.)
    if envelope>=config['responsiveness_threshold']:
        levels=sorted(config['risk_levels'],key=lambda l:(-min(abs(v/envelope-l) for v in safe.values()),l))
        for level in levels:
            target=level*envelope;proposals=[]
            for low in sorted(safe):
                for high in sorted(safe):
                    if high<=low or abs(safe[high]-safe[low])<=1e-12:continue
                    a,b=candidates[low],candidates[high]
                    axes=[k for k in keys if abs(a[k]-b[k])>1e-12]
                    if len(axes)!=1 or not min(safe[low],safe[high])<=target<=max(safe[low],safe[high]):continue
                    axis=axes[0]
                    interior=[i for i in remaining if all(abs(candidates[i][k]-a[k])<=1e-12 for k in keys if k!=axis)
                        and min(a[axis],b[axis])<candidates[i][axis]<max(a[axis],b[axis])]
                    if not interior:continue
                    estimate=a[axis]+(target-safe[low])*(b[axis]-a[axis])/(safe[high]-safe[low])
                    scale=scales[keys.index(axis)]
                    index=min(interior,key=lambda i:(abs(candidates[i][axis]-estimate),i))
                    proposals.append((abs(a[axis]-b[axis])/scale,abs(candidates[index][axis]-estimate)/scale,low,high,index))
            if proposals:return min(proposals)[-1]
    return farthest(coordinates(candidates),remaining,list(observed))


def select(query,config,strategy,seed,predict=None):
    candidates=config['candidates'];count=len(candidates);x=coordinates(candidates);anchors=config['anchors']
    if len(anchors)!=4 or len(set(anchors))!=4 or not 4<=config['query_budget']<=count:
        raise ValueError('Invalid anchors/budget')
    observed={};records=[]
    offset=int(strategy.rsplit('_',1)[1]) if strategy.startswith('random_') else 0
    rest=[i for i in range(count) if i not in anchors]
    order=random.Random(seed+offset*1000003).sample(rest,len(rest))
    for step in range(config['query_budget']):
        remaining=[i for i in range(count) if i not in observed]
        if step<len(anchors):index=anchors[step]
        elif strategy=='axis_bracket':index=next_bracket(candidates,observed,config)
        elif strategy=='gp_straddle':
            stats=(predict or posterior)(x,observed,config)
            assert all(len(stats[k])==count for k in ('mean','std','feasibility_weight'))
            records.append({'step':step,'paid_indices':list(observed),**stats})
            index=choose_gp(stats,remaining,observed,config,step)
        elif strategy=='fixed':index=farthest(x,remaining,list(observed))
        elif strategy.startswith('random_'):index=next(i for i in order if i not in observed)
        else:raise ValueError('Unknown strategy')
        outcome=query(index)
        if index in observed or outcome['index']!=index or not math.isfinite(outcome['paired_risk_delta']):
            raise ValueError('Invalid paid outcome')
        observed[index]=dict(outcome)
    safe={i:o['paired_risk_delta'] for i,o in observed.items() if o['admissible']}
    envelope=max(safe.values(),default=0.);execute=envelope>=config['responsiveness_threshold']
    choices=[min(safe,key=lambda i:(abs(safe[i]/envelope-level),i)) for level in config['risk_levels']] if execute else [None]*4
    return {'queried_indices':list(observed),'paid_envelope':envelope,'execute':execute,
        'candidate_indices':choices,'paid_outcomes':list(observed.values()),'posterior_trace':records}


def replay(query,config,strategy,seed,trace):
    records=iter(trace)
    def cached(x,observed,cfg):
        row=next(records);assert row['paid_indices']==list(observed) and row['step']==len(observed)
        return {k:v for k,v in row.items() if k not in ('step','paid_indices')}
    result=select(query,config,strategy,seed,predict=cached)
    assert next(records,None) is None
    return result
