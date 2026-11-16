"""Prespecified GP heuristic comparison; GP conditioning executes only on lab workers."""
import math
import random
import time


def coordinates(candidates):
    return [[(c['target_ttc']-.8)/.8,(c['speed_gain']-.4)/.4,(c['authority']-2)/2] for c in candidates]


def farthest(x,remaining,paid):
    return min(remaining,key=lambda i:(-min(sum((a-b)**2 for a,b in zip(x[i],x[j])) for j in paid),i))


def posterior(x,observed,config):
    # Do not invoke this function outside the lab workers, even during recovery.
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel,Matern
    start=time.perf_counter();gp=config['gp']
    def fit(indices,y):
        model=GaussianProcessRegressor(kernel=ConstantKernel(gp['variance'],constant_value_bounds='fixed')*
            Matern(gp['length_scale'],length_scale_bounds='fixed',nu=2.5),alpha=gp['noise_variance'],
            optimizer=None,normalize_y=False)
        model.fit(np.asarray(x)[indices],y)
        return model.predict(x,return_std=True)
    safe=[i for i,o in observed.items() if o['admissible']]
    if safe:
        mu,sd=fit(safe,[observed[i]['paired_risk_delta'] for i in safe])
    else:
        mu=np.zeros(len(x));sd=np.full(len(x),math.sqrt(gp['variance']))
    # A clipped Gaussian binary-label regressor is a heuristic weight, not a calibrated probability.
    fm,_=fit(list(observed),[float(o['admissible'])-.5 for o in observed.values()])
    return {'mean':mu.tolist(),'std':sd.tolist(),'feasibility_weight':np.clip(fm+.5,.05,1).tolist(),
            'fit_seconds':time.perf_counter()-start}


def choose_gp(stats,remaining,observed,config,step):
    mu,sd,fw=(stats[k] for k in ('mean','std','feasibility_weight'))
    assert all(math.isfinite(v) for arr in (mu,sd,fw) for v in arr)
    assert all(v>=0 for v in sd) and all(.05<=v<=1 for v in fw)
    safe=[o['paired_risk_delta'] for o in observed.values() if o['admissible']]
    env=max([0.]+safe+[mu[i] for i in range(len(mu)) if fw[i]>=.5])
    if step in config['gp']['envelope_steps'] or not safe:
        utilities={i:fw[i]*max(0.,mu[i]+config['gp']['beta']*sd[i]) for i in remaining}
    else:
        level=min(config['risk_levels'],key=lambda l:(-min(abs(v/max(env,1e-12)-l) for v in safe),l))
        utilities={i:fw[i]*max(0.,config['gp']['beta']*sd[i]-abs(mu[i]-level*env)) for i in remaining}
    best=max(utilities.values())
    return min(i for i in remaining if utilities[i]>=best-1e-12)


def select(query,config,strategy,seed,predict=None):
    candidates=config['candidates'];count=len(candidates);x=coordinates(candidates)
    if config['anchors']!=[0,1] or not 2<=config['query_budget']<=count:
        raise ValueError('Unsupported anchors/budget')
    observed={};records=[]
    offset=int(strategy.rsplit('_',1)[1]) if strategy.startswith('random_') else 0
    order=random.Random(seed+offset*1000003).sample(list(range(2,count)),count-2)
    for step in range(config['query_budget']):
        remaining=[i for i in range(count) if i not in observed]
        if step<2:
            index=config['anchors'][step]
        elif strategy=='gp_straddle':
            stats=(predict or posterior)(x,observed,config)
            assert all(len(stats[k])==count for k in ('mean','std','feasibility_weight'))
            records.append({'step':step,'paid_indices':list(observed),**stats})
            index=choose_gp(stats,remaining,observed,config,step)
        elif strategy=='fixed':
            index=farthest(x,remaining,list(observed))
        elif strategy.startswith('random_'):
            index=next(i for i in order if i not in observed)
        else:
            raise ValueError('Unknown strategy')
        outcome=query(index)
        if outcome['index']!=index or not math.isfinite(outcome['paired_risk_delta']):
            raise ValueError('Invalid outcome')
        observed[index]=dict(outcome)
    safe={i:o['paired_risk_delta'] for i,o in observed.items() if o['admissible']}
    envelope=max(safe.values(),default=0.);execute=envelope>=config['responsiveness_threshold']
    choices=[min(safe,key=lambda i:(abs(safe[i]/envelope-level),i)) for level in config['risk_levels']] if execute else [None]*4
    return {'queried_indices':list(observed),'paid_envelope':envelope,'execute':execute,
            'candidate_indices':choices,'paid_outcomes':list(observed.values()),'posterior_trace':records}


def replay(query,config,strategy,seed,trace):
    records=iter(trace)
    def cached(x,observed,cfg):
        row=next(records)
        assert row['paid_indices']==list(observed) and row['step']==len(observed)
        return {k:v for k,v in row.items() if k not in ('step','paid_indices')}
    result=select(query,config,strategy,seed,predict=cached)
    assert next(records,None) is None
    return result


def same_selection(a,b):
    # Wall timing is not an algorithm output. Worker repeat conditioning is audited separately.
    return all(a[k]==b[k] for k in ('queried_indices','paid_envelope','execute','candidate_indices','paid_outcomes'))
