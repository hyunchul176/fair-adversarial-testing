"""Frozen paid-support rule plus outcome-blind acquisition-family controls."""
import random
from .paid_support_router import select as supported_select
from .multidomain_query_baseline import same_selection as base_same_selection


def select(query,config,strategy,seed,predict=None):
    if not strategy.startswith('blind_router_'):
        return supported_select(query,config,strategy,seed,predict=predict)
    trajectory=int(strategy.rsplit('_',1)[1])
    chosen=random.Random(seed+7919+trajectory*1000003).choice(['axis_bracket','gp_straddle'])
    result=supported_select(query,config,chosen,seed,predict=predict)
    result['routing']={'strategy':chosen,'reason':'outcome_blind_coin_before_queries','level':None}
    return result


def same_selection(a,b):
    return base_same_selection(a,b) and a.get('routing')==b.get('routing')


def replay(query,config,strategy,seed,trace):
    rows=iter(trace)
    def cached(x,observed,cfg):
        row=next(rows)
        assert row['paid_indices']==list(observed) and row['step']==len(observed)
        return {k:v for k,v in row.items() if k not in ('step','paid_indices')}
    result=select(query,config,strategy,seed,predict=cached)
    assert next(rows,None) is None
    return result
