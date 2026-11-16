"""Frozen B5 registry and fixed-cohort analysis; no native execution."""
from src.evaluation.adversarial_b3_core import generated

PROTOCOL_SHA = 'e7714a30640fb8cf74409067f75b799acf12489c35daabc878980a96aa8d6254'
METHODS = tuple(f'constant_{a}' for a in range(6)) + tuple(f'brake_then_enter_{s}s' for s in (2,4,6,8)) + ('causal_entry','braking_entry')


def slot(phase, method, condition):
    return dict(run_id=f'{phase}_{method}_0_fixed_{condition+1:06d}', phase=phase,
                method=method, seed=0, checkpoint='fixed', condition_id=condition,
                input_id=condition//2, H=(0.,.5)[condition%2])


def schedule():
    return ([slot('sentinel','constant_4',2*i+h) for i in range(0,16,2) for h in (0,1)] +
            [slot('evaluation',m,c) for m in METHODS for c in range(32)])


def fixed_action(method, tick):
    if method not in METHODS[:-2] or tick not in range(0,120,5):
        raise ValueError('Unregistered fixed decision')
    if method.startswith('constant_'): return int(method[-1])
    seconds=int(method.removeprefix('brake_then_enter_').removesuffix('s'))
    return 1 if tick/10 < seconds else 4


def summarize(scores, inputs, technical_complete):
    keyed={}
    for r in scores:
        c=r['input_id']*2+int(r['H']==.5); key=r['method'],c
        if r['H'] not in (0.,.5) or r['input_id'] not in range(16) or key[0] not in METHODS or key in keyed:
            raise ValueError('Unexpected/duplicate B5 score')
        keyed[key]=r
    cases={r['input_id']:r['case_id'] for r in inputs}
    if set(cases)!=set(range(16)): raise ValueError('Expected sixteen frozen inputs')
    labels={m:[generated(keyed.get((m,c)),keyed.get(('constant_3',c))) for c in range(32)] for m in METHODS}
    methods={}
    for m in METHODS:
        rows=[keyed[m,c] for c in range(32) if (m,c) in keyed and keyed[m,c]['complete']]
        pairs=[(r,keyed.get(('constant_3',r['input_id']*2+int(r['H']==.5)))) for r in rows]
        pairs=[(r,n) for r,n in pairs if n is not None and n['complete']]
        methods[m]=dict(registered=32,completed=len(rows),missing=32-len(rows),
            valid=sum(r['valid'] for r in rows),invalid=sum(not r['valid'] for r in rows),
            contacts=sum(r['contact'] for r in rows),noncontact_risk=sum(r['event'] and not r['contact'] for r in rows),
            initial_risk=sum(r['initial_event'] for r in rows),
            nominal_risk=sum(n['valid'] and n['event'] for _,n in pairs),
            generated=sum(v is True for v in labels[m]),generated_NA=sum(v is None for v in labels[m]),
            generated_cases=sorted({cases[c//2] for c,v in enumerate(labels[m]) if v is True}),
            mean_return=sum(r['total_return'] for r in rows)/len(rows) if rows else None,
            mean_Q=sum(r['Q'] for r in rows)/len(rows) if rows else None,
            mean_delta_Q_vs_nominal=sum(r['Q']-n['Q'] for r,n in pairs)/len(pairs) if pairs else None,
            delta_Q_complete_pairs=len(pairs),
            per_H={str(h):dict(registered=16,completed=sum(r['H']==h for r in rows),
                valid=sum(r['valid'] and r['H']==h for r in rows),
                generated=sum(labels[m][c] is True for c in range(hi,32,2)),
                generated_NA=sum(labels[m][c] is None for c in range(hi,32,2))) for hi,h in enumerate((0.,.5))})
    agent=methods['braking_entry']
    technical_complete=bool(technical_complete and len(keyed)==384 and all(r['complete'] for r in keyed.values()))
    f1=agent['valid']>=31 and len(agent['generated_cases'])>=2 if technical_complete else None
    paired={}
    for m in METHODS[:-1]:
        a,b=labels['braking_entry'],labels[m]
        available=[c for c in range(32) if a[c] is not None and b[c] is not None]
        gains=[c for c in available if a[c] and not b[c]]; losses=[c for c in available if b[c] and not a[c]]
        gain_cases=sorted({cases[c//2] for c in gains})
        paired[m]=dict(evaluable_conditions=len(available),gained_conditions=gains,lost_conditions=losses,
            gained_cases=gain_cases,evaluable_subset_net=len(gains)-len(losses),
            full_cohort_net=len(gains)-len(losses) if len(available)==32 else None,
            advantage=(len(gains)-len(losses)>=2 and len(gain_cases)>=2) if len(available)==32 else None,
            per_H={str(h):dict(registered=16,evaluable=sum(c%2==hi for c in available),
                gains=sum(c%2==hi for c in gains),losses=sum(c%2==hi for c in losses)) for hi,h in enumerate((0.,.5))})
    f2=None if not technical_complete or any(p['advantage'] is None for p in paired.values()) else f1 and all(p['advantage'] for p in paired.values())
    fixed_complete=all(v is not None for m in METHODS[:-2] for v in labels[m])
    best=max(methods[m]['generated'] for m in METHODS[:-2]) if fixed_complete else None
    oracle=[True if any(labels[m][c] is True for m in METHODS[:-2]) else
            (None if any(labels[m][c] is None for m in METHODS[:-2]) else False) for c in range(32)]
    return dict(methods=methods,paired_comparators=paired,generated_labels=labels,G1=f1,G2=f2,
        hindsight_diagnostics=dict(non_deployable=True,best_single_fixed_generated=best,
            best_single_fixed_ties=[m for m in METHODS[:-2] if methods[m]['generated']==best] if best is not None else [],
            per_condition_fixed_oracle=oracle,oracle_generated=sum(v is True for v in oracle),oracle_NA=sum(v is None for v in oracle)),
        technical_complete=technical_complete,
        interpretation='Research-exposed development diagnostic. Rule-based policy; no training, generalization, novelty or feedback-causal claim.')
