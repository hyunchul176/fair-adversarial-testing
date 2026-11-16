"""Registered B3 slots and three-valued paired analysis; no simulator imports."""
METHODS = ('coast', 'maximum_brake', 'maximum_acceleration',
           'frozen_reactive_42', 'frozen_reactive_123')
PROTOCOL_SHA = '6774f9d96dff9e1bccde1c7379e06614f3bb0af2b596ff2694366dd10d03afce'


def slot(phase, method, condition):
    seed = 42 if phase == 'sentinel' else (int(method.rsplit('_', 1)[1]) if method.startswith('frozen_reactive_') else 0)
    checkpoint = 16128 if seed else 'fixed'
    return dict(run_id=f'{phase}_{method}_{seed}_{checkpoint}_{condition+1:06d}',
                phase=phase, method=method, seed=seed, checkpoint=checkpoint,
                condition_id=condition, input_id=condition//2, H=(0., .5)[condition%2])


def schedule():
    return ([slot('sentinel', 'reactive', 2*i+h) for i in range(0, 16, 2) for h in (0, 1)] +
            [slot('evaluation', m, c) for m in METHODS for c in range(32)])


def generated(score, coast):
    """None means unevaluable, never zero. Preserve protocol precedence."""
    if score is None or not score['complete']:
        return None
    if not score['valid'] or score['initial_event']:
        return False
    if coast is not None and coast['complete'] and coast['valid'] and coast['event']:
        return False
    if coast is None or not coast['complete'] or not coast['valid']:
        return None
    return bool(score['event'])


def summarize(scores, inputs, technical_complete):
    """Fixed32 denominator per method, repeated H/seed measurements kept paired."""
    keyed = {}
    for row in scores:
        key = row['method'], row['input_id']*2 + int(row['H'] == .5)
        if row['H'] not in (0., .5) or key[0] not in METHODS or key[1] not in range(32) or key in keyed:
            raise ValueError('Unexpected or duplicate B3 score identity')
        keyed[key] = row
    cases = {r['input_id']: r['case_id'] for r in inputs}
    labels = {m: [generated(keyed.get((m, c)), keyed.get(('coast', c))) for c in range(32)] for m in METHODS}
    methods = {}
    for method in METHODS:
        rows = [keyed[method, c] for c in range(32) if (method, c) in keyed]
        available = [r for r in rows if r['complete']]
        paired_q = [(r,keyed.get(('coast',r['input_id']*2+int(r['H']==.5)))) for r in available]
        paired_q = [(r,z) for r,z in paired_q if z is not None and z['complete']]
        methods[method] = dict(registered=32, completed=len(available), missing=32-len(available),
            valid=sum(r['valid'] for r in available), invalid=sum(not r['valid'] for r in available),
            contacts=sum(r['contact'] for r in available),
            noncontact_risk=sum(r['event'] and not r['contact'] for r in available),
            initial_risk=sum(r['initial_event'] for r in available),
            generated=sum(v is True for v in labels[method]), generated_NA=sum(v is None for v in labels[method]),
            generated_cases=sorted({cases[c//2] for c, value in enumerate(labels[method]) if value is True}),
            mean_return=sum(r['total_return'] for r in available)/len(available) if available else None,
            mean_Q=sum(r['Q'] for r in available)/len(available) if available else None,
            mean_delta_Q_vs_coast=sum(r['Q']-z['Q'] for r,z in paired_q)/len(paired_q) if paired_q else None,
            delta_Q_complete_pairs=len(paired_q),
            contact_gain_vs_coast=sum(r['contact'] and not z['contact'] for r,z in paired_q),
            contact_loss_vs_coast=sum(z['contact'] and not r['contact'] for r,z in paired_q),
            per_H={str(h): dict(registered=16, completed=sum(r['H']==h for r in available),
                valid=sum(r['valid'] and r['H']==h for r in available),
                generated=sum(labels[method][c] is True for c in range(hi,32,2)),
                generated_NA=sum(labels[method][c] is None for c in range(hi,32,2))) for hi,h in enumerate((0.,.5))})
    comparisons = {}
    d1, d2 = [], []
    for seed in (42, 123):
        method = f'frozen_reactive_{seed}'
        a, b = labels[method], labels['maximum_brake']
        paired = [c for c in range(32) if a[c] is not None and b[c] is not None]
        gains = [c for c in paired if a[c] and not b[c]]
        losses = [c for c in paired if b[c] and not a[c]]
        gains_cases = sorted({cases[c//2] for c in gains})
        m = methods[method]
        functional = m['valid'] >= 31 and len(m['generated_cases']) >= 2 if technical_complete else None
        advantage = (functional and len(gains)-len(losses) >= 2 and len(gains_cases) >= 2
                     if technical_complete and len(paired) == 32 else None)
        comparisons[str(seed)] = dict(evaluable_conditions=len(paired), gained_conditions=gains,
            lost_conditions=losses, gained_cases=gains_cases,
            evaluable_subset_net=len(gains)-len(losses), full_cohort_net=len(gains)-len(losses) if len(paired)==32 else None,
            D1=functional, D2=advantage)
        d1.append(functional); d2.append(advantage)
    # Missing evidence takes precedence over a negative available seed.
    both = lambda values: None if any(v is None for v in values) else all(values)
    return dict(methods=methods, paired_brake=comparisons, generated_labels=labels,
                D1=both(d1), D2=both(d2), technical_complete=technical_complete,
                interpretation='Research-exposed development diagnostic; no learning/feedback causal or generalization claim.')
