"""B54 analysis: the definitions of the B54 design (available on request), from the aw_run outputs.

  detected(v, c, s)  = the v2j policy episode of seed s reports (unified contract) or, when it does not, the refinement
                       of (v, c) finds a success
  confirmed(v, c, s) = detected and the reported failure's reruns (repro phase) collide in >= CONFIRM of REPRO
  rates              = mean over seeds of the per-seed share of the partition's cases (an errored job = not detected)
  attacker-caused    = invalid manoeuvre or the CC also collides in the highway-env replay (A1: v2j policy contacts,
                       A2: collision-seeking contacts)
  A3                 = confirmed rate AW-default - AW-r157, paired case bootstrap (10,000, rng 0), 95% lower bound > 0
  A4                 = AW-r157 confirmed rate >= 0.10;  A5 = reruns colliding / reruns >= 0.8 per variant
Descriptive: failure mode (braking onset after the cut-in is visible, peak deceleration, MRM) against the CC
counterfactual, cost, fidelity, the B29 consensus and R157 Annex 5 grade of the reported failures.

  python -m fairtest.aw_analysis OUT_DIR SUMMARY.json [--partition development]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401

VARIANTS = ('aw_default', 'aw_r157')
SEEDS = (1, 2, 3)
CONFIRM, REPRO, PERCEIVE_M = 2, 4, 1.095
A1_MAX, A2_MIN, A4_MIN, A5_MIN = .05, .8, .1, .8


def _load(out, phase):
    rows = []
    for p in sorted(Path(out).glob(f'{phase}_*.jsonl')):
        rows += [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    return rows


def _mode(trace, t_hit):
    """Failure mode from an Autoware trace: braking onset after the cut-in is visible, peak deceleration, MRM."""
    rows = [r for r in trace if t_hit is None or r['t'] <= t_hit + 1e-6]
    k_in = next((k for k, r in enumerate(rows) if abs(r['adv'][1]) >= PERCEIVE_M), None)
    acc = [r['ego'][3] for r in rows]
    brake = None if k_in is None else next((k for k in range(k_in, len(rows)) if acc[k] is not None and acc[k] <= -1.), None)
    mrm = any(r.get('mrm') and r['mrm'][0] == 2 for r in rows)
    return dict(t_cutin=None if k_in is None else rows[k_in]['t'],
                brake_delay_s=None if brake is None else round(rows[brake]['t'] - rows[k_in]['t'], 1),
                min_acc=min((a for a in acc if a is not None), default=None), mrm=mrm)


def _cc_mode(case, how, payload):
    from fairtest.b29_consensus import _run, features
    f = features(case, _run(case, how, payload, 'cc_r157'))
    return dict(brake_delay_s=f.get('sut_brake_delay_s'), min_acc=f.get('sut_min_acc'))


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]


def main(a):
    from fairtest.b29_consensus import task as consensus
    from fairtest.runner import _STATE, _init
    _init()
    ids = sorted(c for c, v in _STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    pol, ref, rep = _load(a.out, 'policy'), _load(a.out, 'refine'), _load(a.out, 'repro')
    P = {(r['variant'], r['case_id'], r['attacker'], r['seed']): r['result'] for r in pol}
    R = {(r['variant'], r['case_id']): r['result'] for r in ref}
    X = {}
    for r in rep:  # reruns per reported failure
        runs = r['result'].get('runs') or []
        X[(r['variant'], r['case_id'], r['source'])] = dict(contacts=sum(bool(x['contact']) for x in runs), n=len(runs),
                                                            error=r['result'].get('error'))
    S = dict(partition=a.partition, cases=len(ids), definitions=__doc__.split('Descriptive')[0].strip(), variants={})
    per_case = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        out = S['variants'][v] = {}
        errors = dict(policy=sum(1 for (vv, c, at, s), x in P.items() if vv == v and 'error' in x),
                      refine=sum(1 for (vv, c), x in R.items() if vv == v and 'error' in x),
                      repro=sum(1 for (vv, c, src), x in X.items() if vv == v and (x['error'] or x['n'] < REPRO)))
        # A1 / A2: attacker-caused share among contacts
        for attacker, key in (('v2j', 'A1'), ('collide', 'A2')):
            hits = [x for (vv, c, at, s), x in P.items() if vv == v and at == attacker and x.get('contact')]
            caused = [not (x.get('valid') and x.get('fair')) for x in hits]
            out[key] = dict(contacts=len(hits), attacker_caused=int(sum(caused)),
                            share=round(float(np.mean(caused)), 4) if hits else None,
                            episodes=sum(1 for (vv, c, at, s) in P if vv == v and at == attacker))
        out['A1']['supported'] = out['A1']['share'] is not None and out['A1']['share'] <= A1_MAX
        out['A2']['supported'] = out['A2']['share'] is not None and out['A2']['share'] >= A2_MIN
        # detection
        det, conf, cost, failures = {s: [] for s in SEEDS}, {s: [] for s in SEEDS}, [], []
        for c in ids:
            per_case[v][c] = {}
            rf = R.get((v, c))
            for s in SEEDS:
                p = P.get((v, c, 'v2j', s)) or {'error': 'missing'}
                src, found, q = None, False, 1
                if p.get('reportable'):
                    src, found = f'policy_s{s}', True
                elif rf is not None and rf.get('found'):
                    src, found, q = 'refine', True, 1 + rf['queries']
                elif rf is not None and 'queries' in rf:
                    q = 1 + rf['queries']
                x = X.get((v, c, src)) if found else None
                ok = bool(found and x is not None and x['contacts'] >= CONFIRM)
                det[s].append(found)
                conf[s].append(ok)
                per_case[v][c][s] = ok
                if found:
                    cost.append(q)
                    failures.append((c, s, src, x))
        out['detected'] = dict(per_seed={s: round(float(np.mean(det[s])), 4) for s in SEEDS},
                               mean=round(float(np.mean([np.mean(det[s]) for s in SEEDS])), 4))
        out['confirmed'] = dict(per_seed={s: round(float(np.mean(conf[s])), 4) for s in SEEDS},
                                mean=round(float(np.mean([np.mean(conf[s]) for s in SEEDS])), 4),
                                ci=boot_ci([np.mean([per_case[v][c][s] for s in SEEDS]) for c in ids]))
        out['by_stage'] = dict(policy=sum(1 for f in failures if f[2].startswith('policy')),
                               refine=sum(1 for f in failures if f[2] == 'refine'))
        out['queries_per_detection'] = dict(mean=round(float(np.mean(cost)), 2) if cost else None,
                                            median=float(np.median(cost)) if cost else None)
        # A5: reruns of every reported failure (distinct (case, source))
        rr = {(v, c, src): x for (vv, c, src), x in X.items() if vv == v}
        n = sum(x['n'] for x in rr.values())
        out['A5'] = dict(failures=len(rr), reruns=n, colliding=sum(x['contacts'] for x in rr.values()),
                         rate=round(sum(x['contacts'] for x in rr.values()) / n, 4) if n else None,
                         per_failure_hist={k: sum(1 for x in rr.values() if x['contacts'] == k) for k in range(REPRO + 1)})
        out['A5']['supported'] = out['A5']['rate'] is not None and out['A5']['rate'] >= A5_MIN
        out['errors'] = errors
        # fidelity
        eps = [x for (vv, c, at, s), x in P.items() if vv == v and 'error' not in x]
        late = [x['late_max'] for x in eps if x.get('late_max') is not None]
        ages = [x['health']['traj_age_max'] for x in eps if (x.get('health') or {}).get('traj_age_max') is not None]
        out['fidelity'] = dict(episodes=len(eps), late_max_p95=round(float(np.quantile(late, .95)), 4) if late else None,
                               late_max_max=max(late) if late else None,
                               late_mean_mean=round(float(np.mean([x['late_mean'] for x in eps if x.get('late_mean') is not None])), 5) if eps else None,
                               traj_age_p95=round(float(np.quantile(ages, .95)), 3) if ages else None, traj_age_max=max(ages) if ages else None,
                               v0_abs_err_mean=round(float(np.mean([abs(x['v0_error']) for x in eps if x.get('v0_error') is not None])), 3) if eps else None,
                               decide_max_s=max((x.get('decide_max_s') or 0.) for x in eps) if eps else None,
                               restarts=sum(x.get('restarts', 0) for x in eps) + sum(x.get('restarts', 0) for (vv, c), x in R.items() if vv == v))
        # failure mode + consensus, once per distinct reported failure (a refinement failure serves several seeds)
        distinct = {}
        for c, s, src, x in failures:
            if (c, src) in distinct:
                continue
            if src.startswith('policy'):
                p = P[(v, c, 'v2j', s)]
                distinct[(c, src)] = ('actions', p['actions'], p['trace'], p['t_contact'])
            else:
                b = R[(v, c)]['best']
                distinct[(c, src)] = ('theta', b['theta'], b['aw_trace'], b['aw']['t_contact'])
        modes, cons = [], []
        for (c, src), (how, payload, trace, t_hit) in sorted(distinct.items()):
            case = _STATE['cases'][c]
            m = _mode(trace, t_hit)
            m.update(case_id=c, source=src, cc=_cc_mode(case, how, payload))
            modes.append(m)
            cons.append(consensus(('aw', src, c, 'following_H0', how, payload)))
        out['failure_mode'] = dict(
            n=len(modes), mrm_share=round(float(np.mean([m['mrm'] for m in modes])), 4) if modes else None,
            brake_delay_median=float(np.median([m['brake_delay_s'] for m in modes if m['brake_delay_s'] is not None])) if any(m['brake_delay_s'] is not None for m in modes) else None,
            min_acc_median=float(np.median([m['min_acc'] for m in modes if m['min_acc'] is not None])) if modes else None,
            cc_brake_delay_median=float(np.median([m['cc']['brake_delay_s'] for m in modes if m['cc']['brake_delay_s'] is not None])) if any(m['cc']['brake_delay_s'] is not None for m in modes) else None,
            cc_min_acc_median=float(np.median([m['cc']['min_acc'] for m in modes if m['cc']['min_acc'] is not None])) if modes else None,
            rows=modes)
        ok = [r for r in cons if 'error' not in r]
        out['consensus'] = dict(n=len(cons), errors=len(cons) - len(ok),
                                consensus_fair=round(float(np.mean([r['consensus_fair'] for r in ok])), 4) if ok else None,
                                grades={g: sum(1 for r in ok if r['grade'] == g) for g in ('avoidable', 'difficult', 'unavoidable')})
    # per case: the confirmed share over the seeds (B55 compares variants of different runs case by case)
    S['per_case_confirmed'] = {v: {str(c): float(np.mean([per_case[v][c][s] for s in SEEDS])) for c in ids} for v in VARIANTS}
    if 'aw_default' in VARIANTS and 'aw_r157' in VARIANTS:
        d = [np.mean([per_case['aw_default'][c][s] for s in SEEDS]) - np.mean([per_case['aw_r157'][c][s] for s in SEEDS]) for c in ids]
        ci = boot_ci(d)
        S['A3'] = dict(difference=round(float(np.mean(d)), 4), ci=ci, supported=ci[0] > 0)
    if 'aw_r157' in VARIANTS:
        S['A4'] = dict(rate=S['variants']['aw_r157']['confirmed']['mean'], supported=S['variants']['aw_r157']['confirmed']['mean'] >= A4_MIN)
    Path(a.summary).write_text(json.dumps(S, indent=1), encoding='utf-8')
    for v in VARIANTS:
        o = S['variants'][v]
        print(v, 'A1', o['A1'], 'A2', o['A2'])
        print('  detected', o['detected'], 'confirmed', o['confirmed'], 'stage', o['by_stage'], 'q', o['queries_per_detection'])
        print('  A5', o['A5'], 'errors', o['errors'])
        print('  fidelity', o['fidelity'])
        print('  mode', {k: x for k, x in o['failure_mode'].items() if k != 'rows'}, 'consensus', o['consensus'])
    print('A3', S.get('A3'), 'A4', S.get('A4'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('summary')
    ap.add_argument('--partition', default='development')
    ap.add_argument('--variants', default=','.join(VARIANTS), help='B55: aw_r157n,aw_r157nc')
    args = ap.parse_args()
    VARIANTS = tuple(args.variants.split(','))
    main(args)
