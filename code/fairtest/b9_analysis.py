"""Frozen B9 analysis (pre-specified). Input: landscape rows for the reserved cohort + its case pool.

Methods (per case, per SUT; candidate space = frozen 199-plan grid):
  b5          frozen B5 braking_entry policy (its own run against the SUT)
  fixed2s     plan_e2_pre-3_post-3_dgrid
  random      uniformly random CC-preventable candidate (exact expectation)
  boundary    R157-boundary planner: max CC demand (<=1.0) among CC-preventable candidates; 0 SUT queries
  guided_K    first K boundary-ranked candidates queried on the SUT (K=3,5)
  learned_K   frozen per-SUT selector ranking (K=1 is 0 SUT queries; K=3,5 queried); falls back to
              boundary ranking where no model was frozen
  oracle      any CC-preventable candidate fails the SUT (upper bound over the grid)
Fair failure: method's SUT run is a valid contact AND the reference CC run of the same plan has no
contact. Attacker-at-fault: valid SUT contact AND reference CC contact. Primary reference: cc_r157
(0.75 s); sensitivity references: cc_r157_react050, cc_r157_react100.
Realism budgets on candidates: none, max exceedance <= 1.0, <= 0.5 (boundary/guided/learned/oracle).
Uncertainty: recording-cluster bootstrap, 10,000 resamples, percentile 95% CI, seed 20260925.
"""
from __future__ import annotations

import collections
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np

from fairtest.b8_selector import feature_row
from fairtest.boundary_planner import NOMINAL, SUTS, cc_demand, index
from fairtest.features import Realism

REFERENCES = ('cc_r157', 'cc_r157_react050', 'cc_r157_react100')
FIXED2S = 'plan_e2_pre-3_post-3_dgrid'
B5 = 'b5_braking_entry'
BUDGETS = (math.inf, 1., .5)
BOOT, SEED = 10000, 20260925
DEMAND_BINS = (0, .2, .3, .4, .5, .6, .8, 1.5)


def sut_name(s):
    return f'{s[0]}_H{s[1]:g}'


def per_case_outcomes(rows, cases, models, reference='cc_r157', budget=math.inf):
    """{sut: {case_id: {method: (fair, unfair)}}} plus eligibility; exact random expectation."""
    ref = (reference, 0.)
    realism = Realism()
    by = index(rows)
    per_case = collections.defaultdict(list)
    for (c, p), v in by.items():
        per_case[c].append((p, v))
    out = {}
    for sut in SUTS:
        name = sut_name(sut)
        res = {}
        for c in sorted(cases):
            v_nom = by.get((c, NOMINAL), {})
            nom = v_nom.get(sut)
            if nom is None:
                continue
            if nom['contact'] or nom['ttc_event'] or nom['initial_event']:
                res[c] = None  # excluded: nominal already risky for this SUT
                continue
            m = {}
            for key, pol in (('b5', B5), ('fixed2s', FIXED2S)):
                v = by.get((c, pol), {})
                r, cc = v.get(sut), v.get(ref)
                if r is None or cc is None:
                    m[key] = None
                    continue
                contact = r['valid'] and r['contact']
                m[key] = (float(contact and not cc['contact']), float(contact and cc['contact']))
            cand = []
            for p, v in per_case[c]:
                if not p.startswith('plan_') or p == NOMINAL or ref not in v or sut not in v:
                    continue
                cc = v[ref]
                if not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
                    continue
                sc = realism.score(cc['features'])['max_exceedance']
                sc = math.inf if sc is None else sc
                if sc > budget:
                    continue
                hit = bool(v[sut]['valid'] and v[sut]['contact'])
                cand.append(dict(plan=p, hit=hit, demand=cc_demand(cc), exc=sc,
                                 x=feature_row(cases[c], p, cc, min(sc, 20.))))
            hits = [x['hit'] for x in cand]
            n, h = len(hits), sum(hits)
            m['random'] = (h / n if n else 0., 0.)
            m['oracle'] = (float(h > 0), 0.)
            feasible = [x for x in cand if x['demand'] <= 1. + 1e-9] or cand
            bnd = sorted(feasible, key=lambda x: (-x['demand'], x['exc']))
            rest = [x for x in sorted(cand, key=lambda x: (-x['demand'], x['exc'])) if x not in bnd]
            order_b = bnd + rest
            m['boundary'] = (float(bool(order_b) and order_b[0]['hit']), 0.)
            for k in (3, 5):
                m[f'guided_{k}'] = (float(any(x['hit'] for x in order_b[:k])), 0.)
            model = models.get(name)
            if model is not None and cand:
                prob = model.predict_proba(np.array([x['x'] for x in cand], float))[:, 1]
                order_l = [cand[i] for i in np.argsort(-prob, kind='stable')]
            else:
                order_l = order_b
            for k in (1, 3, 5):
                m[f'learned_{k}'] = (float(any(x['hit'] for x in order_l[:k])), 0.)
            res[c] = m
        out[name] = res
    return out


def bootstrap(per_case, recordings, methods, contrasts):
    rng = random.Random(SEED)
    recs = sorted({recordings[c] for c in per_case if per_case[c] is not None})
    agg = {r: collections.defaultdict(lambda: [0., 0., 0]) for r in recs}  # method -> [fair, unfair, n]
    for c, m in per_case.items():
        if m is None:
            continue
        for meth in methods:
            if m.get(meth) is None:
                continue
            a = agg[recordings[c]][meth]
            a[0] += m[meth][0]
            a[1] += m[meth][1]
            a[2] += 1

    def rate(sample, meth, idx):
        num = sum(agg[r][meth][idx] for r in sample)
        den = sum(agg[r][meth][2] for r in sample)
        return num / den if den else float('nan')

    point = {meth: dict(fair=rate(recs, meth, 0), unfair=rate(recs, meth, 1),
                        n=int(sum(agg[r][meth][2] for r in recs))) for meth in methods}
    draws = {meth: [] for meth in methods}
    cdraws = {f'{a}-{b}': [] for a, b in contrasts}
    for _ in range(BOOT):
        sample = [rng.choice(recs) for _ in recs]
        for meth in methods:
            draws[meth].append(rate(sample, meth, 0))
        for a, b in contrasts:
            cdraws[f'{a}-{b}'].append(rate(sample, a, 0) - rate(sample, b, 0))
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    for meth in methods:
        point[meth]['fair_ci95'] = ci(draws[meth])
    diffs = {}
    for a, b in contrasts:
        key = f'{a}-{b}'
        diffs[key] = dict(estimate=point[a]['fair'] - point[b]['fair'], ci95=ci(cdraws[key]))
    return dict(recordings=len(recs), methods=point, contrasts=diffs)


def fragility(rows, reference='cc_r157'):
    ref = (reference, 0.)
    by = index(rows)
    out = {}
    for sut in SUTS:
        tally = collections.defaultdict(lambda: [0, 0])
        for (c, p), v in by.items():
            if not p.startswith('plan_') or p == NOMINAL or ref not in v or sut not in v:
                continue
            cc = v[ref]
            if not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
                continue
            d = cc_demand(cc)
            b = next((f'{lo:g}-{hi:g}' for lo, hi in zip(DEMAND_BINS, DEMAND_BINS[1:]) if lo <= d < hi), None)
            if b:
                tally[b][0] += 1
                tally[b][1] += bool(v[sut]['valid'] and v[sut]['contact'])
        out[sut_name(sut)] = {b: dict(plans=n, sut_failure_rate=k / n) for b, (n, k) in sorted(tally.items())}
    return out


METHODS = ('b5', 'fixed2s', 'random', 'boundary', 'guided_3', 'guided_5', 'learned_1', 'learned_3', 'learned_5', 'oracle')
CONTRASTS = (('boundary', 'b5'), ('boundary', 'fixed2s'), ('learned_1', 'boundary'), ('guided_5', 'boundary'))


def monotone(curve, min_plans=30):
    rates = [v['sut_failure_rate'] for b, v in sorted(curve.items(), key=lambda x: float(x[0].split('-')[0]))
             if v['plans'] >= min_plans]
    return all(b >= a - 1e-12 for a, b in zip(rates, rates[1:]))


def hypotheses(primary, frag=None):
    s = primary['no_limit']
    v = {}
    weak = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
    v['H1_boundary_beats_b5_and_fixed2s'] = {k: (s[k]['contrasts']['boundary-b5']['ci95'][0] > 0 and
                                                s[k]['contrasts']['boundary-fixed2s']['ci95'][0] > 0) for k in weak}
    v['H2_b5_unfair_exceeds_fair'] = {k: s[k]['methods']['b5']['unfair'] > s[k]['methods']['b5']['fair'] for k in s}
    v['H3_guided5_reaches_90pct_oracle'] = {k: (s[k]['methods']['guided_5']['fair'] >= .9 * s[k]['methods']['oracle']['fair'])
                                            for k in s if s[k]['methods']['oracle']['fair'] >= .10}
    v['H4_learned1_at_least_boundary'] = {k: s[k]['contrasts']['learned_1-boundary']['estimate'] >= 0
                                          for k in ('following_H0.5', 'following_H1')}
    o = {k: s[k]['methods']['oracle']['fair'] for k in s}
    v['H5_capability_order'] = (o['following_H0'] >= o['following_H0.5'] >= o['following_H1'] > o['following_aeb_H0']
                                >= o['following_aeb_H0.5'])
    if frag is not None:
        v['H6_fragility_nondecreasing'] = {k: monotone(frag[k]) for k in ('following_H0', 'following_H1') if k in frag}
    return v


def run(rows, cases, models):
    recordings = {c: cases[c]['event']['recording'] for c in cases}
    report = dict(primary={}, sensitivity={})
    for budget in BUDGETS:
        key = 'no_limit' if math.isinf(budget) else f'max_exceedance_{budget:g}'
        outcomes = per_case_outcomes(rows, cases, models, 'cc_r157', budget)
        report['primary'][key] = {s: dict(bootstrap(o, recordings, METHODS, CONTRASTS),
                                          excluded_nominal=sum(1 for m in o.values() if m is None))
                                  for s, o in outcomes.items()}
    for ref in REFERENCES[1:]:
        outcomes = per_case_outcomes(rows, cases, models, ref, math.inf)
        if any(outcomes.values()):
            report['sensitivity'][ref] = {s: bootstrap(o, recordings, ('b5', 'boundary', 'learned_1', 'oracle'), ())
                                          for s, o in outcomes.items()}
    report['fragility'] = fragility(rows)
    report['hypotheses'] = hypotheses(report['primary'], report['fragility'])
    return report


def load_models(path):
    return pickle.loads(Path(path).read_bytes())


if __name__ == '__main__':
    # usage: python -m fairtest.b9_analysis ROWS.jsonl[,ROWS2...] CASE_POOL.json MODELS.pkl OUT.json
    from fairtest.analyze_landscape import load
    rows = []
    for p in sys.argv[1].split(','):
        r, e = load(p)
        if e:
            raise SystemExit(f'{len(e)} error rows in {p}')
        rows += r
    pool = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    cases = {c['case_id']: c for c in pool['cases']}
    rep = run(rows, cases, load_models(sys.argv[3]))
    Path(sys.argv[4]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep['hypotheses'], indent=1))
