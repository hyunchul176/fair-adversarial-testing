"""Frozen B13 analysis: external prospective evaluation (NGSIM US-101; exiD later).

Inputs: landscape rows for the external cohort (200 policies x SUT panel x reference drivers),
continuous-search results (b11_search format) for the same cohort, the cohort case pool and the
frozen B13 selectors (trained on all highD cases).
Methods: B9 set (b5, fixed2s, random, boundary, guided_3/5, learned_1/3/5, oracle) plus
  cont_search  CMA-ES per SUT (budget 36 SUT queries), fair failure found
  hybrid       learned_1 on the grid, else cont_search (queries 1 + cont queries)
Primary reference CC R157 0.75 s; sensitivity: ref_fsm, ref_rss, cc 0.5/1.0 s; consensus.
Uncertainty: event-level bootstrap (10,000, percentile, seed 20260926); per-period table.
"""
from __future__ import annotations

import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from fairtest import b9_analysis as A
import fairtest.boundary_planner as BP

SUTS = (('following', 0.), ('following', .5), ('following', 1.), ('reference_idm', 0.),
        ('following_aeb', 0.), ('following_aeb', .5), ('rss_sut', 0.))
REFS = ('cc_r157', 'ref_fsm', 'ref_rss', 'cc_r157_react050', 'cc_r157_react100')
METHODS = ('b5', 'fixed2s', 'random', 'boundary', 'guided_3', 'guided_5', 'learned_1', 'learned_3', 'learned_5',
           'oracle', 'cont_search', 'hybrid')
BOOT, SEED = 10000, 20260926


def add_continuous(outcomes, b11, reference='cc_r157'):
    """Attach cont_search / hybrid to per-case outcomes (primary reference only)."""
    for sut, per in outcomes.items():
        for c, m in per.items():
            if m is None:
                continue
            r = b11.get(c)
            ss = (r or {}).get('sut_search', {}).get(sut)
            found = bool(ss and ss.get('best') and ss['best']['fair']) if reference == 'cc_r157' else False
            q = ss['sut_queries'] if ss else 0
            m['cont_search'] = (float(found), 0.)
            hit_l = m['learned_1'][0] > 0
            m['hybrid'] = (float(hit_l or found), 0.)
            m['_hybrid_queries'] = 1 if hit_l else 1 + q
    return outcomes


def event_bootstrap(per_case, methods, contrasts):
    rng = random.Random(SEED)
    ids = sorted(c for c, m in per_case.items() if m is not None)
    vals = {meth: np.array([per_case[c][meth][0] if per_case[c].get(meth) else np.nan for c in ids]) for meth in methods}
    unf = {meth: np.array([per_case[c][meth][1] if per_case[c].get(meth) else np.nan for c in ids]) for meth in methods}
    n = len(ids)
    point = {meth: dict(fair=float(np.nanmean(vals[meth])) if n else float('nan'),
                        unfair=float(np.nanmean(unf[meth])) if n else float('nan'), n=n) for meth in methods}
    if n == 0:
        return dict(n=0, methods=point, contrasts={})
    idx = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(BOOT)])
    for meth in methods:
        d = np.nanmean(vals[meth][idx], axis=1)
        point[meth]['fair_ci95'] = [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    cons = {}
    for a, b in contrasts:
        d = np.nanmean(vals[a][idx] - vals[b][idx], axis=1)
        cons[f'{a}-{b}'] = dict(estimate=point[a]['fair'] - point[b]['fair'],
                                ci95=[float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))])
    hq = [per_case[c].get('_hybrid_queries') for c in ids if per_case[c].get('_hybrid_queries') is not None]
    return dict(n=n, methods=point, contrasts=cons, hybrid_mean_queries=float(np.mean(hq)) if hq else None)


def consensus(rows, cases):
    by = BP.index(rows)
    out = {}
    for sut in SUTS:
        n = k = 0
        for c in cases:
            nom = by.get((c, BP.NOMINAL), {}).get(sut)
            if nom is None or nom['contact'] or nom['ttc_event'] or nom['initial_event']:
                continue
            n += 1
            for (cc, p), v in by.items():
                if cc != c or not p.startswith('plan_') or sut not in v:
                    continue
                refs = [v.get((r, 0.)) for r in ('cc_r157', 'ref_fsm', 'ref_rss')]
                if all(x is not None for x in refs) and v[sut]['valid'] and v[sut]['contact'] and not any(x['contact'] for x in refs):
                    k += 1
                    break
        out[f'{sut[0]}_H{sut[1]:g}'] = dict(n=n, oracle_consensus_fair=k / n if n else float('nan'))
    return out


def run(rows, cases, models, b11):
    A.SUTS = SUTS
    BP.SUTS = SUTS
    rep = dict(primary={}, sensitivity={}, per_period={})
    contrasts = (('boundary', 'b5'), ('boundary', 'fixed2s'), ('learned_1', 'boundary'), ('hybrid', 'oracle'))
    for budget in (math.inf, 1.):
        key = 'no_limit' if math.isinf(budget) else 'max_exceedance_1'
        out = A.per_case_outcomes(rows, cases, models, 'cc_r157', budget)
        if key == 'no_limit':
            out = add_continuous(out, b11)
        rep['primary'][key] = {s: event_bootstrap(o, METHODS if key == 'no_limit' else METHODS[:10], contrasts if key == 'no_limit' else ())
                               for s, o in out.items()}
        if key == 'no_limit':
            for s, o in out.items():
                per = collections.defaultdict(lambda: collections.Counter())
                for c, m in o.items():
                    if m is None:
                        continue
                    rec = cases[c]['event']['recording']
                    per[rec]['n'] += 1
                    for meth in ('b5', 'boundary', 'learned_1', 'oracle', 'hybrid'):
                        per[rec][meth] += m[meth][0]
                    per[rec]['b5_unfair'] += m['b5'][1]
                rep['per_period'][s] = {str(k): dict(v) for k, v in sorted(per.items())}
    for ref in REFS[1:]:
        out = A.per_case_outcomes(rows, cases, models, ref, math.inf)
        rep['sensitivity'][ref] = {s: event_bootstrap(o, ('b5', 'boundary', 'guided_5', 'learned_1', 'oracle'), ())
                                   for s, o in out.items()}
    rep['consensus'] = consensus(rows, cases)
    rep['fragility'] = A.fragility(rows)
    rep['hypotheses'] = hypotheses(rep)
    return rep


def hypotheses(rep):
    p = rep['primary']['no_limit']
    weak = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
    v = {}
    v['H1_boundary_beats_b5_and_fixed2s'] = {k: p[k]['contrasts']['boundary-b5']['ci95'][0] > 0 and
                                            p[k]['contrasts']['boundary-fixed2s']['ci95'][0] > 0 for k in weak if p[k]['n']}
    v['H2_b5_unfair_exceeds_fair'] = {k: p[k]['methods']['b5']['unfair'] > p[k]['methods']['b5']['fair'] for k in p if p[k]['n']}
    v['H3_guided5_reaches_90pct_oracle'] = {k: p[k]['methods']['guided_5']['fair'] >= .9 * p[k]['methods']['oracle']['fair']
                                            for k in p if p[k]['n'] and p[k]['methods']['oracle']['fair'] >= .10}
    v['H4_learned1_at_least_boundary'] = {k: p[k]['contrasts']['learned_1-boundary']['estimate'] >= 0
                                          for k in ('following_H0.5', 'following_H1') if p[k]['n']}
    o = {k: p[k]['methods']['oracle']['fair'] for k in p}
    v['H5_capability_order'] = (o['following_H0'] >= o['following_H0.5'] >= o['following_H1'] > o['following_aeb_H0']
                                >= o['following_aeb_H0.5'])
    v['H7_hybrid_at_least_oracle'] = {k: p[k]['methods']['hybrid']['fair'] >= p[k]['methods']['oracle']['fair']
                                      for k in p if p[k]['n'] and p[k]['methods']['oracle']['fair'] >= .10}
    r = p.get('rss_sut_H0')
    if r and r['n']:
        v['H8_rss_sut_no_fair_failures'] = all(r['methods'][m]['fair'] == 0 for m in ('boundary', 'guided_5', 'learned_1', 'oracle', 'hybrid'))
        v['H8b_b5_blames_rss_sut'] = r['methods']['b5']['unfair'] > 0
    v['H9_b5_unfair_under_all_references'] = {ref: rep['sensitivity'][ref]['following_H0']['methods']['b5']['unfair'] >
                                              rep['sensitivity'][ref]['following_H0']['methods']['b5']['fair']
                                              for ref in ('ref_fsm', 'ref_rss') if ref in rep['sensitivity']}
    v['H9_b5_unfair_under_all_references']['cc_r157'] = v['H2_b5_unfair_exceeds_fair'].get('following_H0')
    return v


if __name__ == '__main__':
    # python -m fairtest.b13_analysis ROWS1,ROWS2,... B11.jsonl CASE_POOL.json MODELS.pkl OUT.json
    from fairtest.analyze_landscape import load
    rows = []
    for p in sys.argv[1].split(','):
        r, e = load(p)
        if e:
            raise SystemExit(f'{len(e)} error rows in {p}')
        rows += r
    b11 = {json.loads(l)['case_id']: json.loads(l) for l in open(sys.argv[2], encoding='utf-8')}
    pool = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
    cases = {c['case_id']: c for c in pool['cases']}
    rep = run(rows, cases, A.load_models(sys.argv[4]), b11)
    Path(sys.argv[5]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep['hypotheses'], indent=1))
