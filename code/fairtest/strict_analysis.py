"""Post-B13 reference sensitivity: fair-failure rates judged by cc_r157 vs cc_r157_strict (perception release).

Grid methods are recomputed with the frozen B9 per-case logic using the strict CC rows as the
reference; continuous-search and falsification plans are re-judged from strict_reeval output.

  python -m fairtest.strict_analysis OUT.json
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from fairtest import b9_analysis as A
import fairtest.boundary_planner as BP
from fairtest.analyze_landscape import load

EV = Path(os.environ.get('FAIRTEST_EVIDENCE', 'evidence'))  # local evidence folder of the study (not distributed)
J = EV / 'fairtest_jobs'
E13 = EV / 'b13_execution'
SUT6 = (('following', 0.), ('following', .5), ('following', 1.), ('reference_idm', 0.), ('following_aeb', 0.),
        ('following_aeb', .5))
SUT7 = SUT6 + (('rss_sut', 0.),)
GRID_METHODS = ('b5', 'boundary', 'guided_5', 'learned_1', 'oracle')


def rows_of(paths):
    rows = []
    for p in paths:
        r, e = load(p)
        if e:
            raise RuntimeError(f'{len(e)} error rows in {p}')
        rows += r
    return rows


def grid(rows, cases, models, suts, ref):
    A.SUTS = suts
    BP.SUTS = suts
    out = A.per_case_outcomes(rows, cases, models, ref, math.inf)
    res = {}
    for s, per in out.items():
        vals = [m for m in per.values() if m is not None]
        res[s] = dict(n=len(vals), b5_unfair=float(np.mean([m['b5'][1] for m in vals])),
                      **{k: float(np.mean([m[k][0] for m in vals])) for k in GRID_METHODS})
    return res


def continuous(cont, falsify, reeval, n_cases):
    strict = {(r['source'], r['case_id'], r['key']): r for r in (json.loads(l) for l in open(reeval, encoding='utf-8'))}
    res = collections.defaultdict(dict)
    for line in open(cont, encoding='utf-8'):
        r = json.loads(line)
        for sut, ss in r['sut_search'].items():
            b = ss.get('best')
            c = res[sut].setdefault('cont', collections.Counter())
            if b and b['fair']:
                c['fair_cc'] += 1
                c['fair_strict'] += int(not strict[('cont_search', r['case_id'], sut)]['strict_contact'])
    for line in open(falsify, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            start, sut = key.split('|')
            if start != 'falsify_centre':
                continue
            c = res[sut].setdefault('falsify', collections.Counter())
            b = rec.get('best')
            if b and b['contact']:
                c['collisions'] += 1
                c['fair_cc'] += int(not rec['refs']['cc_r157']['contact'])
                c['fair_strict'] += int(not strict[('falsify', r['case_id'], key)]['strict_contact'])
    return {s: {k: {m: v / n_cases for m, v in c.items()} for k, c in d.items()} for s, d in res.items()}


def main(out):
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    rep = {}
    exposed_rows, _ = rows_all()
    exposed_rows += rows_of([J / 'b10v2-a' / 'landscape.jsonl', J / 'b10v2-b' / 'landscape.jsonl', J / 'strict' / 'strict_exposed.jsonl'])
    exposed_cases = dict(_STATE['cases'])
    b9_pool = EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json'
    b9_cases = {c['case_id']: c for c in json.loads(b9_pool.read_text(encoding='utf-8'))['cases']}
    b9_rows = rows_of([J / s / 'landscape.jsonl' for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')] + [J / 'strict' / 'strict_b9.jsonl'])
    ng_pool = E13 / 'cohort' / 'case_pool.json'
    ng_cases = {c['case_id']: c for c in json.loads(ng_pool.read_text(encoding='utf-8'))['cases']}
    ng_rows = rows_of([J / 'b13-ngsim-a' / 'landscape.jsonl', J / 'b13-ngsim-b' / 'landscape.jsonl', J / 'strict' / 'strict_ngsim.jsonl'])
    sets = dict(
        highd_exposed_development=(exposed_rows, exposed_cases, {}, SUT7,
                                   E13 / 'posthoc_highd' / 'exposed_cont_search_all7.jsonl', E13 / 'falsify' / 'b13_falsify_highd_dev.jsonl',
                                   J / 'strict' / 'strict_reeval_exposed.jsonl'),
        highd_b9_posthoc=(b9_rows, b9_cases, A.load_models(EV / 'b9_design' / 'selectors' / 'models.pkl'), SUT6,
                          E13 / 'posthoc_highd' / 'b9_cont_search_all7_POSTHOC.jsonl', E13 / 'posthoc_highd' / 'b9_falsify_POSTHOC.jsonl',
                          J / 'strict' / 'strict_reeval_b9.jsonl'),
        ngsim_b13=(ng_rows, ng_cases, A.load_models(EV / 'b13_design' / 'selectors' / 'models.pkl'), SUT7,
                   E13 / 'continuous' / 'b13_cont_search.jsonl', E13 / 'falsify' / 'b13_falsify.jsonl', J / 'strict' / 'strict_reeval_ngsim.jsonl'))
    for name, (rows, cases, models, suts, cont, fal, ree) in sets.items():
        rep[name] = dict(grid_cc=grid(rows, cases, models, suts, 'cc_r157'), grid_strict=grid(rows, cases, models, suts, 'cc_r157_strict'),
                         continuous=continuous(cont, fal, ree, len(cases)), n_cases=len(cases))
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for name, r in rep.items():
        print('==', name, r['n_cases'])
        for s in r['grid_cc']:
            g0, g1 = r['grid_cc'][s], r['grid_strict'][s]
            c = r['continuous'].get(s, {})
            cont, fal = c.get('cont', {}), c.get('falsify', {})
            print(f"  {s:20} oracle {100*g0['oracle']:5.1f}->{100*g1['oracle']:5.1f}  boundary {100*g0['boundary']:5.1f}->{100*g1['boundary']:5.1f}"
                  f"  b5 fair {100*g0['b5']:4.1f}->{100*g1['b5']:4.1f} unfair {100*g0['b5_unfair']:4.1f}->{100*g1['b5_unfair']:4.1f}"
                  f"  cont {100*cont.get('fair_cc', 0):5.1f}->{100*cont.get('fair_strict', 0):5.1f}"
                  f"  falsify coll {100*fal.get('collisions', 0):5.1f} fair {100*fal.get('fair_cc', 0):4.1f}->{100*fal.get('fair_strict', 0):4.1f}")


if __name__ == '__main__':
    main(sys.argv[1])
