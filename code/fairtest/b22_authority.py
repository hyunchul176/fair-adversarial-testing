"""B22 (G9): is the NGSIM non-replication explained by the attacker's deceleration authority?

highD and NGSIM do not overlap in deceleration headroom above the generator's 15 m/s floor (highD
exposed: no case below 5 m/s; NGSIM: median 1.8 m/s), so stratification cannot separate "regime" from
"authority". Intervention instead: re-run the highD grid with the floor raised to max(15, v0 - m)
(env FAIRTEST_FLOOR_MARGIN = m) and compare fair-failure rates with NGSIM's (median headroom 1.8 m/s).
Two variants of the raised floor:
  b22r  relative headroom, saturation (primary): floor = max(15, SUT initial speed - r); the actuator releases
        braking before the floor and the plan stays valid, exactly like the protocol's own 15 m/s floor;
  b22   absolute headroom, invalidation (first run): floor = max(15, generator speed - m) and a plan that
        would cross it is invalid. Stricter than NGSIM in highD, where generators start faster than the SUT.

  python -m fairtest.b22_authority OUT.json [--variant b22r|b22]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from fairtest import b16_analysis as B
from fairtest import b9_analysis as A
import fairtest.boundary_planner as BP
from fairtest.b20_validity import RANK
from fairtest.strict_analysis import EV, E13, J, SUT6, SUT7

B22 = J / 'b22'
SUT_KEYS = ('following_H0', 'reference_idm_H0', 'following_H0.5', 'following_H1', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')


SUT_FIELDS = ('case_id', 'policy', 'target', 'H', 'valid', 'contact', 'ttc_event', 'initial_event', 'outcome')


def slim(r):
    return {k: r[k] for k in SUT_FIELDS}


def stream(paths, keep=lambda t: True):
    """Rows of the landscapes; the CC rows keep their features (realism, demand), SUT rows only outcomes."""
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if 'error' not in r and keep(r['target']):
                        yield r if r['target'] == 'cc_r157' else slim(r)


def attack_space(rows):
    """Mean share of the 199 grid plans whose CC run is a valid cut-in (the attack space the grid can use)."""
    per = {}
    for r in rows:
        if r['target'] == 'cc_r157' and r['policy'].get('name', '').startswith('plan_'):
            per.setdefault(r['case_id'], []).append(bool(r['valid'] and r['features'].get('cut_in')))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else None


BINS = (0., 1., 2., 3., 5., math.inf)


def margins(cases, m, kind):
    """Headroom per case: 'abs' = generator speed - 15, 'rel' = SUT speed - 15 (how much slower than the SUT
    the attacker may become), each capped at the intervention value m."""
    out = {}
    for c, case in cases.items():
        g, t = case['physical_initial_state'][:2]
        v = math.hypot(g['vx'], g['vy']) if kind == 'abs' else math.hypot(t['vx'], t['vy'])
        out[c] = min(m, v - 15.)
    return out


def by_margin(rows, cases, suts, m, kind):
    """Grid-oracle fair-failure rate of each SUT per headroom bin."""
    A.SUTS, BP.SUTS = suts, suts
    per = A.per_case_outcomes(rows, cases, {}, 'cc_r157', math.inf)
    mg = margins(cases, m, kind)
    out = {}
    for k, v in per.items():
        out[k] = {}
        for lo, hi in zip(BINS, BINS[1:]):
            ids = [c for c, x in v.items() if x is not None and lo <= mg[c] < hi]
            out[k][f'{lo:g}-{hi:g}'] = dict(n=len(ids), fair=float(np.mean([v[c]['oracle'][0] for c in ids])) if ids else None)
    return out


def summary(grid):
    g = grid['cc_r157']
    ks = [k for k in SUT_KEYS if k in g]
    out = {}
    for m in ('oracle', 'boundary', 'b5_fair', 'b5_unfair'):
        vals = [g[k][m] for k in ks]
        out[m] = dict(tau=float(kendalltau([RANK[k] for k in ks], vals).statistic) if m != 'b5_unfair' else None,
                      spread=float(max(vals) - min(vals)))
    return out


def main(out, variant='b22r'):
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    exposed_cases = dict(_STATE['cases'])
    b9_cases = {c['case_id']: c for c in json.loads((EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json').read_text(encoding='utf-8'))['cases']}
    ng_cases = {c['case_id']: c for c in json.loads((E13 / 'cohort' / 'case_pool.json').read_text(encoding='utf-8'))['cases']}
    def base_exposed():
        rows, _ = rows_all()
        rows = [r if r['target'] == 'cc_r157' else slim(r) for r in rows if r['target'] in ('cc_r157', 'following', 'reference_idm', 'following_aeb')]
        return rows + list(stream([J / 'b10v2-a' / 'landscape.jsonl', J / 'b10v2-b' / 'landscape.jsonl'], lambda t: t == 'rss_sut'))
    if variant == 'b22r':  # primary: relative headroom, actuator saturation
        kind, runs = 'rel', dict(
            highd_exposed_unrestricted=(base_exposed, exposed_cases, SUT7, math.inf),
            highd_exposed_rel3_0=(lambda: list(stream([B22 / 'b22r_exposed_r3.0.jsonl'])), exposed_cases, SUT7, 3.),
            highd_exposed_rel1_6=(lambda: list(stream([B22 / 'b22r_exposed_r1.6.jsonl'])), exposed_cases, SUT7, 1.6),
            highd_b9_unrestricted=(lambda: list(stream([J / s / 'landscape.jsonl' for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')])), b9_cases, SUT6, math.inf),
            highd_b9_rel1_6=(lambda: list(stream([B22 / 'b22r_b9_r1.6.jsonl'])), b9_cases, SUT7, 1.6),
            ngsim=(lambda: list(stream([J / 'b13-ngsim-a' / 'landscape.jsonl', J / 'b13-ngsim-b' / 'landscape.jsonl'])), ng_cases, SUT7, math.inf))
    else:  # first run: absolute headroom, plans crossing the raised floor invalid
        kind, runs = 'abs', dict(
            highd_exposed_unrestricted=(base_exposed, exposed_cases, SUT7, math.inf),
            highd_exposed_margin4=(lambda: list(stream([B22 / 'b22_exposed_m4.0.jsonl'])), exposed_cases, SUT7, 4.),
            highd_exposed_margin1_8=(lambda: list(stream([B22 / 'b22_exposed_m1.8.jsonl'])), exposed_cases, SUT7, 1.8),
            highd_b9_unrestricted=(lambda: list(stream([J / s / 'landscape.jsonl' for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')])), b9_cases, SUT6, math.inf),
            highd_b9_margin1_8=(lambda: list(stream([B22 / 'b22_b9_m1.8.jsonl'])), b9_cases, SUT7, 1.8),
            ngsim=(lambda: list(stream([J / 'b13-ngsim-a' / 'landscape.jsonl', J / 'b13-ngsim-b' / 'landscape.jsonl'])), ng_cases, SUT7, math.inf))
    rep = {}
    for name, (load_rows, cases, suts, m) in runs.items():
        rows = load_rows()
        grid = B.grid(rows, cases, {}, suts, ('cc_r157',))
        rep[name] = dict(n_cases=len(cases), attack_space=attack_space(rows), grid=grid['cc_r157'], validity=summary(grid),
                         by_margin=by_margin(rows, cases, suts, m, kind))
        g = grid['cc_r157']
        print(f"== {name:28} n={len(cases)} valid cut-in plans {100*(rep[name]['attack_space'] or 0):5.1f}%  "
              f"oracle tau {rep[name]['validity']['oracle']['tau']:+.2f} spread {100*rep[name]['validity']['oracle']['spread']:5.1f}")
        print('   oracle', {k: round(100 * g[k]['oracle'], 1) for k in SUT_KEYS if k in g})
        print('   B5 fair/unfair', {k: (round(100 * g[k]['b5_fair'], 1), round(100 * g[k]['b5_unfair'], 1)) for k in SUT_KEYS if k in g})
        print('   boundary', {k: round(100 * g[k]['boundary'], 1) for k in SUT_KEYS if k in g})
        bm = rep[name]['by_margin'].get('following_H0', {})
        print(f'   following H0 by {kind} headroom', {b: (x['n'], None if x['fair'] is None else round(100 * x['fair'], 1)) for b, x in bm.items() if x['n']})
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == '--variant' else 'b22r')
