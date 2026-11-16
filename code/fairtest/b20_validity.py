"""B20 (G1): construct-validity audit of adversarial-testing metrics on the known-capability SUT ladder.

For every tester (adversary x metric) and dataset, per SUT: per-scenario collision rate, per-scenario
fair-failure rate (CC R157 reference) and the conditional attribution rate (fair / collisions, the
CARS-style statistic). Validity statistics per tester and metric:
  Kendall tau-b between the known capability rank and the metric (a valid failure metric is strongly
  negative), with a paired case-bootstrap 95% CI; the value on the longitudinal-RSS SUT (specificity:
  a valid fair metric is ~0); the spread across the ladder.

  python -m fairtest.b20_validity OUT.json
"""
from __future__ import annotations

import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from fairtest import b9_analysis as A
import fairtest.boundary_planner as BP
from fairtest.analyze_landscape import load
from fairtest.strict_analysis import EV, E13, J

RANK = {'following_H0': 1, 'reference_idm_H0': 1, 'following_H0.5': 2, 'following_H1': 3, 'following_aeb_H0': 4,
        'following_aeb_H0.5': 5, 'rss_sut_H0': 6}
SUT_T = {k: (k.rsplit('_H', 1)[0], float(k.rsplit('_H', 1)[1])) for k in RANK}
B14D = EV / 'b14_realism'
B17D = J / 'b17'
BOOT, SEED = 2000, 20260926


def grid_testers(rows, cases):
    """Per SUT and case: B5 collision/fair, grid collision oracle, grid fair oracle (frozen logic)."""
    by = BP.index(rows)
    have = {(r['target'], r['H']) for r in rows}
    suts = tuple(SUT_T[k] for k in RANK if SUT_T[k] in have)
    A.SUTS, BP.SUTS = suts, suts
    per = A.per_case_outcomes(rows, cases, {}, 'cc_r157', math.inf)
    plans = collections.defaultdict(list)
    for (c, p), v in by.items():
        if p.startswith('plan_'):
            plans[c].append(v)
    out = collections.defaultdict(dict)
    for sut in suts:
        k = f'{sut[0]}_H{sut[1]:g}'
        for c, m in per[k].items():
            if m is None:
                continue
            b5 = by.get((c, 'b5_braking_entry'), {})
            r = b5.get(sut)
            coll_b5 = float(bool(r and r['valid'] and r['contact']))
            coll_grid = float(any(v.get(sut) and v[sut]['valid'] and v[sut]['contact'] for v in plans[c]))
            out['B5'].setdefault(k, {})[c] = (coll_b5, m['b5'][0])
            out['grid_collision_oracle'].setdefault(k, {})[c] = (coll_grid, m['oracle'][0])
            out['grid_fair_oracle'].setdefault(k, {})[c] = (m['oracle'][0], m['oracle'][0])
    return out


def falsify_testers(path, cases_ok):
    out = collections.defaultdict(dict)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            start, k = key.split('|')
            if k not in RANK:
                continue
            b = rec['best']
            hit = bool(b and b['contact'])
            fair = hit and not rec['refs']['cc_r157']['contact']
            out[f'falsify_{start.split("_")[1]}'].setdefault(k, {})[r['case_id']] = (float(hit), float(fair))
    return out


def cont_testers(path):
    out = collections.defaultdict(dict)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        for k, ss in r['sut_search'].items():
            if k not in RANK:
                continue
            b = ss.get('best')
            fair = bool(b and b['fair'])
            out['cont_search_fair_cma'].setdefault(k, {})[r['case_id']] = (float(bool(b and b['sut_contact'])), float(fair))
    return out


def b14_testers(path):
    out = collections.defaultdict(dict)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            mode, k = key.split('|')
            if k not in RANK:
                continue
            b = rec['best']
            hit = bool(b and b['hit'])
            if mode == 'falsify_real':
                fair = hit and not rec['refs']['cc_r157']
                out['falsify_real'].setdefault(k, {})[r['case_id']] = (float(hit), float(fair))
            else:
                out['fair_real'].setdefault(k, {})[r['case_id']] = (float(hit), float(hit))
    return out


def rl_testers(path, name):
    out = collections.defaultdict(dict)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        k = r['sut']
        if k not in RANK:
            continue
        coll = bool(r['valid'] and r['contact'])
        fair = coll and not r['refs']['cc_r157']['contact']
        out[name].setdefault(k, {})[r['case_id']] = (float(coll), float(fair))
    return out


def stats(tester):
    """tester: {sut: {case: (collision, fair)}} -> per-SUT rates and validity statistics with paired bootstrap."""
    suts = [k for k in RANK if k in tester]
    common = sorted(set.intersection(*[set(tester[k]) for k in suts])) if suts else []
    if len(suts) < 3 or not common:
        return None
    arr = {k: np.array([tester[k][c] for c in common]) for k in suts}  # (n, 2)
    ranks = [RANK[k] for k in suts]

    def metrics(idx):
        res = {}
        for m, col in (('collision', 0), ('fair', 1)):
            vals = [float(arr[k][idx, col].mean()) for k in suts]
            res[m] = vals
        cond = []
        for k in suts:
            c, f = arr[k][idx, 0].sum(), arr[k][idx, 1].sum()
            cond.append(float(f / c) if c else np.nan)
        res['conditional_attribution'] = cond
        return res

    point = metrics(np.arange(len(common)))
    rng = random.Random(SEED)
    boot = collections.defaultdict(list)
    rng_boot = collections.defaultdict(list)  # dynamic range: weakest level minus the competent SUT (or strongest available)
    lo_k, hi_k = suts[0], suts[-1]
    for _ in range(BOOT):
        idx = np.array([rng.randrange(len(common)) for _ in common])
        mm = metrics(idx)
        for m in ('collision', 'fair'):
            rng_boot[m].append(mm[m][0] - mm[m][-1])
        for m, vals in mm.items():
            ok = [(r, v) for r, v in zip(ranks, vals) if not np.isnan(v)]
            if len(ok) >= 3 and len({v for _, v in ok}) > 1:
                boot[m].append(kendalltau([r for r, _ in ok], [v for _, v in ok]).statistic)
    out = dict(n_cases=len(common), suts=suts, range_between=[lo_k, hi_k])
    for m, vals in point.items():
        ok = [(r, v) for r, v in zip(ranks, vals) if not np.isnan(v)]
        tau = kendalltau([r for r, _ in ok], [v for _, v in ok]).statistic if len(ok) >= 3 and len({v for _, v in ok}) > 1 else float('nan')
        b = boot.get(m, [])
        out[m] = dict(per_sut=dict(zip(suts, vals)), kendall_tau=float(tau),
                      tau_ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if b else None,
                      rss_value=dict(zip(suts, vals)).get('rss_sut_H0'),
                      dynamic_range=float(vals[0] - vals[-1]) if m in ('collision', 'fair') else None,
                      dynamic_range_ci95=[float(np.percentile(rng_boot[m], 2.5)), float(np.percentile(rng_boot[m], 97.5))] if m in rng_boot else None,
                      spread=float(np.nanmax(vals) - np.nanmin(vals)) if not all(np.isnan(vals)) else float('nan'))
    return out


def merge(*dicts):
    out = collections.defaultdict(dict)
    for d in dicts:
        for t, per in d.items():
            for k, cs in per.items():
                out[t].setdefault(k, {}).update(cs)
    return out


def main(out_path):
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    exp_rows, _ = rows_all()
    exp_rows += load(J / 'b10v2-a' / 'landscape.jsonl')[0] + load(J / 'b10v2-b' / 'landscape.jsonl')[0]
    exp_cases = dict(_STATE['cases'])
    b9_cases = {c['case_id']: c for c in json.loads((EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json').read_text(encoding='utf-8'))['cases']}
    b9_rows = sum((load(J / s / 'landscape.jsonl')[0] for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')), [])
    ng_cases = {c['case_id']: c for c in json.loads((E13 / 'cohort' / 'case_pool.json').read_text(encoding='utf-8'))['cases']}
    ng_rows = sum((load(J / s / 'landscape.jsonl')[0] for s in ('b13-ngsim-a', 'b13-ngsim-b')), [])
    dev_ids = {c for c, v in exp_cases.items() if v['event']['partition'] == 'development'}
    sets = {
        'highD_exposed_266': merge(grid_testers(exp_rows, exp_cases), falsify_testers(E13 / 'falsify' / 'b13_falsify_highd_dev.jsonl', None),
                                   cont_testers(E13 / 'posthoc_highd' / 'exposed_cont_search_all7.jsonl'), b14_testers(B14D / 'b14_exposed_development.jsonl')),
        'highD_exposed_dev62_RL': merge(rl_testers(B17D / 'eval_collide_commit_s1_dev.jsonl', 'RL_collide_commit'),
                                        rl_testers(B17D / 'eval_collide_s1_dev.jsonl', 'RL_collide'),
                                        rl_testers(B17D / 'eval_fair_s1_dev.jsonl', 'RL_fair_naive'),
                                        rl_testers(B17D / 'eval_fair_guarded2_s1_dev.jsonl', 'RL_fair_guarded')),
        'highD_B9_148_posthoc': merge(grid_testers(b9_rows, b9_cases), falsify_testers(E13 / 'posthoc_highd' / 'b9_falsify_POSTHOC.jsonl', None),
                                      cont_testers(E13 / 'posthoc_highd' / 'b9_cont_search_all7_POSTHOC.jsonl'), b14_testers(B14D / 'b14_b9_POSTHOC.jsonl'),
                                      rl_testers(B17D / 'eval_collide_commit_s1_b9.jsonl', 'RL_collide_commit'),
                                      rl_testers(B17D / 'eval_fair_guarded2_s1_b9.jsonl', 'RL_fair_guarded')),
        'NGSIM_47': merge(grid_testers(ng_rows, ng_cases), falsify_testers(E13 / 'falsify' / 'b13_falsify.jsonl', None),
                          cont_testers(E13 / 'continuous' / 'b13_cont_search.jsonl'), b14_testers(B14D / 'b14_ngsim_POSTHOC.jsonl')),
    }
    rep = {}
    for name, testers in sets.items():
        rep[name] = {}
        print('==', name)
        for t, per in sorted(testers.items()):
            st = stats(per)
            if st is None:
                continue
            rep[name][t] = st
            c, f, a = st['collision'], st['fair'], st['conditional_attribution']
            fmt = lambda x: 'nan' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:+.2f}'
            rv = lambda x: '  n/a' if x is None else f'{100*x:5.1f}'
            dr = lambda d: f"{100*d['dynamic_range']:5.1f} [{100*d['dynamic_range_ci95'][0]:5.1f},{100*d['dynamic_range_ci95'][1]:5.1f}]"
            print(f"  {t:24} n={st['n_cases']:3d} {st['range_between'][0][:10]}->{st['range_between'][1][:10]} | collision range {dr(c)} RSS {rv(c['rss_value'])} tau {fmt(c['kendall_tau'])}"
                  f" | fair range {dr(f)} RSS {rv(f['rss_value'])} tau {fmt(f['kendall_tau'])}")
    Path(out_path).write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
