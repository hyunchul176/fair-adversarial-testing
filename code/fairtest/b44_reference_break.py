"""B44 (design fixed before the run): can a human-authority attacker make the reference driver itself collide?

Per development case of B30 (deceleration) or B34 (cut-out) and per reference driver: a fixed grid over the plan box
and a GP-BO (budget 72, warm start at the case's stored CC boundary plan) minimising the reference's min gap (contact =
success), both within the braking authority (a_b >= -6 m/s^2, human; -7 = the full box, descriptive). A case 'breaks'
the reference when any plan makes it collide. Every breaking grid plan is also run with the competent followers: if
they avoid, the collision exposes a gap in the reference model rather than an unavoidable plan.

  python -m fairtest.b44_reference_break run OUT.jsonl --scenario dec|cutout --cases CASES --eval EVAL [--workers 8]
  python -m fairtest.b44_analysis ...   (summary)
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp

import numpy as np

from fairtest import cutout_sim as X
from fairtest import dec_sim as D

REFS = {'dec': ('cc_dec', 'cc_dec_aeb', 'cc_dec_r050', 'cc_dec_r100', 'cc_dec_onset'),
        'cutout': ('cc_co', 'cc_co_aeb', 'cc_co_r050', 'cc_co_r100')}
COMPETENT = {'dec': ('idm_d0', 'rss_dec'), 'cutout': ('rss_dec',)}
A_B = {'dec': 1, 'cutout': 3}  # index of the braking level in theta
HUMAN, FULL = -6., -7.
GRID = {  # the full-box grid adds a_b = -6.5, -7
    'dec': (tuple(np.linspace(0., 5., 5)), (-7., -6.5, -6., -5.5, -5.01, -4.99, -4.5, -4., -3., -2., -1.), (.5, 1.5, 2.5, 4.),
            (-1.5, 0., 1.), (-1., 0., 1.)),
    'cutout': (tuple(np.linspace(0., 6., 5)), (3., 4.5, 6.), tuple(np.linspace(0., 6., 5)), (-7., -6.5, -6., -5., -4., -3., -2., -1.),
               (.5, 2., 4., 6.)),
}
BUDGET = 72


def episode(scenario, case, theta, name):
    """(valid, contact, min_gap)"""
    if scenario == 'dec':
        r = D.run(case, theta, name)
        return True, r['contact'], r['min_gap']
    r = X.run(case, theta, name)
    return r['valid'], r['contact'], r['min_gap']


def box(scenario, floor):
    low, high = (D.LOW, D.HIGH) if scenario == 'dec' else (X.LOW, X.HIGH)
    low = low.copy()
    low[A_B[scenario]] = floor
    return low, high


def grid_search(scenario, case, ref):
    out = {}
    plans = list(itertools.product(*GRID[scenario]))
    for label, floor in (('human', HUMAN), ('full', FULL)):
        n_valid = n_break = 0
        weakest, sub5, competent_avoid = None, 0, 0
        for th in plans:
            if th[A_B[scenario]] < floor - 1e-9:
                continue
            valid, contact, _ = episode(scenario, case, th, ref)
            if not valid:
                continue
            n_valid += 1
            if not contact:
                continue
            n_break += 1
            a_b = th[A_B[scenario]]
            sub5 += a_b > -5.
            if weakest is None or a_b > weakest[A_B[scenario]]:
                weakest = [float(x) for x in th]
            competent_avoid += all(not episode(scenario, case, th, f)[1] for f in COMPETENT[scenario])
        out[label] = dict(n_valid=n_valid, n_break=n_break, sub5=sub5, competent_avoid=competent_avoid, weakest=weakest)
    return out


def bo_search(scenario, case, ref, x_first, seed):
    out = {}
    for label, floor in (('human', HUMAN), ('full', FULL)):
        low, high = box(scenario, floor)
        start = None if x_first is None else np.clip(np.asarray(x_first, float), low, high)

        def obj(theta):
            valid, contact, gap = episode(scenario, case, theta, ref)
            if not valid:
                return 30., False, dict(valid=False)
            return (-1. if contact else gap), bool(contact), dict(valid=True)
        best, n = D.bo(obj, np.random.default_rng(seed + [floor == FULL]), BUDGET, x_first=start, low=low, high=high)
        found = bool(best and best['success'])
        rec = dict(found=found, queries=n if found else None)
        if found:
            th = best['theta']
            rec.update(theta=th, a_b=th[A_B[scenario]],
                       competent_avoid=all(not episode(scenario, case, th, f)[1] for f in COMPETENT[scenario]))
        out[label] = rec
    return out


def task(args):
    scenario, case, x_first = args
    out = dict(case_id=case['case_id'], scenario=scenario, refs={})
    for i, ref in enumerate(REFS[scenario]):
        out['refs'][ref] = dict(grid=grid_search(scenario, case, ref),
                                bo=bo_search(scenario, case, ref, x_first, [case['case_id'], 44, i]))
    return out


def main(a):
    cases = {c['case_id']: c for c in json.load(open(a.cases, encoding='utf-8'))['cases'] if c['partition'] == 'development'}
    boundary = {}
    for line in open(a.eval, encoding='utf-8'):
        r = json.loads(line)
        boundary[r['case_id']] = r.get('boundary')
    ids = sorted(cases)
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(task, [(a.scenario, cases[c], boundary.get(c)) for c in ids]):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(scenario=a.scenario, cases=len(ids))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run'])
    ap.add_argument('out')
    ap.add_argument('--scenario', required=True, choices=['dec', 'cutout'])
    ap.add_argument('--cases', required=True)
    ap.add_argument('--eval', required=True, help='B30 / B34 evaluation (the stored CC boundary plans)')
    ap.add_argument('--workers', type=int, default=8)
    main(ap.parse_args())
