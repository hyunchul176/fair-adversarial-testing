"""B51 (B51 design, available on request): v2j with a tester-chosen severity floor.

A reported failure must also hit the SUT with a closing speed (SUT vx - generator vx at contact) >= v_min (3 m/s).
  refine   bo_warm_g4js = bo_warm_g4j (b23_search: boundary start, budget 36, G4 + G1 + G2 + joint support, the same random
           stream) whose success also needs closing >= v_min and whose objective adds 2 * max(0, v_min - closing) for
           SUT-contact plans, so the search does not stop at a light touch
  closing  the closing speed at contact of stored attacks: every policy-stage contact (policy files) and every refinement
           success (refinement files, method bo_warm_g4j) -- the v2j side of the comparison
New module only: the frozen b23_search objective is reproduced here with the severity term added.

  python -m fairtest.b51_severity refine OUT.jsonl --cont STORED_B11.jsonl --partition development --suts ... --workers W
  python -m fairtest.b51_severity closing OUT.jsonl --policy final_s1.jsonl ... --refine REFINE.jsonl ... --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import CC, HIGH, LOW, gap
from fairtest.b23_search import DELTA, FRONT_DX, METHODS, N_CAND, N_INIT, _ei, _record, guard_terms, joint_score

V_MIN, SEV_W = 3., 2.
METHOD = 'bo_warm_g4js'


def objective_g4js(case, theta, name, h, v_min=V_MIN):
    """b23_search._objective_g4(guarded=True, joint=True) + the severity floor (success) and penalty (objective)."""
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_margin import clearance
    runs = []
    for target, hh in (CC, (name, h)):
        plan = ContinuousPlan.from_theta(theta)
        raw = runner.run_episode(case, plan, target, hh, keep_trace=True)
        runs.append((raw['trace'], runner.compact(raw, case, plan, target, hh)))
    (cc_trace, cc), (trace, row) = runs
    clear = clearance(cc_trace)
    last = trace[-1]
    dx = last['generator']['x'] - last['target']['x']
    closing = float(last['target']['vx'] - last['generator']['vx'])
    fair = bool(row['valid'] and row['contact'] and not cc['contact'])
    robust = bool(fair and dx >= FRONT_DX and clear >= DELTA)
    drac = float(cc.get('max_drac') or 0.)
    feint, exc = guard_terms(cc_trace, cc['features'])
    realistic = exc is not None and exc <= 1.
    guard_pen = (10. if feint else 0.) + (10. if exc is None else 5. * max(0., exc - 1.))
    js, tau = joint_score(cc['features'])
    realistic = realistic and js is not None and js <= tau
    guard_pen += 10. if js is None else 5. * max(0., js - tau)
    severe = bool(row['contact']) and closing >= v_min
    sev_pen = SEV_W * max(0., v_min - closing) if row['contact'] else 0.
    success = robust and not feint and realistic and severe
    f = gap(row) + (20. if cc['contact'] else 10. * max(0., DELTA - clear) + guard_pen + sev_pen) - (5. if success else 0.)
    row['g4'] = dict(fair=fair, robust=robust, cc_clearance=round(clear, 3), contact_dx=round(dx, 2), cc_demand=round(drac, 3),
                     feint=feint, realism_exceedance=None if exc is None else round(exc, 3),
                     joint_score=None if js is None else round(js, 3))
    row['closing'] = round(closing, 3)
    return f, success, row, cc


def bo_search_s(case, name, h, budget, rng, x_first=None, v_min=V_MIN):
    """b23_search.bo_search with objective_g4js (same GP, acquisition, candidate sampling and stopping rule)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    scale = HIGH - LOW
    X, y, best = [], [], None
    init = [] if x_first is None else [np.clip((np.asarray(x_first) - LOW) / scale, 0, 1)]
    while len(init) < N_INIT:
        init.append(rng.random(5))
    for n in range(1, budget + 1):
        if n <= len(init):
            x = init[n - 1]
        else:
            gp = GaussianProcessRegressor(ConstantKernel(1.) * Matern(length_scale=np.full(5, .3), nu=2.5) + WhiteKernel(1e-3),
                                          normalize_y=True, n_restarts_optimizer=0, random_state=0)
            gp.fit(np.array(X), np.array(y))
            xb = np.array(X)[int(np.argmin(y))]
            cand = np.vstack([rng.random((N_CAND, 5)), np.clip(xb + rng.normal(0, .08, (N_CAND // 4, 5)), 0, 1)])
            mu, sd = gp.predict(cand, return_std=True)
            x = cand[int(np.argmax(_ei(mu, sd, float(np.min(y)))))]
        theta = LOW + x * scale
        f, ok, row, cc = objective_g4js(case, theta, name, h, v_min)
        X.append(x)
        y.append(f)
        if best is None or f < best['f']:
            best = _record(theta, f, ok, row, cc)
            best['closing'] = row['closing']
        if ok:
            return best, n
    return best, budget


def refine_task(args):
    case_id, x_boundary, suts, budget, v_min = args
    runner._init()
    case = runner._STATE['cases'][case_id]
    out = dict(case_id=case_id, v_min=v_min, runs={})
    t0 = time.time()
    for si, (name, h) in enumerate(suts):
        key = f'{name}_H{h:g}'
        rng = np.random.default_rng([case_id, si, METHODS.index('bo_warm')])  # the bo_warm_g4j stream
        try:
            best, n = bo_search_s(case, name, h, budget, rng, x_first=x_boundary, v_min=v_min)
            out['runs'][f'{METHOD}|{key}'] = dict(best=best, sut_queries=n, first_fair=n if best and best['fair'] else None)
        except Exception as e:  # recorded, never dropped
            out['runs'][f'{METHOD}|{key}'] = dict(error=f'{type(e).__name__}: {e}')
    out['seconds'] = round(time.time() - t0, 1)
    return out


def refine_main(a):
    runner._init()
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    ids = ids[:a.cases] if a.cases else ids
    jobs = [(c, boundary.get(c), suts, a.budget, a.v_min) for c in ids]
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(refine_task, jobs):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(finished=len(jobs))))


def closing_task(job):
    kind, seed, case_id, sut, how, payload = job
    from fairtest.b29_consensus import _run
    runner._init()
    name, h = sut.rsplit('_H', 1)
    try:
        r = _run(runner._STATE['cases'][case_id], how, payload, name, float(h))
        last = r['trace'][-1]
        return dict(kind=kind, seed=seed, case_id=case_id, sut=sut, sut_contact=bool(r['pair_contact']),
                    closing=round(float(last['target']['vx'] - last['generator']['vx']), 3))
    except Exception as e:  # recorded, never dropped
        return dict(kind=kind, seed=seed, case_id=case_id, sut=sut, error=f'{type(e).__name__}: {e}')


def closing_main(a):
    jobs = []
    for path in a.policy:
        seed = re.search(r'_s(\d)', Path(path).name).group(1)
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r.get('valid') and r.get('contact') and r.get('actions'):
                jobs.append(('policy', seed, r['case_id'], r['sut'], 'actions', r['actions']))
    for path in a.refine:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == a.method and 'error' not in rec and rec.get('first_fair') is not None:
                    jobs.append((m, None, r['case_id'], s, 'theta', rec['best']['theta']))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(closing_task, jobs, chunksize=4):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(replayed=len(jobs))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('refine')
    p.add_argument('out')
    p.add_argument('--cont', required=True)
    p.add_argument('--partition', default='development')
    p.add_argument('--suts', required=True)
    p.add_argument('--budget', type=int, default=36)
    p.add_argument('--v-min', type=float, default=V_MIN)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cases', type=int, help='first N cases only (checks)')
    q = sub.add_parser('closing')
    q.add_argument('out')
    q.add_argument('--policy', nargs='*', default=[])
    q.add_argument('--refine', nargs='*', default=[])
    q.add_argument('--method', default='bo_warm_g4j')
    q.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    refine_main(a) if a.cmd == 'refine' else closing_main(a)
