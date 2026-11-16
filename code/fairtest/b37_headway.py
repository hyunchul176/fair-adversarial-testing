"""B37 (design fixed before the run): does forbidding entries inside the follower's headway remove unfair collisions?

The B34/B35 principle: unfair collisions arise when the attacking vehicle itself can enter the follower's headway
(cut-in); where the conflict lies beyond it (deceleration, cut-out) a human-authority attacker produces none. The
intervention, inside the cut-in scenario: a collision-seeking attacker restricted to plans whose time headway at the
lane-intrusion point is >= h_min. THW_int = bumper gap / SUT speed at the R157 5.2.5.2 reference point (b33_mandate:
the front-wheel outer edge 0.3 m past the marking), computed from the SUT's own run; a plan without an intrusion is
allowed. Search: GP-BO in the B11 plan space (5-D box), cold start (5 uniform initial points, as b23 bo_cold), budget
36 SUT queries, objective = SUT min gap + (10 + 10 * deficit if THW_int < h_min); success = the first valid SUT
contact that respects h_min. Every success is replayed with the CC (attacker-at-fault = the CC collides too).
h_min in {0 (unrestricted), 0.5, 1.0, 1.5 s}; the 7-SUT capability ladder; random streams keyed (case, SUT, h_min).

  python -m fairtest.b37_headway OUT.jsonl --partition development --hmins 0,0.5,1,1.5 --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import CC, HIGH, LOW, ContinuousPlan, gap
from fairtest.b23_search import N_CAND, N_INIT, SUTS, _ei
from fairtest.b33_mandate import labels


def objective(case, theta, name, h, h_min):
    plan = ContinuousPlan.from_theta(theta)
    raw = runner.run_episode(case, plan, name, h, keep_trace=True)
    row = runner.compact(raw, case, plan, name, h)
    lab = labels(case, raw['trace'])
    thw = lab.get('thw_int') if lab['intrusion'] else None
    ok = thw is None or thw >= h_min
    deficit = 0. if ok else h_min - thw
    f = gap(row) + (0. if ok else 10. + 10. * deficit)
    success = bool(row['valid'] and row['contact'] and ok)
    return f, success, dict(sut_contact=bool(row['contact']), valid=bool(row['valid']), thw_int=thw,
                            ttc_int=lab.get('ttc_int'), intrusion=lab['intrusion'])


def search(case, name, h, h_min, budget, rng):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    scale = HIGH - LOW
    X, y, best = [], [], None
    init = [rng.random(5) for _ in range(N_INIT)]
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
        f, ok, rec = objective(case, theta, name, h, h_min)
        X.append(x)
        y.append(f)
        if best is None or f < best['f']:
            best = dict(rec, f=float(f), theta=[float(v) for v in theta], success=ok)
        if ok:
            return best, n
    return best, None


def task(args):
    case_id, hmins, budget = args
    runner._init()
    case = runner._STATE['cases'][case_id]
    out = dict(case_id=case_id, runs={})
    t0 = time.time()
    for si, (name, h) in enumerate(SUTS):
        for hi, h_min in enumerate(hmins):
            key = f'{h_min:g}|{name}_H{h:g}'
            try:
                best, q = search(case, name, h, h_min, budget, np.random.default_rng([case_id, si, hi]))
                rec = dict(best=best, first_collision=q)
                if q is not None:
                    cc = runner.run_episode(case, ContinuousPlan.from_theta(best['theta']), *CC, keep_trace=True)
                    from fairtest.b23_margin import clearance
                    rec.update(cc_contact=bool(cc['pair_contact']), cc_clearance=round(clearance(cc['trace']), 3))
                out['runs'][key] = rec
            except Exception as e:  # recorded, never dropped
                out['runs'][key] = dict(error=f'{type(e).__name__}: {e}')
    out['seconds'] = round(time.time() - t0, 1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--partition', default='development')
    ap.add_argument('--hmins', default='0,0.5,1,1.5')
    ap.add_argument('--budget', type=int, default=36)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    runner._init()
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    hmins = tuple(float(x) for x in a.hmins.split(','))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(task, [(c, hmins, a.budget) for c in ids]):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(cases=len(ids), hmins=hmins)), flush=True)
