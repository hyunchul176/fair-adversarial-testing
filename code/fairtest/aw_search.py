"""B54: the v2j refinement stage (bo_warm_g4j) with Autoware as the SUT.

b23_search._objective_g4(guarded=True, joint=True) with the SUT run replaced by an Autoware episode of the plan's
adversary trajectory (fairtest.aw_sut.replay; the adversary's motion depends only on the plan, carla_search.trajectory_row).
The CC counterfactual, G1 (no lateral retreat after the CC perception threshold), G2 (realism <= 1 IQR) and the joint
support (5-NN <= tau) come from the plan's highway-env CC run, exactly as in v2j. The Gaussian process, acquisition,
candidate sampling, warm start (the stored CC boundary plan) and stopping rule (first success) are b23_search's.
  success = SUT contact (valid manoeuvre) with front contact (contact_dx >= 3.5 m), CC avoids with clearance >= 0.5 m, no
            feint, realistic, inside the joint support
  f       = SUT gap (-1 at contact, 50 without overlap) + 20 [CC contact] or 10 max(0, 0.5 - CC clearance) + guard penalties
            - 5 [success]
"""
from __future__ import annotations

import numpy as np

from fairtest import runner
from fairtest.b11_search import CC, HIGH, LOW
from fairtest.b23_search import DELTA, FRONT_DX, METHODS, N_CAND, N_INIT, _ei, guard_terms, joint_score

_CC_CACHE = {}


def cc_twin(case, theta):
    """(CC trace, CC compact row) of a plan in highway-env; cached per (case, theta)."""
    from fairtest.b11_search import ContinuousPlan
    key = (case['case_id'], tuple(round(float(x), 9) for x in theta))
    if key not in _CC_CACHE:
        plan = ContinuousPlan.from_theta(theta)
        raw = runner.run_episode(case, plan, CC[0], CC[1], keep_trace=True)
        _CC_CACHE[key] = (raw['trace'], runner.compact(raw, case, plan, CC[0], CC[1]))
    return _CC_CACHE[key]


def objective_aw(replay, case, theta):
    """replay(row) -> an Autoware episode of the row's adversary trajectory (aw_sut.replay on one instance)."""
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_margin import clearance
    from fairtest.carla_search import trajectory_row
    cc_trace, cc = cc_twin(case, theta)
    valid, row = trajectory_row(case, ContinuousPlan.from_theta(theta))
    aw = replay(row)
    clear = clearance(cc_trace)
    contact = bool(aw['contact'])
    gap = -1. if contact else (50. if aw['min_gap'] is None else float(aw['min_gap']))
    fair = bool(valid and contact and not cc['contact'])
    robust = bool(fair and aw['contact_dx'] is not None and aw['contact_dx'] >= FRONT_DX and clear >= DELTA)
    feint, exc = guard_terms(cc_trace, cc['features'])
    realistic = exc is not None and exc <= 1.
    guard_pen = (10. if feint else 0.) + (10. if exc is None else 5. * max(0., exc - 1.))
    js, tau = joint_score(cc['features'])
    realistic = realistic and js is not None and js <= tau
    guard_pen += 10. if js is None else 5. * max(0., js - tau)
    success = robust and not feint and realistic
    f = gap + (20. if cc['contact'] else 10. * max(0., DELTA - clear) + guard_pen) - (5. if success else 0.)
    rec = dict(f=float(f), theta=[float(x) for x in theta], success=bool(success), fair=fair, robust=robust, valid=bool(valid),
               sut_contact=contact, cc_contact=bool(cc['contact']), cc_clearance=round(float(clear), 3), contact_dx=aw['contact_dx'],
               cc_demand=round(float(cc.get('max_drac') or 0.), 3), feint=feint, realism_exceedance=None if exc is None else round(exc, 3),
               joint_score=None if js is None else round(js, 3), aw={k: aw.get(k) for k in ('t_contact', 'min_gap', 'sut_min_acc', 'v0_error',
                                                                                        'late_max', 'min_clearance', 'health')},
               aw_trace=aw['trace'])
    return f, success, rec


def bo_search_aw(replay, case, budget, rng, x_first=None):
    """b23_search.bo_search with objective_aw (same GP, acquisition, candidates and stopping rule)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    scale = HIGH - LOW
    X, y, best, queries = [], [], None, []
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
        f, ok, rec = objective_aw(replay, case, theta)
        X.append(x)
        y.append(f)
        queries.append({k: v for k, v in rec.items() if k != 'aw_trace'})
        if best is None or f < best['f']:
            best = rec
        if ok:
            return best, n, queries
    return best, budget, queries


def refine(replay, case, boundary, budget=36, panel_index=0):
    rng = np.random.default_rng([case['case_id'], panel_index, METHODS.index('bo_warm')])  # the bo_warm_g4j stream
    best, n, queries = bo_search_aw(replay, case, budget, rng, x_first=boundary)
    return dict(found=bool(best and best['success']), queries=n, best=best, history=queries)


def bo_search_aw_screened(replay, case, budget, rng, x_first=None):
    """B59: b23_search.bo_search_screened (B56 bo_warm_g4s) with Autoware as the SUT. Every candidate is replayed against
    the CC in highway-env first (b23_search.admission) and only admissible ones run on Autoware (objective_aw). Warm
    start, candidates, GP, acquisition order and stopping rules are bo_search_screened's; SCREEN_MAX bounds the replays."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from fairtest.b23_search import SCREEN_MAX, admission
    scale = HIGH - LOW
    X, y, best, queries = [], [], None, []
    st = dict(cc_replays=0, stop=None, warm=None)
    since = [0]

    def run(x):
        nonlocal best
        theta = LOW + x * scale
        f, ok, rec = objective_aw(replay, case, theta)
        X.append(x)
        y.append(f)
        q = {k: v for k, v in rec.items() if k != 'aw_trace'}
        q['cc_replays'] = since[0]
        since[0] = 0
        queries.append(q)
        if best is None or f < best['f']:
            best = rec
        return ok

    def screen(x):
        st['cc_replays'] += 1
        since[0] += 1
        return admission(case, LOW + x * scale)[0]

    warm = []
    if x_first is not None:
        x0 = np.clip((np.asarray(x_first) - LOW) / scale, 0, 1)
        cloud = np.clip(x0 + rng.normal(0, .08, (N_CAND // 4, 5)), 0, 1)
        warm = [x0] + list(cloud[np.argsort(np.linalg.norm(cloud - x0, axis=1), kind='stable')])
    for x in warm:
        if st['cc_replays'] >= SCREEN_MAX:
            break
        if screen(x):
            st['warm'] = 'boundary' if x is warm[0] else 'nearest'
            if run(x):
                st['stop'] = 'success'
                return best, len(X), queries, st
            break
    draws = 0
    while len(X) < min(N_INIT, budget) and st['cc_replays'] < SCREEN_MAX and draws < N_CAND:
        x = rng.random(5)
        draws += 1
        if screen(x) and run(x):
            st['stop'] = 'success'
            return best, len(X), queries, st
    if not X:
        st['stop'] = 'no_admissible_initial'
        return best, 0, queries, st
    while len(X) < budget:
        gp = GaussianProcessRegressor(ConstantKernel(1.) * Matern(length_scale=np.full(5, .3), nu=2.5) + WhiteKernel(1e-3),
                                      normalize_y=True, n_restarts_optimizer=0, random_state=0)
        gp.fit(np.array(X), np.array(y))
        xb = np.array(X)[int(np.argmin(y))]
        cand = np.vstack([rng.random((N_CAND, 5)), np.clip(xb + rng.normal(0, .08, (N_CAND // 4, 5)), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True)
        chosen = None
        for i in np.argsort(-_ei(mu, sd, float(np.min(y))), kind='stable'):
            if st['cc_replays'] >= SCREEN_MAX:
                break
            if screen(cand[i]):
                chosen = cand[i]
                break
        if chosen is None:
            st['stop'] = 'screen_max' if st['cc_replays'] >= SCREEN_MAX else 'no_admissible_candidate'
            return best, len(X), queries, st
        if run(chosen):
            st['stop'] = 'success'
            return best, len(X), queries, st
    st['stop'] = 'budget'
    return best, len(X), queries, st


def refine_screened(replay, case, boundary, budget=36, panel_index=0):
    """B59: the screened refinement against Autoware; the same random stream as refine (bo_warm)."""
    rng = np.random.default_rng([case['case_id'], panel_index, METHODS.index('bo_warm')])
    best, n, queries, st = bo_search_aw_screened(replay, case, budget, rng, x_first=boundary)
    return dict(found=bool(best and best['success']), queries=n, best=best, history=queries, **st)
