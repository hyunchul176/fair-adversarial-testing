"""B23 (E3): query efficiency of fair-failure search — optimiser baselines in the B11 plan space.

Same plan space (B11 theta, 5-D box), same objective f = SUT min gap + 20*[CC contact] - 5*[fair failure],
same SUT budget (36) and the same stop-at-first-fair-failure rule as the B11 CMA-ES (`cont_search`), so
only the optimiser differs. Every evaluation = 1 SUT run + 1 CC run; the SUT query index of the first fair
failure is recorded (anytime curve).
  random      uniform in the box
  cmaes_cold  the B11 CMA-ES started at the box centre (the stored `cont_search` starts at the R157 boundary)
  bo_cold     GP-BO with expected improvement: Matern-5/2 GP on the unit box, 5 uniform initial points
  bo_warm     the same with the R157-boundary plan as the first initial point (+4 uniform)
The boundary plan comes from the stored B11 `cc_boundary` search of the case (CC-only information).
A method with the suffix _g4 (random_g4, bo_cold_g4, bo_warm_g4) targets ROBUST fair failures (G4: front contact,
contact_dx >= 3.5 m, CC clearance >= 0.5 m): it stops at the first robust one and its objective adds
10*max(0, 0.5 - CC clearance), pushing the search off the CC boundary where the warm start begins.
bo_warm_g4c additionally demands the R157 Annex 5 'avoidable' grade (the CC's peak DRAC < 5 m/s^2) and adds
4*max(0, DRAC - 5): fair failures a careful driver avoids with moderate braking.
bo_warm_g4g applies the RL agent's guards in plan space (B29): G1 no lateral retreat > 0.2 m after the CC perception
threshold (a plan with the natural initial lateral speed and a late entry can approach, retreat and re-enter, which
exploits the CC's perception latch) and G2 a realistic cut-in (CC-replay realism exceedance <= 1 IQR); penalties
10*[feint] + 5*max(0, exceedance - 1) (10 without a cut-in).
bo_warm_g4j (B46) = bo_warm_g4g + joint realism: the CC replay's six cut-in features must also lie inside the natural
joint support (5-NN distance <= the natural leave-one-out 99th percentile, b45_joint_realism.Support); penalty
5*max(0, score - tau) (10 when a feature is missing). Same random stream and start as bo_warm_g4g.
bo_warm_g4s (B56) = bo_warm_g4j whose every SUT query is admissible: each candidate is first replayed against the CC
(no SUT run) and runs on the SUT only if the CC side of the g4j objective is zero (CC avoids with clearance >= 0.5 m,
no feint, realism exceedance <= 1, joint score <= tau). Candidates, GP, EI and objective are g4j's; inadmissible ones are
skipped in descending EI order (`bo_search_screened`). Same random stream as bo_warm_g4j. With --log-queries every
executed query of the g4j/g4s methods is also recorded (admission terms, SUT contact, success; g4s: CC replays spent).

  python -m fairtest.b23_search OUT.jsonl --cont STORED_B11.jsonl --partition development --workers W [--suts ...]
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
from fairtest.b11_search import CC, HIGH, LOW, evaluate, gap, search

METHODS = ('random', 'cmaes_cold', 'bo_cold', 'bo_warm', 'random_g4', 'bo_cold_g4', 'bo_warm_g4', 'bo_warm_g4c', 'bo_warm_g4g',
           'bo_warm_g4j', 'bo_warm_g4s', 'random_g4s')  # append only: the index of a base method seeds its random stream
SCREEN_MAX = 50000  # B56: CC replays one bo_warm_g4s search may spend (pilot and smoke test before the freeze: ~220/query)
DELTA, FRONT_DX = .5, 3.5  # G4
AVOIDABLE = 5.  # R157 Annex 5 'avoidable' grade: the CC needs < 5 m/s^2 (the _g4c methods demand it)
PERCEIVE_M, RETREAT_M = 1.095, .2  # _g4g (G1 in plan space): no lateral retreat > 0.2 m after the CC perception threshold
_REALISM = []


def guard_terms(cc_trace, cc_features):
    """(feint, realism exceedance) of an attack from its CC counterfactual -- the RL guards G1/G2 in plan space."""
    ys = [abs(r['generator']['y']) for r in cc_trace]
    k = next((i for i, y in enumerate(ys) if y >= PERCEIVE_M), None)
    retreat, peak = 0., 0.
    if k is not None:
        peak = ys[k]
        for y in ys[k:]:
            peak = max(peak, y)
            retreat = max(retreat, peak - y)
    if not _REALISM:
        from fairtest.features import Realism
        _REALISM.append(Realism())
    exc = _REALISM[0].score(cc_features)['max_exceedance'] if cc_features.get('cut_in') else None
    return retreat > RETREAT_M, (None if exc is None else float(exc))


_SUPPORT = []


def joint_score(cc_features):
    """(5-NN distance to natural cut-ins, tau) of the CC replay (B45); the distance is None without a complete cut-in."""
    if not _SUPPORT:
        from fairtest.b45_joint_realism import Support
        _SUPPORT.append(Support())
    s = _SUPPORT[0]
    return (s.score(cc_features) if cc_features.get('cut_in') else None), s.tau
SUTS = (('following', 0.), ('following', .5), ('following', 1.), ('reference_idm', 0.), ('following_aeb', 0.),
        ('following_aeb', .5), ('rss_sut', 0.))
N_INIT, N_CAND = 5, 4096


def _objective(case, theta, name, h, robust=False, cap=None, guarded=False, joint=False):
    if robust:
        return _objective_g4(case, theta, name, h, cap, guarded, joint)
    cc = evaluate(case, theta, *CC)
    row = evaluate(case, theta, name, h)
    fair = bool(row['valid'] and row['contact'] and not cc['contact'])
    f = gap(row) + (20. if cc['contact'] else 0.) - (5. if fair else 0.)
    return f, fair, row, cc


def _objective_g4(case, theta, name, h, cap=None, guarded=False, joint=False):
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_margin import clearance
    runs = []
    for target, hh in (CC, (name, h)):
        plan = ContinuousPlan.from_theta(theta)
        raw = runner.run_episode(case, plan, target, hh, keep_trace=True)
        runs.append((raw['trace'], runner.compact(raw, case, plan, target, hh)))
    (cc_trace, cc), (trace, row) = runs
    clear = clearance(cc_trace)
    dx = trace[-1]['generator']['x'] - trace[-1]['target']['x']
    fair = bool(row['valid'] and row['contact'] and not cc['contact'])
    robust = bool(fair and dx >= FRONT_DX and clear >= DELTA)
    drac = float(cc.get('max_drac') or 0.)
    over = 0. if cap is None else max(0., drac - cap)
    success = robust and over == 0. and (cap is None or drac < cap)
    guard_pen, feint, exc, js = 0., None, None, None
    if guarded:
        feint, exc = guard_terms(cc_trace, cc['features'])
        realistic = exc is not None and exc <= 1.
        guard_pen = (10. if feint else 0.) + (10. if exc is None else 5. * max(0., exc - 1.))
        if joint:  # B46: also inside the natural joint support
            js, tau = joint_score(cc['features'])
            realistic = realistic and js is not None and js <= tau
            guard_pen += 10. if js is None else 5. * max(0., js - tau)
        success = success and not feint and realistic
    f = gap(row) + (20. if cc['contact'] else 10. * max(0., DELTA - clear) + 4. * over + guard_pen) - (5. if success else 0.)
    row['g4'] = dict(fair=fair, robust=robust, cc_clearance=round(clear, 3), contact_dx=round(dx, 2), cc_demand=round(drac, 3),
                     feint=feint, realism_exceedance=None if exc is None else round(exc, 3))
    if joint:
        row['g4']['joint_score'] = None if js is None else round(js, 3)
    return f, success, row, cc


def admission(case, theta):
    """B56: the CC side of the bo_warm_g4j objective, without an SUT run. admissible <=> that side's penalty is zero:
    the CC avoids with clearance >= DELTA, no feint (G1), realism exceedance <= 1 (G2), joint score <= tau."""
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_margin import clearance
    plan = ContinuousPlan.from_theta(theta)
    raw = runner.run_episode(case, plan, *CC, keep_trace=True)
    cc_trace, cc = raw['trace'], runner.compact(raw, case, plan, *CC)
    clear = clearance(cc_trace)
    feint, exc = guard_terms(cc_trace, cc['features'])
    js, tau = joint_score(cc['features'])
    guard_pen = (10. if feint else 0.) + (10. if exc is None else 5. * max(0., exc - 1.)) + (10. if js is None else 5. * max(0., js - tau))
    pen = 20. if cc['contact'] else 10. * max(0., DELTA - clear) + guard_pen
    return pen == 0., dict(cc=cc, clear=clear, feint=feint, exc=exc, js=js, pen=pen)


def _objective_g4s(case, theta, name, h, adm):
    """The bo_warm_g4j objective of an admissible plan: its CC replay (adm) is reused, only the SUT runs."""
    from fairtest.b11_search import ContinuousPlan
    plan = ContinuousPlan.from_theta(theta)
    raw = runner.run_episode(case, plan, name, h, keep_trace=True)
    trace, row = raw['trace'], runner.compact(raw, case, plan, name, h)
    cc = adm['cc']
    dx = trace[-1]['generator']['x'] - trace[-1]['target']['x']
    fair = bool(row['valid'] and row['contact'] and not cc['contact'])
    robust = bool(fair and dx >= FRONT_DX and adm['clear'] >= DELTA)
    f = gap(row) + adm['pen'] - (5. if robust else 0.)  # admissible: no feint, realistic, inside the joint support
    row['g4'] = dict(fair=fair, robust=robust, cc_clearance=round(adm['clear'], 3), contact_dx=round(dx, 2),
                     cc_demand=round(float(cc.get('max_drac') or 0.), 3), feint=adm['feint'],
                     realism_exceedance=None if adm['exc'] is None else round(adm['exc'], 3),
                     joint_score=None if adm['js'] is None else round(adm['js'], 3))
    return f, robust, row, cc


def _query_log(n, row, cc, success, replays=None):
    """One executed SUT query: its admission terms (from the CC replay), SUT contact and success."""
    g = row.get('g4', {})
    _, tau = joint_score({})
    js, exc = g.get('joint_score'), g.get('realism_exceedance')
    adm = bool(not cc['contact'] and g.get('cc_clearance', 0.) >= DELTA and not g.get('feint') and exc is not None and exc <= 1.
               and js is not None and js <= tau)
    out = dict(n=n, admissible=adm, cc_contact=bool(cc['contact']), cc_clearance=g.get('cc_clearance'), feint=g.get('feint'),
               exc=exc, js=js, sut_contact=bool(row['contact']), valid=bool(row['valid']), success=bool(success))
    if replays is not None:
        out['cc_replays'] = replays
    return out


def _record(theta, f, fair, row, cc):
    # `features` is the SUT run's (kept for the stored analyses); the realism guards use the CC replay's (cc_features)
    out = dict(f=float(f), theta=[float(x) for x in theta], fair=fair, cc_contact=bool(cc['contact']),
               sut_contact=bool(row['contact']), features=row['features'], cc_features=cc.get('features'))
    if 'g4' in row:  # _g4/_g4c methods: 'fair' is the success verdict (robust [and avoidable]); the plain one is kept
        out.update(robust=row['g4']['robust'], plain_fair=row['g4']['fair'], cc_clearance=row['g4']['cc_clearance'],
                   contact_dx=row['g4']['contact_dx'], cc_demand=row['g4']['cc_demand'], feint=row['g4']['feint'],
                   realism_exceedance=row['g4']['realism_exceedance'])
        if 'joint_score' in row['g4']:
            out['joint_score'] = row['g4']['joint_score']
    return out


def random_search(case, name, h, budget, rng, robust=False, cap=None, guarded=False):
    best = None
    for n in range(1, budget + 1):
        theta = LOW + rng.random(5) * (HIGH - LOW)
        f, fair, row, cc = _objective(case, theta, name, h, robust, cap, guarded)
        if best is None or f < best['f']:
            best = _record(theta, f, fair, row, cc)
        if fair:
            return best, n
    return best, budget


def random_search_screened(case, name, h, budget, rng, log=None):
    """B61 random_g4s: uniform plans, each screened like bo_warm_g4s (admission), the admissible ones run on the SUT
    until the first success of the g4j contract or `budget` SUT queries (SCREEN_MAX CC replays at most)."""
    best, st, since, n = None, dict(cc_replays=0, stop='budget', warm=None), 0, 0
    while n < budget:
        if st['cc_replays'] >= SCREEN_MAX:
            st['stop'] = 'screen_max'
            break
        theta = LOW + rng.random(5) * (HIGH - LOW)
        st['cc_replays'] += 1
        since += 1
        ok, adm = admission(case, theta)
        if not ok:
            continue
        n += 1
        f, success, row, cc = _objective_g4s(case, theta, name, h, adm)
        if log is not None:
            log.append(_query_log(n, row, cc, success, since))
        since = 0
        if best is None or f < best['f']:
            best = _record(theta, f, success, row, cc)
        if success:
            st['stop'] = 'success'
            break
    return best, n, st


def _ei(mu, sd, best):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-9)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def bo_search(case, name, h, budget, rng, x_first=None, robust=False, cap=None, guarded=False, joint=False, log=None):
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
        f, fair, row, cc = _objective(case, theta, name, h, robust, cap, guarded, joint)
        if log is not None:
            log.append(_query_log(n, row, cc, fair))
        X.append(x)
        y.append(f)
        if best is None or f < best['f']:
            best = _record(theta, f, fair, row, cc)
        if fair:
            return best, n
    return best, budget


def bo_search_screened(case, name, h, budget, rng, x_first=None, log=None):
    """B56 bo_warm_g4s. Returns (best, executed SUT queries, stats).

    Initial design: the boundary plan if admissible, else the admissible plan nearest to it among N_CAND/4 local
    perturbations (normal 0.08 in the unit box, screened in order of distance); then uniform draws from the stream, each
    screened, until N_INIT plans ran on the SUT or N_CAND draws were screened. (The pilot found the boundary plan
    admissible in 14 of 62 development cases, mostly failing G2.) Then per SUT query: the g4j GP on the executed plans, the g4j candidates
    (N_CAND uniform + N_CAND/4 around the best), screened in descending EI order; the first admissible one runs on the
    SUT. Ends at the first success, at `budget` SUT queries, when no candidate of an iteration is admissible, or when
    SCREEN_MAX CC replays have been spent."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    scale = HIGH - LOW
    X, y, best = [], [], None
    st = dict(cc_replays=0, stop=None, warm=None)
    since = 0

    def run(x, adm):
        nonlocal best, since
        theta = LOW + x * scale
        f, ok, row, cc = _objective_g4s(case, theta, name, h, adm)
        X.append(x)
        y.append(f)
        if log is not None:
            log.append(_query_log(len(X), row, cc, ok, since))
        since = 0
        if best is None or f < best['f']:
            best = _record(theta, f, ok, row, cc)
        return ok

    def screen(x):
        nonlocal since
        st['cc_replays'] += 1
        since += 1
        return admission(case, LOW + x * scale)

    warm = []
    if x_first is not None:
        x0 = np.clip((np.asarray(x_first) - LOW) / scale, 0, 1)
        cloud = np.clip(x0 + rng.normal(0, .08, (N_CAND // 4, 5)), 0, 1)
        warm = [x0] + list(cloud[np.argsort(np.linalg.norm(cloud - x0, axis=1), kind='stable')])
    for x in warm:
        if st['cc_replays'] >= SCREEN_MAX:
            break
        ok, adm = screen(x)
        if ok:
            st['warm'] = 'boundary' if x is warm[0] else 'nearest'
            if run(x, adm):
                st['stop'] = 'success'
                return best, len(X), st
            break
    draws = 0
    while len(X) < min(N_INIT, budget) and st['cc_replays'] < SCREEN_MAX and draws < N_CAND:
        x = rng.random(5)
        draws += 1
        ok, adm = screen(x)
        if ok and run(x, adm):
            st['stop'] = 'success'
            return best, len(X), st
    if not X:
        st['stop'] = 'no_admissible_initial'
        return best, 0, st
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
            ok, adm = screen(cand[i])
            if ok:
                chosen = (cand[i], adm)
                break
        if chosen is None:
            st['stop'] = 'screen_max' if st['cc_replays'] >= SCREEN_MAX else 'no_admissible_candidate'
            return best, len(X), st
        if run(*chosen):
            st['stop'] = 'success'
            return best, len(X), st
    st['stop'] = 'budget'
    return best, len(X), st


def task(args):
    case_id, x_boundary, suts, methods, budget = args[:5]
    only = args[5] if len(args) > 5 else None  # run only these SUT keys; the stream index stays the position in suts
    log_queries = bool(args[6]) if len(args) > 6 else False
    runner._init()
    case = runner._STATE['cases'][case_id]
    out = dict(case_id=case_id, runs={})
    t0 = time.time()
    for si, (name, h) in enumerate(suts):
        key = f'{name}_H{h:g}'
        if only and key not in only:
            continue
        for method in methods:
            cap = AVOIDABLE if method.endswith('_g4c') else None
            joint = method.endswith(('_g4j', '_g4s'))
            guarded = method.endswith(('_g4g', '_g4j', '_g4s'))
            robust = method.endswith(('_g4', '_g4c', '_g4g', '_g4j', '_g4s'))
            base = method[:-4] if (cap or guarded) else (method[:-3] if robust else method)
            rng = np.random.default_rng([case_id, si, METHODS.index(base)])  # _g4 shares its base's random stream
            log = [] if log_queries and method.endswith(('_g4j', '_g4s')) else None
            st = None
            try:
                if method == 'random_g4s':  # B61: screened random search (the 'random' stream)
                    best, n, st = random_search_screened(case, name, h, budget, rng, log=log)
                elif method.endswith('_g4s'):
                    best, n, st = bo_search_screened(case, name, h, budget, rng, x_first=x_boundary, log=log)
                elif base == 'random':
                    best, n = random_search(case, name, h, budget, rng, robust, cap)
                elif base == 'cmaes_cold':
                    best, n, _ = search(case, name, h, (LOW + HIGH) / 2, budget)
                elif base == 'bo_cold':
                    best, n = bo_search(case, name, h, budget, rng, robust=robust, cap=cap)
                else:
                    best, n = bo_search(case, name, h, budget, rng, x_first=x_boundary, robust=robust, cap=cap, guarded=guarded,
                                        joint=joint, log=log)
                rec = dict(best=best, sut_queries=n, first_fair=n if best and best['fair'] else None)
                if st is not None:
                    rec.update(cc_replays=st['cc_replays'], stop=st['stop'], warm=st['warm'])
                if log is not None:
                    rec['queries'] = log
                out['runs'][f'{method}|{key}'] = rec
            except Exception as e:  # recorded, never dropped
                out['runs'][f'{method}|{key}'] = dict(error=f'{type(e).__name__}: {e}')
    out['seconds'] = round(time.time() - t0, 1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True, help='stored B11 output with the per-case cc_boundary plan')
    ap.add_argument('--partition', default='development', help="case partition, or 'all'")
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--suts', help='comma-separated name:H (default: the 7-SUT panel)')
    ap.add_argument('--budget', type=int, default=36)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--only', help='comma-separated SUT keys (name_H<h>) to run; streams keep their --suts position')
    ap.add_argument('--log-queries', action='store_true', help='B56: record every executed query of the g4j/g4s methods')
    ap.add_argument('--split-suts', action='store_true', help='B56: one job (and output line) per case and SUT')
    a = ap.parse_args()
    runner._init()
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    suts = SUTS if not a.suts else tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    methods = tuple(a.methods.split(','))
    only = frozenset(a.only.split(',')) if a.only else None
    jobs = [(c, boundary.get(c), suts, methods, a.budget, only, a.log_queries) for c in ids]
    if a.split_suts:  # B56: one job per (case, SUT) for load balance; streams keep their --suts position
        keys = [f'{n}_H{h:g}' for n, h in suts]
        jobs = [(c, boundary.get(c), suts, methods, a.budget, frozenset({k}), a.log_queries) for c in ids for k in keys
                if not only or k in only]
    t0 = time.time()
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 5 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), elapsed_s=round(time.time() - t0, 1))), flush=True)
