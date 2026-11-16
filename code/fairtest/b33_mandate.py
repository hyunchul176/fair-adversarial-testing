"""B33 (post hoc): the R157 explicit cut-in requirement (5.2.5.2) against the careful-driver benchmark.

R157 (Rev.1 / 01 series; same wording in the original series) 5.2.5.2: the ALKS shall avoid a collision with a
cutting-in vehicle provided
  (a) the cutting-in vehicle maintains its longitudinal speed, which is lower than the ALKS vehicle's;
  (b) its lateral movement has been visible for at least 0.72 s before the reference point for TTC_LaneIntrusion;
  (c) TTC_LaneIntrusion > v_rel / (2 * 6 m/s^2) + 0.35 s,
TTC_LaneIntrusion being the TTC when the outside of the tyre of the intruding vehicle's front wheel closest to the
lane markings crosses a line 0.3 m beyond the outside edge of the visible lane marking. (c) is the gap a follower
needs when it brakes at 6 m/s^2 from 0.35 s after that point: gap > 0.35 v_rel + v_rel^2 / 12. Outside 5.2.5.2 (and
5.2.4, 5.2.5.1) R157 asks for the careful-driver level (Rev.1 5.2.7) -- this study's benchmark. A collision is
  mandated   (a)-(c) hold: R157 explicitly requires the ALKS to avoid it, whatever a careful driver could do;
  cc-level   otherwise: the careful-driver benchmark decides.
Geometry (highway-env): the lane marking lies midway between the two lane centres; the front-wheel outer edge is the
body side at the front axle, 1.4 m ahead of the centre (heading included); ticks are linearly interpolated.
Choices that can only add 'mandated' labels (conservative for the conflict count): the tyre edge is the body side; a
cut-in already moving laterally when the episode starts (highD initial state) is visible from before the episode
(b_lenient, primary; b_strict counts from t = 0). (a): |longitudinal acceleration| <= 0.5 m/s^2 (sensitivity 0.25,
1.0) from the lateral onset (or t = 0) until contact or until the ALKS stops closing, and slower than the ALKS at the
intrusion point. The ALKS is the vehicle of the episode itself (the CC in part A, the SUT in parts B and C).

  python -m fairtest.b33_mandate grid OUT.jsonl --targets cc_r157:0,following:0 [--partition development] --workers W
  python -m fairtest.b33_mandate jobs OUT.jsonl [--final DIR --dataset DS --refine E3.jsonl --method M]
                                               [--baselines EVAL.jsonl ...] --workers W
  python -m fairtest.b33_mandate probe OUT.jsonl [--partition development] --workers W
probe (added after the design, reported as such): the constant-speed plans (pre = post = 0) with the entry time on a
0.05 s grid (0-8 s) and both duration settings, so that the (c) boundary is sampled densely -- the coarse B7 grid
steps the entry by 0.5-1 s.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os

from fairtest import devboot  # noqa: F401
from fairtest.cc_target import ORIGIN_LANE_Y

DT = .1
INTRUSION_M = .3     # the line 0.3 m beyond the lane marking
FRONT_AXLE_M = 1.4   # front axle ahead of the body centre
VISIBLE_S = .72      # (b)
ONSET_VY = .1        # lateral movement counts as visible from this lateral speed (m/s)
ACC_TOLS = (.25, .5, 1.)


def _edge(g, dirn):
    """Signed lateral position (toward the ALKS lane) of the outer edge of the front wheel closest to it."""
    h = g['heading']
    return dirn * (g['y'] + FRONT_AXLE_M * math.sin(h)) + g['width'] / 2 * math.cos(h)


def labels(case, trace):
    g0, t0 = case['physical_initial_state']
    dirn = 1. if t0['y'] > ORIGIN_LANE_Y else -1.
    line = dirn * (ORIGIN_LANE_Y + t0['y']) / 2 + INTRUSION_M
    states = [(0., g0, t0, None)] + [((k + 1) * DT, r['generator'], r['target'], r['acceleration'][0])
                                      for k, r in enumerate(trace)]
    moving0 = dirn * g0['vy'] >= ONSET_VY
    onset = next((t for t, g, _, _ in states if dirn * g['vy'] >= ONSET_VY), None)
    onset_lenient, onset_strict = (-math.inf, 0.) if moving0 else (onset, onset)
    hit = None
    if _edge(g0, dirn) >= line:
        hit = (0., g0, t0)
    else:
        for (ta, ga, xa, _), (tb, gb, xb, _) in zip(states, states[1:]):
            sa, sb = _edge(ga, dirn) - line, _edge(gb, dirn) - line
            if sa < 0 <= sb:
                f = -sa / (sb - sa)
                mix = lambda a, b: {k: a[k] + (b[k] - a[k]) * f for k in ('x', 'vx', 'length')}  # noqa: E731
                hit = (ta + f * DT, mix(ga, gb), mix(xa, xb))
                break
    if hit is None:
        return dict(intrusion=False, mandated=False, mandated_strict=False, mandated_loose=False)
    t_int, g, x = hit
    gap = g['x'] - x['x'] - (g['length'] + x['length']) / 2
    v_rel = x['vx'] - g['vx']
    c_ok = gap > 0 and (v_rel <= 0 or gap > .35 * v_rel + v_rel * v_rel / 12.)
    faster = v_rel > 0
    b_len = onset_lenient is not None and t_int - onset_lenient >= VISIBLE_S - 1e-9
    b_strict = onset_strict is not None and t_int - onset_strict >= VISIBLE_S - 1e-9
    start = max(0., onset_strict if onset_strict is not None else 0.)
    end = next((t for t, gg, xx, _ in states if t > t_int and xx['vx'] - gg['vx'] <= 0), states[-1][0])
    accs = [abs(a) for t, _, _, a in states if a is not None and start < t <= end + 1e-9]
    amax = max(accs) if accs else 0.
    a_ok = {tol: amax <= tol + 1e-9 for tol in ACC_TOLS}
    return dict(intrusion=True, t_int=round(t_int, 3), gap_int=round(gap, 3), v_rel_int=round(v_rel, 3),
                v_alks_int=round(x['vx'], 3), thw_int=round(gap / x['vx'], 3) if x['vx'] > 0 else None,  # B37
                ttc_int=round(gap / v_rel, 3) if faster and gap > 0 else None, c_ok=c_ok, faster=faster,
                b_lenient=b_len, b_strict=b_strict, moving_at_start=moving0, adv_max_abs_acc=round(amax, 3),
                mandated=a_ok[.5] and faster and b_len and c_ok,
                mandated_strict=a_ok[.25] and faster and b_strict and c_ok,
                mandated_loose=a_ok[1.] and faster and b_len and c_ok)


def grid_task(job):
    case_id, spec, target, h = job
    from fairtest import runner
    runner._init()
    case = runner._STATE['cases'][case_id]
    try:
        row = runner.run_episode(case, runner.make_policy(spec), target, h, keep_trace=True)
        return dict(case_id=case_id, spec=spec, target=target, H=h, contact=bool(row['pair_contact']),
                    valid=bool(row['maneuver_valid_precontact']), **labels(case, row['trace']))
    except Exception as e:  # recorded, never dropped
        return dict(case_id=case_id, spec=spec, target=target, H=h, error=f'{type(e).__name__}: {e}')


def job_task(job):
    source, stage, case_id, sut, how, payload, stored_cc = job
    from fairtest import runner
    from fairtest.b29_consensus import _run
    runner._init()
    case = runner._STATE['cases'][case_id]
    name, h = sut.rsplit('_H', 1)
    try:
        raw = _run(case, how, payload, name, float(h))
        return dict(source=source, stage=stage, case_id=case_id, sut=sut, cc_contact=stored_cc,
                    sut_contact=bool(raw['pair_contact']), **labels(case, raw['trace']))
    except Exception as e:  # recorded, never dropped
        return dict(source=source, stage=stage, case_id=case_id, sut=sut, error=f'{type(e).__name__}: {e}')


def _pool(fn, jobs, out, workers):
    from fairtest import runner
    with mp.Pool(workers, initializer=runner._init) as pool, open(out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(fn, jobs, chunksize=4):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(jobs=len(jobs))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('grid', 'jobs', 'probe'))
    ap.add_argument('out')
    ap.add_argument('--targets', default='cc_r157:0')
    ap.add_argument('--partition', default='all')
    ap.add_argument('--final')
    ap.add_argument('--dataset')
    ap.add_argument('--refine', nargs='*', default=[])
    ap.add_argument('--method', default='bo_warm_g4g')
    ap.add_argument('--baselines', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.command in ('grid', 'probe'):
        from fairtest import runner
        runner._init()
        ids = sorted(c for c, case in runner._STATE['cases'].items()
                     if a.partition == 'all' or case['event']['partition'] == a.partition)
        specs = runner.policy_specs('grid') if a.command == 'grid' else \
            [(round(.05 * k, 2), 0., 0., d) for k in range(161) for d in (None, 4.)]
        jobs = [(c, s, t, h) for c in ids for s in specs for (t, h) in runner.parse_targets(a.targets)]
        _pool(grid_task, jobs, a.out, a.workers)
    else:
        from fairtest.b29_consensus import jobs_from
        from fairtest.b32_aeb_reeval import baseline_jobs
        jobs = [j + (False,) for j in jobs_from(a.final, a.dataset, a.refine, a.method, True)] if a.final else []
        jobs += baseline_jobs(a.baselines)
        _pool(job_task, jobs, a.out, a.workers)
