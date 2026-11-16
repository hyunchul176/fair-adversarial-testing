"""B12: does fair adversarial testing give better design guidance than unconstrained testing?

Designer's problem: add AEB + anticipation to the weak `following` H0 SUT. Tunable software
parameters: anticipation horizon H in {0, .5, 1, 1.5} s and AEB trigger TTC threshold tau in
{1.0, 1.5, 2.0, 2.5, 3.0} s (AEB capability fixed at 8 m/s^2, 15 m/s^3) -> 20 configurations.
Scenario sources per exposed case (all SUT-independent, fixed before tuning):
  NAT    natural continuation of the recorded lane change, 4 s, no braking (plan_e0_pre0_post0_d4)
  FAIR   the grid R157-boundary plan (hardest CC-preventable plan; chosen from CC runs only)
  UNFAIR the frozen B5 policy (mostly attacker-at-fault collisions, B7/B9)
Tuning on calibration cases: pick the configuration minimising collisions on the tuning source
subject to a natural false-alarm rate <= 5% (false alarm: applied braking < -3 m/s^2 in NAT);
ties -> fewer false alarms -> smaller (tau, H). Evaluation on development cases.

  python -m fairtest.b12_tuning run OUT.jsonl --workers W      (simulate all configurations)
  python -m fairtest.b12_tuning analyse OUT.jsonl REPORT.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.plans import FrozenB5, Plan

HS = (0., .5, 1., 1.5)
TAUS = (1., 1.5, 2., 2.5, 3.)
CONFIGS = [(h, tau) for h in HS for tau in TAUS]
NAT = Plan(0., 0., 0., 4.)
FALSE_ALARM_MPS2 = -3.
FA_BUDGET = .05


def warm():
    path = Path(os.environ.get('FAIRTEST_DATA', Path(__file__).parent / 'data')) / 'warm_starts.json'
    return {int(k): v for k, v in json.loads(path.read_text(encoding='utf-8')).items()}


def job(args):
    case_id, source, spec, h, tau = args
    runner._init()
    case = runner._STATE['cases'][case_id]
    if source == 'NAT':
        policy = NAT
    elif source == 'UNFAIR':
        policy = FrozenB5()
    else:
        e, pre, post, dur = spec
        policy = Plan(math.inf if e >= 99 else e, pre, post, dur)
    target = f'following_aeb@{tau:g}' if tau is not None else 'following'
    try:
        r = runner.run_episode(case, policy, target, h)
        return dict(case_id=case_id, partition=case['event']['partition'], source=source, H=h, tau=tau,
                    contact=bool(r['valid'] and r['contact']), min_acc=r['target_min_acc'],
                    min_gap=r['min_gap_overlap'], valid=r['valid'])
    except Exception as ex:
        return dict(case_id=case_id, source=source, H=h, tau=tau, error=f'{type(ex).__name__}: {ex}')


def run(out, workers):
    runner._init()
    ws = warm()
    jobs = []
    for c in sorted(runner._STATE['cases']):
        for h, tau in CONFIGS + [(0., None)]:  # (0, None): base SUT without AEB
            jobs.append((c, 'NAT', None, h, tau))
            jobs.append((c, 'UNFAIR', None, h, tau))
            if c in ws:
                jobs.append((c, 'FAIR', ws[c], h, tau))
    t0 = time.time()
    with mp.Pool(workers, initializer=runner._init) as pool, open(out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            f.write(json.dumps(r) + '\n')
            if (i + 1) % 2000 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), s=round(time.time() - t0, 1))), flush=True)


def analyse(path, out):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    tab = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        tab[(r['H'], r['tau'])][(r['partition'], r['source'])].append(r)

    def rate(cfg, part, source, fn):
        v = tab[cfg][(part, source)]
        return sum(fn(r) for r in v) / len(v) if v else float('nan')

    coll = lambda r: r['contact']  # noqa: E731
    fa = lambda r: r['min_acc'] < FALSE_ALARM_MPS2  # noqa: E731
    grid = {}
    for cfg in CONFIGS + [(0., None)]:
        grid[f'H{cfg[0]:g}_tau{cfg[1]}'] = {f'{p}_{s}': round(100 * rate(cfg, p, s, coll if s != 'NAT' else fa), 1)
                                            for p in ('calibration', 'development') for s in ('NAT', 'FAIR', 'UNFAIR')}
    choice = {}
    for source in ('FAIR', 'UNFAIR', 'NAT'):
        feasible = [c for c in CONFIGS if rate(c, 'calibration', 'NAT', fa) <= FA_BUDGET]
        if source == 'NAT':
            key = lambda c: (rate(c, 'calibration', 'NAT', fa), c[1], c[0])  # noqa: E731
        else:
            key = lambda c, s=source: (rate(c, 'calibration', s, coll), rate(c, 'calibration', 'NAT', fa), c[1], c[0])  # noqa: E731
        best = min(feasible or CONFIGS, key=key)
        choice[f'tuned_on_{source}'] = dict(config=dict(H=best[0], tau=best[1]),
                                            test_fair_collision=round(100 * rate(best, 'development', 'FAIR', coll), 1),
                                            test_unfair_collision=round(100 * rate(best, 'development', 'UNFAIR', coll), 1),
                                            test_false_alarm=round(100 * rate(best, 'development', 'NAT', fa), 1),
                                            train_objective=round(100 * key(best)[0], 1))
    base = (0., None)
    choice['base_no_aeb'] = dict(test_fair_collision=round(100 * rate(base, 'development', 'FAIR', coll), 1),
                                 test_unfair_collision=round(100 * rate(base, 'development', 'UNFAIR', coll), 1),
                                 test_false_alarm=round(100 * rate(base, 'development', 'NAT', fa), 1))
    rep = dict(errors=len(errors), rows=len(rows), choice=choice, grid=grid, false_alarm_budget=FA_BUDGET)
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(dict(choice=choice, errors=len(errors)), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('run', 'analyse'))
    ap.add_argument('path')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.command == 'run':
        run(a.path, a.workers)
    else:
        analyse(a.path, a.out)
