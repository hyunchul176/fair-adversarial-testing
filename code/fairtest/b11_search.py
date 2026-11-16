"""B11: continuous adversary search beyond the 199-plan grid (development, exposed cases).

Continuous plan theta = (entry_s in [0,8], pre in [-3,1], post in [-3,1], post2 in [-3,1],
lateral duration in [2,6] s); post2 replaces post 2 s after entry (brake-then-release shapes).
Decisions stay at 2 Hz, so entry time is effective at 0.5 s resolution.

Two searches per case, both warm-started at the grid R157-boundary plan when available:
  cc_boundary  (SUT-agnostic, 0 SUT queries): maximise CC demand subject to CC no-contact;
               the result is then evaluated once on every SUT.
  sut_search   (per SUT, budgeted SUT queries): CMA-ES on
               f = SUT min bumper gap (<=0 means contact) + 20*[CC contact] - 5*[fair failure],
               stopping at the first fair failure. Each evaluation = 1 SUT run + 1 CC run.
Realism features of the best plan are stored for later envelope analysis.

  python -m fairtest.b11_search OUT.jsonl --workers W [--shard k/n] [--suts following:1,following_aeb:0]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.boundary_planner import CC_MAX

LOW = np.array([0., -3., -3., -3., 2.])
HIGH = np.array([8., 1., 1., 1., 6.])
CC = (os.environ.get('FAIRTEST_JUDGE_CC', 'cc_r157'), 0.)  # B38: a stricter reference, e.g. cc_r157@r1.25 (unset = R157)


@dataclass(frozen=True)
class ContinuousPlan:
    entry_s: float
    pre_mps2: float
    post_mps2: float
    post2_mps2: float
    duration_s: float

    @classmethod
    def from_theta(cls, theta):
        t = np.clip(np.asarray(theta, float), LOW, HIGH)
        return cls(float(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4]))

    def durations(self):
        # at least duration_s: shortest feasible among duration_s and the longer B2 grid values
        d = float(self.duration_s)
        return tuple(sorted({d} | {g for g in (2., 3., 4., 5., 6.) if g > d + 1e-9}))

    def act(self, env):
        now = env.tick * .1 + 1e-9
        if now < self.entry_s:
            return (0., self.pre_mps2)
        return (4., self.post_mps2 if now < self.entry_s + 2. else self.post2_mps2)

    def to_json(self):
        return dict(name='continuous', entry_s=self.entry_s, pre_mps2=self.pre_mps2, post_mps2=self.post_mps2,
                    post2_mps2=self.post2_mps2, duration_s=self.duration_s)


def evaluate(case, theta, target, h=0.):
    return runner.run_episode(case, ContinuousPlan.from_theta(theta), target, h)


def demand(row):
    return min(1.5, (row.get('max_drac') or 0.) / CC_MAX)


def gap(row):
    g = row.get('min_gap_overlap')
    if row['contact']:
        return -1.
    return 50. if g is None else float(g)


def warm_start(case_id):
    """Grid boundary plan of the case from the B7 landscape index, if loaded; else box centre."""
    ws = runner._STATE.get('warm', {}).get(case_id)
    if ws is None:
        return (LOW + HIGH) / 2
    e, pre, post, dur = ws
    return np.array([min(8., e), pre, post, post, 2. if dur is None else dur])


def search(case, target, h, x0, budget, sigma=.8, fair_only=True):
    """CMA-ES in the unit box. target='cc_r157' runs the SUT-agnostic CC-demand search."""
    import cma
    scale = HIGH - LOW
    es = cma.CMAEvolutionStrategy(list(np.clip((np.asarray(x0) - LOW) / scale, 0, 1)), sigma / 4.,
                                  dict(bounds=[0, 1], popsize=6, seed=int(case['case_id']) + 1, verbose=-9))
    best, n_sut, n_cc = None, 0, 0
    while not es.stop():
        spent = n_cc if target == 'cc_r157' else n_sut
        if spent >= budget:
            break
        xs, fs = es.ask(), []
        for x in xs:
            theta = LOW + np.asarray(x) * scale
            cc = evaluate(case, theta, *CC)
            n_cc += 1
            if target == 'cc_r157':
                row, fair = cc, False
                f = -demand(cc) + (20. if cc['contact'] or not cc['valid'] else 0.)
            else:
                row = evaluate(case, theta, target, h)
                n_sut += 1
                fair = bool(row['valid'] and row['contact'] and not cc['contact'])
                f = gap(row) + (20. if cc['contact'] else 0.) - (5. if fair else 0.)
            fs.append(f)
            if best is None or f < best['f']:
                best = dict(f=f, theta=theta.tolist(), fair=fair, cc_contact=bool(cc['contact']),
                            sut_contact=bool(row['contact']), cc_demand=demand(cc), features=row['features'])
            if fair and fair_only:
                return best, n_sut, n_cc
        es.tell(xs, fs)
    return best, n_sut, n_cc


def task(args):
    case_id, suts, budget = args
    runner._init()
    case = runner._STATE['cases'][case_id]
    t0 = time.time()
    out = dict(case_id=case_id, event_id=case['event']['event_id'], partition=case['event']['partition'],
               recording=case['event']['recording'])
    try:
        x0 = warm_start(case_id)
        bnd, _, n_cc = search(case, 'cc_r157', 0., x0, budget, fair_only=False)
        out['cc_boundary'] = dict(best=bnd, cc_sims=n_cc)
        if bnd is not None and not bnd['cc_contact']:
            evals = {}
            for name, h in suts:
                r = evaluate(case, bnd['theta'], name, h)
                evals[f'{name}_H{h:g}'] = dict(contact=bool(r['contact'] and r['valid']), min_gap=r['min_gap_overlap'])
            out['cc_boundary']['sut_eval'] = evals
        out['sut_search'] = {}
        for name, h in suts:
            best, n_sut, n_cc = search(case, name, h, x0 if bnd is None else np.asarray(bnd['theta']), budget)
            out['sut_search'][f'{name}_H{h:g}'] = dict(best=best, sut_queries=n_sut, cc_sims=n_cc)
    except Exception as e:  # recorded, never dropped
        out['error'] = f'{type(e).__name__}: {e}'
    out['seconds'] = round(time.time() - t0, 1)
    return out


def load_warm_starts():
    """Grid boundary plan per case (CC-only information), shipped as warm_starts.json in the data dir."""
    path = Path(os.environ.get('FAIRTEST_DATA', '')) / 'warm_starts.json'
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding='utf-8')).items()}
    return {}


def _init_worker():
    runner._init()
    runner._STATE['warm'] = load_warm_starts()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--shard')
    ap.add_argument('--cases', type=int)
    ap.add_argument('--budget', type=int, default=36)
    ap.add_argument('--suts', default='following:1,following_aeb:0,following:0.5')
    a = ap.parse_args()
    runner._init()
    ids = sorted(runner._STATE['cases'])
    if a.shard:
        ks, n = a.shard.split('/')
        keep, n = {int(k) for k in ks.split('+')}, int(n)
        ids = [i for i in ids if i % n in keep]
    if a.cases:
        ids = ids[:a.cases]
    suts = [(s.split(':')[0], float(s.split(':')[1])) for s in a.suts.split(',')]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, [(c, suts, a.budget) for c in ids])):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 10 == 0:
                print(json.dumps(dict(done=i + 1, of=len(ids), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(ids), elapsed_s=round(time.time() - t0, 1))), flush=True)
