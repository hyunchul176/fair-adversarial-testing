"""B14 (post-B13 development): realism-constrained searches — does naturalism imply fairness?

Both searches use the b11/b13 plan space, bounds, CMA-ES settings and SUT budget. Realism of a plan is
scored on the CC R157 run of the same plan against the highD natural cut-in envelope
(max exceedance beyond p1..p99 in IQR units; the grid methods' convention). Violation
v = 0 if the CC run is a cut-in with exceedance <= 1, else min(4, exceedance - 1) (4 when unscorable).

  falsify_real  collision-seeking with a naturalism prior (plausibility-constrained falsification):
                f = SUT gap + 10 v; hit = valid SUT contact with v = 0; start at the box centre.
  fair_real     fairness- and realism-constrained adversary:
                f = SUT gap + 20 [CC contact] - 5 [fair] + 10 v; hit = fair failure with v = 0;
                start at the grid R157-boundary plan within the realism limit (warm_starts_real.json,
                0 SUT queries), else the continuous CC boundary, else the grid boundary plan.
Hits are classified post hoc against every reference driver including cc_r157_strict.

  python -m fairtest.b14_realism_search run OUT.jsonl --workers W [--b11 B11.jsonl] [--suts ...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import HIGH, LOW, evaluate, gap, load_warm_starts, warm_start
from fairtest.b13_falsify import boundary_starts
from fairtest.features import Realism

CC = ('cc_r157', 0.)
REFS = (('cc_r157', 0.), ('cc_r157_strict', 0.), ('ref_fsm', 0.), ('ref_rss', 0.))
_REALISM = {}


def realism():
    if 'r' not in _REALISM:
        path = Path(os.environ.get('FAIRTEST_DATA', '')) / 'natural_reference.json'
        rows = json.loads(path.read_text(encoding='utf-8'))['rows'] if path.exists() else None
        _REALISM['r'] = Realism(natural=rows)
    return _REALISM['r']


def violation(cc_row):
    feats = cc_row.get('features') or {}
    if not feats.get('cut_in'):
        return 4., None
    exc = realism().score(feats)['max_exceedance']
    if exc is None:
        return 4., None
    return (0. if exc <= 1. else min(4., exc - 1.)), float(exc)


def search(case, target, h, x0, budget, mode, sigma=.8):
    import cma
    scale = HIGH - LOW
    es = cma.CMAEvolutionStrategy(list(np.clip((np.asarray(x0) - LOW) / scale, 0, 1)), sigma / 4.,
                                  dict(bounds=[0, 1], popsize=6, seed=int(case['case_id']) + 1, verbose=-9))
    best, n = None, 0
    while not es.stop():
        if n >= budget:
            break
        xs, fs = es.ask(), []
        for x in xs:
            theta = LOW + np.asarray(x) * scale
            cc = evaluate(case, theta, *CC)
            row = evaluate(case, theta, target, h)
            n += 1
            v, exc = violation(cc)
            contact = bool(row['valid'] and row['contact'])
            fair = contact and not cc['contact']
            if mode == 'falsify_real':
                f, hit = gap(row) + 10. * v, contact and v == 0.
            else:
                f, hit = gap(row) + (20. if cc['contact'] else 0.) - (5. if fair else 0.) + 10. * v, fair and v == 0.
            fs.append(f)
            if best is None or f < best['f']:
                best = dict(f=f, theta=theta.tolist(), hit=hit, contact=contact, cc_contact=bool(cc['contact']),
                            exceedance=exc, violation=v)
            if hit:
                return best, n
        es.tell(xs, fs)
    return best, n


def classify(case, theta):
    return {name: bool(evaluate(case, theta, name, h)['contact']) for name, h in REFS}


def task(args):
    case_id, suts, budget, starts = args
    runner._init()
    case = runner._STATE['cases'][case_id]
    t0 = time.time()
    out = dict(case_id=case_id, event_id=case['event']['event_id'], recording=case['event']['recording'], runs={})
    try:
        for mode in ('falsify_real', 'fair_real'):
            for name, h in suts:
                if mode == 'falsify_real':
                    x0 = (LOW + HIGH) / 2
                elif case_id in runner._STATE.get('warm_real', {}):
                    e, pre, post, dur = runner._STATE['warm_real'][case_id]
                    x0 = np.array([min(8., e), pre, post, post, 2. if dur is None else dur])
                else:
                    x0 = np.asarray(starts.get(case_id, warm_start(case_id)))
                best, n = search(case, name, h, x0, budget, mode)
                rec = dict(best=best, sut_queries=n)
                if best is not None and best['hit']:
                    rec['refs'] = classify(case, best['theta'])
                out['runs'][f'{mode}|{name}_H{h:g}'] = rec
    except Exception as e:  # recorded, never dropped
        out['error'] = f'{type(e).__name__}: {e}'
    out['seconds'] = round(time.time() - t0, 1)
    return out


def load_real_starts():
    path = Path(os.environ.get('FAIRTEST_DATA', '')) / 'warm_starts_real.json'
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding='utf-8')).items()}
    return {}


def _init_worker():
    runner._init()
    runner._STATE['warm'] = load_warm_starts()
    runner._STATE['warm_real'] = load_real_starts()
    realism()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('out')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--cases', type=int)
    p.add_argument('--budget', type=int, default=36)
    p.add_argument('--suts', default='following:0,following:0.5,following:1,reference_idm:0,following_aeb:0,following_aeb:0.5,rss_sut:0')
    p.add_argument('--b11')
    a = ap.parse_args()
    runner._init()
    ids = sorted(runner._STATE['cases'])[:a.cases] if a.cases else sorted(runner._STATE['cases'])
    suts = [(s.split(':')[0], float(s.split(':')[1])) for s in a.suts.split(',')]
    starts = boundary_starts(a.b11)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, [(c, suts, a.budget, starts) for c in ids])):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 10 == 0:
                print(json.dumps(dict(done=i + 1, of=len(ids), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(ids), elapsed_s=round(time.time() - t0, 1))), flush=True)
