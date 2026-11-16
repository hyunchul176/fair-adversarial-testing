"""B13 addendum: generic falsification baseline (collision-seeking CMA-ES without R157 information).

Same plan parameterisation, bounds, optimiser settings (popsize 6, sigma .8/4, seed case_id+1) and
SUT-query budget as cont_search (b11_search), but
  objective  f = SUT minimum bumper gap while overlapping (contact = -1), nothing else;
  stop       at the first valid SUT contact;
  start      falsify_centre:   box centre (no reference-driver information, the literature baseline)
             falsify_boundary: the same start as cont_search (continuous CC boundary, else the grid
                               boundary plan), isolating the effect of the fairness term.
The collision plan is classified post hoc against every reference driver (CC R157 0.75/0.5/1.0 s,
FSM, RSS): fair if that reference avoids contact on the same plan, attacker-at-fault otherwise.

  python -m fairtest.b13_falsify run OUT.jsonl --workers W --suts following:0,... [--b11 B11.jsonl]
  python -m fairtest.b13_falsify analyze OUT.jsonl CASE_POOL.json RESULT.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import HIGH, LOW, evaluate, gap, load_warm_starts, warm_start

REFS = (('cc_r157', 0.), ('ref_fsm', 0.), ('ref_rss', 0.), ('cc_r157_react050', 0.), ('cc_r157_react100', 0.))
STARTS = ('falsify_centre', 'falsify_boundary')
BOOT, SEED = 10000, 20260926


def falsify(case, target, h, x0, budget, sigma=.8):
    """CMA-ES on the SUT gap only; budget checked per generation exactly as in b11_search.search."""
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
            row = evaluate(case, theta, target, h)
            n += 1
            f = gap(row)
            fs.append(f)
            hit = bool(row['valid'] and row['contact'])
            if best is None or f < best['f']:
                best = dict(f=f, theta=theta.tolist(), contact=hit, valid=bool(row['valid']),
                            min_gap=row.get('min_gap_overlap'), features=row['features'])
            if hit:
                return best, n
        es.tell(xs, fs)
    return best, n


def classify(case, theta):
    out = {}
    for name, h in REFS:
        r = evaluate(case, theta, name, h)
        out[name] = dict(contact=bool(r['contact']), valid=bool(r['valid']), features=r['features'] if name == 'cc_r157' else None)
    return out


def task(args):
    case_id, suts, budget, starts = args
    runner._init()
    case = runner._STATE['cases'][case_id]
    t0 = time.time()
    out = dict(case_id=case_id, event_id=case['event']['event_id'], recording=case['event']['recording'], runs={})
    try:
        for start in STARTS:
            for name, h in suts:
                key = f'{name}_H{h:g}'
                x0 = (LOW + HIGH) / 2 if start == 'falsify_centre' else np.asarray(starts.get(case_id, warm_start(case_id)))
                best, n = falsify(case, name, h, x0, budget)
                rec = dict(best=best, sut_queries=n)
                if best is not None and best['contact']:
                    rec['refs'] = classify(case, best['theta'])
                out['runs'][f'{start}|{key}'] = rec
    except Exception as e:  # recorded, never dropped
        out['error'] = f'{type(e).__name__}: {e}'
    out['seconds'] = round(time.time() - t0, 1)
    return out


def boundary_starts(b11_path):
    """cont_search start per case, exactly as b11_search.task: continuous CC boundary theta if any."""
    starts = {}
    if not b11_path:
        return starts
    for line in open(b11_path, encoding='utf-8'):
        r = json.loads(line)
        bnd = (r.get('cc_boundary') or {}).get('best')
        if bnd is not None:
            starts[r['case_id']] = bnd['theta']
    return starts


def _init_worker():
    runner._init()
    runner._STATE['warm'] = load_warm_starts()


def run(a):
    runner._init()
    ids = sorted(runner._STATE['cases'])
    if a.cases:
        ids = ids[:a.cases]
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


def outcomes(rows):
    """Per (start, SUT): list over cases of dicts with found / fair_<ref> / unfair_<ref> / consensus_fair."""
    per = {}
    for r in rows:
        if r.get('error'):
            raise RuntimeError(f"error row case {r['case_id']}: {r['error']}")
        for k, rec in r['runs'].items():
            best = rec['best']
            found = bool(best and best['contact'])
            o = dict(case_id=r['case_id'], recording=r['recording'], found=float(found), queries=rec['sut_queries'])
            for name, _ in REFS:
                ref = (rec.get('refs') or {}).get(name)
                o[f'fair_{name}'] = float(found and not ref['contact']) if found else 0.
                o[f'unfair_{name}'] = float(found and ref['contact']) if found else 0.
            o['consensus_fair'] = float(found and not any(rec['refs'][n]['contact'] for n in ('cc_r157', 'ref_fsm', 'ref_rss'))) if found else 0.
            o['features'] = best['features'] if found else None
            per.setdefault(k, []).append(o)
    return per


def bootstrap(vals):
    rng = random.Random(SEED)
    v = np.asarray(vals, float)
    n = len(v)
    if n == 0:
        return dict(n=0)
    idx = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(BOOT)])
    d = v[idx].mean(axis=1)
    return dict(n=n, mean=float(v.mean()), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


def analyze(rows, realism=None):
    per = outcomes(rows)
    rep = {}
    for k, os_ in sorted(per.items()):
        found = [o['found'] for o in os_]
        n_found = int(sum(found))
        item = dict(n=len(os_), collision_rate=bootstrap(found), mean_queries=float(np.mean([o['queries'] for o in os_])))
        for name, _ in REFS:
            item[f'fair_rate_{name}'] = float(np.mean([o[f'fair_{name}'] for o in os_]))
            item[f'unfair_rate_{name}'] = float(np.mean([o[f'unfair_{name}'] for o in os_]))
            item[f'attacker_at_fault_share_{name}'] = (sum(o[f'unfair_{name}'] for o in os_) / n_found) if n_found else None
        item['consensus_fair_rate'] = float(np.mean([o['consensus_fair'] for o in os_]))
        item['unfair_rate_cc_ci95'] = bootstrap([o['unfair_cc_r157'] for o in os_]).get('ci95')
        if realism is not None and n_found:
            exc = [realism.score(o['features'])['max_exceedance'] for o in os_ if o['found']]
            exc = [x for x in exc if x is not None]
            item['realism_max_exceedance_median'] = float(np.median(exc)) if exc else None
            item['realism_within_1iqr_share'] = float(np.mean([x <= 1. for x in exc])) if exc else None
        rep[k] = item
    rep['hypotheses'] = hypotheses(rep)
    return rep


def hypotheses(rep):
    v = {}
    r = rep.get('falsify_centre|rss_sut_H0')
    if r:
        v['H10_falsifier_reports_failures_on_rss_sut'] = r['collision_rate']['mean'] > 0
        v['H10b_all_rss_sut_falsifier_collisions_attacker_at_fault_cc'] = r['fair_rate_cc_r157'] == 0
    v['H11_attacker_at_fault_majority_competent_suts_cc'] = {
        k: rep[f'falsify_centre|{k}']['attacker_at_fault_share_cc_r157'] > .5
        for k in ('rss_sut_H0', 'following_aeb_H0', 'following_aeb_H0.5')
        if f'falsify_centre|{k}' in rep and rep[f'falsify_centre|{k}']['attacker_at_fault_share_cc_r157'] is not None}
    return v


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
    q = sub.add_parser('analyze')
    q.add_argument('rows')
    q.add_argument('out')
    a = ap.parse_args()
    if a.cmd == 'run':
        run(a)
    else:
        from fairtest.features import Realism
        rows = [json.loads(l) for l in open(a.rows, encoding='utf-8')]
        rep = analyze(rows, Realism())
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
        print(json.dumps(rep['hypotheses'], indent=1))
