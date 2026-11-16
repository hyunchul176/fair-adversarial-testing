"""B40 (design fixed before the run): margin-escalating refinement -- the v2 refinement, then the remaining budget spent
on failures that a slower careful driver still avoids.

Stage 1 is the v2 refinement unchanged (bo_warm_g4g: boundary warm start, robust objective, guards, the b23 random
stream), so its first success must reproduce the stored v2 result. After a success at reference reaction r, the search
continues with the next reaction level (1.0, 1.25, 1.5, 2.0 s), warm-started at the last success, until the 36-query
budget is spent. The reported failure is the success at the highest level reached (fallback: the stage-1 failure, so
detection is unchanged by construction). Its reaction margin r_up is then measured on the B36 grid (the CC avoids
at every grid value up to r_up).

  python -m fairtest.b40_margin_search OUT.jsonl --cont STORED_B11.jsonl --partition development --suts ... --workers W
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
from fairtest import b23_search as S
from fairtest.b36_reaction_margin import UP

LEVELS = (.75, 1., 1.25, 1.5, 2.)


def _ref(r):
    return ('cc_r157', 0.) if r == .75 else (f'cc_r157@r{r:g}', 0.)


def r_up(case, theta):
    from fairtest.b11_search import ContinuousPlan
    best = .75
    for r in UP:
        raw = runner.run_episode(case, ContinuousPlan.from_theta(theta), f'cc_r157@r{r:g}', 0., keep_trace=False)
        if raw['contact']:
            break
        best = r
    return best


def escalate(case, name, h, budget, rng, x_boundary):
    S.CC = _ref(.75)  # b23 _objective_g4 reads the module-level reference
    best, n = S.bo_search(case, name, h, budget, rng, x_first=x_boundary, robust=True, guarded=True)
    out = dict(stage1=dict(best=best, first_fair=n if best and best['fair'] else None))
    if not (best and best['fair']):
        return out
    used, level, cur = n, 0, best
    trail = [dict(level=.75, queries=n)]
    while used < budget and level + 1 < len(LEVELS):
        S.CC = _ref(LEVELS[level + 1])
        b, m = S.bo_search(case, name, h, budget - used, rng, x_first=cur['theta'], robust=True, guarded=True)
        used += m
        if not (b and b['fair']):
            break
        level, cur = level + 1, b
        trail.append(dict(level=LEVELS[level], queries=m))
    S.CC = _ref(.75)
    out.update(final=cur, level_reached=LEVELS[level], queries_used=used, trail=trail)
    return out


def task(args):
    case_id, x_boundary, suts, budget = args
    runner._init()
    case = runner._STATE['cases'][case_id]
    out = dict(case_id=case_id, runs={})
    t0 = time.time()
    for si, (name, h) in enumerate(suts):
        key = f'{name}_H{h:g}'
        rng = np.random.default_rng([case_id, si, S.METHODS.index('bo_warm')])  # the v2 stream (bo_warm_g4g)
        try:
            rec = escalate(case, name, h, budget, rng, x_boundary)
            if 'final' in rec:
                rec['r_up_stage1'] = r_up(case, rec['stage1']['best']['theta'])
                rec['r_up_final'] = r_up(case, rec['final']['theta'])
            out['runs'][key] = rec
        except Exception as e:  # recorded, never dropped
            out['runs'][key] = dict(error=f'{type(e).__name__}: {e}')
    out['seconds'] = round(time.time() - t0, 1)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True)
    ap.add_argument('--partition', default='development')
    ap.add_argument('--suts', required=True)
    ap.add_argument('--budget', type=int, default=36)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    runner._init()
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(task, [(c, boundary.get(c), suts, a.budget) for c in ids]):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(cases=len(ids))), flush=True)
