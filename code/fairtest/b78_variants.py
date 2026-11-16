"""B78 (B78 design, fixed before the run, available on request): the final agent's
screened refinement re-run with a bound on how hard the cut-in vehicle may slow down, a follow-up question of 2026-10-09
(the agent's cut-ins slow down while cutting in, which natural cut-ins rarely do).

bo_warm_g4v of b75_variants (contract G2 1.0, joint q99, CC margin 0.5 m, no severity floor) with one more admission
condition on the CC replay: the cut-in vehicle's mean acceleration in the 3 s before (acc_pre) and the 2 s after (acc_post)
the 50% lateral crossing must be at least the natural highD p-th percentile of that feature (features.natural_reference(),
the 463 eligible cut-ins of the calibration and development partitions; p = 1 gives -1.12 / -1.26 m/s^2, p = 5 gives
-0.47 / -0.49 m/s^2). A plan below the bound is inadmissible (penalty 5 x shortfall per feature, 10 without a cut-in), so it
is never executed against the SUT. b23_search and b75_variants are not edited: each worker replaces b23_search.admission and
_query_log. Without --acc-pct the method is b75_variants' with the contract values (pre-check against the stored B56 runs).
Output keys are bo_warm_g4v (as B75) and every line records the parameters, including the bounds.

  python -m fairtest.b78_variants OUT.jsonl --cont STORED_B11.jsonl --suts following:0,... --only following_H0,... \
         [--acc-pct 1|5] [--workers 24] [--cases N] [--log-queries]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import b23_search as B
from fairtest import b75_variants as V
from fairtest import runner

BOUND = dict(pct=None, pre=None, post=None)


def natural_bounds(pct):
    from fairtest.features import natural_reference
    nat = natural_reference()
    pre = np.array([r['acc_pre'] for r in nat if r.get('acc_pre') is not None], float)
    post = np.array([r['acc_post'] for r in nat if r.get('acc_post') is not None], float)
    return float(np.percentile(pre, pct)), float(np.percentile(post, pct))


def shortfall(features):
    """Penalty of the braking bound (0 when admissible or when no bound is set)."""
    if BOUND['pct'] is None:
        return 0.
    if not features.get('cut_in'):
        return 10.
    pen = 0.
    for key, lo in (('acc_pre', BOUND['pre']), ('acc_post', BOUND['post'])):
        x = features.get(key)
        pen += 10. if x is None else 5. * max(0., lo - x)
    return pen


def admission_b78(case, theta):
    ok, info = V.admission_v(case, theta)
    extra = shortfall(info['cc']['features'])
    info['braking_pen'] = extra
    info['pen'] = info['pen'] + extra
    return bool(ok and extra == 0.), info


def _query_log_b78(n, row, cc, success, replays=None):
    out = V._query_log_v(n, row, cc, success, replays)
    feats = cc.get('features') or {}
    out['acc_pre'], out['acc_post'] = feats.get('acc_pre'), feats.get('acc_post')
    out['admissible'] = bool(out['admissible'] and shortfall(feats) == 0.)
    return out


def _init_worker(cfg, bound):
    V._init_worker(cfg)
    BOUND.update(bound)
    B.admission = admission_b78
    B._query_log = _query_log_b78


def task(args):
    out = V.task(args)
    out['cfg'] = dict(out['cfg'], acc_pct=BOUND['pct'], acc_min_pre=BOUND['pre'], acc_min_post=BOUND['post'])
    return out


def main(a):
    runner._init()
    cfg = dict(g2=1., q=99., delta=.5, v_min=-1000., tau=V._tau(99.))
    bound = dict(pct=None, pre=None, post=None)
    if a.acc_pct is not None:
        pre, post = natural_bounds(a.acc_pct)
        bound = dict(pct=a.acc_pct, pre=pre, post=post)
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if v['event'].get('partition') == 'development')
    ids = ids[:a.cases] if a.cases else ids
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    keys = [f'{n}_H{h:g}' for n, h in suts]
    only = set(a.only.split(',')) if a.only else set(keys)
    jobs = [(c, boundary.get(c), suts, ('bo_warm_g4s',), 36, frozenset({k}), a.log_queries) for c in ids for k in keys if k in only]
    print(json.dumps(dict(jobs=len(jobs), cfg=cfg, bound=bound)), flush=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(cfg, bound)) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 25 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), cfg=cfg, bound=bound, elapsed_s=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True)
    ap.add_argument('--suts', required=True, help='the 17 SUTs in the B46 order (the random streams keep their position)')
    ap.add_argument('--only', help='comma-separated SUT keys to run')
    ap.add_argument('--acc-pct', type=float, help='natural percentile of the braking bound (none: contract only)')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--cases', type=int, help='first N cases only (pre-check)')
    ap.add_argument('--log-queries', action='store_true')
    main(ap.parse_args())
