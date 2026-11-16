"""B79 (B79 design, fixed before the run, available on request): can the screened
refinement attack by overtaking, a follow-up question of 2026-10-10 (the agent's attacks start ahead of the follower and slow
down; could it accelerate from behind and cut in?).

The contract is that of B75/B78 (bo_warm_g4v of b75_variants: G2 1.0, joint q99, CC margin 0.5 m, no severity floor).
Three switches, each recorded on every output line:
  --case-pool FILE   replace the development cases by the cases of FILE (b79_cases.py: the same highD events started
                     earlier, with the cut-in vehicle behind the follower); without a boundary plan the search starts cold.
  --acc-high A       upper bound of the plan's three accelerations (b11_search.HIGH[1:4], stored 1.0 m/s^2). The realism
                     contract (G1, G2, joint support on the CC replay) still applies to every executed plan.
  --mode overtake    one more admission condition on the CC replay: the cut-in vehicle is at least as fast as the follower
                     at the 50% lateral crossing (closing_p50 <= 0); penalty 5 x closing_p50 (10 without a cut-in).
  --method bo|random screened GP-BO (bo_warm_g4s; cold without a boundary plan) or screened random search (random_g4s, B61).
  --acc-pct P        (design amendment 2026-10-10 10:10, before these runs) B78's braking bound on the CC replay as one more
                     admission condition: acc_pre and acc_post at least the natural highD P-th percentile (b78_variants).
b23_search, b11_search and b75_variants are not edited: each worker replaces b23_search.admission and _query_log and sets the
shared bound array in place. Without switches the method is b75_variants' with the contract values (pre-check against the
stored B56 runs; --method random against the stored B61 runs).

  python -m fairtest.b79_variants OUT.jsonl --cont STORED_B11.jsonl --suts following:0,... --only following_H0,... \
         [--case-pool FILE] [--acc-high 2] [--mode overtake] [--method bo|random] [--workers 24] [--cases N] [--log-queries]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from fairtest import devboot  # noqa: F401
from fairtest import b11_search
from fairtest import b23_search as B
from fairtest import b75_variants as V
from fairtest import b78_variants as N
from fairtest import runner

OPT = dict(mode=None, acc_high=1.0)


def mode_penalty(features):
    if OPT['mode'] != 'overtake':
        return 0.
    if not features.get('cut_in') or features.get('closing_p50') is None:
        return 10.
    return 5. * max(0., float(features['closing_p50']))


def admission_b79(case, theta):
    ok, info = V.admission_v(case, theta)
    extra = mode_penalty(info['cc']['features']) + N.shortfall(info['cc']['features'])
    info['mode_pen'] = extra
    info['pen'] = info['pen'] + extra
    return bool(ok and extra == 0.), info


def _query_log_b79(n, row, cc, success, replays=None):
    out = V._query_log_v(n, row, cc, success, replays)
    feats = cc.get('features') or {}
    out['closing_p50'], out['acc_pre'], out['acc_post'] = feats.get('closing_p50'), feats.get('acc_pre'), feats.get('acc_post')
    out['admissible'] = bool(out['admissible'] and mode_penalty(feats) == 0. and N.shortfall(feats) == 0.)
    return out


def _init_worker(cfg, opt, pool_file):
    V._init_worker(cfg)
    OPT.update(opt)
    N.BOUND.update(opt.get('bound') or dict(pct=None, pre=None, post=None))
    b11_search.HIGH[1:4] = opt['acc_high']  # the array shared by b11_search and b23_search
    B.admission = admission_b79
    B._query_log = _query_log_b79
    if pool_file:
        runner._init()
        cases = json.loads(Path(pool_file).read_text(encoding='utf-8'))['cases']
        runner._STATE['cases'].update({c['case_id']: c for c in cases})


def task(args):
    out = V.task(args)
    out['cfg'] = dict(out['cfg'], acc_high=OPT['acc_high'], mode=OPT['mode'], pool=OPT.get('pool'), method=OPT.get('method'),
                      acc_pct=(OPT.get('bound') or {}).get('pct'), acc_min_pre=(OPT.get('bound') or {}).get('pre'),
                      acc_min_post=(OPT.get('bound') or {}).get('post'))
    return out


def main(a):
    runner._init()
    cfg = dict(g2=1., q=99., delta=.5, v_min=-1000., tau=V._tau(99.))
    opt = dict(mode=a.mode, acc_high=a.acc_high, pool=Path(a.case_pool).name if a.case_pool else None, method=a.method, bound=None)
    if a.acc_pct is not None:
        pre, post = N.natural_bounds(a.acc_pct)
        opt['bound'] = dict(pct=a.acc_pct, pre=pre, post=post)
    if a.case_pool:
        cases = json.loads(Path(a.case_pool).read_text(encoding='utf-8'))['cases']
        ids = sorted(c['case_id'] for c in cases)
        boundary = {}
    else:
        ids = sorted(c for c, v in runner._STATE['cases'].items() if v['event'].get('partition') == 'development')
        boundary = {}
        for line in open(a.cont, encoding='utf-8'):
            r = json.loads(line)
            b = (r.get('cc_boundary') or {}).get('best')
            boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = ids[:a.cases] if a.cases else ids
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    keys = [f'{n}_H{h:g}' for n, h in suts]
    only = set(a.only.split(',')) if a.only else set(keys)
    method = ('bo_warm_g4s',) if a.method == 'bo' else ('random_g4s',)
    jobs = [(c, boundary.get(c), suts, method, 36, frozenset({k}), a.log_queries) for c in ids for k in keys if k in only]
    print(json.dumps(dict(jobs=len(jobs), cfg=cfg, opt=opt)), flush=True)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(cfg, opt, a.case_pool)) as pool, \
            open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 25 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), cfg=cfg, opt=opt, elapsed_s=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True)
    ap.add_argument('--suts', required=True, help='the 17 SUTs in the B46 order (the random streams keep their position)')
    ap.add_argument('--only', help='comma-separated SUT keys to run')
    ap.add_argument('--case-pool', help='b79_cases.py output (cases started from behind); default: the development cases')
    ap.add_argument('--acc-high', type=float, default=1.0, help='upper bound of the plan accelerations (stored 1.0)')
    ap.add_argument('--mode', choices=('overtake',), help='admission: cut-in vehicle not slower than the follower at the crossing')
    ap.add_argument('--method', choices=('bo', 'random'), default='bo')
    ap.add_argument('--acc-pct', type=float, help='B78 braking bound: natural percentile of acc_pre / acc_post (amendment)')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--cases', type=int, help='first N cases only (pre-check)')
    ap.add_argument('--log-queries', action='store_true')
    main(ap.parse_args())
