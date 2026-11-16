"""B75 and B76 (B75 and B76 designs, fixed before the runs, available on request): the final agent's screened
refinement bo_warm_g4s re-run with other contract parameters, review of manuscript v7 items R4 and R5.

bo_warm_g4v = b23_search's bo_warm_g4s (B56: admission screening against the CC, boundary start, random stream, budget 36,
stopping rules; run through b23_search.task) with four parameters set per run:
  --g2      G2 tolerance in IQR (contract 1.0)             admission: realism exceedance <= g2
  --q       joint-support percentile (contract 99)          admission: 5-NN score <= tau(q) (b63_realism_sensitivity.taus)
  --delta   CC clearance margin in metres (contract 0.5)    admission and robust success: CC clearance >= delta
  --v-min   severity floor in m/s (contract: none, -1000)   success also needs closing >= v_min; contacts below pay 2 x shortfall
b23_search is not edited: each worker process sets b23_search.DELTA and replaces admission, _objective_g4s, _record and
_query_log (the severity objective is b64_severity_screened's). With the contract values the method reproduces the stored
bo_warm_g4s runs (pre-check, as B64). Output keys are renamed bo_warm_g4s -> bo_warm_g4v and every line records the parameters.

  python -m fairtest.b75_variants OUT.jsonl --cont STORED_B11.jsonl --partition development --suts following:0,... \
         [--g2 1] [--q 99] [--delta .5] [--v-min -1000] [--workers 24] [--cases N] [--log-queries]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

from fairtest import devboot  # noqa: F401
from fairtest import b23_search as B
from fairtest import b64_severity_screened as S
from fairtest import runner

METHOD = 'bo_warm_g4v'
CFG = dict(g2=1., q=99., delta=.5, v_min=-1000., tau=None)


def _tau(q):
    from fairtest.b63_realism_sensitivity import taus
    sup, t = taus() if q in (95., 99., 99.9) else (None, None)
    if t is not None:
        return t[q]
    import numpy as np
    from fairtest.b45_joint_realism import K, Support
    sup = Support()
    d = np.sqrt(((sup.z[:, None, :] - sup.z[None, :, :]) ** 2).sum(-1))
    return float(np.percentile(np.sort(d, 1)[:, K], q))


def admission_v(case, theta):
    """b23_search.admission with the run's G2 tolerance, joint threshold and CC margin (B.DELTA)."""
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_margin import clearance
    plan = ContinuousPlan.from_theta(theta)
    raw = runner.run_episode(case, plan, *B.CC, keep_trace=True)
    cc_trace, cc = raw['trace'], runner.compact(raw, case, plan, *B.CC)
    clear = clearance(cc_trace)
    feint, exc = B.guard_terms(cc_trace, cc['features'])
    js, _ = B.joint_score(cc['features'])
    g2, tau = CFG['g2'], CFG['tau']
    guard_pen = (10. if feint else 0.) + (10. if exc is None else 5. * max(0., exc - g2)) + (10. if js is None else 5. * max(0., js - tau))
    pen = 20. if cc['contact'] else 10. * max(0., B.DELTA - clear) + guard_pen
    return pen == 0., dict(cc=cc, clear=clear, feint=feint, exc=exc, js=js, pen=pen)


def _query_log_v(n, row, cc, success, replays=None):
    g = row.get('g4', {})
    js, exc = g.get('joint_score'), g.get('realism_exceedance')
    adm = bool(not cc['contact'] and g.get('cc_clearance', 0.) >= B.DELTA and not g.get('feint') and exc is not None
               and exc <= CFG['g2'] and js is not None and js <= CFG['tau'])
    out = dict(n=n, admissible=adm, cc_contact=bool(cc['contact']), cc_clearance=g.get('cc_clearance'), feint=g.get('feint'),
               exc=exc, js=js, sut_contact=bool(row['contact']), valid=bool(row['valid']), success=bool(success))
    if 'closing' in row:
        out['closing'] = row['closing']
    if replays is not None:
        out['cc_replays'] = replays
    return out


def _init_worker(cfg):
    CFG.update(cfg)
    B.DELTA = CFG['delta']
    S.STATE['v_min'] = CFG['v_min']
    B.admission = admission_v
    B._objective_g4s = S._objective_g4ss
    B._record = S._record_s
    B._query_log = _query_log_v


def task(args):
    out = B.task(args)
    out['runs'] = {k.replace('bo_warm_g4s|', METHOD + '|', 1): v for k, v in out['runs'].items()}
    out['cfg'] = {k: CFG[k] for k in ('g2', 'q', 'delta', 'v_min', 'tau')}
    return out


def main(a):
    runner._init()
    cfg = dict(g2=a.g2, q=a.q, delta=a.delta, v_min=a.v_min, tau=_tau(a.q))
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    ids = ids[:a.cases] if a.cases else ids
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    keys = [f'{n}_H{h:g}' for n, h in suts]
    only = set(a.only.split(',')) if a.only else set(keys)
    jobs = [(c, boundary.get(c), suts, ('bo_warm_g4s',), a.budget, frozenset({k}), a.log_queries) for c in ids for k in keys if k in only]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(cfg,)) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 25 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), cfg=cfg, elapsed_s=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True)
    ap.add_argument('--partition', default='development')
    ap.add_argument('--suts', required=True, help='the 17 SUTs in the B46 order (the random streams keep their position)')
    ap.add_argument('--only', help='comma-separated SUT keys to run (streams keep their --suts position)')
    ap.add_argument('--budget', type=int, default=36)
    ap.add_argument('--g2', type=float, default=1.)
    ap.add_argument('--q', type=float, default=99.)
    ap.add_argument('--delta', type=float, default=.5)
    ap.add_argument('--v-min', type=float, default=-1000.)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--cases', type=int, help='first N cases only (pre-check)')
    ap.add_argument('--log-queries', action='store_true')
    main(ap.parse_args())
