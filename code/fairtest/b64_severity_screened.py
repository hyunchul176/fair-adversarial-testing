"""B64 (B64 design, available on request): the final agent's screened refinement with
a tester-chosen severity floor, bo_warm_g4ss.

bo_warm_g4ss = b23_search's bo_warm_g4s (B56: the same admission screening against the CC, boundary start, random stream,
budget 36 and stopping rules, run through b23_search.task with --split-suts) whose objective adds the B51 severity terms:
success also needs the closing speed at contact (SUT vx - generator vx) >= v_min, and SUT-contact plans pay
SEV_W * max(0, v_min - closing). b23_search is not edited: each worker process replaces its module-level _objective_g4s
(and wraps _record and _query_log to keep the closing speed). Output keys are renamed bo_warm_g4s -> bo_warm_g4ss.
With v_min = -1000 the method reproduces the stored bo_warm_g4s runs (pre-check).

  python -m fairtest.b64_severity_screened OUT.jsonl --cont STORED_B11.jsonl --partition development|all \
         --suts following:0,... [--v-min 3] [--workers 24] [--cases N] [--log-queries]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

from fairtest import devboot  # noqa: F401
from fairtest import b23_search as B
from fairtest import runner
from fairtest.b11_search import gap
from fairtest.b51_severity import SEV_W, V_MIN

METHOD = 'bo_warm_g4ss'
STATE = dict(v_min=V_MIN)
ORIG = dict(record=B._record, qlog=B._query_log)


def _objective_g4ss(case, theta, name, h, adm):
    """b23_search._objective_g4s (the admissible plan's CC replay is reused, only the SUT runs) + the severity floor."""
    from fairtest.b11_search import ContinuousPlan
    v_min = STATE['v_min']
    plan = ContinuousPlan.from_theta(theta)
    raw = runner.run_episode(case, plan, name, h, keep_trace=True)
    trace, row = raw['trace'], runner.compact(raw, case, plan, name, h)
    cc = adm['cc']
    last = trace[-1]
    dx = last['generator']['x'] - last['target']['x']
    closing = float(last['target']['vx'] - last['generator']['vx'])
    fair = bool(row['valid'] and row['contact'] and not cc['contact'])
    robust = bool(fair and dx >= B.FRONT_DX and adm['clear'] >= B.DELTA)
    severe = bool(row['contact']) and closing >= v_min
    sev_pen = SEV_W * max(0., v_min - closing) if row['contact'] else 0.
    success = robust and severe
    f = gap(row) + adm['pen'] + sev_pen - (5. if success else 0.)
    row['g4'] = dict(fair=fair, robust=robust, cc_clearance=round(adm['clear'], 3), contact_dx=round(dx, 2),
                     cc_demand=round(float(cc.get('max_drac') or 0.), 3), feint=adm['feint'],
                     realism_exceedance=None if adm['exc'] is None else round(adm['exc'], 3),
                     joint_score=None if adm['js'] is None else round(adm['js'], 3))
    row['closing'] = round(closing, 3)
    return f, success, row, cc


def _record_s(theta, f, fair, row, cc):
    out = ORIG['record'](theta, f, fair, row, cc)
    if 'closing' in row:
        out['closing'] = row['closing']
    return out


def _query_log_s(n, row, cc, success, replays=None):
    out = ORIG['qlog'](n, row, cc, success, replays)
    if 'closing' in row:
        out['closing'] = row['closing']
    return out


def _init_worker(v_min):
    STATE['v_min'] = v_min
    B._objective_g4s = _objective_g4ss
    B._record = _record_s
    B._query_log = _query_log_s


def task(args):
    out = B.task(args)  # bo_warm_g4s with the replaced objective
    out['runs'] = {k.replace('bo_warm_g4s|', METHOD + '|', 1): v for k, v in out['runs'].items()}
    out['v_min'] = STATE['v_min']
    return out


def main(a):
    runner._init()
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    ids = ids[:a.cases] if a.cases else ids
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    keys = [f'{n}_H{h:g}' for n, h in suts]
    jobs = [(c, boundary.get(c), suts, ('bo_warm_g4s',), a.budget, frozenset({k}), a.log_queries) for c in ids for k in keys]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(a.v_min,)) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 25 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), v_min=a.v_min, elapsed_s=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True)
    ap.add_argument('--partition', default='development')
    ap.add_argument('--suts', required=True)
    ap.add_argument('--budget', type=int, default=36)
    ap.add_argument('--v-min', type=float, default=V_MIN)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cases', type=int, help='first N cases only (pre-check)')
    ap.add_argument('--log-queries', action='store_true')
    main(ap.parse_args())
