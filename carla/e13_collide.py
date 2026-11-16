"""E13 (design addendum 8): run the collision-only search inside CARLA and judge the collisions it finds post hoc.

The search uses the earlier version's `collide` objective (`carla_search.search_bo` as is): start at the box centre, objective = minimum SUT gap (-1 on
collision), 36 SUT queries, stop at the first SUT collision. Trajectories with a SUT collision are also run with CARLA's C (as before). Queries are organised as in E4:
one CARLA server is used by a query-only process (e4_search.service) and GP-BO runs in several workers in parallel (only `carla_eval` is swapped in).
The plan of the first collision found is run through `b23_search.admission` in the highway-env twin with the two C of E4 (S0 default, S1 matched) to check the
final version's C-side conditions (C gap >= 0.5 m, G1, G2, joint realism).

  FAIRTEST_DATA=data python e13_collide.py OUT.jsonl --port 2030 [--kinds ...] [--workers 6] [--limit N]
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
sys.path[:0] = [str(W), str(W / 'code')]
import e4_search as E4  # noqa: E402

# v2: the search runs in the default twin geometry (U) and this process judges S0 (U) only. S1 (matched twin M) is judged separately post hoc by e13_twin.py.
TWIN = {'S0': ('cc_r157', 0.)}


def worker(wid, job_q, req_q, res_q, out_q):
    from fairtest import runner
    import fairtest.b23_search as B23
    import fairtest.carla_search as CS
    from fairtest.b11_search import HIGH, LOW
    runner._init()
    seq = [0]

    def carla_run(kind, row):
        seq[0] += 1
        rid = seq[0]
        for _ in range(3):
            req_q.put((wid, rid, kind, row))
            t_end = time.time() + E4.TIMEOUT_S
            while time.time() < t_end:
                try:
                    got, res = res_q.get(timeout=max(1., t_end - time.time()))
                except queue.Empty:
                    break
                if got == rid:
                    if 'error' in res:
                        raise RuntimeError(res['error'])
                    return res
        raise RuntimeError('carla service timeout (3 tries)')

    def carla_eval(world, road, bp, base, case, kind, objective, theta):
        """Same computation as the 'collide' case of carla_search.carla_eval. Only the CARLA run is sent to the service."""
        valid, row = CS.adversary_row(case, theta)
        if not valid:
            return 60., None
        s = carla_run(kind, row)
        gap = -1. if s['contact'] else (50. if s['min_gap'] is None else s['min_gap'])
        cc = carla_run('cc', row) if s['contact'] else None
        cc_contact = bool(cc and cc['contact'])
        fair = bool(s['contact'] and cc is not None and not cc_contact)
        clear = None if cc is None else cc['min_clearance']
        robust = bool(fair and (s.get('contact_dx') or 0.) >= 3.5 and clear is not None and clear >= .5)
        return gap, dict(f=gap, theta=[float(x) for x in theta], sut_contact=bool(s['contact']), cc_contact=cc_contact, fair=fair,
                         robust=robust, guarded=None, feint=None, realism_exceedance=None, sut_min_acc=s['sut_min_acc'],
                         v0_error=s['v0_error'], contact_dx=s.get('contact_dx'), cc_min_clearance=clear)

    CS.carla_eval = carla_eval
    while True:
        job = job_q.get()
        if job is None:
            break
        case_id, kind = job
        t0 = time.time()
        case = runner._STATE['cases'][case_id]
        try:
            res = CS.search_bo(None, None, None, runner._STATE['base'], case, kind, 'collide', (LOW + HIGH) / 2, E4.BUDGET)
            rec = dict(case_id=case_id, kind=kind, **res)
            if res.get('found'):
                rec['twin'] = {}
                for arm, cc in TWIN.items():
                    B23.CC = cc
                    ok, adm = B23.admission(case, res['found']['theta'])
                    rec['twin'][arm] = dict(admissible=bool(ok), cc_contact=bool(adm['cc']['contact']), clear=round(float(adm['clear']), 3),
                                            feint=adm['feint'], exc=None if adm['exc'] is None else round(float(adm['exc']), 3),
                                            js=None if adm['js'] is None else round(float(adm['js']), 3), pen=round(float(adm['pen']), 3))
        except Exception as e:
            rec = dict(case_id=case_id, kind=kind, error=f'{type(e).__name__}: {e}')
        rec['seconds'] = round(time.time() - t0, 1)
        out_q.put(rec)


def main(a):
    from fairtest import runner
    runner._init()
    ids = sorted(c for c, v in runner._STATE['cases'].items() if v['event'].get('partition') == 'development')
    kinds = a.kinds.split(',') if a.kinds else E4.KINDS
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['case_id'], r['kind']))
    jobs = [(c, k) for k in kinds for c in ids if (c, k) not in done]
    rng = np.random.default_rng(0)
    jobs = [jobs[i] for i in rng.permutation(len(jobs))]
    if a.limit:
        jobs = jobs[:a.limit]
    print(json.dumps(dict(todo=len(jobs), done_before=len(done))), flush=True)
    ctx = mp.get_context('fork')
    req_q, job_q, out_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    res_qs = [ctx.Queue() for _ in range(a.workers)]
    for j in jobs:
        job_q.put(j)
    for _ in range(a.workers):
        job_q.put(None)
    log = W / f'logs/carla_server_{a.port}.log'

    def new_service():
        if f':{a.port} ' not in subprocess.run(['ss', '-ltn'], capture_output=True, text=True).stdout:
            E4.start_server(a.port, log)
        ready = ctx.Event()
        p = ctx.Process(target=E4.service, args=(a.port, req_q, res_qs, ready), daemon=True)
        p.start()
        ready.wait(300)
        return p

    svc = new_service()
    ws = [ctx.Process(target=worker, args=(i, job_q, req_q, res_qs[i], out_q)) for i in range(a.workers)]
    for p in ws:
        p.start()
    n_out, t0 = 0, time.time()
    with open(a.out, 'a', encoding='utf-8') as f:
        while n_out < len(jobs):
            try:
                rec = out_q.get(timeout=20.)
            except queue.Empty:
                rec = None
            if rec is not None:
                f.write(json.dumps(rec) + '\n')
                f.flush()
                n_out += 1
                if n_out % 10 == 0:
                    print(json.dumps(dict(done=n_out, of=len(jobs), elapsed_s=round(time.time() - t0))), flush=True)
            if not svc.is_alive():
                print(json.dumps(dict(service_restart=time.strftime('%T'), exitcode=svc.exitcode)), flush=True)
                svc = new_service()
            if not any(p.is_alive() for p in ws) and out_q.empty():
                break
    req_q.put(None)
    for p in ws:
        p.join(5)
    print(json.dumps(dict(finished=n_out, elapsed_s=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--port', type=int, default=2030)
    ap.add_argument('--kinds')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    main(ap.parse_args())
