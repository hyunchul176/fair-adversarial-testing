"""E4 (design addendum 3): run the final version's screened refinement (b23_search.bo_search_screened, B56 bo_warm_g4s) as is, but issue only the SUT
queries in CARLA. The screening C differs per arm: S0 = default C (cc_r157, as in the final version), S1 = C matched to CARLA (cc_r157_cal).

Structure: one CARLA server is used in turn by a query-only process (service), and the screening (CPU) runs in several workers in parallel. A worker sends the
plan's trajectory to the service and waits for the result. If the service dies (watchdog etc.), the main process restarts the server and the service, and
the worker resends the same query after a timeout. A successful attack is also run once with CARLA's C as a check.

  FAIRTEST_DATA=data python e4_search.py OUT.jsonl --cont CONT.jsonl --port 2040 [--arms S0,S1] [--kinds ...] [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import subprocess
import time
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
KINDS = ('ladder:following:0', 'ladder:following_aeb:0', 'ladder:rss_sut:0', 'ladder:rss_lat_sut:0', 'ba:cautious', 'ba:normal',
         'ba:aggressive', 'tm:default')
SI = {'ladder:following:0': 0, 'ladder:following_aeb:0': 4, 'ladder:rss_sut:0': 6, 'ladder:rss_lat_sut:0': 10,  # order of the 17 B56 kinds
      'ba:cautious': 100, 'ba:normal': 101, 'ba:aggressive': 102, 'tm:default': 103}
# v2 (harness H2): both arms use the R157 C. The 'matching' of S1 is geometric, given by process environment variables (FAIRTEST_CC_PERCEIVE, VEH_W, VEH_L).
# So S0 and S1 run in separately started processes (design v2 §2).
ARMS = {'S0': ('cc_r157', 0.), 'S1': ('cc_r157', 0.)}
BUDGET, TIMEOUT_S = 36, 600.


def service(port, req_q, res_qs, ready):
    """CARLA queries only: (worker, number, kind, trajectory) -> carla_replay.run. Ticks every 30 s even when idle (watchdog)."""
    from fairtest import carla_replay as cr
    from fairtest.runner import _STATE, _init
    _init()
    base = _STATE['base']
    world = cr.connect(port)
    road = cr.Road(world)
    bp = world.get_blueprint_library().find(cr.BLUEPRINT)
    ready.set()
    while True:
        try:
            item = req_q.get(timeout=30.)
        except queue.Empty:
            world.tick()
            continue
        if item is None:
            break
        wid, rid, kind, row = item
        for attempt in range(3):                   # transient errors (vehicle left at the spawn point, BehaviorAgent start failure): clear and retry
            try:
                res = cr.run(world, road, bp, row, kind, base)
                break
            except Exception as e:
                res = dict(error=f'{type(e).__name__}: {e}', attempts=attempt + 1)
                for actor in world.get_actors().filter('vehicle.*'):
                    actor.destroy()
                world.tick()
        res_qs[wid].put((rid, res))


def worker(wid, job_q, req_q, res_q, out_q, cont):
    from fairtest import runner
    import fairtest.b23_search as B23
    from fairtest.carla_search import adversary_row
    runner._init()
    seq = [0]

    def carla_run(kind, row):
        seq[0] += 1
        rid = seq[0]
        for _ in range(3):
            req_q.put((wid, rid, kind, row))
            t_end = time.time() + TIMEOUT_S
            while time.time() < t_end:
                try:
                    got, res = res_q.get(timeout=max(1., t_end - time.time()))
                except queue.Empty:
                    break
                if got == rid:
                    return res
        return dict(error='carla service timeout (3 tries)')

    while True:
        job = job_q.get()
        if job is None:
            break
        case_id, kind, arm = job
        t0 = time.time()
        case = runner._STATE['cases'][case_id]
        B23.CC = ARMS[arm]                                    # screening C used by admission()
        hit = {}

        def objective(case_, theta, name, h, adm):
            valid, row = adversary_row(case_, theta)
            if not valid:
                s, sut_contact, f, success = None, False, 60., False
            else:
                s = carla_run(kind, row)
                if 'error' in s:
                    raise RuntimeError(s['error'])
                sut_contact = bool(s['contact'])
                gap = -1. if sut_contact else (50. if s.get('min_gap') is None else s['min_gap'])
                success = bool(sut_contact and (s.get('contact_dx') or -1e9) >= B23.FRONT_DX and adm['clear'] >= B23.DELTA)
                f = gap + adm['pen'] - (5. if success else 0.)
                if success:
                    hit['row'] = row
            out = dict(valid=valid, contact=sut_contact, features=None, carla=s,
                       g4=dict(fair=bool(valid and sut_contact and not adm['cc']['contact']), robust=success,
                               cc_clearance=round(adm['clear'], 3), contact_dx=None if s is None else s.get('contact_dx'),
                               cc_demand=round(float(adm['cc'].get('max_drac') or 0.), 3), feint=adm['feint'],
                               realism_exceedance=None if adm['exc'] is None else round(adm['exc'], 3),
                               joint_score=None if adm['js'] is None else round(adm['js'], 3)))
            return f, success, out, adm['cc']

        B23._objective_g4s = objective
        rng = np.random.default_rng([case_id, SI[kind], B23.METHODS.index('bo_warm')])
        log = []
        try:
            best, n, st = B23.bo_search_screened(case, kind, 0., BUDGET, rng, x_first=cont.get(case_id), log=log)
            rec = dict(case_id=case_id, kind=kind, arm=arm, sut_queries=n, first_success=n if best and best['fair'] else None,
                       best=best, cc_replays=st['cc_replays'], stop=st['stop'], warm=st['warm'], queries=log)
            if best and best['fair'] and 'row' in hit:          # also run the attack found with CARLA's C
                rec['carla_cc'] = carla_run('cc', hit['row'])
        except Exception as e:
            rec = dict(case_id=case_id, kind=kind, arm=arm, error=f'{type(e).__name__}: {e}')
        rec['seconds'] = round(time.time() - t0, 1)
        out_q.put(rec)


def kill_server(port):
    """Kill only the CARLA processes of that port (shells and scripts whose command line contains the same string are left alone)."""
    out = subprocess.run(['pgrep', '-a', '-f', f'carla-rpc-port={port}'], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, _, cmd = line.partition(' ')
        if any(t.split('/')[-1].startswith('CarlaUE4') for t in cmd.split(' ')[:2]):   # the ./CarlaUE4.sh wrapper and the binary
            subprocess.run(['kill', pid], check=False)


def start_server(port, log):
    kill_server(port)
    time.sleep(5)
    env = {k: v for k, v in os.environ.items() if k != 'DISPLAY'}
    extra = os.environ.get('FAIRTEST_CARLA_SERVER_ARGS', '')  # e.g. '-graphicsadapter=1' (RTX 4090) on the second machine
    subprocess.Popen(f'cd $HOME/carla_0916 && exec nice -n 5 ./CarlaUE4.sh /Game/Carla/Maps/Town04 -RenderOffScreen -nosound '
                     f'-carla-rpc-port={port} -quality-level=Low {extra} >> {log} 2>&1', shell=True, env=env, start_new_session=True)
    for _ in range(60):
        if f':{port} ' in subprocess.run(['ss', '-ltn'], capture_output=True, text=True).stdout:
            break
        time.sleep(3)
    time.sleep(15)


def main(a):
    from fairtest import runner
    runner._init()
    cont = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        cont[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if v['event'].get('partition') == 'development')
    kinds = a.kinds.split(',') if a.kinds else KINDS
    arms = a.arms.split(',')
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf-8'):
            r = json.loads(line)
            if 'error' not in r:
                done.add((r['case_id'], r['kind'], r['arm']))
    jobs = [(c, k, arm) for k in kinds for arm in arms for c in ids if (c, k, arm) not in done]
    rng = np.random.default_rng(0)
    jobs = [jobs[i] for i in rng.permutation(len(jobs))]       # mix long jobs (competent SUTs) with short ones
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
            start_server(a.port, log)
        ready = ctx.Event()
        p = ctx.Process(target=service, args=(a.port, req_q, res_qs, ready), daemon=True)
        p.start()
        ready.wait(300)
        return p

    svc = new_service()
    ws = [ctx.Process(target=worker, args=(i, job_q, req_q, res_qs[i], out_q, cont)) for i in range(a.workers)]
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
            if not svc.is_alive():                            # watchdog or server crash: restart the server and the service
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
    ap.add_argument('--cont', required=True)
    ap.add_argument('--port', type=int, default=2040)
    ap.add_argument('--arms', default='S0,S1')
    ap.add_argument('--kinds')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    main(ap.parse_args())
