"""B54 runner: Autoware instances in parallel, one worker process per instance (the shield's highway-env replays are
Python and would contend for one interpreter), jobs split statically by index so the assignment is reproducible.

Phases (each writes OUT_DIR/<phase>_<instance>.jsonl; finished jobs are skipped on a re-run):
  policy   v2j policy stage (fair_guarded2 s1-3, window shield, CC margin 0.5 m) and the collision-seeking baseline
           (collide_commit s1-3, no shield), closed loop, one episode per (variant, case, seed, attacker)
  refine   v2j refinement (aw_search.refine, budget 36, stored boundary start) for every (variant, case) where some seed's
           policy episode gave no reportable failure (unified contract: robust fair + G1 + G2 + joint support); the
           refinement does not depend on the policy stage, so one run per (variant, case) serves the three seeds
  repro    each reported failure's adversary trajectory replayed REPRO more times against the same variant
Infrastructure failures are not SUT outcomes: an episode whose reset fails, whose bridge errs or whose planner stalls
(aw_sut.Episode.health) restarts the instance (container and bridge) and is run again, at most MAX_TRIES times; the
restarts are recorded per job and a job that never runs healthy is recorded as an error, never as an outcome.

  python -m fairtest.aw_run OUT_DIR --phase policy --instances a1:81:5481:aw_default,... --partition development [--cases N]
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):  # 8 workers next to 8 Autoware instances
    os.environ.setdefault(_v, '1')

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

AW = Path(os.path.expanduser('~/fairtest/aw'))
MODELS = {'v2j': [os.path.expanduser('~/fairtest/b17/fair_guarded2_s1/ppo_fair_guarded2_s1_1500000.zip'),
                  os.path.expanduser('~/fairtest/b23/fair_guarded2_s2/ppo_fair_guarded2_s2_1500000.zip'),
                  os.path.expanduser('~/fairtest/b23/fair_guarded2_s3/ppo_fair_guarded2_s3_1500000.zip')],
          'collide': [os.path.expanduser('~/fairtest/b17/collide_commit_s1/ppo_collide_commit_s1_1500000.zip'),
                      os.path.expanduser('~/fairtest/b23/collide_commit_s2/ppo_collide_commit_s2_1500000.zip'),
                      os.path.expanduser('~/fairtest/b23/collide_commit_s3/ppo_collide_commit_s3_1500000.zip')]}
MARGIN, FRONT_DX, REPRO, MAX_TRIES = .5, 3.5, 4, 3
# B58: perceived-adversary latency (s) per variant; these variants also treat a start > V0_MAX m/s off the case speed
# as an infrastructure failure (aw_sut.Episode)
LATENCY = {'aw_r157nc_l0': 0., 'aw_r157nc_l300': .3, 'aw_r157nc_l600': .6}
# B60: measurement noise (sd along m, across m, speed m/s) and missed-detection bursts (start probability per tick, ticks)
PERCEPTION = {**{v: dict(latency_s=t) for v, t in LATENCY.items()}, 'aw_r157nc_nz': dict(noise=(.5, .3, .5)),
              'aw_r157nc_dp': dict(dropout=(.05, 5))}
V0_MAX = .5


class Unhealthy(RuntimeError):
    pass


class Instance:
    """One Autoware instance: its bridge connection and the restart-and-retry of infrastructure failures."""

    def __init__(self, inst):
        self.inst, self.br, self.restarts, self.log = inst, None, 0, []
        v = inst[3]
        self.perception = dict(PERCEPTION[v], v0_max=V0_MAX) if v in PERCEPTION else {}

    def bridge(self):
        from fairtest import aw_sut
        if self.br is None:
            self.br = aw_sut.Bridge(self.inst[2])
        return self.br

    def drop(self):
        try:
            self.br.sock.close()
        except Exception:
            pass
        self.br = None

    def restart(self):
        name, dom, port, variant = self.inst
        r = subprocess.run(['bash', str(AW / 'aw_restart.sh'), name, str(dom), str(port), variant], capture_output=True, cwd=AW)
        self.restarts += 1
        return r.returncode == 0

    def run(self, fn, *args, **kw):
        """fn(bridge, *args) -> an episode result (with 'health'); retried after a restart on an error or a stall."""
        for attempt in range(MAX_TRIES):
            try:
                res = fn(self.bridge(), *args, **kw)
                if not res['health']['healthy']:
                    raise Unhealthy(json.dumps(res['health']))
                return res
            except Exception as e:
                self.log.append(dict(attempt=attempt, error=f'{type(e).__name__}: {e}'[:500], t=round(time.time(), 1)))
                self.drop()
                if attempt == MAX_TRIES - 1:
                    raise
                self.restart()

    def replay(self, row):
        from fairtest import aw_sut
        return self.run(aw_sut.replay, row, **self.perception)


def reportable(res, case_id):
    """The v2j unified contract for a policy episode (G2 and the joint support from the CC replay of its actions)."""
    from fairtest.fair_agent_v2 import _cc_realism
    if not (res.get('contact') and res.get('valid') and res.get('fair')):
        return False, None
    robust = res['contact_dx'] is not None and res['contact_dx'] >= FRONT_DX and res['cc_min_clearance'] >= MARGIN
    real = _cc_realism(case_id, [tuple(a) for a in res['actions']])
    return bool(robust and real['g2'] and real['joint']), real


def adv_traj(res):
    """The realised adversary trajectory of a policy episode (t = 0 .. 12 s, constant velocity after the episode end)."""
    tr = [dict(t=round(r['t'], 1), x=r['x'], y=r['y'], vx=r['vx'], vy=r['vy']) for r in res['adv_traj']]
    while len(tr) < 121:
        p = tr[-1]
        tr.append(dict(t=round(p['t'] + .1, 1), x=p['x'] + p['vx'] * .1, y=p['y'] + p['vy'] * .1, vx=p['vx'], vy=p['vy']))
    return tr[:121]


def worker(args):
    inst, jobs, phase, out_dir = args
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from fairtest import aw_search, aw_sut, b17_rl, devboot  # noqa: F401
    from fairtest.runner import _STATE, _init
    _init()
    b17_rl.SHIELD_MARGIN = MARGIN
    name, dom, port, variant = inst
    out = Path(out_dir) / f'{phase}_{name}.jsonl'
    done = set()
    if out.exists():
        for line in open(out, encoding='utf-8'):
            if line.strip():
                done.add(json.loads(line)['key'])
    models, I = {}, Instance(inst)
    with open(out, 'a', encoding='utf-8') as f:
        for job in jobs:
            key = json.dumps(job, sort_keys=True)
            if key in done:
                continue
            case = _STATE['cases'][job['case_id']]
            I.log, r0, t0 = [], I.restarts, time.time()
            try:
                if phase == 'policy':
                    mk = (job['attacker'], job['seed'])
                    if mk not in models:
                        models[mk] = PPO.load(MODELS[job['attacker']][job['seed'] - 1], device='cpu')
                    res = I.run(aw_sut.policy_episode, case, models[mk], shield=job['attacker'] == 'v2j', **I.perception)
                    if job['attacker'] == 'v2j':
                        res['reportable'], res['realism'] = reportable(res, job['case_id'])
                elif phase == 'refine':
                    fn = aw_search.refine_screened if job.get('mode') == 'screened' else aw_search.refine  # B59: screened
                    res = fn(I.replay, case, job['boundary'], budget=36, panel_index=0)
                else:  # repro
                    row = dict(sut0=job['sut0'], cruise=job['cruise'], trajectory=job['trajectory'])
                    res = dict(runs=[{k: v for k, v in I.replay(row).items() if k != 'trace'} for _ in range(REPRO)])
            except Exception as e:  # recorded, never dropped and never counted as an outcome
                res = dict(error=f'{type(e).__name__}: {e}'[:1000])
            res.update(seconds=round(time.time() - t0, 1), restarts=I.restarts - r0, infra_log=I.log)
            rec = dict(key=key, variant=variant, instance=name, **{k: v for k, v in job.items() if k != 'trajectory'}, result=res)
            f.write(json.dumps(rec) + '\n')
            f.flush()
            print(json.dumps(dict(phase=phase, instance=name, case=job['case_id'], seed=job.get('seed'), attacker=job.get('attacker'),
                                  contact=res.get('contact'), found=res.get('found'), queries=res.get('queries'),
                                  reportable=res.get('reportable'), error=res.get('error'), restarts=res['restarts'],
                                  s=res['seconds'])), flush=True)
    I.drop()
    return name


def main(a):
    from fairtest import devboot  # noqa: F401
    from fairtest.runner import _STATE, _init
    _init()
    insts = [tuple(x.split(':')) for x in a.instances.split(',')]
    insts = [(n, int(d), int(p), v) for n, d, p, v in insts]
    variants = sorted({v for *_, v in insts})
    ids = sorted(c for c, v in _STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    ids = ids[:a.cases] if a.cases else ids
    if a.case_ids:
        ids = [int(x) for x in a.case_ids.split(',')]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {v: [] for v in variants}
    if a.phase == 'policy':
        attackers = a.attackers.split(',')
        for v in variants:
            for c in ids:
                for attacker in attackers:
                    for s in (1, 2, 3):
                        jobs[v].append(dict(case_id=c, seed=s, attacker=attacker))
    elif a.phase == 'refine':
        boundary = {}
        for line in open(a.cont, encoding='utf-8'):
            r = json.loads(line)
            b = (r.get('cc_boundary') or {}).get('best')
            boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
        found = {}
        for p in out.glob('policy_*.jsonl'):
            for line in open(p, encoding='utf-8'):
                r = json.loads(line)
                if r['attacker'] == 'v2j' and 'error' not in r['result']:
                    found.setdefault((r['variant'], r['case_id']), {})[r['seed']] = bool(r['result'].get('reportable'))
        for v in variants:
            for c in ids:
                seeds = found.get((v, c), {})
                if not (len(seeds) == 3 and all(seeds.values())):
                    jobs[v].append(dict(case_id=c, boundary=boundary.get(c), **({'mode': 'screened'} if a.refine_mode == 'screened' else {})))
    else:  # repro: every reported failure
        from fairtest.b11_search import ContinuousPlan
        from fairtest.carla_search import trajectory_row
        for p in sorted(out.glob('policy_*.jsonl')) + sorted(out.glob('refine_*.jsonl')):
            for line in open(p, encoding='utf-8'):
                r = json.loads(line)
                res = r['result']
                case = _STATE['cases'][r['case_id']]
                if p.name.startswith('policy') and a.repro_sources == 'refine':  # B59: policy failures reproduced earlier
                    continue
                if p.name.startswith('policy') and r['attacker'] == 'v2j' and res.get('reportable'):
                    t0 = case['physical_initial_state'][1]
                    jobs[r['variant']].append(dict(case_id=r['case_id'], source=f"policy_s{r['seed']}",
                                                   sut0=dict(x=t0['x'], y=t0['y'], vx=t0['vx'], vy=t0['vy']),
                                                   cruise=case['event']['metrics']['p10']['target_speed'], trajectory=adv_traj(res)))
                elif p.name.startswith('refine') and res.get('found'):
                    _, row = trajectory_row(case, ContinuousPlan.from_theta(res['best']['theta']))
                    jobs[r['variant']].append(dict(case_id=r['case_id'], source='refine', sut0=row['sut0'], cruise=row['cruise'],
                                                   trajectory=row['trajectory']))
    tasks = []
    for v in variants:
        mine = [i for i in insts if i[3] == v]
        for k, inst in enumerate(mine):
            tasks.append((inst, jobs[v][k::len(mine)], a.phase, str(out)))
    print(json.dumps(dict(phase=a.phase, jobs={v: len(j) for v, j in jobs.items()}, instances=len(insts))), flush=True)
    with mp.get_context('spawn').Pool(len(tasks)) as pool:
        for name in pool.imap_unordered(worker, tasks):
            print(json.dumps(dict(finished=name)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--phase', choices=('policy', 'refine', 'repro'), required=True)
    ap.add_argument('--instances', required=True, help='name:domain:port:variant,...')
    ap.add_argument('--partition', default='development')
    ap.add_argument('--cases', type=int)
    ap.add_argument('--case-ids', help='explicit case ids (smoke runs)')
    ap.add_argument('--attackers', default='v2j,collide')
    ap.add_argument('--cont', default=os.path.expanduser('~/fairtest/b23/exposed_cont_search_all7.jsonl'))
    ap.add_argument('--refine-mode', choices=('g4j', 'screened'), default='g4j', help='B59: screened = aw_search.refine_screened')
    ap.add_argument('--repro-sources', choices=('all', 'refine'), default='all', help='B59: rerun only the refinement failures')
    main(ap.parse_args())
