"""E7: export adversary trajectories from highway-env RL evaluations for replay in CARLA.

For every selected evaluation row (agent, case, SUT) the realised action sequence (then its latched last action,
the CC-replay semantics) is replayed in highway-env against the omniscient braking target `oracle_t0`, which almost
never collides, so the adversary's full 12 s trajectory is recorded (its low-level control depends only on its own
state, so the trajectory is the same whatever the target). Exported with the SUT's initial state and the
highway-env verdicts of the same attack (SUT contact; CC, FSM, RSS contact).

  python -m fairtest.carla_export OUT.jsonl EVAL.jsonl [EVAL.jsonl ...] [--suts following_H0,...] [--cases N] --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from fairtest import devboot  # noqa: F401


def _task(job):
    source, row = job
    from fairtest.b23_crossjudge import replay
    from fairtest.runner import _STATE, _init
    _init()
    case = _STATE['cases'][row['case_id']]
    r = replay(case, row['actions'], 'oracle_t0', 0.)
    g_init = r['physical_initial_state'][0]  # trace row k is the state after tick k: prepend the t = 0 state
    traj = [dict(t=0., x=g_init['x'], y=g_init['y'], vx=g_init['vx'], vy=g_init['vy'], heading=g_init['heading'])]
    traj += [dict(t=round((k + 1) * .1, 1), x=s['generator']['x'], y=s['generator']['y'], vx=s['generator']['vx'],
                  vy=s['generator']['vy'], heading=s['generator']['heading']) for k, s in enumerate(r['trace'])]
    # if even the omniscient target was hit, extend with the adversary's last velocity to the 12 s horizon
    while len(traj) < 121:
        last = traj[-1]
        traj.append(dict(t=round(last['t'] + .1, 1), x=last['x'] + last['vx'] * .1, y=last['y'] + last['vy'] * .1,
                         vx=last['vx'], vy=last['vy'], heading=last['heading']))
    g0, t0 = case['physical_initial_state'][:2]
    refs = row.get('refs') or {}
    return dict(source=source, case_id=row['case_id'], sut=row['sut'], cruise=case['event']['metrics']['p10']['target_speed'],
                sut0=dict(x=t0['x'], y=t0['y'], vx=t0['vx'], vy=t0['vy']), gen0=dict(x=g0['x'], y=g0['y']),
                hwenv=dict(valid=bool(row['valid']), sut_contact=bool(row['contact']),
                           cc_contact=None if not refs else refs['cc_r157']['contact'],
                           fsm_contact=None if not refs else refs['ref_fsm']['contact'],
                           rss_contact=None if not refs else refs['ref_rss']['contact'],
                           ghost_contact=bool(r['pair_contact'])),
                trajectory=traj)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('evals', nargs='+')
    ap.add_argument('--suts', help='comma-separated SUT keys to export (default: all)')
    ap.add_argument('--cases', type=int, help='first N case ids only')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    keep = set(a.suts.split(',')) if a.suts else None
    jobs = []
    for path in a.evals:
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        ids = sorted({r['case_id'] for r in rows})[:a.cases] if a.cases else None
        for r in rows:
            if (keep is None or r['sut'] in keep) and (ids is None or r['case_id'] in ids) and r['valid']:
                jobs.append((Path(path).name, r))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(_task, jobs, chunksize=2):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(exported=len(jobs))))
