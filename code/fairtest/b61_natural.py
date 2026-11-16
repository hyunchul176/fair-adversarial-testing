"""B61 control (attack-free): each case's natural cut-in continued without adversarial modification, one SUT query per
case and SUT, judged by the v2j success contract (b23_search._objective_g4 with the guards and the joint support).

The natural plan starts the lane change at once from the case's initial state (10% of the natural lateral progress, with
its natural lateral speed), keeps the speed (all accelerations 0) and uses the natural 10-90% lateral duration clipped
to the plan box (2-6 s). It approximates the recorded cut-in; how many natural cut-ins already give a fair failure
shows what the adversary adds.

  python -m fairtest.b61_natural OUT.jsonl --partition development --suts name:H,... --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import HIGH, LOW


def natural_theta(case):
    d = float(case['event']['metrics'].get('duration_10_90') or 4.)
    return np.clip(np.array([0., 0., 0., 0., d]), LOW, HIGH)


def task(job):
    from fairtest.b23_search import _objective
    case_id, suts = job
    runner._init()
    case = runner._STATE['cases'][case_id]
    theta = natural_theta(case)
    out = dict(case_id=case_id, theta=[float(x) for x in theta], runs={})
    for name, h in suts:
        key = f'{name}_H{h:g}'
        try:
            f, ok, row, cc = _objective(case, theta, name, h, robust=True, guarded=True, joint=True)
            g = row.get('g4', {})
            out['runs'][key] = dict(success=bool(ok), sut_contact=bool(row['contact']), cc_contact=bool(cc['contact']),
                                    valid=bool(row['valid']), fair=g.get('fair'), robust=g.get('robust'), feint=g.get('feint'),
                                    realism_exceedance=g.get('realism_exceedance'), joint_score=g.get('joint_score'),
                                    cc_clearance=g.get('cc_clearance'))
        except Exception as e:  # recorded, never dropped
            out['runs'][key] = dict(error=f'{type(e).__name__}: {e}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--partition', default='development')
    ap.add_argument('--suts', required=True)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    runner._init()
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(task, [(c, suts) for c in ids]):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(cases=len(ids), suts=len(suts))))
