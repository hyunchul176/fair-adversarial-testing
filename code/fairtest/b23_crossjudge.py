"""B23: judge every RL adversary's collisions with the Dynasto validity rules as well (G2-style cross-judgement).

Reads B17-format evaluation rows (with the realised action sequence), replays each valid SUT collision against
the SUT with the full trace, and applies `rl_baselines.dynasto_validity` (ISO 34502 safe distances, tau 0.2 s):
the collision is 'Dynasto-valid' unless the adversary cut in or braked unsafely, or rear-ended the ego.
The CC/FSM/RSS judgements of the same collisions are already in the evaluation rows.

  python -m fairtest.b23_crossjudge OUT.jsonl EVAL.jsonl [EVAL.jsonl ...] --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from fairtest import devboot  # noqa: F401


def replay(case, actions, target, h, durations=(2., 3., 4., 5., 6.)):
    from fairtest.env import B7Env
    from fairtest.runner import _STATE
    env = B7Env(_STATE['base'], case, _STATE['b1'], _STATE['b2'], case['physical_initial_state'],
                target=target, target_h=h, durations=durations)
    try:
        env.reset()
        done = False
        for a in actions:
            if done:
                break
            _, _, done, _, _ = env.step(tuple(a))
        while not done:
            _, _, done, _, _ = env.step(tuple(actions[-1]))
        return env.result()
    finally:
        env.close()


def _task(job):
    source, row = job
    from fairtest import rl_baselines as BL
    from fairtest.runner import _STATE, _init
    _init()
    name, h = row['sut'].rsplit('_H', 1)
    r = replay(_STATE['cases'][row['case_id']], row['actions'], name, float(h))
    ok, why = BL.dynasto_validity(r['trace'])
    return dict(source=source, case_id=row['case_id'], sut=row['sut'], contact=bool(r['pair_contact']),
                reproduced=bool(r['pair_contact']) == bool(row['contact']), dynasto_valid=bool(ok),
                violation=None if why is None else why[0],
                cc_contact=row['refs']['cc_r157']['contact'], fsm_contact=row['refs']['ref_fsm']['contact'],
                rss_contact=row['refs']['ref_rss']['contact'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('evals', nargs='+')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = []
    for path in a.evals:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r['valid'] and r['contact'] and r.get('refs'):
                jobs.append((Path(path).name, r))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(_task, jobs, chunksize=4):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(finished=len(jobs))))
