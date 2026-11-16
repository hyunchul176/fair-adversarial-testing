"""E1b (post hoc, outside the design): re-judge each E1 attack in highway-env with the C matched to CARLA braking (cc_r157_cal) to see whether it predicts the
CARLA C outcomes. If it does, 'match the C of the screening and the shield to the run environment' becomes a way to keep the attacks transferring.

  python e1b_calibrated.py OUT.jsonl --b26 B26_FINAL_DIR [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

W = Path(__file__).resolve().parent


def _task(job):
    from fairtest.b23_crossjudge import replay
    from fairtest.b23_margin import clearance
    from fairtest.runner import _STATE, _init, run_episode
    from fairtest.b11_search import ContinuousPlan
    _init()
    key, case_id, actions, theta = job
    case = _STATE['cases'][case_id]
    out = dict(key=key)
    for target in ('cc_r157', 'cc_r157_cal'):
        if actions is not None:
            r = replay(case, actions, target, 0.)
        else:
            r = run_episode(case, ContinuousPlan.from_theta(theta), target, 0., keep_trace=True)
        hit = bool(r['pair_contact'])
        out[target] = dict(contact=hit, clearance=None if hit else round(float(clearance(r['trace'])), 3))
    return out


def main(a):
    rows = {}
    for s in ('s1', 's2', 's3'):
        for line in open(Path(a.b26) / f'final_{s}_dev.jsonl', encoding='utf-8'):
            r = json.loads(line)
            rows[f"policy_{s}|{r['case_id']}|{r['sut']}"] = r['actions']
    jobs = []
    for line in open(W / 'runs/E1_trajectories.jsonl', encoding='utf-8'):
        r = json.loads(line)
        key = f"{r['source']}|{r['case_id']}|{r['sut']}"
        if r['source'].startswith('policy_'):
            jobs.append((key, r['case_id'], rows[key], None))
        else:
            jobs.append((key, r['case_id'], None, r['hwenv']['theta']))
    if a.limit:
        jobs = jobs[:a.limit]
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap(_task, jobs):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(done=len(jobs))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--b26', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    main(ap.parse_args())
