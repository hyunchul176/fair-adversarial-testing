"""B65 re-judging: b57_rejudge unchanged except that only the populations in LABELS run (the two B57 main populations,
the paired bracket that perceives at the CC's 1.095 m, and the appended bracket that perceives later, at 1.5 m). The
driver indices of the population file are not touched, so a driver's marks equal its B57 marks for the same job.

  FAIRTEST_HUMAN_POP=population.json python -m fairtest.b65_rejudge JOBS.jsonl OUT.jsonl --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

from fairtest import b57_rejudge as B

LABELS = ('attentive', 'less_attentive', 'att_perceive_1.095', 'att_perceive_1.5')
_all_pops = B.pops


def pops():
    return [p for p in _all_pops() if p[0] in LABELS]


B.pops = pops  # b57_rejudge.task looks the name up at call time; forked workers inherit the patch


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    missing = set(LABELS) - {p[0] for p in _all_pops()}
    if missing:
        raise SystemExit(f'population file lacks {sorted(missing)}')
    jobs = [json.loads(line) for line in open(a.jobs, encoding='utf-8') if line.strip()]
    done = set()
    if os.path.exists(a.out):  # resume: finished jobs are skipped
        done = {json.loads(line)['job'] for line in open(a.out, encoding='utf-8') if line.strip()}
    todo = [j for j in jobs if j['job'] not in done]
    t0 = time.time()
    with mp.Pool(a.workers) as pool, open(a.out, 'a', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(B.task, todo, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(done=i + 1, of=len(todo), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(todo), elapsed_s=round(time.time() - t0, 1))), flush=True)
