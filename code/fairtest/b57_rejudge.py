"""B57 re-judging: every job of b57_jobs.py replayed against the R157 CC (must reproduce the stored verdict: the edited
CCDriver is unchanged at its defaults) and against the human-like reference drivers of FAIRTEST_HUMAN_POP (target
cc_r157@h<k>). The main populations run for every job, the bracket populations (sensitivity) only for jobs flagged
`bracket`. No SUT runs.

Per job: cc_contact, cc_matches_stored, cc_clearance, and `collide` = {population: one character per driver ('1'
contact, '0' avoided, 'e' replay error)}, so P(avoid) = share of '0' among the drivers that ran.

  FAIRTEST_HUMAN_POP=population.json python -m fairtest.b57_rejudge JOBS.jsonl OUT.jsonl --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

from fairtest import devboot  # noqa: F401
from fairtest.b29_consensus import _run

_POPS = []


def pops():
    """[(label, is_bracket, [driver indices])] of the population file, in file order."""
    if not _POPS:
        d = json.load(open(os.environ['FAIRTEST_HUMAN_POP'], encoding='utf-8'))
        brackets = set(d['spec'].get('brackets', {}))
        order = []
        for drv in d['drivers']:
            if not order or order[-1][0] != drv['pop']:
                order.append((drv['pop'], drv['pop'] in brackets, []))
            order[-1][2].append(drv['k'])
        _POPS.append(order)
    return _POPS[0]


def task(job):
    from fairtest import runner
    from fairtest.b23_margin import clearance
    runner._init()
    case = runner._STATE['cases'][job['case_id']]
    t0 = time.time()
    out = {k: v for k, v in job.items() if k != 'payload'}
    try:
        raw = _run(case, job['how'], job['payload'], 'cc_r157')
        cc = bool(raw['pair_contact'])
        out.update(cc_contact=cc, cc_matches_stored=cc == bool(job['stored_cc']), cc_clearance=round(float(clearance(raw['trace'])), 3))
    except Exception as e:  # recorded, never dropped
        out.update(cc_error=f'{type(e).__name__}: {e}'[:300])
    collide = {}
    for label, is_bracket, ks in pops():
        if is_bracket and not job.get('bracket'):
            continue
        marks = []
        for k in ks:
            try:
                marks.append('1' if _run(case, job['how'], job['payload'], f'cc_r157@h{k}')['pair_contact'] else '0')
            except Exception:
                marks.append('e')
        collide[label] = ''.join(marks)
    out.update(collide=collide, seconds=round(time.time() - t0, 1))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = [json.loads(line) for line in open(a.jobs, encoding='utf-8') if line.strip()]
    done = set()
    if os.path.exists(a.out):  # resume: finished jobs are skipped
        done = {json.loads(line)['job'] for line in open(a.out, encoding='utf-8') if line.strip()}
    todo = [j for j in jobs if j['job'] not in done]
    t0 = time.time()
    with mp.Pool(a.workers) as pool, open(a.out, 'a', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, todo, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(done=i + 1, of=len(todo), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(todo), elapsed_s=round(time.time() - t0, 1))), flush=True)
