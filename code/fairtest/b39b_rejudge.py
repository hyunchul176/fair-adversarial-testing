"""B39b: re-judge with the R157 CC every success of a refinement run whose reference was another model.

For each (case, SUT) success of a b23_search output (method bo_warm_g4g), the best plan is replayed with cc_r157: the
human-reference verdict (contact, clearance) of an attack found under a feasibility-style reference.

  python -m fairtest.b39b_rejudge OUT.jsonl E3.jsonl --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

from fairtest import devboot  # noqa: F401


def task(job):
    case_id, sut, theta = job
    from fairtest import runner
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_margin import clearance
    runner._init()
    try:
        raw = runner.run_episode(runner._STATE['cases'][case_id], ContinuousPlan.from_theta(theta), 'cc_r157', 0., keep_trace=True)
        return dict(case_id=case_id, sut=sut, cc_contact=bool(raw['pair_contact']), cc_clearance=round(clearance(raw['trace']), 3))
    except Exception as e:  # recorded, never dropped
        return dict(case_id=case_id, sut=sut, error=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('e3')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = []
    for line in open(a.e3, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == 'bo_warm_g4g' and rec.get('first_fair') and rec.get('best'):
                jobs.append((r['case_id'], s, rec['best']['theta']))
    from fairtest import runner
    with mp.Pool(a.workers, initializer=runner._init) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(task, jobs, chunksize=4):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(jobs=len(jobs))))
