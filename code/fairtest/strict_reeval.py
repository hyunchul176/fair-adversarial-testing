"""Post-B13 reference sensitivity: re-judge continuous-search and falsification plans with cc_r157_strict.

cc_r157_strict is the R157 CC driver whose cut-in perception is released as soon as the cutting-in
vehicle returns below the perception threshold (1.095 m) instead of below its wandering zone
(0.375 m). Every stored plan that was a SUT contact is re-run once with the strict reference:
  cont_search  best theta of every (case, SUT) whose search reported a fair failure;
  falsify      best theta of every (case, start, SUT) whose search reported a SUT contact.

  python -m fairtest.strict_reeval OUT.jsonl --cont CONT.jsonl [--falsify FALSIFY.jsonl] --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import evaluate

STRICT = ('cc_r157_strict', 0.)


def jobs(cont, falsify):
    out = []
    for line in open(cont, encoding='utf-8'):
        r = json.loads(line)
        for sut, ss in r['sut_search'].items():
            b = ss.get('best')
            if b and b['fair']:
                out.append(('cont_search', r['case_id'], sut, b['theta']))
    if falsify:
        for line in open(falsify, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                b = rec.get('best')
                if b and b['contact']:
                    out.append(('falsify', r['case_id'], key, b['theta']))
    return out


def task(job):
    source, case_id, key, theta = job
    runner._init()
    row = evaluate(runner._STATE['cases'][case_id], theta, *STRICT)
    return dict(source=source, case_id=case_id, key=key, strict_contact=bool(row['contact']),
                strict_valid=bool(row['valid']), strict_emergency=row.get('cc_emergency'),
                strict_perceived_at=row.get('cc_perceived_at'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True)
    ap.add_argument('--falsify')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    runner._init()
    todo = jobs(a.cont, a.falsify)
    with mp.Pool(a.workers, initializer=runner._init) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(task, todo, chunksize=4):
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(finished=len(todo))), flush=True)
