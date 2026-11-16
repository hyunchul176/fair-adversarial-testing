"""B16: re-judge stored SUT-contact plans with additional reference drivers (generalises strict_reeval).

Sources: cont_search fair failures (b11 format), falsification contacts (b13_falsify format) and B14 hits
(b14_realism_search format). Each plan is re-run once per reference.

  python -m fairtest.b16_reeval OUT.jsonl --refs oracle_t0,oracle_react0 --cont C.jsonl --falsify F.jsonl --b14 B.jsonl --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import evaluate


def jobs(cont, falsify, b14):
    out = []
    if cont:
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
    if b14:
        for line in open(b14, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                b = rec.get('best')
                if b and b['hit']:
                    out.append(('b14', r['case_id'], key, b['theta']))
    return out


def task(args):
    job, refs = args
    source, case_id, key, theta = job
    runner._init()
    case = runner._STATE['cases'][case_id]
    res = {}
    for name in refs:
        row = evaluate(case, theta, name, 0.)
        res[name] = dict(contact=bool(row['contact']), valid=bool(row['valid']))
    return dict(source=source, case_id=case_id, key=key, refs=res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--refs', required=True)
    ap.add_argument('--cont')
    ap.add_argument('--falsify')
    ap.add_argument('--b14')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    runner._init()
    refs = a.refs.split(',')
    todo = [(j, refs) for j in jobs(a.cont, a.falsify, a.b14)]
    with mp.Pool(a.workers, initializer=runner._init) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(task, todo, chunksize=4):
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(finished=len(todo))), flush=True)
