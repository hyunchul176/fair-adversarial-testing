"""B67 (B67 design, available on request): the final agent's screened refinement
bo_warm_g4s re-run with other random streams.

b23_search seeds each (case, SUT, method) with np.random.default_rng([case_id, si, METHODS.index(base)]). Stream s >= 1
appends s, default_rng([case_id, si, index, s]); stream 0 is the stored run. b23_search is not edited: each worker replaces
that module's numpy reference by a shim whose random.default_rng appends the stream index (stream 0 passes the seed through
unchanged: the transparency pre-check) and records the seeds it was given; every other attribute is numpy itself.
Jobs as b23_search --split-suts (one per case and SUT, streams keep their --suts position), method bo_warm_g4s only.

  python -m fairtest.b67_streams run OUT.jsonl --stream S --cont STORED_B11.jsonl --partition development \
         --suts following:0,... [--budget 36] [--workers 24] [--cases N]
  python -m fairtest.b67_streams check NEW.jsonl STORED_B56.jsonl --stream S --expect same|different
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as _np

from fairtest import devboot  # noqa: F401

METHOD = 'bo_warm_g4s'
SEEN = []


class _Random:
    def __init__(self, stream):
        self.stream = stream

    def default_rng(self, seed=None):
        seed = list(seed) if self.stream == 0 else list(seed) + [self.stream]
        SEEN.append([int(x) for x in seed])
        return _np.random.default_rng(seed)

    def __getattr__(self, k):
        return getattr(_np.random, k)


class _Numpy:
    def __init__(self, stream):
        self.random = _Random(stream)

    def __getattr__(self, k):
        return getattr(_np, k)


STATE = {}


def _init_worker(stream):
    from fairtest import b23_search as B
    B.np = _Numpy(stream)
    STATE['stream'] = stream


def task(args):
    from fairtest import b23_search as B
    SEEN.clear()
    out = B.task(args)
    out.update(stream=STATE['stream'], seeds=list(SEEN))
    return out


def run(a):
    from fairtest import b23_search as B
    from fairtest import runner
    runner._init()
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    if a.cases:
        ids = ids[:a.cases]
    suts = tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    keys = [f'{n}_H{h:g}' for n, h in suts]
    jobs = [(c, boundary.get(c), suts, (METHOD,), a.budget, frozenset({k}), False) for c in ids for k in keys]
    assert METHOD in B.METHODS
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(a.stream,)) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, jobs, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if (i + 1) % 20 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), stream=a.stream, elapsed_s=round(time.time() - t0, 1))), flush=True)


def runs(path):
    out, lines = {}, []
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        lines.append(r)
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == METHOD:
                out[(r['case_id'], s)] = rec
    return out, lines


def check(a):
    """same: every (case, SUT) equals the stored run (first success, SUT queries, best plan); different: the stream's own
    seeds only, no errors, and at least one pair differs from the stored run."""
    from fairtest import b23_search as B
    new, lines = runs(a.new)
    old, _ = runs(a.stored)
    idx = B.METHODS.index('bo_warm')
    bad_seeds = []
    for r in lines:
        want = [r['case_id'], None, idx] + ([] if a.stream == 0 else [a.stream])
        for sd in r.get('seeds', []):
            if len(sd) != len(want) or sd[0] != want[0] or sd[2:] != want[2:]:
                bad_seeds.append([r['case_id'], sd])
        if r.get('stream') != a.stream or len(r.get('seeds', [])) != 1:
            bad_seeds.append([r['case_id'], r.get('stream'), len(r.get('seeds', []))])
    errors = [k for k, v in new.items() if 'error' in v]
    diff = []
    for k, x in new.items():
        y = old.get(k)
        if y is None or 'error' in x or 'error' in y:
            diff.append([*k, 'missing or error'])
            continue
        if not (x.get('first_fair') == y.get('first_fair') and x.get('sut_queries') == y.get('sut_queries')
                and all(abs(p - q) < 1e-9 for p, q in zip(x['best']['theta'], y['best']['theta']))):
            diff.append([*k, x.get('first_fair'), y.get('first_fair'), x.get('sut_queries'), y.get('sut_queries')])
    rep = dict(stream=a.stream, expect=a.expect, pairs=len(new), errors=len(errors), bad_seeds=bad_seeds[:10],
               differing=len(diff), identical=len(new) - len(diff), rows=diff[:10])
    ok = bool(new) and not errors and not bad_seeds and (not diff if a.expect == 'same' else len(diff) > 0)
    rep['ok'] = ok
    print(json.dumps(rep))
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run')
    p.add_argument('out')
    p.add_argument('--stream', type=int, required=True)
    p.add_argument('--cont', required=True)
    p.add_argument('--partition', default='development')
    p.add_argument('--suts', required=True)
    p.add_argument('--budget', type=int, default=36)
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--cases', type=int, default=0)
    c = sub.add_parser('check')
    c.add_argument('new')
    c.add_argument('stored')
    c.add_argument('--stream', type=int, required=True)
    c.add_argument('--expect', choices=('same', 'different'), required=True)
    a = ap.parse_args()
    if a.cmd == 'run':
        run(a)
    else:
        sys.exit(check(a))
