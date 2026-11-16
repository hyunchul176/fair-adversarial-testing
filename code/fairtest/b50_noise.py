"""B50 runs (B50 design, available on request): the v2j callable against the
perception-noise SUT ladder (sut_noise), and the reproducibility replays of its reported failures.

  attack  POLICY.zip OUT.jsonl --cont STORED_B11.jsonl [--workers 24]
          FairAdversaryV2(joint=True).attack on the development cases x {following, following_aeb, rss_sut} x sigma
          {0, 1, 2, 4}; the refinement's random stream uses the base SUT's B49 panel position (0, 4, 6)
  repro   ATTACKS.jsonl OUT.jsonl [--reps 20] [--workers 24]
          every found attack at sigma > 0, fixed (actions or plan), replayed against the same noisy SUT with fresh
          noise realisations 1..reps (FAIRTEST_NOISE_REP); records the SUT contact of each replicate
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

from fairtest import devboot  # noqa: F401
from fairtest import sut_noise

BASES = (('following', 0., 0), ('following_aeb', 0., 4), ('rss_sut', 0., 6))  # name, H, B49 panel position
SIGMAS = (0., 1., 2., 4.)
_AGENT = []


def _attack(job):
    policy, case_id, sut, boundary, panel = job
    from fairtest.fair_agent_v2 import FairAdversaryV2
    if not _AGENT:
        _AGENT.append(FairAdversaryV2(policy, joint=True))
    try:
        return _AGENT[0].attack(case_id, sut, boundary=boundary, panel_index=panel)
    except Exception as e:  # recorded, never dropped
        return dict(case_id=case_id, sut=f'{sut[0]}_H{sut[1]:g}', error=f'{type(e).__name__}: {e}')


def attack_main(a):
    from fairtest import runner
    from fairtest.fair_agent_v2 import _init_worker
    _init_worker()
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if v['event'].get('partition') == 'development')
    jobs = [(a.policy, c, (f'{name}~n{s:g}', h), boundary.get(c), panel) for c in ids for name, h, panel in BASES for s in SIGMAS]
    with mp.Pool(a.workers, initializer=_init_worker) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(_attack, jobs):
            f.write(json.dumps(res) + '\n')
            f.flush()
    print(json.dumps(dict(finished=len(jobs))))


def _repro(job):
    case_id, sut, how, payload, rep = job
    os.environ['FAIRTEST_NOISE_REP'] = str(rep)
    from fairtest import runner
    from fairtest.b29_consensus import _run
    runner._init()
    name, h = sut.rsplit('_H', 1)
    try:
        r = _run(runner._STATE['cases'][case_id], how, payload, name, float(h))
        return dict(case_id=case_id, sut=sut, rep=rep, sut_contact=bool(r['pair_contact']))
    except Exception as e:  # recorded, never dropped
        return dict(case_id=case_id, sut=sut, rep=rep, error=f'{type(e).__name__}: {e}')


def repro_main(a):
    jobs = []
    for line in open(a.attacks, encoding='utf-8'):
        r = json.loads(line)
        if 'error' in r or not r.get('found'):
            continue
        if sut_noise.split(r['sut'].rsplit('_H', 1)[0])[1] in (None, 0.):
            continue
        for rep in range(1, a.reps + 1):
            jobs.append((r['case_id'], r['sut'], r['attack']['how'], r['attack']['payload'], rep))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(_repro, jobs, chunksize=4):
            f.write(json.dumps(res) + '\n')
            f.flush()
    print(json.dumps(dict(replays=len(jobs))))


if __name__ == '__main__':
    sut_noise.install()  # before the pools fork
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('attack')
    p.add_argument('policy')
    p.add_argument('out')
    p.add_argument('--cont', required=True)
    p.add_argument('--workers', type=int, default=24)
    q = sub.add_parser('repro')
    q.add_argument('attacks')
    q.add_argument('out')
    q.add_argument('--reps', type=int, default=20)
    q.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    attack_main(a) if a.cmd == 'attack' else repro_main(a)
