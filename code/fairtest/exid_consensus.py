"""exiD step 7 (exiD protocol draft §8, available on request): judge every failure the final agent v2s reports on the competent SUTs with the
held-out reference drivers (b29_consensus.task, unchanged): consensus-fair = C, strict CC, FSM and the RSS reference
driver all avoid the collision.

Reported failures (the unified contract, as b41_analysis / exid_analysis compose them):
  policy stage  final_s<k>.jsonl episodes that are robust fair (margin.jsonl) and inside the joint support with G2
                (policy_replay.jsonl), replayed from their action sequences; one row per seed
  refinement    the first success of METHOD (default bo_warm_g4s) per (case, SUT), replayed from its plan theta; the same
                plan serves every seed whose policy episode did not report
Every SUT in SUTS is judged (default: the eight competent SUTs of the manuscript). Rows keep stage, seed (policy), case,
SUT, consensus_fair, grade, refs_contact and any error (errors are recorded, never dropped).

  python -m fairtest.exid_consensus DIR OUT.jsonl --refine REFINE.jsonl [...] [--method bo_warm_g4s] [--workers 30]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from fairtest import devboot  # noqa: F401

COMPETENT8 = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
              'drl_sut@3_H0', 'drl_sut@6_H0')


def jobs_for(d, refine_paths, method='bo_warm_g4s', suts=COMPETENT8):
    from fairtest.b41_analysis import load
    d = Path(d)
    evals, realistic, _, _, _ = load(d)
    jobs = []
    for path in sorted(d.glob('final_s[0-9].jsonl')):
        seed = path.stem.rsplit('_s', 1)[1]
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'valid' not in r or r['sut'] not in suts:
                continue
            m = evals['final'][seed].get((r['case_id'], r['sut']), {})
            if m.get('robust') and realistic.get((seed, r['case_id'], r['sut']), False):
                jobs.append((path.name, 'policy', r['case_id'], r['sut'], 'actions', r['actions']))
    seen = set()
    for path in refine_paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == method and s in suts and rec.get('first_fair') and rec.get('best') and (r['case_id'], s) not in seen:
                    seen.add((r['case_id'], s))
                    jobs.append((Path(path).name, 'refine', r['case_id'], s, 'theta', rec['best']['theta']))
    return jobs


def run_job(job):
    from fairtest.b29_consensus import task
    res = task(job)
    if job[1] == 'policy':
        res['seed'] = job[0].rsplit('_s', 1)[1].split('.')[0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('out')
    ap.add_argument('--refine', nargs='+', required=True)
    ap.add_argument('--method', default='bo_warm_g4s')
    ap.add_argument('--suts', default=','.join(COMPETENT8))
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = jobs_for(a.dir, a.refine, a.method, tuple(a.suts.split(',')))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(run_job, jobs, chunksize=1):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(jobs=len(jobs), policy=sum(j[1] == 'policy' for j in jobs), refine=sum(j[1] == 'refine' for j in jobs))))


if __name__ == '__main__':
    main()
