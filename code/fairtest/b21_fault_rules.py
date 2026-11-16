"""B21 (G2/G10): agreement of fault oracles on the same collisions (exposed highD, generic falsification).

For every SUT collision found by falsify_centre, seven judgements of "the SUT is at fault":
  nuplan      benchmark-style geometric rule: the SUT is the active party if its front end hits the other
              vehicle (rear-end / front contact); a side-swipe onto the lane-keeping SUT is not SUT fault;
  oracle_t0   omniscient longitudinal solvability (B16);  oracle_react0  ideal ADS (B16);
  cc_distance R157 CC, fixed 1.095 m perception;  cc_time  R157 CC, 0.4 s time-based perception (B21);
  fsm, rss    Mattas et al. references (B10).
Reports the share attributed to the SUT per rule and SUT, and pairwise Cohen's kappa.

  python -m fairtest.b21_fault_rules geometry OUT.jsonl --workers W   (re-runs the collisions with traces)
  python -m fairtest.b21_fault_rules analyze GEOMETRY.jsonl OUT.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

from fairtest.b21_reference import kappa
from fairtest.strict_analysis import E13, J

FRONT_DX_M = 3.5  # generator centre at least this far ahead of the SUT centre at contact = SUT front contact
ORDER = ('following_H0', 'reference_idm_H0', 'following_H0.5', 'following_H1', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')


def _geom(job):
    case_id, key, theta = job
    from fairtest import runner
    from fairtest.b11_search import ContinuousPlan
    runner._init()
    name, h = key.rsplit('_H', 1)
    row = runner.run_episode(runner._STATE['cases'][case_id], ContinuousPlan.from_theta(theta), name, float(h), keep_trace=True)
    last = row['trace'][-1]
    g, t = last['generator'], last['target']
    dx, dy = g['x'] - t['x'], g['y'] - t['y']
    return dict(case_id=case_id, sut=key, contact=bool(row['pair_contact']), dx=float(dx), dy=float(dy),
                front=bool(row['pair_contact'] and dx >= FRONT_DX_M), sut_speed=float(t['vx']))


def geometry(a):
    jobs = []
    for line in open(E13 / 'falsify' / 'b13_falsify_highd_dev.jsonl', encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            start, k = key.split('|')
            if start == 'falsify_centre' and rec['best'] and rec['best']['contact']:
                jobs.append((r['case_id'], k, rec['best']['theta']))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(_geom, jobs, chunksize=4):
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(finished=len(jobs))))


def analyze(a):
    geo = {(r['case_id'], r['sut']): r for r in (json.loads(l) for l in open(a.geometry, encoding='utf-8'))}
    b16 = {(r['case_id'], r['key']): r['refs'] for r in (json.loads(l) for l in open(J / 'b16' / 'b16_reeval_exposed.jsonl', encoding='utf-8')) if r['source'] == 'falsify'}
    b21 = {(r['case_id'], r['key']): r['refs'] for r in (json.loads(l) for l in open(J / 'b21' / 'reeval_exposed_cctime.jsonl', encoding='utf-8')) if r['source'] == 'falsify'}
    rules = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(E13 / 'falsify' / 'b13_falsify_highd_dev.jsonl', encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            start, k = key.split('|')
            if start != 'falsify_centre' or not (rec['best'] and rec['best']['contact']):
                continue
            c = r['case_id']
            g = geo.get((c, k))
            if g is None or not g['contact']:
                continue
            j = {'nuplan': g['front'],
                 'oracle_t0': not b16[(c, key)]['oracle_t0']['contact'],
                 'oracle_react0': not b16[(c, key)]['oracle_react0']['contact'],
                 'cc_distance': not rec['refs']['cc_r157']['contact'],
                 'cc_time': not b21[(c, key)]['cc_r157_time']['contact'],
                 'fsm': not rec['refs']['ref_fsm']['contact'],
                 'rss': not rec['refs']['ref_rss']['contact']}
            for name, v in j.items():
                rules[k][name].append(v)
                rules['_all'][name].append(v)
    names = ('nuplan', 'oracle_t0', 'oracle_react0', 'cc_distance', 'cc_time', 'fsm', 'rss')
    rep = dict(attributed={k: {n: float(np.mean(rules[k][n])) for n in names} | dict(n=len(rules[k]['nuplan'])) for k in list(ORDER) + ['_all'] if k in rules},
               kappa={f'{x}~{y}': kappa(rules['_all'][x], rules['_all'][y]) for x, y in itertools.combinations(names, 2)})
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('share of collisions attributed to the SUT')
    for k, d in rep['attributed'].items():
        print(f'  {k:20} n={d["n"]:5d}', ' '.join(f'{n} {100*d[n]:5.1f}' for n in names))
    print('kappa', {k: round(v, 2) for k, v in rep['kappa'].items()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('geometry')
    g.add_argument('out')
    g.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    q = sub.add_parser('analyze')
    q.add_argument('geometry')
    q.add_argument('out')
    a = ap.parse_args()
    geometry(a) if a.cmd == 'geometry' else analyze(a)
