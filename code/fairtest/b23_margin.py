"""B23 (G4): decision margin of fair failures in highway-env.

A fair failure (SUT collides, CC does not) is only as convincing as the careful driver's escape. For every fair
failure of the RL adversaries (evaluation rows with their action sequences) and of the fair CMA-ES (stored plans),
the attack is replayed against the CC with the full trace and the SUT with the trace, and we record
  cc_clearance  smallest distance between the CC's and the adversary's bounding boxes (m; 0 = contact)
  contact_dx    adversary centre minus SUT centre at the SUT's contact (>= 3.5 m: front contact)
A robust fair failure has cc_clearance >= delta (reported for 0.25, 0.5, 1.0 m) and a front contact.
Severity of the CC's escape (L5, following traffic): cc_min_acc (m/s^2), cc_hard_s (s below -4 m/s^2) and the CC's
speed profile cc_v (0.1 s), from which a follower behind the CC is simulated post hoc.

  python -m fairtest.b23_margin OUT.jsonl --evals EVAL.jsonl ... [--cont CONT.jsonl ...] --workers W
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

from fairtest import devboot  # noqa: F401


def corners(s):
    c, n = math.cos(s['heading']), math.sin(s['heading'])
    hl, hw = s['length'] / 2, s['width'] / 2
    return [(s['x'] + dx * c - dy * n, s['y'] + dx * n + dy * c) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _overlap(p, q):
    for poly in (p, q):
        for i in range(4):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
            ax, ay = y1 - y2, x2 - x1
            pa = [ax * x + ay * y for x, y in p]
            qa = [ax * x + ay * y for x, y in q]
            if max(pa) < min(qa) or max(qa) < min(pa):
                return False
    return True


def _seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = max(0., min(1., ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def box_distance(a, b):
    p, q = corners(a), corners(b)
    if _overlap(p, q):
        return 0.
    d = math.inf
    for u, v in ((p, q), (q, p)):
        for x, y in u:
            for i in range(4):
                d = min(d, _seg(x, y, *v[i], *v[(i + 1) % 4]))
    return d


def clearance(trace):
    return min(box_distance(r['generator'], r['target']) for r in trace)


def escape(case, trace):
    """The CC's speed profile from t = 0 and its braking severity."""
    t0 = case['physical_initial_state'][1]
    v = [math.hypot(t0['vx'], t0['vy'])] + [math.hypot(r['target']['vx'], r['target']['vy']) for r in trace]
    acc = [(b - a) / .1 for a, b in zip(v, v[1:])]
    return dict(cc_min_acc=round(min(acc), 2), cc_hard_s=round(.1 * sum(a < -4. for a in acc), 1), cc_v=[round(x, 2) for x in v])


def _task(job):
    source, case_id, sut, how, payload = job
    from fairtest import runner
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_crossjudge import replay
    runner._init()
    case = runner._STATE['cases'][case_id]
    name, h = sut.rsplit('_H', 1)
    if how == 'actions':
        cc = replay(case, payload, 'cc_r157', 0.)
        s = replay(case, payload, name, float(h))
    else:
        cc = runner.run_episode(case, ContinuousPlan.from_theta(payload), 'cc_r157', 0., keep_trace=True)
        s = runner.run_episode(case, ContinuousPlan.from_theta(payload), name, float(h), keep_trace=True)
    last = s['trace'][-1]
    return dict(source=source, case_id=case_id, sut=sut, cc_contact=bool(cc['pair_contact']), sut_contact=bool(s['pair_contact']),
                cc_clearance=round(clearance(cc['trace']), 3), contact_dx=round(last['generator']['x'] - last['target']['x'], 2),
                **escape(case, cc['trace']))


def jobs_from(evals, conts):
    jobs = []
    for path in evals:
        if Path(path).name.startswith(('cross_', 'margin_')):  # outputs sharing the *_<dataset>.jsonl pattern
            continue
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'valid' not in r:
                continue
            if r['valid'] and r['contact'] and r.get('refs') and not r['refs']['cc_r157'].get('contact', True):  # unjudged: skip
                jobs.append((Path(path).name, r['case_id'], r['sut'], 'actions', r['actions']))
    for path in conts:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for k, ss in (r.get('sut_search') or {}).items():
                b = ss.get('best')
                if b and b['fair']:
                    jobs.append((Path(path).name, r['case_id'], k, 'theta', b['theta']))
    return jobs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--evals', nargs='*', default=[])
    ap.add_argument('--cont', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = jobs_from(a.evals, a.cont)
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(_task, jobs, chunksize=2):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(finished=len(jobs))))
