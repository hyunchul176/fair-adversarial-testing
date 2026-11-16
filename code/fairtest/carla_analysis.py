"""E7 analysis: in-CARLA search outcomes per SUT and objective.

For each SUT kind (our ladder controllers in CARLA physics, CARLA BehaviorAgent profiles) and objective:
  fair     share of cases with a fair failure found (SUT collides, CARLA-CC avoids), queries to find it
  collide  share of cases with a SUT collision found, and the share of those the CARLA-CC could not avoid either
           (attacker-at-fault), i.e. the CARS-style post-hoc attribution of generic falsification
plus the per-scenario fair-failure rate that post-hoc filtering of the collision search yields.

  python -m fairtest.carla_analysis OUT.json SEARCH.jsonl [SEARCH.jsonl ...]
  python -m fairtest.carla_analysis --rl OUT.json RL_<agent>.jsonl [...]   (closed-loop RL adversaries, 1 query)
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ('ladder:following:0', 'ladder:following_aeb:0', 'ladder:rss_sut:0', 'ba:cautious', 'ba:normal', 'ba:aggressive',
         'tm:default')


def main(out, paths):
    runs = collections.defaultdict(list)
    n_cases = 0
    for path in paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            n_cases += 1
            for key, v in r['runs'].items():
                runs[key].append(v)
    rep = {}
    for kind in ORDER:
        fair, col = runs.get(f'fair|{kind}', []), runs.get(f'collide|{kind}', [])
        err = sum('error' in v for v in fair + col)
        f_ok = [v for v in fair if 'error' not in v]
        c_ok = [v for v in col if 'error' not in v]
        found = [v['found'] for v in f_ok if v.get('found')]
        hits = [v['found'] for v in c_ok if v.get('found')]
        g4 = [x for x in found if 'cc_min_clearance' in x]  # recorded at search time since B24
        robust = sum(1 for x in g4 if x['fair'] and (x.get('contact_dx') or -1e9) >= 3.5 and (x.get('cc_min_clearance') or 0.) >= .5)
        rep[kind] = dict(
            cases=len(f_ok), errors=err,
            fair_found=len(found) / len(f_ok) if f_ok else None,
            fair_robust_05=(robust / len(f_ok)) if f_ok and g4 and len(g4) == len(found) else None,
            fair_queries_median=float(np.median([x['query'] for x in found])) if found else None,
            collision_found=len(hits) / len(c_ok) if c_ok else None,
            collision_attacker_at_fault=(sum(x['cc_contact'] for x in hits) / len(hits)) if hits else None,
            collision_posthoc_fair=(sum(x['fair'] for x in hits) / len(c_ok)) if c_ok else None,
            collision_queries_median=float(np.median([x['query'] for x in hits])) if hits else None)
    Path(out).write_text(json.dumps(dict(cases=n_cases, per_sut=rep), indent=1), encoding='utf-8')
    print(f'cases {n_cases}')
    print(f"{'SUT':24} {'fair found':>10} {'robust':>6} {'q':>4} | {'coll found':>10} {'att-fault':>9} {'post-hoc fair':>13} {'q':>4}  err")
    for kind, d in rep.items():
        pct = lambda x: '   n/a' if x is None else f'{100 * x:6.1f}'
        print(f"{kind:24} {pct(d['fair_found']):>10} {pct(d['fair_robust_05']):>6} {d['fair_queries_median'] or '-':>4} | {pct(d['collision_found']):>10} "
              f"{pct(d['collision_attacker_at_fault']):>9} {pct(d['collision_posthoc_fair']):>13} {d['collision_queries_median'] or '-':>4}  {d['errors']}")


def main_rl(out, paths):
    rep = {}
    for path in paths:
        agent = Path(path).stem.replace('rl_dev_', '')
        per = collections.defaultdict(list)
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for kind, v in r['runs'].items():
                per[kind].append(v)
        rep[agent] = {}
        for kind in list(ORDER) + sorted(k for k in per if k not in ORDER):  # e.g. ':hw' perception kinds
            vs = [v for v in per.get(kind, []) if 'error' not in v]
            if not vs:
                continue
            col = [v for v in vs if v['contact'] and v['valid']]
            robust = [v for v in vs if v['fair'] and 'cc_min_clearance' in v]  # G4 fields recorded since B24
            rep[agent][kind] = dict(n=len(vs), errors=len(per[kind]) - len(vs), collision=len(col) / len(vs),
                                    fair=sum(v['fair'] for v in vs) / len(vs),
                                    fair_robust_05=(sum(v['contact_dx'] >= 3.5 and v['cc_min_clearance'] >= .5 for v in robust) / len(vs))
                                    if any('cc_min_clearance' in v for v in vs) else None,
                                    attacker_at_fault=(sum(bool(v['cc_contact']) for v in col) / len(col)) if col else None,
                                    feint_share=sum(v['feints'] > 0 for v in vs) / len(vs))
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    pct = lambda x: '   n/a' if x is None else f'{100 * x:6.1f}'
    for agent, d in rep.items():
        print(f'== {agent}')
        for kind, x in d.items():
            print(f"   {kind:24} n={x['n']:3d} collision {pct(x['collision'])} fair {pct(x['fair'])} robust(0.5 m, front) {pct(x['fair_robust_05'])} "
                  f"attacker-at-fault {pct(x['attacker_at_fault'])} feints {pct(x['feint_share'])} err {x['errors']}")


if __name__ == '__main__':
    if sys.argv[1] == '--rl':
        main_rl(sys.argv[2], sys.argv[3:])
    else:
        main(sys.argv[1], sys.argv[2:])
