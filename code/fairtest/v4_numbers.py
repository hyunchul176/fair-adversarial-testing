"""Stored-data numbers for manuscript v4 (internal review notes R3, R8 and R12, available on request). No simulation.

  efficiency  per dataset (development in-sample; TGSIM and MiTra post hoc) and SUT group: the final agent's detection at
              1 / 6 / 37 SUT queries, the screened search alone at 1 / 6 / 36, the smallest budget at which the search alone
              matches the policy's one-query rate, and the policy stage's share of the final agent's detection (one query over
              37, the nine detectable SUTs summed)
  duplicate   development: the final agent's rates (37 queries) without the second case of each vehicle that appears in
              two cases, as generator (60:332) or as follower (58:291, 60:738); maximum change over the 17 SUTs
  recordings  development: the final agent's rates (37 queries) per recording

  python -m fairtest.v4_numbers OUT.json --jobs fairtest_jobs --pool fairtest/data/case_pool.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest import b56_analysis as A
from fairtest.audit_analyses import GROUPS, search_only

DETECTABLE = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'bc_human_H0', 'ads@2.0_H0', 'drl_sut@p1.5_H0',
              'drl_sut@p2_H0', 'drl_sut@p2.5_H0')
RUNS = dict(dev='b56/b56_dev.jsonl', tgsim='b56/b56_tgsim.jsonl', mitra='b56/b56_mitra.jsonl')


def main(a):
    J = Path(a.jobs)
    rep = dict(efficiency={})
    for ds, run in RUNS.items():
        robust, inside = A.policy(J, ds)
        seeds = sorted({k[0] for k in robust})
        g4s = A.refine([J / run], 'bo_warm_g4s')[0]
        cases_of = lambda s: sorted({k[1] for k in robust if k[2] == s})  # noqa: E731
        comp = lambda s, k: float(A.compose(robust, inside, g4s, seeds, cases_of(s), s, k).mean())  # noqa: E731
        e = dict(cases=len({k[1] for k in robust}), groups={})
        for g, suts in GROUPS.items():
            pol = float(np.mean([comp(s, 1) for s in suts]))
            curve = [float(np.mean([search_only(g4s, cases_of(s), s, k) for s in suts])) for k in range(1, 37)]
            match = next((k for k, v in enumerate(curve, 1) if v >= pol - 1e-9), None) if pol > 0 else None
            e['groups'][g] = dict(agent=[round(comp_k, 4) for comp_k in (pol, float(np.mean([comp(s, 6) for s in suts])),
                                                                       float(np.mean([comp(s, 37) for s in suts])))],
                                  search_only=[round(curve[k - 1], 4) for k in (1, 6, 36)], search_queries_to_match_policy_k1=match)
        k1 = sum(comp(s, 1) for s in DETECTABLE)
        k37 = sum(comp(s, 37) for s in DETECTABLE)
        e['policy_share_of_detection'] = round(k1 / k37, 4)
        rep['efficiency'][ds] = e
    robust, inside = A.policy(J, 'dev')
    seeds = sorted({k[0] for k in robust})
    g4s = A.refine([J / RUNS['dev']], 'bo_warm_g4s')[0]
    pool = json.load(open(a.pool, encoding='utf-8'))
    dev = [c for c in pool['cases'] if c['event'].get('partition') == 'development']
    seen, drop, shared = collections.defaultdict(list), set(), {}
    for c in sorted(dev, key=lambda c: c['case_id']):
        for role, v in (('generator', c['event']['vehicle']), ('follower', c.get('follower'))):
            if v is None:
                continue
            key = (c['event']['recording'], v)
            if seen[key]:
                drop.add(c['case_id'])
                shared[f"{key[0]}:{key[1]}"] = seen[key] + [c['case_id']]
            seen[key].append(c['case_id'])
    suts = sorted({k[2] for k in robust})
    cases_of = lambda s: sorted({k[1] for k in robust if k[2] == s})  # noqa: E731
    rates = {s: dict(all=round(float(A.compose(robust, inside, g4s, seeds, cases_of(s), s, 37).mean()), 4),
                     without=round(float(A.compose(robust, inside, g4s, seeds, [c for c in cases_of(s) if c not in drop], s, 37).mean()), 4))
             for s in suts}
    rep['duplicate'] = dict(shared=shared, dropped=sorted(drop), rates=rates,
                            max_change_pp=round(max(abs(v['all'] - v['without']) for v in rates.values()) * 100, 2))
    rec = {c['case_id']: c['event']['recording'] for c in dev}
    rep['recordings'] = dict(cases=dict(collections.Counter(rec.values())),
                             rates={s: {str(r): round(float(A.compose(robust, inside, g4s, seeds, [c for c in cases_of(s) if rec[c] == r], s, 37).mean()), 4)
                                        for r in sorted(set(rec.values()))} for s in suts})
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for ds, e in rep['efficiency'].items():
        print(ds, 'cases', e['cases'], 'policy share', e['policy_share_of_detection'])
        for g, v in e['groups'].items():
            print(f'  {g:24s} agent {v["agent"]} search {v["search_only"]} match {v["search_queries_to_match_policy_k1"]}')
    print('duplicate', rep['duplicate']['shared'], 'max change pp', rep['duplicate']['max_change_pp'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--pool', required=True)
    main(ap.parse_args())
