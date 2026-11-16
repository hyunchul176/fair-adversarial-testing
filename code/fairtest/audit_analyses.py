"""Stored-data analyses for the 2026-09-30 audit response (audit response notes §3, available on request). No simulation.

  budget     detection within k SUT queries (development): the unscreened and the screened BO alone (no policy), v2j and
             v2s, by SUT group (audit A5)
  seeds      per-seed v2j detection at 1 and 37 queries, development and MiTra, and P1 per seed (A16, A7)
  recordings per-recording v2j and v2s detection at 37 queries on the three development recordings (A16)
  duplicate  rates without the second case of a cutting-in vehicle that appears twice (A16)
  references attacker-caused share of the baselines' collisions under each stored reference (A4)
  intervals  the per-case values behind two identical intervals (AW-default confirmed, following H0 at 37; A35)
  efficiency (2026-09-30, the learned policy's contribution) per SUT group: the policy's one-query detection, the screened
             search alone at 1..36 queries, and the smallest budget at which the search alone matches the policy's one
             query; groups split by the PPO training panel (b17_rl.PANEL) and unseen SUTs; Autoware from the B59 summary

  python -m fairtest.audit_analyses OUT.json --jobs fairtest_jobs --pool fairtest/data/case_pool.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest import b56_analysis as A

GROUPS = {'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'), 'BC human': ('bc_human_H0',),
          'ADS 2.0 m': ('ads@2.0_H0',), 'learned lane-assignment': ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
          'competent': ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
                        'drl_sut@3_H0', 'drl_sut@6_H0')}
KEY = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'bc_human_H0', 'ads@2.0_H0', 'drl_sut@p1.5_H0',
       'drl_sut@p2_H0', 'drl_sut@p2.5_H0', 'rss_sut_H0', 'following_aeb_H0')
BASELINES = ('collide_commit', 'cars', 'dynasto', 'naive_fair')


def search_only(ref, cases, s, k):
    return float(np.mean([ref.get((c, s)) is not None and ref[(c, s)] <= k for c in cases]))


def per_seed(robust, inside, ref, seeds, cases, s, k):
    return [float(np.mean([(robust.get((x, c, s), False) and inside.get((x, c, s), False))
                           or (k >= 2 and ref.get((c, s)) is not None and 1 + ref[(c, s)] <= k) for c in cases])) for x in seeds]


def main(a):
    J = Path(a.jobs)
    rep = {}
    robust, inside = A.policy(J, 'dev')
    seeds = sorted({k[0] for k in robust})
    stored = A.refine([J / 'b46' / 'joint_dev_rule.jsonl', J / 'b46' / 'joint_dev_learned.jsonl'], 'bo_warm_g4j')[0]
    g4j = A.refine([J / 'b56' / 'b56_dev.jsonl'], 'bo_warm_g4j')[0]
    g4s = A.refine([J / 'b56' / 'b56_dev.jsonl'], 'bo_warm_g4s')[0]
    cases_of = lambda s: sorted({k[1] for k in robust if k[2] == s})  # noqa: E731
    rep['budget'] = {}
    for g, suts in GROUPS.items():
        e = {}
        for tag, fn in (('g4j_only', lambda s, k: search_only(g4j, cases_of(s), s, k)), ('g4s_only', lambda s, k: search_only(g4s, cases_of(s), s, k)),
                        ('v2j', lambda s, k: float(A.compose(robust, inside, g4j, seeds, cases_of(s), s, k).mean())),
                        ('v2s', lambda s, k: float(A.compose(robust, inside, g4s, seeds, cases_of(s), s, k).mean()))):
            ks = (1, 6, 36) if tag.endswith('only') else (1, 6, 37)
            e[tag] = [round(float(np.mean([fn(s, k) for s in suts])), 4) for k in ks]
        rep['budget'][g] = e
    rep['seeds'] = {}
    mr, mi = A.policy(J, 'mitra')
    mref = A.refine([J / 'b41' / 'refine.jsonl'], 'bo_warm_g4j')[0]
    for ds, (rb, ins, ref) in (('dev', (robust, inside, stored)), ('mitra', (mr, mi, mref))):
        sd = sorted({k[0] for k in rb})
        e = {}
        for s in KEY:
            cs = sorted({k[1] for k in rb if k[2] == s})
            e[s] = dict(k1=[round(v, 4) for v in per_seed(rb, ins, ref, sd, cs, s, 1)], k37=[round(v, 4) for v in per_seed(rb, ins, ref, sd, cs, s, 37)])
        h0, rss = e['following_H0']['k37'], e['rss_sut_H0']['k37']
        e['P1_per_seed'] = [round(x - y, 4) for x, y in zip(h0, rss)]
        rep['seeds'][ds] = e
    pool = json.load(open(a.pool, encoding='utf-8'))
    dev = [c for c in pool['cases'] if c['event'].get('partition') == 'development']
    rec = {c['case_id']: c['event']['recording'] for c in dev}
    rep['recordings'] = {'cases': dict(collections.Counter(rec.values()))}
    for s in KEY:
        rep['recordings'][s] = {str(r): dict(v2j=round(float(A.compose(robust, inside, stored, seeds, [c for c in cases_of(s) if rec[c] == r], s, 37).mean()), 4),
                                             v2s=round(float(A.compose(robust, inside, g4s, seeds, [c for c in cases_of(s) if rec[c] == r], s, 37).mean()), 4))
                                for r in sorted(set(rec.values()))}
    veh = collections.defaultdict(list)
    for c in dev:
        veh[(c['event']['recording'], c['event']['vehicle'])].append(c['case_id'])
    dup = {f'{k[0]}:{k[1]}': v for k, v in veh.items() if len(v) > 1}
    drop = {v[1] for v in veh.values() if len(v) > 1}
    rep['duplicate'] = dict(shared_vehicles=dup, rates={s: dict(all=round(float(A.compose(robust, inside, stored, seeds, cases_of(s), s, 37).mean()), 4),
                                                                  without=round(float(A.compose(robust, inside, stored, seeds, [c for c in cases_of(s) if c not in drop], s, 37).mean()), 4))
                                                          for s in KEY})
    rep['references'] = {}
    for att in BASELINES:
        tot, n = collections.Counter(), 0
        for p in sorted((J / 'b23' / 'eval').glob(f'{att}_s*_dev.jsonl')):
            for line in open(p, encoding='utf-8'):
                r = json.loads(line)
                if not (r.get('valid') and r.get('contact') and r.get('refs')):
                    continue
                n += 1
                for ref, v in r['refs'].items():
                    if 'contact' in v:
                        tot[ref] += bool(v['contact'])
        rep['references'][att] = dict(collisions=n, attacker_caused={k: round(v / n, 4) for k, v in sorted(tot.items())})
    h0 = A.compose(robust, inside, stored, seeds, cases_of('following_H0'), 'following_H0', 37)
    aw = json.load(open(J / 'b55' / 'b55_dev' / 'B54_SUMMARY_percase.json', encoding='utf-8'))['per_case_confirmed']['aw_default']
    awv = np.array([aw[str(c)] for c in cases_of('following_H0')])
    rep['intervals'] = dict(following_H0=sorted(collections.Counter(np.round(h0, 3).tolist()).items()),
                            aw_default=sorted(collections.Counter(np.round(awv, 3).tolist()).items()),
                            same_vector=bool(np.allclose(h0, awv)), means=[round(float(h0.mean()), 4), round(float(awv.mean()), 4)])
    rep['efficiency'] = {}
    for g, suts in (('training-panel weak', ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')),
                    ('unseen learned lane-assignment', ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')),
                    ('unseen BC human', ('bc_human_H0',)), ('unseen ADS 2.0 m', ('ads@2.0_H0',))):
        pol = float(np.mean([A.compose(robust, inside, g4s, seeds, cases_of(s), s, 1).mean() for s in suts]))
        curve = [float(np.mean([search_only(g4s, cases_of(s), s, k) for s in suts])) for k in range(1, 37)]
        match = next((k for k, v in enumerate(curve, 1) if v >= pol - 1e-9), None)
        rep['efficiency'][g] = dict(policy_k1=round(pol, 4), search_only={k: round(curve[k - 1], 4) for k in (1, 6, 12, 24, 36)},
                                    search_queries_to_match_policy_k1=match)
    b59 = J / 'b59' / 'b59_dev' / 'B59_SUMMARY.json'
    if b59.exists():
        s59 = json.load(open(b59, encoding='utf-8'))
        c59 = json.load(open(J / 'b59' / 'b59_dev' / 'B59_COMPARE_corrected.json', encoding='utf-8'))
        rep['efficiency']['Autoware'] = {v: dict(policy_stage_reported=s59['variants'][v]['by_stage']['policy'],
                                                 refine_reported=s59['variants'][v]['by_stage']['refine'],
                                                 episodes=s59['variants'][v]['A1']['episodes'],
                                                 refine_autoware_queries_per_success=c59['refine']['b59'][v]['queries_per_found'])
                                         for v in ('aw_default', 'aw_r157')}
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep['budget'], indent=1))
    print(json.dumps(rep['efficiency'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--pool', required=True, help='the development case pool (fairtest/data/case_pool.json, md5 b4048cb9...)')
    main(ap.parse_args())
