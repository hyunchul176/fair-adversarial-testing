"""B61 analysis (B61 design, fixed before the run, available on request).

Inputs in JOBS/b61 (B41 layout, read with b41_analysis.load): final_s<k>.jsonl (v2j policy stage against the SUTs of known
capability), collide_shield_s<k>.jsonl (the collision-seeking policy with the window shield, 17 SUTs), margin.jsonl,
policy_replay.jsonl (v2j) and policy_replay_c.jsonl (collide + shield), refine.jsonl (bo_warm_g4j, bo_warm_g4s against the
known SUTs), random_g4s.jsonl (17 SUTs), natural.jsonl (17 SUTs), case_pool.json. Composition: the unified contract
(robust + joint support + G2 at the policy stage, else the refinement's first success), k = 1 / 6 / 37.
  K1  SUT = C: detection 0 at every budget (implementation check)
  K2  SUTs stronger than C in one factor (reaction 0.5 s, braking 8.5 m/s^2): v2j and v2s k37 <= 3%
  K3  along each degraded factor (reaction +0.25 / +0.5 / +1.0 s; braking 6.5 / 5.5; perception 1.5 / 2.0 m) v2s k37 is
      non-decreasing and the most degraded level's 95% lower bound is > 0
  S   collide + shield: attacker-caused policy contacts 0 (implementation check); detection at one query and the
      discrimination following H0 - longitudinal RSS, next to v2j's (descriptive)
  R   screened random search: detection within 1 / 6 / 36 queries, next to the screened BO alone (B56; descriptive)
  N   attack-free natural cut-ins: share of cases whose natural plan meets the contract (descriptive)

  python -m fairtest.b61_analysis OUT.json --jobs fairtest_jobs [--b56 fairtest_jobs/b56/b56_dev.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest.b45_joint_realism import Support
from fairtest.b56_analysis import refine

KNOWN = {'C': 'cc_r157_H0', 'reaction 0.5 s (stronger)': 'cc_r157@r0.5_H0', 'reaction +0.25 s': 'cc_r157@r1_H0',
         'reaction +0.5 s': 'cc_r157@r1.25_H0', 'reaction +1.0 s': 'cc_r157@r1.75_H0', 'braking 8.5 (stronger)': 'cc_r157@h4_H0',
         'braking 6.5': 'cc_r157@h0_H0', 'braking 5.5': 'cc_r157@h1_H0', 'perception 1.5 m': 'cc_r157@h2_H0',
         'perception 2.0 m': 'cc_r157@h3_H0'}
FACTORS = {'reaction': ('C', 'reaction +0.25 s', 'reaction +0.5 s', 'reaction +1.0 s'), 'braking': ('C', 'braking 6.5', 'braking 5.5'),
           'perception': ('C', 'perception 1.5 m', 'perception 2.0 m')}
STRONGER = ('reaction 0.5 s (stronger)', 'braking 8.5 (stronger)')
STD = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0',
       'bc_human_H0', 'ads@0.5_H0', 'ads@2.0_H0', 'rss_lat_sut_H0', 'drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0', 'drl_sut@p1.5_H0',
       'drl_sut@p2_H0', 'drl_sut@p2.5_H0')


def boot(vals, n=4000, seed=61):
    vals = np.asarray(vals, float)
    m = vals[np.random.default_rng(seed).integers(0, len(vals), (n, len(vals)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def realistic_from(path, sup):
    out = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        sc = sup.score(r['features']) if r['cut_in'] else None
        out[(str(r['seed']), r['case_id'], r['sut'])] = bool(sc is not None and sc <= sup.tau and r.get('exceedance') is not None
                                                             and r['exceedance'] <= 1.)
    return out


def compose(robust, inside, ref, seeds, cases, s, k):
    return np.mean([np.array([(robust.get((x, c, s), False) and inside.get((x, c, s), False))
                              or (k >= 2 and ref.get((c, s)) is not None and 1 + ref[(c, s)] <= k) for c in cases], float)
                    for x in seeds], axis=0)


def main(a):
    from fairtest.b41_analysis import load
    d = Path(a.jobs) / 'b61'
    evals, realistic, _, _, _ = load(d)
    sup = Support()
    rep = dict(known={}, collide_shield={}, random={}, natural={})
    g4j, _, err_j = refine([d / 'refine.jsonl'], 'bo_warm_g4j')
    g4s, _, err_s = refine([d / 'refine.jsonl'], 'bo_warm_g4s')
    rep['errors'] = dict(g4j=err_j, g4s=err_s)
    fin = evals['final']
    seeds = sorted(fin)
    robust = {(x, c, s): m['robust'] for x in seeds for (c, s), m in fin[x].items()}
    unfair = sum(m['unfair'] for x in seeds for m in fin[x].values())
    rep['known_policy_attacker_caused'] = int(unfair)
    for label, s in KNOWN.items():
        cases = sorted({c for (x, c, ss) in robust if ss == s})
        if not cases:
            continue
        e = dict(n_cases=len(cases))
        for tag, ref in (('v2j', g4j), ('v2s', g4s)):
            for k in (1, 6, 37):
                v = compose(robust, realistic, ref, seeds, cases, s, k)
                e[f'{tag}_k{k}'] = dict(mean=round(float(v.mean()), 4), ci=boot(v))
        rep['known'][label] = e
    kn = rep['known']
    rep['K1'] = dict(supported=all(kn['C'][f'{t}_k{k}']['mean'] == 0 for t in ('v2j', 'v2s') for k in (1, 6, 37))) if 'C' in kn else None
    rep['K2'] = dict(values={l: (kn[l]['v2j_k37']['mean'], kn[l]['v2s_k37']['mean']) for l in STRONGER if l in kn},
                     supported=all(kn[l]['v2j_k37']['mean'] <= .03 and kn[l]['v2s_k37']['mean'] <= .03 for l in STRONGER if l in kn))
    k3 = {}
    for f, levels in FACTORS.items():
        v = [kn[l]['v2s_k37']['mean'] for l in levels if l in kn]
        lo = kn[levels[-1]]['v2s_k37']['ci'][0] if levels[-1] in kn else None
        k3[f] = dict(v2s_k37=v, monotone=all(b >= a_ for a_, b in zip(v, v[1:])), last_lower_bound=lo,
                     supported=bool(len(v) == len(levels) and all(b >= a_ for a_, b in zip(v, v[1:])) and lo is not None and lo > 0))
    rep['K3'] = dict(factors=k3, supported=all(x['supported'] for x in k3.values()))
    # collide + shield
    cs = evals.get('collide_shield', {})
    if cs:
        real_c = realistic_from(d / 'policy_replay_c.jsonl', sup)
        cseeds = sorted(cs)
        rep['collide_shield_attacker_caused'] = int(sum(m['unfair'] for x in cseeds for m in cs[x].values()))
        for s in STD:
            cases = sorted({c for x in cseeds for (c, ss) in cs[x] if ss == s})
            if not cases:
                continue
            v = np.mean([np.array([cs[x].get((c, s), {}).get('robust', False) and real_c.get((x, c, s), False) for c in cases], float)
                         for x in cseeds], axis=0)
            rep['collide_shield'][s] = dict(k1=round(float(v.mean()), 4), ci=boot(v),
                                            collisions=int(sum(cs[x].get((c, s), {}).get('collision', False) for x in cseeds for c in cases)))
        if 'following_H0' in rep['collide_shield'] and 'rss_sut_H0' in rep['collide_shield']:
            rep['collide_shield_discrimination_k1'] = round(rep['collide_shield']['following_H0']['k1'] - rep['collide_shield']['rss_sut_H0']['k1'], 4)
    # screened random search and, for comparison, the screened BO alone (B56)
    rnd, _, err_r = refine([d / 'random_g4s.jsonl'], 'random_g4s')
    bo = refine([Path(a.b56)], 'bo_warm_g4s')[0] if a.b56 else {}
    rep['errors']['random_g4s'] = err_r
    for s in STD:
        cases = sorted({c for (c, ss) in rnd if ss == s})
        if not cases:
            continue
        rep['random'][s] = {f'random_k{k}': round(float(np.mean([rnd.get((c, s)) is not None and rnd[(c, s)] <= k for c in cases])), 4)
                            for k in (1, 6, 36)}
        if bo:
            rep['random'][s].update({f'screened_bo_k{k}': round(float(np.mean([bo.get((c, s)) is not None and bo[(c, s)] <= k for c in cases])), 4)
                                     for k in (1, 6, 36)})
    # attack-free natural cut-ins
    nat = [json.loads(l) for l in open(d / 'natural.jsonl', encoding='utf-8') if l.strip()]
    for s in STD:
        rows = [r['runs'][s] for r in nat if s in r['runs'] and 'error' not in r['runs'][s]]
        if rows:
            rep['natural'][s] = dict(n=len(rows), success=round(float(np.mean([x['success'] for x in rows])), 4),
                                     sut_contact=round(float(np.mean([x['sut_contact'] for x in rows])), 4),
                                     cc_contact=round(float(np.mean([x['cc_contact'] for x in rows])), 4))
    rep['natural_errors'] = sum('error' in v for r in nat for v in r['runs'].values())
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('errors', 'known_policy_attacker_caused', 'K1', 'K2', 'K3')}, indent=1))
    for label, e in rep['known'].items():
        print(f"{label:26s} v2j {100*e['v2j_k1']['mean']:5.1f} {100*e['v2j_k37']['mean']:5.1f} | v2s {100*e['v2s_k37']['mean']:5.1f} {e['v2s_k37']['ci']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--b56', help='B56 b56_dev.jsonl, for the screened BO alone next to the screened random search')
    main(ap.parse_args())
