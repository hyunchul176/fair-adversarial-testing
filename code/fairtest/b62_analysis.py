"""B62 analysis (B62 design, fixed before training, available on request).

B62 = the population-trained policy stage (JOBS/b62: final_s<k>.jsonl, margin.jsonl, policy_replay.jsonl, read with
b41_analysis.load); v2j = the stored v2j policy stage (b56_analysis.policy, dev). Unified contract at the policy stage
(robust + joint support + G2); k = 6 / 37 compose the stored B56 bo_warm_g4s (policy-independent). Development partition.
  Q1  unseen detectable SUTs (UNSEEN): mean k1 difference B62 - v2j >= +10 points and the paired case-bootstrap 95% lower
      bound > 0 (per case: mean over the 5 SUTs and the seeds; 10,000 resamples, rng 0)
  Q2  training-panel weak SUTs (PANEL_WEAK): mean k1 difference >= -5 points (point estimate)
  Q3  competent SUTs (b56_analysis.COMPETENT): B62 k1 <= 3% each
  Q4  policy-stage attacker-caused contacts (invalid manoeuvre or the CC also collides) = 0
  descriptive: per SUT k1 / k6 / k37 for both policies with g4s; queries the screened search alone needs to match k1

  python -m fairtest.b62_analysis OUT.json --jobs fairtest_jobs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest import b56_analysis as A

UNSEEN = ('bc_human_H0', 'ads@2.0_H0', 'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')
PANEL_WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
Q1_MIN, Q2_MIN, Q3_MAX = .10, -.05, .03


def boot(vals, n=10000, seed=0):
    vals = np.asarray(vals, float)
    m = vals[np.random.default_rng(seed).integers(0, len(vals), (n, len(vals)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def match_k(g4s, cases, s, target):
    for k in range(1, 37):
        if np.mean([g4s.get((c, s)) is not None and g4s[(c, s)] <= k for c in cases]) >= target - 1e-9:
            return k
    return None


def main(a):
    from fairtest.b41_analysis import load
    J = Path(a.jobs)
    evals, inside_b, _, _, _ = load(J / 'b62')
    fin = evals['final']
    seeds_b = sorted(fin)
    robust_b = {(x, c, s): m['robust'] for x in seeds_b for (c, s), m in fin[x].items()}
    robust_v, inside_v = A.policy(J, 'dev')
    seeds_v = sorted({k[0] for k in robust_v})
    g4s = A.refine([J / 'b56' / 'b56_dev.jsonl'], 'bo_warm_g4s')[0]
    rep = dict(seeds=dict(b62=seeds_b, v2j=seeds_v), suts={})
    rep['Q4'] = dict(attacker_caused=int(sum(m['unfair'] for x in seeds_b for m in fin[x].values())),
                     unjudged=int(sum(m['unjudged'] for x in seeds_b for m in fin[x].values())),
                     episodes=int(sum(len(fin[x]) for x in seeds_b)))
    rep['Q4']['supported'] = rep['Q4']['attacker_caused'] == 0 and rep['Q4']['episodes'] > 0
    per_case = {}
    for s in sorted({k[2] for k in robust_b}):
        cases = sorted({c for (x, c, ss) in robust_b if ss == s} & {c for (x, c, ss) in robust_v if ss == s})
        e = dict(n_cases=len(cases))
        for tag, (rb, ins, sd) in (('b62', (robust_b, inside_b, seeds_b)), ('v2j', (robust_v, inside_v, seeds_v))):
            for k in (1, 6, 37):
                v = A.compose(rb, ins, g4s, sd, cases, s, k)
                e[f'{tag}_k{k}'] = round(float(v.mean()), 4)
                if k == 1:
                    per_case[(tag, s)] = dict(zip(cases, v))
            e[f'{tag}_search_match_k'] = match_k(g4s, cases, s, e[f'{tag}_k1'])
        e['diff_k1'] = round(e['b62_k1'] - e['v2j_k1'], 4)
        rep['suts'][s] = e

    def group_diff(group):
        cases = sorted(set.intersection(*[set(per_case[('b62', s)]) for s in group]))
        d = [np.mean([per_case[('b62', s)][c] - per_case[('v2j', s)][c] for s in group]) for c in cases]
        return round(float(np.mean(d)), 4), boot(d), len(cases)

    m1, ci1, n1 = group_diff(UNSEEN)
    rep['Q1'] = dict(diff_k1=m1, ci=ci1, cases=n1, b62=round(float(np.mean([rep['suts'][s]['b62_k1'] for s in UNSEEN])), 4),
                     v2j=round(float(np.mean([rep['suts'][s]['v2j_k1'] for s in UNSEEN])), 4), supported=bool(m1 >= Q1_MIN and ci1[0] > 0))
    m2, ci2, n2 = group_diff(PANEL_WEAK)
    rep['Q2'] = dict(diff_k1=m2, ci=ci2, cases=n2, supported=bool(m2 >= Q2_MIN))
    c3 = {s: rep['suts'][s]['b62_k1'] for s in A.COMPETENT if s in rep['suts']}
    rep['Q3'] = dict(b62_k1=c3, supported=bool(len(c3) == len(A.COMPETENT) and all(v <= Q3_MAX for v in c3.values())))
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('Q1', 'Q2', 'Q3', 'Q4')}, indent=1))
    for s, e in rep['suts'].items():
        print(f"{s:20s} b62 {100*e['b62_k1']:5.1f} {100*e['b62_k6']:5.1f} {100*e['b62_k37']:5.1f} | v2j {100*e['v2j_k1']:5.1f} "
              f"{100*e['v2j_k6']:5.1f} {100*e['v2j_k37']:5.1f} | match b62 {e['b62_search_match_k']} v2j {e['v2j_search_match_k']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    main(ap.parse_args())
