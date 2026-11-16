"""B34 analysis (B34 design, available on request).

  X1  discrimination: the agent's robust fair-failure rate (36 queries), weakest SUT - rss_dec >= 30 pp
  X2  specificity: rss_dec <= 5%
  X3  collision seeking does not discriminate: collision-rate range <= 20 pp and >= 80% of the competent SUTs'
      (rss_dec, idm_d0) collisions attacker-at-fault (not judged below 10 such collisions)
  X4  >= 95% of the agent's failures fair under every careful-driver variant

  python -m fairtest.b34_analysis EVAL.jsonl OUT.json
"""
from __future__ import annotations

import collections
import json
import sys

from scipy.stats import kendalltau

from fairtest.cutout_sim import CC_VARIANTS, SUTS

KS = (1, 5, 36)
COMPETENT = ('rss_dec', 'idm_d0')
RANK = {'idm_d10': 1, 'acc_prod': 1, 'idm_d5': 2, 'idm_d0': 3, 'acc_prod_aeb': 4, 'rss_dec': 5}  # as B30 (descriptive)


def main(path, out):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    n = len(rows)
    agent = {s: {k: 0 for k in KS} for s in SUTS}
    col, att, grades, judged = collections.Counter(), collections.Counter(), collections.Counter(), []
    boundary = sum(r['boundary'] is not None for r in rows)
    for r in rows:
        for s in SUTS:
            a = r['runs'][f'agent|{s}']
            q = a.get('first_robust')
            for k in KS:
                agent[s][k] += q is not None and q <= k
            if q is not None:
                grades[a['grade']] += 1
                judged.append(all(not v['contact'] for v in a['judged'].values()))
            c = r['runs'][f'collide|{s}']
            if c.get('first_collision') is not None:
                col[s] += 1
                att[s] += bool(c['best'].get('cc_contact'))
    rate = {s: {k: agent[s][k] / n for k in KS} for s in SUTS}
    crate = {s: col[s] / n for s in SUTS}
    share = {s: (att[s] / col[s] if col[s] else None) for s in SUTS}
    comp_n = sum(col[s] for s in COMPETENT)
    comp_share = sum(att[s] for s in COMPETENT) / comp_n if comp_n else None
    top = max(rate[s][36] for s in SUTS)
    rep = dict(cases=n, boundary_found=boundary, agent_rate=rate, collide_rate=crate, collide_attacker_share=share,
               agent_grades=dict(grades), agent_failures=len(judged))
    rep['X1'] = dict(weakest=max(SUTS, key=lambda s: rate[s][36]), diff=top - rate['rss_dec'][36],
                     supported=top - rate['rss_dec'][36] >= .30)
    rep['X2'] = dict(rss=rate['rss_dec'][36], supported=rate['rss_dec'][36] <= .05)
    rng = max(crate.values()) - min(crate.values())
    rep['X3'] = dict(range=rng, competent_collisions=comp_n, competent_attacker_share=comp_share,
                     judged=comp_n >= 10,
                     supported=(rng <= .20 and comp_share is not None and comp_share >= .80) if comp_n >= 10 else None)
    fair_all = sum(judged) / len(judged) if judged else None
    rep['X4'] = dict(fair_under_all_variants=fair_all, variants=list(CC_VARIANTS),
                     supported=fair_all is not None and fair_all >= .95)
    rep['tau_agent'] = float(kendalltau([RANK[s] for s in SUTS], [rate[s][36] for s in SUTS]).statistic)
    rep['tau_collide'] = float(kendalltau([RANK[s] for s in SUTS], [crate[s] for s in SUTS]).statistic)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
