"""Cost of v2j on an external evaluation, reconstructed from its stage outputs (B41 layout; B49 measured the callable).

Per (seed, case, SUT): SUT search queries = 1 if the policy stage passes the one contract, else 1 + q (bo_warm_g4j's
first success q), else 37; shield CC rollouts of the policy episode; refinement CC replays = the BO run's queries when
the policy stage failed. The training-free boundary-start BO (bo_warm_g4j alone): queries = its run's queries, found =
q is not None. Per case, SUT-independent: the CC grid (200 plans) and the CC-boundary search (b11 `cc_sims`, if present).
Certificates are not part of an external evaluation run, so verification replays are not counted here.

  python -m fairtest.v2j_cost DIR OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fairtest.b41_analysis import _rows, load

GROUPS = {
    'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'),
    'BC human': ('bc_human_H0',),
    'ADS 2.0 m': ('ads@2.0_H0',),
    'learned': ('drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0', 'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
    'competent': ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0'),
}


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=round(float(x.mean()), 2), median=round(float(np.median(x)), 2), p90=round(float(np.percentile(x, 90)), 2)) if len(x) else None


def summary(recs):
    found = sum(r['found'] for r in recs)
    out = dict(attacks=len(recs), detection=round(found / len(recs), 4) if recs else None)
    for k in ('search', 'shield', 'refine_cc'):
        v = [r[k] for r in recs]
        out[k] = stats(v)
        out[k + '_per_detection'] = round(sum(v) / found, 2) if found else None
    return out


def main(d, out):
    d = Path(d)
    evals, realistic, refine, pool, _ = load(d)
    shield = {}
    for path in sorted(d.glob('final_s[0-9].jsonl')):
        seed = path.stem.rsplit('_s', 1)[1]
        for r in _rows(path):
            if 'valid' in r:
                shield[(seed, r['case_id'], r['sut'])] = (r.get('shield') or {}).get('cc_rollouts', 0)
    used = {}
    for r in _rows(d / 'refine.jsonl'):
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == 'bo_warm_g4j' and 'error' not in rec:
                used[(r['case_id'], s)] = rec.get('sut_queries')
    rep = dict(n_cases=len(pool['cases']), groups={})
    for g, suts in GROUPS.items():
        agent, bo = [], []
        for x, e in evals['final'].items():
            for (c, s), m in e.items():
                if s not in suts:
                    continue
                q = refine['bo_warm_g4j'].get((c, s))
                p1 = m['robust'] and realistic.get((x, c, s), False)
                n_bo = used.get((c, s)) or 0
                agent.append(dict(found=bool(p1 or q is not None), search=1 if p1 else (1 + q if q is not None else 37),
                                  shield=shield.get((x, c, s), 0), refine_cc=0 if p1 else n_bo))
        for (c, s), q in refine['bo_warm_g4j'].items():
            if s in suts and (c, s) in used:
                bo.append(dict(found=q is not None, search=used[(c, s)], shield=0, refine_cc=used[(c, s)]))
        if agent:
            rep['groups'][g] = dict(v2j_agent=summary(agent), bo_only=summary(bo))
    b11 = d / 'b11.jsonl'
    if b11.exists():
        sims = [(r.get('cc_boundary') or {}).get('cc_sims') for r in _rows(b11)]
        rep['precompute_per_case'] = dict(cc_grid=200, cc_boundary_search=stats([x for x in sims if x is not None]))
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    for g, e in rep['groups'].items():
        a, b = e['v2j_agent'], e['bo_only']
        print(f"{g:16s} v2j {100*a['detection']:5.1f}% search {a['search']['mean']:5.1f} (median {a['search']['median']:g}, per det {a['search_per_detection']}) "
              f"shield {a['shield']['mean']:5.1f} refine CC {a['refine_cc']['mean']:5.1f} | BO {100*b['detection']:5.1f}% search {b['search']['mean']:5.1f} "
              f"(per det {b['search_per_detection']})")
    return rep


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
