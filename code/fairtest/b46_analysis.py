"""B46 analysis (B46 design, available on request).

Exact per-case composition (as b26_analysis final):
  v2   the policy stage's robust fair failure, else the refinement bo_warm_g4g (stored b29/guarded_dev.jsonl)
  v2j  the policy stage's robust fair failure that is also inside the natural joint support (B45 replay), else the
       refinement bo_warm_g4j (B46)
  J1  every bo_warm_g4j success inside the joint support;  J2 following H0/H0.5/H1 and IDM: v2j k37 >= v2 k37 - 5 pp;
  J3  AEB H0/H0.5, RSS, lateral RSS, ADS 0.5 m: v2j k37 <= 3%

  python -m fairtest.b46_analysis OUT.json --jobs fairtest_jobs B46_a.jsonl [B46_b.jsonl ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage, refine_stage
from fairtest.b45_joint_realism import Support

KS = (1, 6, 37)
WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0')


def policy_tgsim(jobs):
    """b26_analysis.policy_stage for TGSIM (B28 layout: b28/final_s<k>.jsonl with b28/margin.jsonl)."""
    margin = {}
    for line in open(jobs / 'b28' / 'margin.jsonl', encoding='utf-8'):
        r = json.loads(line)
        margin[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= 3.5
                                                             and r['cc_clearance'] >= .5)
    robust = {}
    for path in sorted((jobs / 'b28').glob('final_s[0-9].jsonl')):
        seed = path.stem.split('_')[1]
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'valid' not in r:
                continue
            col = bool(r['valid'] and r['contact'])
            fair = col and not r['refs']['cc_r157']['contact']
            robust[(seed, r['case_id'], r['sut'])] = bool(fair and margin.get((path.name, r['case_id'], r['sut']), False))
    return robust


def main(a):
    jobs = Path(a.jobs)
    sup = Support()
    ds = a.dataset
    robust = policy_tgsim(jobs) if ds == 'tgsim' else policy_stage(jobs / 'b26', ds)[0]
    replay = Path(a.replay) if a.replay else jobs / 'b45' / 'replay_dev.jsonl'
    inside = {}
    for line in open(replay, encoding='utf-8'):
        r = json.loads(line)
        if r['source'] == 'v2_policy':
            sc = sup.score(r['features']) if r['cut_in'] else None
            ok = sc is not None and sc <= sup.tau
            if a.policy_g2:  # W5 P1-C: one success contract -- the policy stage must also pass the per-feature G2
                ok = ok and r.get('exceedance') is not None and r['exceedance'] <= 1.
            inside[(f"s{r['seed']}", r['case_id'], r['sut'])] = ok
    g4g = refine_stage(jobs / 'b26', ds, 'bo_warm_g4g', [str(jobs / 'b29' / 'guarded_{ds}.jsonl')])
    g4j, scores, errors = {}, [], 0
    for path in a.b46:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m != 'bo_warm_g4j':
                    continue
                if 'error' in rec:
                    errors += 1
                    continue
                g4j[(r['case_id'], s)] = rec.get('first_fair')
                if rec.get('first_fair') is not None:
                    scores.append(rec['best'].get('joint_score'))
    seeds = sorted({k[0] for k in robust})
    rng = np.random.default_rng(20260928)
    rep = dict(errors=errors, J1=dict(successes=len(scores), inside=sum(x is not None and x <= sup.tau for x in scores)), suts={})
    rep['J1']['supported'] = rep['J1']['inside'] == len(scores)
    for s in sorted({k[1] for k in g4j}):
        cases = sorted({k[1] for k in robust if k[2] == s})
        if not all((c, s) in g4j for c in cases):
            continue
        entry = dict(n_cases=len(cases))
        for label, ref, need_inside in (('v2', g4g, False), ('v2j', g4j, True)):
            for k in KS:
                per_seed = [np.array([(robust.get((x, c, s), False) and (inside.get((x, c, s), False) or not need_inside))
                                      or (k >= 2 and ref.get((c, s)) is not None and 1 + ref[(c, s)] <= k) for c in cases], float)
                            for x in seeds]
                vals = np.mean(per_seed, axis=0)
                boot = vals[rng.integers(0, len(vals), (4000, len(vals)))].mean(1)
                entry[f'{label}_k{k}'] = dict(mean=round(float(vals.mean()), 4),
                                              ci=[round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)])
            entry[f'{label}_refine_only'] = round(float(np.mean([ref.get((c, s)) is not None for c in cases])), 4)
        entry['policy_robust_inside_share'] = round(float(np.mean([inside.get((x, c, s), False) for x in seeds for c in cases
                                                                  if robust.get((x, c, s), False)] or [np.nan])), 4)
        rep['suts'][s] = entry
    j2 = {s: (rep['suts'][s]['v2j_k37']['mean'], rep['suts'][s]['v2_k37']['mean']) for s in WEAK if s in rep['suts']}
    rep['J2'] = dict(v2j_vs_v2=j2, supported=len(j2) == len(WEAK) and all(a_ >= b_ - .05 for a_, b_ in j2.values()))
    j3 = {s: rep['suts'][s]['v2j_k37']['mean'] for s in COMPETENT if s in rep['suts']}
    rep['J3'] = dict(v2j_k37=j3, supported=len(j3) == len(COMPETENT) and all(v <= .03 for v in j3.values()))
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('errors', 'J1', 'J2', 'J3')}, indent=1))
    for s, e in rep['suts'].items():
        print(f"{s:22s} v2 {100*e['v2_k1']['mean']:5.1f} {100*e['v2_k6']['mean']:5.1f} {100*e['v2_k37']['mean']:5.1f} | "
              f"v2j {100*e['v2j_k1']['mean']:5.1f} {100*e['v2j_k6']['mean']:5.1f} {100*e['v2j_k37']['mean']:5.1f} | "
              f"refine g4g {100*e['v2_refine_only']:5.1f} g4j {100*e['v2j_refine_only']:5.1f}  policy-inside {e['policy_robust_inside_share']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--dataset', default='dev', choices=['dev', 'b9', 'ngsim', 'tgsim'], help='B47 (post hoc): the other datasets')
    ap.add_argument('--replay', help='policy-stage replay (b45 replay --dataset); default the dev replay of B45')
    ap.add_argument('--policy-g2', action='store_true', help='v2j stage 1 also requires the per-feature G2 (exceedance <= 1)')
    ap.add_argument('b46', nargs='+')
    main(ap.parse_args())
