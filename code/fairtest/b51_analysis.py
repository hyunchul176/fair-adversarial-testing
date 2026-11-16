"""B51 analysis (B51 design, fixed before the run, available on request).

Exact per-case composition (policy seeds 1-3, development cases), all under the v2j contract:
  v2j            policy stage success (robust fair + G2 + joint on its CC replay), else bo_warm_g4j's first success
  v2j severe     the same reports, counted only when the SUT's closing speed at contact is >= v_min (post-hoc filter)
  v2j-S          policy success with closing >= v_min, else bo_warm_g4js's first success (severity floor in the search)
  SV1  weak rule-based SUTs pooled: v2j-S - v2j severe >= 5 pp (case bootstrap interval reported)
  SV2  each competent SUT (AEB H0/H0.5, longitudinal RSS, lateral RSS, ADS 0.5 m): v2j-S <= 3%
  SV3  P1 (following H0 - longitudinal RSS): v2j-S >= v2j severe

  python -m fairtest.b51_analysis OUT.json --jobs fairtest_jobs --g4js b51/refine_g4js_dev.jsonl --closing b51/closing_dev.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage
from fairtest.b45_joint_realism import Support

V_MIN, K_FULL, B, SEED = 3., 37, 10000, 20260928
WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0')
GROUPS = {'weak rule-based': WEAK, 'BC human': ('bc_human_H0',), 'ADS 2.0 m': ('ads@2.0_H0',),
          'learned': ('drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0', 'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
          'competent': COMPETENT}


def _rows(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def first_success(paths, method):
    out, errors = {}, 0
    for p in paths:
        for r in _rows(p):
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m != method:
                    continue
                if 'error' in rec:
                    errors += 1
                    continue
                out[(r['case_id'], s)] = dict(q=rec.get('first_fair'), used=rec.get('sut_queries'),
                                              closing=(rec.get('best') or {}).get('closing'))
    return out, errors


def main(a):
    jobs = Path(a.jobs)
    sup = Support()
    robust, _ = policy_stage(jobs / 'b26', 'dev')
    ok = {}
    for r in _rows(jobs / 'b45' / 'replay_dev.jsonl'):
        if r['source'] == 'v2_policy':
            sc = sup.score(r['features']) if r['cut_in'] else None
            ok[(f"s{r['seed']}", r['case_id'], r['sut'])] = (sc is not None and sc <= sup.tau and r.get('exceedance') is not None
                                                            and r['exceedance'] <= 1.)
    g4j, e1 = first_success([jobs / 'b46' / 'joint_dev_rule.jsonl', jobs / 'b46' / 'joint_dev_learned.jsonl'], 'bo_warm_g4j')
    g4js, e2 = first_success([a.g4js], 'bo_warm_g4js')
    cl_pol, cl_ref, e3 = {}, {}, 0
    for r in _rows(a.closing):
        if 'error' in r:
            e3 += 1
            continue
        if r['kind'] == 'policy':
            cl_pol[(f"s{r['seed']}", r['case_id'], r['sut'])] = r['closing']
        else:
            cl_ref[(r['case_id'], r['sut'])] = r['closing']
    seeds = sorted({k[0] for k in robust})
    suts = sorted({s for (_, s) in g4js})
    cases = sorted({c for (c, _) in g4js})
    per = collections.defaultdict(dict)  # arm -> sut -> {case: seed-averaged success}
    cost = collections.defaultdict(list)
    missing = 0
    for s in suts:
        for c in cases:
            hits = collections.defaultdict(list)
            for x in seeds:
                p1 = robust.get((x, c, s), False) and ok.get((x, c, s), False)
                cp = cl_pol.get((x, c, s))
                if p1 and cp is None:
                    missing += 1
                j, js = g4j.get((c, s)) or {}, g4js.get((c, s)) or {}
                cj = cl_ref.get((c, s)) if j.get('q') is not None else None
                if j.get('q') is not None and cj is None:
                    missing += 1
                v2j = p1 or j.get('q') is not None
                sev = (p1 and cp is not None and cp >= V_MIN) or (not p1 and j.get('q') is not None and cj is not None and cj >= V_MIN)
                p1s = p1 and cp is not None and cp >= V_MIN
                v2js = p1s or js.get('q') is not None
                hits['v2j'].append(float(v2j))
                hits['v2j_severe'].append(float(sev))
                hits['v2j_S'].append(float(v2js))
                cost[s].append(dict(v2j=1 if p1 else (1 + j['q'] if j.get('q') is not None else K_FULL),
                                    v2jS=1 if p1s else (1 + js['q'] if js.get('q') is not None else K_FULL), found=v2js))
            for arm, v in hits.items():
                per[arm].setdefault(s, {})[c] = float(np.mean(v))
    rng = np.random.default_rng(SEED)

    def rate(arm, members):
        return float(np.mean([per[arm][s][c] for s in members if s in per[arm] for c in cases]))

    def pooled(arm, members):
        return np.array([np.mean([per[arm][s][c] for s in members if s in per[arm]]) for c in cases])

    rep = dict(errors=dict(g4j=e1, g4js=e2, closing=e3), missing_closing=missing, n_cases=len(cases), v_min=V_MIN,
               suts={s: {arm: round(float(np.mean(list(per[arm][s].values()))), 4) for arm in per} for s in suts},
               groups={g: {arm: round(rate(arm, m), 4) for arm in per} for g, m in GROUPS.items()})
    d = pooled('v2j_S', WEAK) - pooled('v2j_severe', WEAK)
    boot = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    rep['SV1'] = dict(diff=round(float(d.mean()), 4), ci=[round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
                      supported=bool(d.mean() >= .05))
    sv2 = {s: rep['suts'][s]['v2j_S'] for s in COMPETENT if s in rep['suts']}
    rep['SV2'] = dict(v2j_S=sv2, supported=all(v <= .03 for v in sv2.values()))
    p1 = {arm: rep['suts']['following_H0'][arm] - rep['suts']['rss_sut_H0'][arm] for arm in per}
    rep['SV3'] = dict(P1={k: round(v, 4) for k, v in p1.items()}, supported=p1['v2j_S'] >= p1['v2j_severe'])
    rep['queries'] = {g: dict(v2j_per_attack=round(float(np.mean([r['v2j'] for s in m for r in cost[s]])), 2),
                              v2jS_per_attack=round(float(np.mean([r['v2jS'] for s in m for r in cost[s]])), 2))
                      for g, m in GROUPS.items()}
    sev_closing = [(g4js.get((c, s)) or {}).get('closing') for s in WEAK for c in cases if (g4js.get((c, s)) or {}).get('q') is not None]
    rep['g4js_weak_closing_median'] = float(np.median(sev_closing)) if sev_closing else None
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('errors', 'missing_closing', 'SV1', 'SV2', 'SV3')}, indent=1))
    for g, e in rep['groups'].items():
        print(f"{g:16s} v2j {100*e['v2j']:5.1f}  v2j severe {100*e['v2j_severe']:5.1f}  v2j-S {100*e['v2j_S']:5.1f}  queries {rep['queries'][g]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--g4js', required=True)
    ap.add_argument('--closing', required=True)
    main(ap.parse_args())
