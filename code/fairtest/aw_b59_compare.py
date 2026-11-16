"""B59 (B59 design, available on request): the screened refinement (B56 bo_warm_g4s) against
Autoware, paired by case with B54 (same policy stage, same refinement cases, same boundary plans; B54's refinement is
bo_warm_g4j). Confirmed detection = aw_analysis.py definition (policy stage from B54, refinement from B59, reruns of both).

  W1  the screened refinement runs no inadmissible plan on Autoware (implementation check)
  W2  confirmed detection B59 - B54 >= -3 points per variant (point estimate; paired case bootstrap 10,000, rng 0)
  W3 (descr.)  Autoware queries of the refinement stage: total, inadmissible, per refinement success; CC replays; the
               refinement jobs' found share
  also: B54's inadmissible Autoware queries and the Autoware contacts on them (from B54's stored query histories)

  python -m fairtest.aw_b59_compare B54_SUMMARY_percase.json B59_SUMMARY.json B54_DIR B59_DIR OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V = ('aw_default', 'aw_r157')
# the joint-support threshold as the search used it (b45_joint_realism.Support().tau = 2.7714...); the frozen version
# rounded it to 2.77 and flagged one admissible query (joint score 2.771) -- corrected after the run, disclosed in B59 RESULT
_TAU = []


def tau():
    if not _TAU:
        from fairtest.b45_joint_realism import Support
        _TAU.append(Support().tau)
    return _TAU[0]


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, size=(n, len(x)), replace=True).mean(1)
    return [round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]


def admissible(h):
    return bool(not h['cc_contact'] and (h['cc_clearance'] or 0.) >= .5 and not h['feint'] and h['realism_exceedance'] is not None
                and h['realism_exceedance'] <= 1. and h['joint_score'] is not None and h['joint_score'] <= round(tau(), 3))


def refine_stats(d):
    out = {}
    for v in V:
        e = dict(jobs=0, errors=0, found=0, autoware_queries=0, inadmissible=0, contacts_inadmissible=0, contacts_inadmissible_cc_contact=0,
                 cc_replays=0, hours=0.)
        for p in sorted(Path(d).glob('refine_*.jsonl')):
            for line in open(p, encoding='utf-8'):
                r = json.loads(line)
                if r['variant'] != v:
                    continue
                res = r['result']
                if 'error' in res:
                    e['errors'] += 1
                    continue
                e['jobs'] += 1
                e['found'] += bool(res.get('found'))
                e['hours'] += res.get('seconds', 0.) / 3600
                e['cc_replays'] += res.get('cc_replays', 0)
                for h in res['history']:
                    e['autoware_queries'] += 1
                    if not admissible(h):
                        e['inadmissible'] += 1
                        e['contacts_inadmissible'] += bool(h['sut_contact'])
                        e['contacts_inadmissible_cc_contact'] += bool(h['sut_contact'] and h['cc_contact'])
        e['hours'] = round(e['hours'], 2)
        e['queries_per_found'] = round(e['autoware_queries'] / e['found'], 2) if e['found'] else None
        out[v] = e
    return out


def main(b54, b59, d54, d59, out):
    s4, s9 = json.load(open(b54, encoding='utf-8')), json.load(open(b59, encoding='utf-8'))
    R = dict(detection={}, refine=dict(b54=refine_stats(d54), b59=refine_stats(d59)))
    for v in V:
        p4, p9 = s4['per_case_confirmed'][v], s9['per_case_confirmed'][v]
        ids = sorted(p4, key=int)
        assert sorted(p9, key=int) == ids, 'the runs must cover the same cases'
        d = [p9[c] - p4[c] for c in ids]
        R['detection'][v] = dict(b54=round(float(np.mean([p4[c] for c in ids])), 4), b59=round(float(np.mean([p9[c] for c in ids])), 4),
                                 difference=round(float(np.mean(d)), 4), ci=boot(d))
    R['W1'] = dict(inadmissible={v: R['refine']['b59'][v]['inadmissible'] for v in V},
                   supported=all(R['refine']['b59'][v]['inadmissible'] == 0 for v in V) and all(R['refine']['b59'][v]['autoware_queries'] > 0 for v in V))
    R['W2'] = dict(difference={v: R['detection'][v]['difference'] for v in V},
                   supported=all(R['detection'][v]['difference'] >= -.03 for v in V))
    S9 = s9['variants']
    R['guarantee_policy_stage'] = {v: S9[v]['A1'] for v in V}
    R['reproduction'] = {v: {k: S9[v]['A5'][k] for k in ('failures', 'reruns', 'colliding', 'rate')} for v in V}
    R['by_stage'] = {v: S9[v]['by_stage'] for v in V}
    json.dump(R, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
