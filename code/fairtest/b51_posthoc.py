"""B51 POST HOC on an external evaluation in the B41 layout (MiTra): v2j, v2j severe-only (closing >= 3 m/s) and v2j-S
(bo_warm_g4js), composed exactly per case as b51_analysis does for the development cases, plus the same by attack
authority (relative deceleration headroom bins, authority_strata).

  python -m fairtest.b51_posthoc OUT.json DIR --g4js REFINE_G4JS.jsonl --closing CLOSING.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest.b41_analysis import load
from fairtest.b51_analysis import COMPETENT, GROUPS, V_MIN, WEAK, _rows, first_success

BINS = ((-99, 2), (2, 5), (5, 10), (10, 99))


def main(a):
    d = Path(a.dir)
    evals, realistic, refine, pool, _ = load(d)
    head = {c['case_id']: c['physical_initial_state'][1]['vx'] - 15. for c in pool['cases']}
    g4j = {k: dict(q=v) for k, v in refine['bo_warm_g4j'].items()}
    g4js, err = first_success([a.g4js], 'bo_warm_g4js')
    cl_pol, cl_ref = {}, {}
    for r in _rows(a.closing):
        if 'error' in r:
            continue
        if r['kind'] == 'policy':
            cl_pol[(str(r['seed']), r['case_id'], r['sut'])] = r['closing']
        else:
            cl_ref[(r['case_id'], r['sut'])] = r['closing']
    seeds = sorted(evals['final'])
    cases = sorted(c['case_id'] for c in pool['cases'])
    suts = sorted({s for (_, s) in g4js})
    per = collections.defaultdict(dict)
    for s in suts:
        for c in cases:
            hits = collections.defaultdict(list)
            for x in seeds:
                p1 = evals['final'][x].get((c, s), {}).get('robust', False) and realistic.get((x, c, s), False)
                cp = cl_pol.get((x, c, s))
                q = (g4j.get((c, s)) or {}).get('q')
                cj = cl_ref.get((c, s))
                qs = (g4js.get((c, s)) or {}).get('q')
                hits['v2j'].append(float(p1 or q is not None))
                hits['v2j_severe'].append(float((p1 and cp is not None and cp >= V_MIN) or
                                                (not p1 and q is not None and cj is not None and cj >= V_MIN)))
                hits['v2j_S'].append(float((p1 and cp is not None and cp >= V_MIN) or qs is not None))
            for arm, v in hits.items():
                per[arm].setdefault(s, {})[c] = float(np.mean(v))

    def rate(arm, s, cs=None):
        cs = cases if cs is None else cs
        return float(np.mean([per[arm][s][c] for c in cs])) if cs else None
    rep = dict(n_cases=len(cases), g4js_errors=err, v_min=V_MIN,
               suts={s: {arm: round(rate(arm, s), 4) for arm in per} for s in suts},
               groups={g: {arm: round(float(np.mean([rate(arm, s) for s in m if s in per[arm]])), 4) for arm in per}
                       for g, m in GROUPS.items()})
    rep['P1'] = {arm: round(rate(arm, 'following_H0') - rate(arm, 'rss_sut_H0'), 4) for arm in per}
    rep['by_authority'] = {}
    for lo, hi in BINS:
        cs = [c for c in cases if lo <= head[c] < hi]
        if cs:
            rep['by_authority'][f'{lo},{hi}'] = dict(n=len(cs), **{arm: dict(following_H0=round(rate(arm, 'following_H0', cs), 4),
                                                                              rss=round(rate(arm, 'rss_sut_H0', cs), 4)) for arm in per})
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print('P1', rep['P1'])
    for g, e in rep['groups'].items():
        print(f"{g:16s} v2j {100*e['v2j']:5.1f}  severe {100*e['v2j_severe']:5.1f}  v2j-S {100*e['v2j_S']:5.1f}")
    for k, v in rep['by_authority'].items():
        print('headroom', k, v)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('dir')
    ap.add_argument('--g4js', required=True)
    ap.add_argument('--closing', required=True)
    main(ap.parse_args())
