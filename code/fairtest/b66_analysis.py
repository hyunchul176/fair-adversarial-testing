"""B66 analysis (B66 design, available on request).

Per condition, seed, case and SUT (verdicts by the CC, cc_r157): collision, fair (CC avoids), attacker-caused (CC collides),
robust fair (G4 margin: front contact dx >= 3.5 m and CC clearance >= 0.5 m), feint (lane reversal after entering,
b23_analysis.feints). Rates are seed-averaged case shares (62 cases).
  BL1  carsG1: longitudinal RSS robust-fair rate <= 5%
  BL2  carsG1: share of collisions the CC cannot avoid, pooled over the weak-4 ladder SUTs and longitudinal RSS, >= 50%
  BL3  feasr0sh and feast0sh: P1 by the CC (robust fair, following H0 - longitudinal RSS) < 50 pp
Controls: cars (same policy, no commitment) and the stored unshielded B52 feasibility evaluation (--feas-noshield).

  python -m fairtest.b66_analysis OUT.json DIR --feas-noshield B52_EVAL_DEV
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


DELTA, FRONT = .5, 3.5
def feints(actions):
    """Lane reversals after the first entry (lane goal 4 -> 0), as b23_analysis.feints (copied: no plotting imports)."""
    n, entered, last = 0, False, 0.
    for lane, _ in actions:
        if lane == 4.:
            entered = True
        elif entered and last == 4.:
            n += 1
        last = lane
    return n

WEAK4 = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')


def rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def margins(path):
    out = {}
    if Path(path).exists():
        for r in rows(path):
            out[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= FRONT
                                                             and r['cc_clearance'] >= DELTA)
    return out


def condition(d, tag, margin):
    """{sut: {metric: seed-averaged case share}}, plus pooled counts."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))  # sut -> metric -> [per (seed, case) 0/1]
    files = sorted(Path(d).glob(f'{tag}_s[0-9].jsonl'))
    for path in files:
        for r in rows(path):
            if 'valid' not in r:
                continue
            col = bool(r['valid'] and r['contact'])
            refs = r.get('refs') or {}
            judged = not col or 'contact' in refs.get('cc_r157', {})
            fair = bool(col and judged and not refs['cc_r157']['contact'])
            unfair = bool(col and judged and refs['cc_r157']['contact'])
            fe = feints(r.get('actions') or []) > 0
            rob = bool(fair and margin.get((path.name, r['case_id'], r['sut']), False))
            m = per[r['sut']]
            m['collision'].append(col)
            m['fair'].append(fair)
            m['robust_fair'].append(rob)
            m['attacker_caused'].append(unfair)
            m['fair_feint'].append(fair and fe)
            m['fair_no_feint'].append(fair and not fe)
            m['unjudged'].append(col and not judged)
    out = {s: {k: round(float(np.mean(v)), 4) for k, v in m.items()} for s, m in per.items()}
    pool = [s for s in WEAK4 + ('rss_sut_H0',) if s in per]
    cols = sum(sum(per[s]['collision']) for s in pool)
    unf = sum(sum(per[s]['attacker_caused']) for s in pool)
    return dict(seeds=len(files), suts=out, pooled_weak4_rss=dict(collisions=cols, attacker_caused=unf,
                                                                 share=round(unf / cols, 4) if cols else None))


def main(a):
    d = Path(a.dir)
    margin = margins(d / 'margin.jsonl')
    rep = dict(conditions={})
    for tag in ('carsG1', 'cars', 'feasr0sh', 'feast0sh'):
        rep['conditions'][tag] = condition(d, tag, margin)
    if a.feas_noshield:
        m2 = margins(Path(a.feas_noshield) / 'margin.jsonl')
        for tag in ('feasr0', 'feast0'):
            rep['conditions'][f'{tag}_noshield'] = condition(a.feas_noshield, tag, m2)

    def p1(tag):
        c = rep['conditions'][tag]['suts']
        return round(c['following_H0']['robust_fair'] - c['rss_sut_H0']['robust_fair'], 4)
    cg = rep['conditions']['carsG1']
    rep['BL1'] = dict(rss_robust_fair=cg['suts']['rss_sut_H0']['robust_fair'], supported=cg['suts']['rss_sut_H0']['robust_fair'] <= .05)
    rep['BL2'] = dict(share=cg['pooled_weak4_rss']['share'], supported=bool(cg['pooled_weak4_rss']['share'] is not None
                                                                           and cg['pooled_weak4_rss']['share'] >= .5))
    rep['BL3'] = {tag: dict(P1=p1(tag), supported=p1(tag) < .5) for tag in ('feasr0sh', 'feast0sh')}
    rep['P1'] = {tag: p1(tag) for tag in rep['conditions']}
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for tag, c in rep['conditions'].items():
        s = c['suts']
        print(f"{tag:16s} seeds {c['seeds']} | H0 col {s['following_H0']['collision'] * 100:5.1f} robust {s['following_H0']['robust_fair'] * 100:5.1f} "
              f"| RSS col {s['rss_sut_H0']['collision'] * 100:5.1f} fair {s['rss_sut_H0']['fair'] * 100:5.1f} feint {s['rss_sut_H0']['fair_feint'] * 100:5.1f} "
              f"robust {s['rss_sut_H0']['robust_fair'] * 100:5.1f} unfair {s['rss_sut_H0']['attacker_caused'] * 100:5.1f} "
              f"| pooled attacker-caused {c['pooled_weak4_rss']['share']} | P1 {rep['P1'][tag] * 100:5.1f}")
    print('BL1', rep['BL1'], '| BL2', rep['BL2'], '| BL3', rep['BL3'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('dir')
    ap.add_argument('--feas-noshield')
    main(ap.parse_args())
