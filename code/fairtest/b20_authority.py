"""B20 (G1-3): when does the collision rate separate SUTs again? Restrict the attack space and watch.

From stored grid landscapes (no new simulation), the collision-seeking oracle (any SUT collision among the
allowed plans) and the fair oracle are recomputed per SUT under two families of restrictions:
  authority  |pre|, |post| <= a_max (the attacker's braking authority, m/s^2);
  severity   the plan's CC braking demand <= s (fraction of 7.59 m/s^2; R157 difficulty in continuous form).
Validity per restriction: Kendall tau-b of the collision oracle against the known capability rank, spread,
and the share of collisions that are attacker-at-fault.

  python -m fairtest.b20_authority OUT.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import fairtest.boundary_planner as BP
from fairtest.analyze_landscape import load
from fairtest.b20_validity import RANK, SUT_T
from fairtest.strict_analysis import EV, J

AMAX = (0., 1., 1.5, 2., 3.)
SEVERITY = (.1, .2, .3, .4, .6, .8, 1., 1.5)
CC = ('cc_r157', 0.)


def parse(name):
    parts = name.split('_')
    return float(parts[2][3:]), float(parts[3][4:])


def sweep(rows):
    by = BP.index(rows)
    have = {(r['target'], r['H']) for r in rows}
    suts = [k for k in RANK if SUT_T[k] in have]
    cases = sorted({c for c, _ in by})
    per_case = collections.defaultdict(list)
    for (c, p), v in by.items():
        if p.startswith('plan_') and p != BP.NOMINAL and CC in v:
            pre, post = parse(p)
            per_case[c].append((max(abs(pre), abs(post)), BP.cc_demand(v[CC]), v))
    out = {}
    for fam, levels, key in (('authority', AMAX, 0), ('severity', SEVERITY, 1)):
        out[fam] = {}
        for lv in levels:
            res = {}
            for k in suts:
                sut = SUT_T[k]
                coll = fair = unfair = 0
                for c in cases:
                    allowed = [v for a, d, v in per_case[c] if (a if key == 0 else d) <= lv + 1e-9 and sut in v]
                    hits = [v for v in allowed if v[sut]['valid'] and v[sut]['contact']]
                    coll += bool(hits)
                    fair += any(not v[CC]['contact'] for v in hits)
                    unfair += bool(hits) and all(v[CC]['contact'] for v in hits)
                n = len(cases)
                res[k] = dict(collision=coll / n, fair=fair / n, only_unfair=unfair / n)
            ranks = [RANK[k] for k in suts]
            cv = [res[k]['collision'] for k in suts]
            fv = [res[k]['fair'] for k in suts]
            tau_c = kendalltau(ranks, cv).statistic if len(set(cv)) > 1 else float('nan')
            tau_f = kendalltau(ranks, fv).statistic if len(set(fv)) > 1 else float('nan')
            out[fam][f'{lv:g}'] = dict(per_sut=res, collision_tau=float(tau_c), fair_tau=float(tau_f),
                                       collision_spread=float(max(cv) - min(cv)), fair_spread=float(max(fv) - min(fv)))
    return out


def main(out_path):
    from fairtest.report_b7 import rows_all
    exp, _ = rows_all()
    exp += load(J / 'b10v2-a' / 'landscape.jsonl')[0] + load(J / 'b10v2-b' / 'landscape.jsonl')[0]
    b9 = sum((load(J / s / 'landscape.jsonl')[0] for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')), [])
    rep = {}
    for name, rows in (('highD_exposed_266', exp), ('highD_B9_148_posthoc', b9)):
        rep[name] = sweep(rows)
        print('==', name)
        for fam, levels in rep[name].items():
            for lv, d in levels.items():
                ps = d['per_sut']
                row = ' '.join(f"{k.replace('following_', 'f').replace('reference_idm', 'IDM').replace('rss_sut', 'RSS')}:{100*ps[k]['collision']:3.0f}/{100*ps[k]['fair']:3.0f}" for k in ps)
                print(f"  {fam:9} {lv:>4}  coll tau {d['collision_tau']:+.2f} spread {100*d['collision_spread']:3.0f} | fair tau {d['fair_tau']:+.2f} | {row}")
    Path(out_path).write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
