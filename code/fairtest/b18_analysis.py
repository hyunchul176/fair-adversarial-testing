"""B18 analysis: where does the behaviour-cloned human SUT sit on the capability ladder, and how do the
testers judge it? Grid oracle / boundary (frozen B9 logic, CC reference), cont_search, generic
falsification (collisions and attacker-at-fault share) and B14 searches, per dataset.

  python -m fairtest.b18_analysis OUT.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from fairtest import b9_analysis as A
import fairtest.boundary_planner as BP
from fairtest.analyze_landscape import load
from fairtest.strict_analysis import EV, E13, J

B18 = J / 'b18'
SUT = ('bc_human', 0.)
KEY = 'bc_human_H0'


def cc_rows(name):
    if name == 'exposed':
        from fairtest.report_b7 import rows_all
        rows, _ = rows_all()
        return [r for r in rows if r['target'] == 'cc_r157']
    shards = {'b9': ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b'), 'ngsim': ('b13-ngsim-a', 'b13-ngsim-b')}[name]
    out = []
    for s in shards:
        out += [r for r in load(J / s / 'landscape.jsonl')[0] if r['target'] == 'cc_r157']
    return out


def cases_of(name):
    if name == 'exposed':
        from fairtest.runner import _STATE, _init
        _init()
        return dict(_STATE['cases'])
    pool = {'b9': EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json',
            'ngsim': E13 / 'cohort' / 'case_pool.json'}[name]
    return {c['case_id']: c for c in json.loads(pool.read_text(encoding='utf-8'))['cases']}


def main(out):
    rep = {}
    for name in ('exposed', 'b9', 'ngsim'):
        rows = cc_rows(name) + load(B18 / f'b18_{name}_grid.jsonl')[0]
        cases = cases_of(name)
        A.SUTS, BP.SUTS = (SUT,), (SUT,)
        per = A.per_case_outcomes(rows, cases, {}, 'cc_r157', math.inf)[KEY]
        vals = [m for m in per.values() if m is not None]
        grid = {k: float(np.mean([m[k][0] for m in vals])) for k in ('b5', 'boundary', 'guided_5', 'oracle')}
        grid['b5_unfair'] = float(np.mean([m['b5'][1] for m in vals]))
        grid['eligible'] = len(vals)
        n = len(cases)
        cont = [json.loads(l) for l in open(B18 / f'b18_{name}_cont.jsonl', encoding='utf-8')]
        cont_fair = sum(1 for r in cont if (r['sut_search'][KEY].get('best') or {}).get('fair')) / n
        fal = [json.loads(l) for l in open(B18 / f'b18_{name}_falsify.jsonl', encoding='utf-8')]
        coll = [r['runs'][f'falsify_centre|{KEY}'] for r in fal if (r['runs'][f'falsify_centre|{KEY}']['best'] or {}).get('contact')]
        unfair = sum(1 for rec in coll if rec['refs']['cc_r157']['contact'])
        b14 = [json.loads(l) for l in open(B18 / f'b18_{name}_b14.jsonl', encoding='utf-8')]
        fr = [r['runs'][f'falsify_real|{KEY}'] for r in b14]
        fr_hits = [x for x in fr if (x['best'] or {}).get('hit')]
        fair_real = sum(1 for r in b14 if (r['runs'][f'fair_real|{KEY}']['best'] or {}).get('hit')) / n
        rep[name] = dict(n_cases=n, grid=grid, cont_search_fair=cont_fair,
                         falsify_collision_rate=len(coll) / n, falsify_attacker_at_fault_share=(unfair / len(coll)) if coll else None,
                         falsify_real_collision_rate=len(fr_hits) / n,
                         falsify_real_attacker_at_fault_share=(sum(1 for x in fr_hits if x['refs']['cc_r157']) / len(fr_hits)) if fr_hits else None,
                         fair_real=fair_real)
        print(name, json.dumps({k: (round(100 * v, 1) if isinstance(v, float) else v) for k, v in rep[name].items() if k != 'grid'}),
              {k: round(100 * v, 1) if isinstance(v, float) else v for k, v in grid.items()})
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
