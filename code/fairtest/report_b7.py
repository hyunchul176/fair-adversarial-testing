"""Assemble the B7 landscape result: v1 (local) + v2 (server shards) -> analysis JSON + figures.

  python -m fairtest.report_b7 OUTDIR
"""
from __future__ import annotations

import collections
import json
import math
import os
import sys
from pathlib import Path

from fairtest.analyze_landscape import analyse, load
from fairtest.boundary_planner import CC, NOMINAL, SUTS, cc_demand, evaluate, index
from fairtest.figures import make_all
from fairtest.guided_search import curves

EV = Path(os.environ.get('FAIRTEST_EVIDENCE', 'evidence'))  # local evidence folder of the study (not distributed)
V1 = EV / 'b7_landscape' / 'landscape_v1.jsonl'
V2 = [EV / 'fairtest_jobs' / lab / 'landscape.jsonl' for lab in
      ('landscape-v2-gpu01', 'landscape-v2-gpu03', 'landscape-v2-rest', 'landscape-v2-gpu02b', 'landscape-v2-gpu04b')]


def rows_all():
    v1, e1 = load(V1)
    v2, e2 = [], []
    for p in V2:
        r, e = load(p)
        v2 += r
        e2 += e
    c1 = collections.Counter(r['case_id'] for r in v1)
    c2 = collections.Counter(r['case_id'] for r in v2)
    check = dict(v1_rows=len(v1), v1_errors=len(e1), v2_rows=len(v2), v2_errors=len(e2),
                 v1_complete_cases=sum(1 for n in c1.values() if n == 800),
                 v2_complete_cases=sum(1 for n in c2.values() if n == 600))
    return v1 + v2, check


def fragility(rows, bins=(0, .2, .3, .4, .5, .6, .8, 1.5)):
    """P(SUT contact | CC-preventable plan, CC demand bin), pooled over cases."""
    by = index(rows)
    out = {}
    for s in SUTS:
        name = f'{s[0]}_H{s[1]:g}'
        tally = collections.defaultdict(lambda: [0, 0])
        for (c, p), v in by.items():
            if not p.startswith('plan_') or p == NOMINAL or CC not in v or s not in v:
                continue
            cc = v[CC]
            if not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
                continue
            d = cc_demand(cc)
            b = next((f'{lo:g}-{hi:g}' for lo, hi in zip(bins, bins[1:]) if lo <= d < hi), None)
            if b is None:
                continue
            tally[b][0] += 1
            tally[b][1] += bool(v[s]['valid'] and v[s]['contact'])
        if tally:
            out[name] = {b: dict(plans=n, sut_failure_rate=k / n) for b, (n, k) in sorted(tally.items())}
    return out


def by_partition(rows):
    out = {}
    for part in ('calibration', 'development'):
        sub = [r for r in rows if r['partition'] == part]
        res = analyse(sub)
        out[part] = {k: {m: v[m] for m in ('eligible_cases', 'oracle_cases_with_fair_failure', 'b5_fair',
                                            'b5_unpreventable', 'fixed2s_fair', 'fixed2s_unpreventable')}
                     for k, v in res['suts'].items()}
    return out


def main(outdir):
    outdir = Path(outdir)
    rows, check = rows_all()
    res = make_all(rows, outdir)
    report = dict(completeness=check, analysis=res,
                  boundary=evaluate(rows, severities=(1.,), exceedances=(math.inf, 2., 1., .5)),
                  guided_search={'no_limit': curves(rows), 'max_exceedance_1': curves(rows, max_exceedance=1.)},
                  fragility=fragility(rows), partitions=by_partition(rows))
    (outdir / 'B7_REPORT.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(check))
    return report


if __name__ == '__main__':
    main(sys.argv[1])
