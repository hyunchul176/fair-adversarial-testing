"""B56 pilot (before the design freeze; CC replays only, no SUT run, so nothing about detection is seen).

Per development case: the share of admissible plans (b23_search.admission) among uniform draws from the B11 plan box,
and whether the case's boundary plan is admissible. Used only to set SCREEN_MAX and to predict the run's cost.
Own random stream (case_id, 56), not the search streams.

  python -m fairtest.b56_pilot OUT.json --cont STORED_B11.jsonl --draws 256 --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b23_search import HIGH, LOW, admission


def task(job):
    case_id, x_boundary, draws = job
    runner._init()
    case = runner._STATE['cases'][case_id]
    rng = np.random.default_rng([case_id, 56])
    t0 = time.time()
    reasons = dict(cc_contact=0, clearance=0, feint=0, realism=0, joint=0)
    ok = 0
    for _ in range(draws):
        good, adm = admission(case, LOW + rng.random(5) * (HIGH - LOW))
        ok += good
        if not good:
            if adm['cc']['contact']:
                reasons['cc_contact'] += 1
            elif adm['clear'] < .5:
                reasons['clearance'] += 1
            elif adm['feint']:
                reasons['feint'] += 1
            elif adm['exc'] is None or adm['exc'] > 1.:
                reasons['realism'] += 1
            else:
                reasons['joint'] += 1
    boundary = None if x_boundary is None else bool(admission(case, np.asarray(x_boundary, float))[0])
    return dict(case_id=case_id, draws=draws, admissible=ok, share=round(ok / draws, 4), boundary_admissible=boundary,
                first_reason=reasons, seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cont', required=True)
    ap.add_argument('--draws', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    runner._init()
    boundary = {}
    for line in open(a.cont, encoding='utf-8'):
        r = json.loads(line)
        b = (r.get('cc_boundary') or {}).get('best')
        boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if v['event'].get('partition') == 'development')
    with mp.Pool(a.workers) as pool:
        rows = sorted(pool.map(task, [(c, boundary.get(c), a.draws) for c in ids]), key=lambda r: r['case_id'])
    sh = np.array([r['share'] for r in rows])
    summary = dict(cases=len(rows), draws=a.draws, share_quantiles=dict(zip(('min', 'p10', 'p25', 'median', 'p75', 'max'),
                   [round(float(np.quantile(sh, q)), 4) for q in (0, .1, .25, .5, .75, 1)])),
                   cases_share_below_1pct=int((sh < .01).sum()), boundary_admissible=sum(bool(r['boundary_admissible']) for r in rows),
                   boundary_known=sum(r['boundary_admissible'] is not None for r in rows),
                   seconds_per_replay=round(sum(r['seconds'] for r in rows) / (len(rows) * (a.draws + 1)), 4))
    json.dump(dict(summary=summary, cases=rows), open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(summary, indent=1))
