"""B10: learned-selector learning curve and cross-SUT transfer (offline, existing landscape rows).

Learning curve: train on k random calibration cases (5 repeats), evaluate success@1 on all
development cases. Transfer: train on SUT A (all calibration), evaluate on SUT B (development).
Zero SUT queries at evaluation; training uses the calibration landscape of the training SUT.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from fairtest.b8_selector import candidates
from fairtest.boundary_planner import index
from fairtest.features import Realism

PARAMS = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=0)
SUTS = (('following', .5), ('following', 1.), ('following_aeb', 0.))


def name(s):
    return f'{s[0]}_H{s[1]:g}'


def fit(rows_):
    X = np.array([r[1] for r in rows_], float)
    y = np.array([r[2] for r in rows_], int)
    if y.sum() < 2:
        return None
    return HistGradientBoostingClassifier(**PARAMS).fit(X, y)


def success1(model, cand, cases, fallback_boundary=True):
    hits = []
    for c in cases:
        rs = cand.get(c)
        if rs is None:
            continue
        if not rs:
            hits.append(0.)
            continue
        if model is None:
            best = max(rs, key=lambda r: (r[3], -r[4]))
        else:
            p = model.predict_proba(np.array([r[1] for r in rs], float))[:, 1]
            best = rs[int(np.argmax(p))]
        hits.append(float(best[2]))
    return float(np.mean(hits)) if hits else float('nan')


def main(out):
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    rows, _ = rows_all()
    by = index(rows)
    realism = Realism()
    cases = _STATE['cases']
    cal = sorted(c for c, v in cases.items() if v['event']['partition'] == 'calibration')
    dev = sorted(c for c, v in cases.items() if v['event']['partition'] == 'development')
    cand = {name(s): candidates(by, cases, realism, s, math.inf) for s in SUTS}
    report = dict(learning_curve={}, transfer={}, boundary_dev={}, oracle_dev={})
    for s in SUTS:
        n = name(s)
        report['boundary_dev'][n] = success1(None, cand[n], dev)
        report['oracle_dev'][n] = float(np.mean([any(r[2] for r in cand[n][c]) for c in dev if c in cand[n]]))
        curve = {}
        for k in (10, 25, 50, 100, len(cal)):
            vals = []
            for rep in range(5 if k < len(cal) else 1):
                sub = random.Random(1000 * k + rep).sample(cal, k)
                model = fit([r for c in sub if c in cand[n] for r in cand[n][c]])
                vals.append(success1(model, cand[n], dev))
            curve[k] = dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)))
        report['learning_curve'][n] = curve
    for a in SUTS:
        model = fit([r for c in cal if c in cand[name(a)] for r in cand[name(a)][c]])
        for b in SUTS:
            report['transfer'][f'{name(a)}->{name(b)}'] = success1(model, cand[name(b)], dev)
    Path(out).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
