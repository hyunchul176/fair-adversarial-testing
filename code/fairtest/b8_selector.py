"""B8a: learned SUT-specific plan selector (zero SUT queries at test time).

For each (case, candidate plan) the selector sees only SUT-free information: the case's
initial conditions, the plan parameters, the generator's own kinematic features and the
R157 CC reference-driver simulation of that plan. It predicts P(fair failure | SUT) with a
gradient-boosted classifier trained on calibration cases (where the SUT was queried during
training), and at test time ranks the CC-preventable candidates of a new case by that
probability. Evaluated on development cases against the R157-boundary ranking and the oracle.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from fairtest.boundary_planner import CC, NOMINAL, SUTS, cc_demand, index
from fairtest.features import FEATURES, Realism

CASE_KEYS = ('generator_speed', 'target_speed', 'gap', 'headway')


def plan_params(name):
    # plan_e{entry}_pre{pre}_post{post}_d{dur}
    parts = name.split('_')
    entry = parts[1][1:]
    return [99. if entry == 'never' else float(entry), float(parts[2][3:]), float(parts[3][4:]),
            0. if parts[4][1:] == 'grid' else float(parts[4][1:])]


def feature_row(case, plan, cc, realism_score):
    p10 = case['event']['metrics']['p10']
    f = cc['features']
    x = [p10[k] for k in CASE_KEYS] + [case['event']['metrics']['duration_10_90'],
                                       p10['target_speed'] - p10['generator_speed']]
    x += plan_params(plan)
    x += [f.get(k) if f.get(k) is not None else np.nan for k in FEATURES]
    x += [f.get('peak_abs_acc', np.nan), f.get('min_acc', np.nan)]
    x += [cc_demand(cc), cc['min_gap_overlap'] if cc['min_gap_overlap'] is not None else np.nan,
          cc['min_ttc'] if cc['min_ttc'] is not None else np.nan,
          cc.get('cc_emergency_onset_s') if cc.get('cc_emergency_onset_s') is not None else np.nan,
          min(50., cc.get('cc_drac_at_onset')) if cc.get('cc_drac_at_onset') is not None else np.nan,
          realism_score]
    return x


def candidates(by, cases, realism, sut, max_exceedance=math.inf):
    """Yield (case_id, plan, features, fair_label, cc_demand) for CC-preventable candidates."""
    per_case = {}
    for (c, p), v in by.items():
        per_case.setdefault(c, []).append((p, v))
    out = {}
    for c in cases:
        nom = by.get((c, NOMINAL), {}).get(sut)
        if nom is None or nom['contact'] or nom['ttc_event'] or nom['initial_event']:
            continue
        rows = []
        for p, v in per_case[c]:
            if not p.startswith('plan_') or p == NOMINAL or CC not in v or sut not in v:
                continue
            cc = v[CC]
            if not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
                continue
            sc = realism.score(cc['features'])['max_exceedance']
            sc = math.inf if sc is None else sc
            if sc > max_exceedance:
                continue
            label = bool(v[sut]['valid'] and v[sut]['contact'])
            rows.append((p, feature_row(cases[c], p, cc, min(sc, 20.)), label, cc_demand(cc), sc))
        out[c] = rows
    return out


def success_at(ranked, ks):
    return {k: float(np.mean([any(lbl for lbl in r[:k]) for r in ranked])) if ranked else 0. for k in ks}


def run(rows, cases_meta, ks=(1, 2, 3, 5, 10), max_exceedance=math.inf, seed=0):
    realism = Realism()
    by = index(rows)
    train_ids = {c for c, m in cases_meta.items() if m['event']['partition'] == 'calibration'}
    test_ids = {c for c, m in cases_meta.items() if m['event']['partition'] == 'development'}
    report = {}
    for sut in SUTS:
        name = f'{sut[0]}_H{sut[1]:g}'
        if not any(sut in v for v in by.values()):
            continue
        cand = candidates(by, cases_meta, realism, sut, max_exceedance)
        tr = [r for c in train_ids if c in cand for r in cand[c]]
        if not tr or sum(r[2] for r in tr) < 5:
            report[name] = dict(note='too few positive training examples', positives=sum(r[2] for r in tr))
            continue
        X = np.array([r[1] for r in tr], float)
        y = np.array([r[2] for r in tr], int)
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=31,
                                               l2_regularization=1., random_state=seed)
        model.fit(X, y)
        learned, boundary, oracle = [], [], 0
        n_eval = 0
        for c in sorted(test_ids):
            if c not in cand:
                continue
            n_eval += 1
            rs = cand[c]
            if not rs:
                learned.append([])
                boundary.append([])
                continue
            prob = model.predict_proba(np.array([r[1] for r in rs], float))[:, 1]
            order = np.argsort(-prob, kind='stable')
            learned.append([rs[i][2] for i in order])
            boundary.append([r[2] for r in sorted(rs, key=lambda r: (-r[3], r[4]))])
            oracle += any(r[2] for r in rs)
        report[name] = dict(train_candidates=len(tr), train_positives=int(y.sum()), test_cases=n_eval,
                            learned=success_at(learned, ks), boundary=success_at(boundary, ks),
                            oracle=oracle / max(1, n_eval))
    return report


if __name__ == '__main__':
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    all_rows, _ = rows_all()
    out = {}
    for exc in (math.inf, 1.):
        out['no_limit' if math.isinf(exc) else f'max_exceedance_{exc:g}'] = run(all_rows, _STATE['cases'], max_exceedance=exc)
    text = json.dumps(out, indent=1)
    print(text)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text + '\n', encoding='utf-8')
