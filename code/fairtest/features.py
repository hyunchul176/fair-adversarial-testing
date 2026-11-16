"""Cut-in kinematic features and naturalistic reference (exposed highD partitions only).

Anchor p50: generator lateral centre crossing y=2 (half-way between lane centres 0 and 4),
matching the highD extractor's 50% lateral-progress anchor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

EVIDENCE = Path(os.environ.get('FAIRTEST_EVIDENCE', 'evidence'))  # local evidence folder of the study (not distributed)
NATURAL = EVIDENCE / 'highd_cutin_joint' / 'result'
DT = .1
FEATURES = ('gap_p50', 'thw_p50', 'closing_p50', 'acc_pre', 'acc_post', 'duration_10_90')


def _mean_acc(t, v, lo, hi, min_span=.5):
    m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if m.sum() < 2:
        return None
    tt, vv = t[m], v[m]
    if tt[-1] - tt[0] < min_span:
        return None
    return float((vv[-1] - vv[0]) / (tt[-1] - tt[0]))


def _crossing(times, ys, level):
    for i in range(1, len(ys)):
        if ys[i - 1] < level <= ys[i]:
            return times[i - 1] + (times[i] - times[i - 1]) * (level - ys[i - 1]) / (ys[i] - ys[i - 1])
    return None


def simulated_features(row):
    """Features of one simulated episode (row = B7Env.result())."""
    tr = row['trace']
    init = row['physical_initial_state']
    t = np.arange(len(tr) + 1) * DT
    gy = np.array([init[0]['y']] + [r['generator']['y'] for r in tr])
    gx = np.array([init[0]['x']] + [r['generator']['x'] for r in tr])
    gv = np.array([init[0]['vx']] + [r['generator']['vx'] for r in tr])
    tx = np.array([init[1]['x']] + [r['target']['x'] for r in tr])
    tv = np.array([init[1]['vx']] + [r['target']['vx'] for r in tr])
    out = dict.fromkeys(FEATURES)
    # last upward crossing of y=2 is the realised cut-in
    k50 = None
    for i in range(1, len(gy)):
        if gy[i - 1] < 2. <= gy[i]:
            k50 = i
    acc = np.diff(gv) / DT
    out['peak_abs_acc'] = float(np.max(np.abs(acc))) if len(acc) else 0.
    out['peak_abs_jerk'] = float(np.max(np.abs(np.diff(acc) / DT))) if len(acc) > 1 else 0.
    out['min_acc'] = float(np.min(acc)) if len(acc) else 0.
    if k50 is None:
        out['cut_in'] = False
        return out
    out['cut_in'] = True
    gap = gx[k50] - tx[k50] - 5.
    out['gap_p50'] = float(gap)
    out['closing_p50'] = float(tv[k50] - gv[k50])
    out['thw_p50'] = float(gap / tv[k50]) if tv[k50] > .1 else None
    out['acc_pre'] = _mean_acc(t, gv, t[k50] - 3., t[k50])
    out['acc_post'] = _mean_acc(t, gv, t[k50], t[k50] + 2.)
    start = None
    for i in range(k50, 0, -1):  # last upward crossing of y=0.4 before p50
        if gy[i - 1] < .4 <= gy[i]:
            start = t[i - 1] + DT * (.4 - gy[i - 1]) / (gy[i] - gy[i - 1])
            break
    if start is None and gy[0] >= .4:
        start = 0.  # reset state is the recorded 10% crossing
    end = _crossing(t[k50:], gy[k50:], 3.6)
    out['duration_10_90'] = float(end - start) if start is not None and end is not None else None
    return out


def natural_reference():
    """Features of eligible natural cut-ins in the exposed partitions (same definitions).

    On compute nodes the serialised copy `natural_reference.json` in FAIRTEST_DATA is used (identical rows).
    """
    bundled = os.environ.get('FAIRTEST_DATA')
    if bundled and (Path(bundled) / 'natural_reference.json').exists():
        return json.loads((Path(bundled) / 'natural_reference.json').read_text(encoding='utf-8'))['rows']
    events = {}
    with (NATURAL / 'events.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r.get('eligible') and r['partition'] in ('calibration', 'development'):
                events[r['event_id']] = r
    rows = []
    with (NATURAL / 'windows.jsonl').open() as f:
        for line in f:
            w = json.loads(line)
            r = events.get(w['event_id'])
            if r is None:
                continue
            s = w['sign']
            g = w['generator']
            t = np.array(g['time'])
            v = np.array(g['vx']) * s
            p50 = w['times'][1]
            m = r['metrics']
            acc = np.diff(v) / np.diff(t)
            rows.append(dict(event_id=w['event_id'], partition=r['partition'],
                             gap_p50=m['p50']['gap'], thw_p50=m['p50']['headway'],
                             closing_p50=m['p50']['closing_speed'],
                             acc_pre=_mean_acc(t, v, p50 - 3., p50), acc_post=_mean_acc(t, v, p50, p50 + 2.),
                             duration_10_90=m['duration_10_90'], peak_abs_acc=m.get('peak_ax')))
    return rows


class Realism:
    """Per-feature two-sided empirical tail score and exceedance beyond natural p1..p99."""

    def __init__(self, natural=None):
        natural = natural or natural_reference()
        self.samples = {f: np.sort(np.array([r[f] for r in natural if r.get(f) is not None], float))
                        for f in FEATURES}
        self.bounds = {f: (np.percentile(v, 1), np.percentile(v, 99), np.subtract(*np.percentile(v, [75, 25])))
                       for f, v in self.samples.items()}

    def score(self, feats):
        tails, exceed = {}, {}
        for f in FEATURES:
            x = feats.get(f)
            if x is None:
                continue
            v = self.samples[f]
            u = np.searchsorted(v, x, side='right') / len(v)
            tails[f] = float(max(u, 1 - u))
            lo, hi, iqr = self.bounds[f]
            exceed[f] = float(max(0., (x - hi) / iqr, (lo - x) / iqr))
        return dict(tail=tails, exceedance=exceed,
                    max_tail=max(tails.values()) if tails else None,
                    max_exceedance=max(exceed.values()) if exceed else None)
