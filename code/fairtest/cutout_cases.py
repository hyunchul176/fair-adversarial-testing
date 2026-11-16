"""B34: steady three-car platoons from the exposed highD recordings (the cut-out scenario's initial states).

A snapshot = a follower E, its preceding car L1 and L1's preceding car L2, all cars in one lane, both preceding
relations constant for >= 3 s with all three >= 15 m/s, THW(E->L1) 0.5-3.0 s, THW(L1->L2) 0.5-4.0 s and |relative
speeds| < 3 m/s throughout; the state 1 s into that window is taken (one snapshot per E-L1 pair, the first
qualifying window). Partition and selection as dec_cases (B30): calibration recordings 1-10, 15-24, development
58-60; sha256(SALT + id) ranking, participant-disjoint greedy, at most ceil(target / recordings) per recording.

  python -m fairtest.cutout_cases HIGHD_DATA_DIR OUT.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

from fairtest.dec_cases import CALIBRATION, DEVELOPMENT, OFFSET_S, WINDOW_S

TARGETS = dict(calibration=150, development=60)
SALT = 'acarl-b34-cutout-v1:'
COLS = ('frame', 'x', 'width', 'xVelocity', 'xAcceleration', 'precedingId')


def snapshots(data, rec):
    fps = float(pd.read_csv(data / f'{rec:02d}_recordingMeta.csv').iloc[0]['frameRate'])
    meta = pd.read_csv(data / f'{rec:02d}_tracksMeta.csv').set_index('id')
    tr = pd.read_csv(data / f'{rec:02d}_tracks.csv', usecols=list(COLS) + ['id', 'laneId'])
    by = {}
    for vid, g in tr.groupby('id'):
        g = g.sort_values('frame')
        by[int(vid)] = {c: g[c].to_numpy() for c in COLS}
    car = lambda v: v > 0 and v in by and meta.loc[v, 'class'] == 'Car'  # noqa: E731

    def at(v, frame):
        i = int(frame - by[v]['frame'][0])
        return i if 0 <= i < len(by[v]['frame']) else None

    win, off = int(WINDOW_S * fps), int(OFFSET_S * fps)
    out = []
    for vid, g in by.items():
        if meta.loc[vid, 'class'] != 'Car':
            continue
        sign = 1. if int(meta.loc[vid, 'drivingDirection']) == 2 else -1.
        gap = lambda xb, wb, xf, wf: (xf - (xb + wb)) if sign > 0 else (xb - (xf + wf))  # noqa: E731  back, front
        run, key, done = 0, None, set()
        for i, frame in enumerate(g['frame']):
            l1 = int(g['precedingId'][i])
            if l1 in done:  # one snapshot per E-L1 pair (the first qualifying window)
                run, key = 0, None
                continue
            i1 = at(l1, frame) if car(l1) else None
            l2 = int(by[l1]['precedingId'][i1]) if i1 is not None else 0
            i2 = at(l2, frame) if car(l2) else None
            good = False
            if i2 is not None:
                d1 = gap(g['x'][i], g['width'][i], by[l1]['x'][i1], by[l1]['width'][i1])
                d2 = gap(by[l1]['x'][i1], by[l1]['width'][i1], by[l2]['x'][i2], by[l2]['width'][i2])
                v0, v1, v2 = abs(g['xVelocity'][i]), abs(by[l1]['xVelocity'][i1]), abs(by[l2]['xVelocity'][i2])
                good = (min(v0, v1, v2) >= 15 and d1 > 0 and d2 > 0 and .5 <= d1 / v0 <= 3. and .5 <= d2 / v1 <= 4.
                        and abs(v1 - v0) < 3 and abs(v2 - v1) < 3)
            run = run + 1 if good and (l1, l2) == key else (1 if good else 0)
            key = (l1, l2) if good else None
            if run >= win:
                s = i - run + 1 + off
                f = int(g['frame'][s])
                j1, j2 = at(l1, f), at(l2, f)
                out.append(dict(snapshot_id=f'highd-{rec}:{vid}:{l1}:{l2}:{f}', recording=rec, follower=int(vid),
                                lead=int(l1), lead2=int(l2), frame=f,
                                gap=float(gap(g['x'][s], g['width'][s], by[l1]['x'][j1], by[l1]['width'][j1])),
                                gap2=float(gap(by[l1]['x'][j1], by[l1]['width'][j1], by[l2]['x'][j2], by[l2]['width'][j2])),
                                v_follower=float(abs(g['xVelocity'][s])), v_lead=float(abs(by[l1]['xVelocity'][j1])),
                                v_lead2=float(abs(by[l2]['xVelocity'][j2])),
                                length_follower=float(g['width'][s]), length_lead=float(by[l1]['width'][j1]),
                                length_lead2=float(by[l2]['width'][j2])))
                done.add(l1)
                run, key = 0, None
    return out


def select(cands, partition, recordings):
    cap = math.ceil(TARGETS[partition] / len(recordings))
    cases = []
    for rec in recordings:
        used = set()
        pool = sorted((c for c in cands if c['recording'] == rec),
                      key=lambda c: hashlib.sha256((SALT + c['snapshot_id']).encode()).hexdigest())
        n = 0
        for c in pool:
            if n >= cap or {c['follower'], c['lead'], c['lead2']} & used:
                continue
            used |= {c['follower'], c['lead'], c['lead2']}
            cases.append(dict(c, partition=partition))
            n += 1
    return cases


def main(data, out):
    data = Path(data)
    cands, cases = {}, []
    for part, recs in (('calibration', CALIBRATION), ('development', DEVELOPMENT)):
        allc = []
        for rec in recs:
            s = snapshots(data, rec)
            allc += s
            print(json.dumps(dict(recording=rec, snapshots=len(s))), flush=True)
        cands[part] = len(allc)
        cases += select(allc, part, recs)
    for i, c in enumerate(cases):
        c['case_id'] = i
    Path(out).write_text(json.dumps(dict(salt=SALT, candidates=cands, cases=cases), indent=0), encoding='utf-8')
    print(json.dumps(dict(candidates=cands, selected={p: sum(c['partition'] == p for c in cases) for p in TARGETS})))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
