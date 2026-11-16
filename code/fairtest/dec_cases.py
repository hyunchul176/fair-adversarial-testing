"""B30: steady car-following snapshots from the exposed highD recordings (the deceleration scenario's initial states).

A snapshot = a follower car and its preceding car in the same lane, the preceding id constant for >= 3 s with both cars
>= 15 m/s, time headway 0.5-3.0 s and |relative speed| < 3 m/s throughout; the state 1 s into that window is taken
(one snapshot per pair, the first qualifying window). Recording partition as the cut-in cases (calibration 1-10,
15-24; development 58-60). Selection per recording: sha256(SALT + snapshot id) ranking, participant-disjoint greedy,
at most ceil(target / recordings) per recording (targets: calibration 150, development 60).

  python -m fairtest.dec_cases HIGHD_DATA_DIR OUT.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CALIBRATION = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
DEVELOPMENT = [58, 59, 60]
TARGETS = dict(calibration=150, development=60)
SALT = 'acarl-b30-dec-v1:'
WINDOW_S, OFFSET_S = 3., 1.


def snapshots(data, rec):
    fps = float(pd.read_csv(data / f'{rec:02d}_recordingMeta.csv').iloc[0]['frameRate'])
    meta = pd.read_csv(data / f'{rec:02d}_tracksMeta.csv').set_index('id')
    tr = pd.read_csv(data / f'{rec:02d}_tracks.csv', usecols=['frame', 'id', 'x', 'width', 'xVelocity', 'xAcceleration',
                                                              'precedingId', 'laneId'])
    by = {}
    for vid, g in tr.groupby('id'):
        g = g.sort_values('frame')
        by[int(vid)] = {c: g[c].to_numpy() for c in ('frame', 'x', 'width', 'xVelocity', 'xAcceleration', 'precedingId')}
    win, off = int(WINDOW_S * fps), int(OFFSET_S * fps)
    out = []
    for vid, g in by.items():
        if meta.loc[vid, 'class'] != 'Car':
            continue
        sign = 1. if int(meta.loc[vid, 'drivingDirection']) == 2 else -1.
        pre, frames = g['precedingId'], g['frame']
        k = 0
        while k < len(frames):
            lid = int(pre[k])
            j = k
            while j < len(frames) and pre[j] == lid:
                j += 1
            if lid > 0 and lid in by and meta.loc[lid, 'class'] == 'Car' and j - k >= win:
                lead = by[lid]
                li = frames[k:j] - lead['frame'][0]
                inside = (li >= 0) & (li < len(lead['frame']))
                li = np.clip(li, 0, len(lead['frame']) - 1)
                xf, wf, vxf, axf = g['x'][k:j], g['width'][k:j], g['xVelocity'][k:j], g['xAcceleration'][k:j]
                xl, wl, vxl, axl = lead['x'][li], lead['width'][li], lead['xVelocity'][li], lead['xAcceleration'][li]
                gap = np.where(sign > 0, xl - (xf + wf), xf - (xl + wl))
                vf, vl = np.abs(vxf), np.abs(vxl)
                thw = np.where(vf > 0, gap / np.maximum(vf, 1e-9), np.inf)
                ok = inside & (vf >= 15) & (vl >= 15) & (thw >= .5) & (thw <= 3.) & (np.abs(vl - vf) < 3.) & (gap > 0)
                run = 0
                for i, good in enumerate(ok):
                    run = run + 1 if good else 0
                    if run >= win:
                        s = i - run + 1 + off
                        out.append(dict(snapshot_id=f'highd-{rec}:{vid}:{lid}:{int(frames[k + s])}', recording=rec,
                                        follower=int(vid), lead=int(lid), frame=int(frames[k + s]), gap=float(gap[s]),
                                        v_follower=float(vf[s]), v_lead=float(vl[s]), a_follower=float(sign * axf[s]),
                                        a_lead=float(sign * axl[s]), length_follower=float(wf[s]), length_lead=float(wl[s]),
                                        thw=float(thw[s])))
                        break
            k = j
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
            if n >= cap or {c['follower'], c['lead']} & used:
                continue
            used |= {c['follower'], c['lead']}
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
