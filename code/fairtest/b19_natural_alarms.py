"""B19: natural false-alarm cost of a perception-threshold ADS (highD exposed recordings only).

A benign overtaking = a vehicle F passing a lane-keeping vehicle L (no lane changes) that is F's
left/right preceding vehicle. The perception-threshold ADS `ads@d_p` treats L as a cut-in once L's
lateral displacement from its own lane centre, towards F's lane, reaches d_p; it then brakes at
8 m/s^2 if the longitudinal bumper TTC <= 2 s persists beyond its 0.3 s latency. Such braking at a
vehicle that never leaves its lane is a phantom brake. For each d_p report phantom-brake events per
1000 overtakings and per driven hour, plus the natural lane-keeping wander distribution.
The same count for whole driver models (B21/B19): the R157 time-based perception (0.4 s outside the
0.375 m wandering zone) with the ADS latency (0.3 s) or the careful driver's reaction (0.75 s), and the
fixed-distance careful driver (1.095 m, 0.75 s) — how often each reference would itself brake at a
vehicle that never leaves its lane.

  python -m fairtest.b19_natural_alarms HIGHD_DATA_DIR OUT.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPOSED = [r for r in range(1, 61) if not (11 <= r <= 14 or 25 <= r <= 57)]
THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9, 1.0, 1.095, 1.2, 1.3, 1.4, 1.6, 2.0)
FPS, LATENCY_S, TTC_S = 25., .3, 2.
WANDER_M, PERCEPTION_S, CC_REACTION_S, CC_DISTANCE_M = .375, .4, .75, 1.095
MODELS = ('time_perception_ads_latency', 'time_perception_cc_reaction', 'distance_perception_cc_reaction')


def _run_lengths(mask):
    out, r = np.zeros(len(mask), int), 0
    for i, m in enumerate(mask):
        r = r + 1 if m else 0
        out[i] = r
    return out


def recording(data, rec):
    meta = pd.read_csv(data / f'{rec:02d}_recordingMeta.csv').iloc[0]
    tm = pd.read_csv(data / f'{rec:02d}_tracksMeta.csv').set_index('id')
    tr = pd.read_csv(data / f'{rec:02d}_tracks.csv')
    centre_of = {}
    # lane centre per laneId: median vehicle centre y in that lane (robust to marking conventions)
    tr['yc'] = tr['y'] + tr['height'] / 2
    for lid in sorted(set(tr['laneId'])):
        centre_of[lid] = float(tr.loc[tr['laneId'] == lid, 'yc'].median())
    tr['xc'] = tr['x'] + tr['width'] / 2
    by = {vid: g.set_index('frame') for vid, g in tr.groupby('id')}
    keep = set(tm.index[tm['numLaneChanges'] == 0])
    wander = []
    events = {d: 0 for d in THRESHOLDS}
    model_events = dict.fromkeys(MODELS, 0)
    pairs = 0
    hours = float(tm['numFrames'].sum()) / FPS / 3600.
    for vid, g in by.items():
        if tm.loc[vid, 'class'] != 'Car':
            continue
        direction = 1. if tm.loc[vid, 'drivingDirection'] == 2 else -1.
        for side in ('leftPrecedingId', 'rightPrecedingId'):
            ids = g[side].to_numpy()
            for lead in set(ids[ids > 0]):
                if lead not in keep or lead not in by:
                    continue
                frames = g.index[ids == lead]
                L = by[lead]
                common = frames.intersection(L.index)
                if len(common) < int(FPS):
                    continue
                pairs += 1
                f, l = g.loc[common], L.loc[common]
                gap = (l['xc'] - f['xc']) * direction - (l['width'] + f['width']) / 2
                closing = (f['xVelocity'] - l['xVelocity']) * direction
                ttc = np.where((gap > 0) & (closing > 0), gap / np.maximum(closing, 1e-6), np.inf)
                towards = np.sign(f['yc'].to_numpy() - l['yc'].to_numpy())
                disp = (l['yc'].to_numpy() - np.array([centre_of[int(k)] for k in l['laneId']])) * towards
                wander.append(float(np.max(disp)))
                need = int(round(LATENCY_S * FPS))
                for d in THRESHOLDS:
                    cond = (disp >= d) & (ttc <= TTC_S)
                    run = best = 0
                    for c in cond:
                        run = run + 1 if c else 0
                        best = max(best, run)
                    if best > need:
                        events[d] += 1
                timed = _run_lengths(disp >= WANDER_M) >= int(round(PERCEPTION_S * FPS))
                for m, perceived, react in ((MODELS[0], timed, LATENCY_S), (MODELS[1], timed, CC_REACTION_S),
                                            (MODELS[2], disp >= CC_DISTANCE_M, CC_REACTION_S)):
                    if _run_lengths(perceived & (ttc <= TTC_S)).max() > int(round(react * FPS)):
                        model_events[m] += 1
    return dict(recording=rec, pairs=pairs, hours=hours, events=events, model_events=model_events, wander=wander)


def main(data, out):
    data = Path(data)
    res = [recording(data, r) for r in EXPOSED if (data / f'{r:02d}_tracks.csv').exists()]
    pairs = sum(r['pairs'] for r in res)
    hours = sum(r['hours'] for r in res)
    wander = np.concatenate([r['wander'] for r in res]) if res else np.array([])
    rep = dict(recordings=[r['recording'] for r in res], overtakings=pairs, vehicle_hours=hours,
               wander_max_displacement_towards_quantiles={f'p{q}': float(np.percentile(wander, q)) for q in (50, 90, 99, 99.9)},
               phantom_brakes={f'{d:g}': dict(events=sum(r['events'][d] for r in res),
                                             per_1000_overtakings=1000. * sum(r['events'][d] for r in res) / max(1, pairs),
                                             per_vehicle_hour=sum(r['events'][d] for r in res) / max(1e-9, hours)) for d in THRESHOLDS},
               reference_models={m: dict(events=sum(r['model_events'][m] for r in res),
                                         per_1000_overtakings=1000. * sum(r['model_events'][m] for r in res) / max(1, pairs),
                                         per_vehicle_hour=sum(r['model_events'][m] for r in res) / max(1e-9, hours)) for m in MODELS})
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
