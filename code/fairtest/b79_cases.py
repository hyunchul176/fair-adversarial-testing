"""B79 start states "from behind" (follow-up question of 2026-10-10: can the agent attack by accelerating from behind and cutting in?).

Every case of the stored pool starts at the 10% lateral crossing of a recorded highD cut-in, with the cut-in vehicle already
ahead of the follower (the extractor requires a positive gap from 10% to 90%). For each development case this script goes
back in the same highD recording to the latest frame, at most LOOKBACK_S before the 10% crossing, at which the cut-in
vehicle's centre was at least BEHIND_M behind the follower's centre, with the cut-in vehicle in its original lane and the
follower in the destination lane. The recorded speeds of both vehicles at that frame become the new start; the cut-in
vehicle is put on its lane centre (y 0, vy 0) and the follower on its lane (y 4), ahead by the recorded centre distance.
Cases without such a frame (the cut-in vehicle never was behind the follower in the window) are left out. The start must pass
the same initial-domain checks as the stored pool (speeds 15-40 / 15-35 m/s, no overlap). No simulation, no outcome.

  python -m fairtest.b79_cases DATA_DIR HIGHD_DIR OUT_POOL.json [--behind 6] [--lookback 15]
DATA_DIR: the development data (case_pool.json); HIGHD_DIR: highD data/ with NN_tracks.csv, NN_tracksMeta.csv,
NN_recordingMeta.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main(a):
    pool_path = Path(a.data) / 'case_pool.json'
    pool = json.loads(pool_path.read_text(encoding='utf-8'))
    parts = set(a.partitions.split(','))
    dev = [c for c in pool['cases'] if c['event']['partition'] in parts]
    out, ledger, hashes = [], [], {}
    for rec in sorted({c['event']['recording'] for c in dev}):
        paths = {s: Path(a.highd) / f'{rec:02d}_{s}.csv' for s in ('tracks', 'tracksMeta', 'recordingMeta')}
        hashes[rec] = {s: sha(p) for s, p in paths.items()}
        fps = float(pd.read_csv(paths['recordingMeta']).iloc[0]['frameRate'])
        meta = pd.read_csv(paths['tracksMeta']).set_index('id')
        df = pd.read_csv(paths['tracks'], usecols=['frame', 'id', 'x', 'y', 'width', 'height', 'xVelocity', 'laneId'])
        tracks = {int(i): g.set_index('frame') for i, g in df.groupby('id')}
        for c in [c for c in dev if c['event']['recording'] == rec]:
            ev = c['event']
            g_id, f_id = int(ev['vehicle']), int(c['follower'])
            direction = int(meta.loc[g_id, 'drivingDirection'])
            sign = -1. if direction == 1 else 1.
            g, f = tracks[g_id], tracks[f_id]
            t10 = int(np.ceil(ev['times'][0] * fps - 1e-8))
            old_lane = int(g.loc[t10, 'laneId']) if t10 in g.index else None
            new_lane = int(f.loc[t10, 'laneId']) if t10 in f.index else None
            row = dict(case_id=c['case_id'], event_id=ev['event_id'], follower=f_id, t10_frame=t10, fps=fps)
            found = None
            for fr in range(t10 - 1, t10 - int(a.lookback * fps) - 1, -1):
                if fr not in g.index or fr not in f.index:
                    break
                if int(g.loc[fr, 'laneId']) != old_lane or int(f.loc[fr, 'laneId']) != new_lane:
                    break
                sg = sign * (g.loc[fr, 'x'] + g.loc[fr, 'width'] / 2)
                sf = sign * (f.loc[fr, 'x'] + f.loc[fr, 'width'] / 2)
                if sg - sf <= -a.behind:
                    found = dict(frame=fr, centre_distance=float(sg - sf), v_gen=float(abs(g.loc[fr, 'xVelocity'])),
                                 v_fol=float(abs(f.loc[fr, 'xVelocity'])), lead_time_s=(t10 - fr) / fps)
                    break
            if found is None:
                row.update(included=False, reason='cut-in vehicle not behind the follower within the window')
                ledger.append(row)
                continue
            d = -found['centre_distance']  # follower ahead of the cut-in vehicle (centre to centre, m)
            gen = dict(x=2000.0, y=0.0, vx=found['v_gen'], vy=0.0, heading=0.0, length=5.0, width=2.0)
            fol = dict(x=2000.0 + d, y=4.0, vx=found['v_fol'], vy=0.0, heading=0.0, length=5.0, width=2.0)
            checks = dict(generator_speed=15. <= gen['vx'] <= 40., target_speed=15. <= fol['vx'] <= 35., no_overlap=d >= 5.0)
            row.update(found, checks=checks, included=all(checks.values()))
            ledger.append(row)
            if row['included']:
                case = {k: v for k, v in c.items() if k != 'physical_initial_state'}
                case['physical_initial_state'] = [gen, fol]
                case['initial_state_source'] = 'b79_from_behind'
                case['b79'] = dict(frame=found['frame'], lead_time_s=found['lead_time_s'], follower_ahead_m=d,
                                   stored_start=c['physical_initial_state'])
                out.append(case)
    manifest = dict(source=str(pool_path), source_sha256=sha(pool_path), highd_hashes=hashes, behind_m=a.behind,
                    lookback_s=a.lookback, partitions=sorted(parts), source_cases=len(dev), included=len(out),
                    rule='latest frame within lookback before the 10% crossing with the cut-in vehicle centre >= behind_m '
                         'behind the follower centre, both in their lanes')
    Path(a.out).write_text(json.dumps(dict(manifest=manifest, cases=out), indent=1), encoding='utf-8')
    Path(a.out).with_suffix('.ledger.json').write_text(json.dumps(ledger, indent=1), encoding='utf-8')
    inc = [r for r in ledger if r['included']]
    if inc:
        dv = np.array([r['v_gen'] - r['v_fol'] for r in inc])
        lt = np.array([r['lead_time_s'] for r in inc])
        print(json.dumps(dict(source_cases=len(dev), included=len(inc), dv_p10_50_90=np.percentile(dv, [10, 50, 90]).round(2).tolist(),
                              lead_time_p10_50_90=np.percentile(lt, [10, 50, 90]).round(2).tolist())))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('highd')
    ap.add_argument('out')
    ap.add_argument('--behind', type=float, default=6.)
    ap.add_argument('--lookback', type=float, default=15.)
    ap.add_argument('--partitions', default='development', help='comma-separated: development, calibration')
    main(ap.parse_args())
