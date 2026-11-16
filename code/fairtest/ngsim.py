"""NGSIM US-101 cut-in extraction (B13 external dataset), same event semantics as the highD extractor.

Differences handled explicitly:
  * units: feet -> metres; 10 Hz frames;
  * NGSIM Local_Y is the FRONT bumper longitudinal position, Local_X the lateral centre position;
  * noise: Savitzky-Golay smoothing (window 21 frames = 2 s, order 2) of lateral and longitudinal
    position; speed = smoothed derivative of smoothed longitudinal position;
  * lanes: mainline 1-5 and auxiliary 6 are driving lanes; ramps 7/8 excluded; lane centres are
    the median smoothed lateral position per lane over the period;
  * vehicle class 2 (automobile) for both generator and follower.
Event rule (as highD): adjacent-lane flip with 1 s stable labels either side; lateral progress
p10/p50/p90 crossings with the p50 crossing within 0.5 s of the flip; follower = nearest vehicle
behind in the new lane at the p50 frame, observed continuously in that lane from p10 to p90;
speeds >= 15 m/s at p10 and p50; positive bumper gap throughout. No simulation, no outcomes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FT = .3048
FPS = 10.
COLS = ['id', 'frame', 'total', 'gtime', 'lx', 'ly', 'gx', 'gy', 'length', 'width', 'cls', 'vel', 'acc',
        'lane', 'pre', 'fol', 'shw', 'thw']
DRIVING = {1, 2, 3, 4, 5, 6}
CFG = dict(stable_s=1., search_s=6., tol_s=.5, min_speed=15., window=21, order=2)


def load(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLS, engine='c')
    df = df.sort_values(['id', 'frame']).drop_duplicates(['id', 'frame'])
    return df


def tracks_of(df):
    tracks = {}
    for vid, g in df.groupby('id', sort=True):
        g = g.reset_index(drop=True)
        if len(g) < CFG['window'] or not np.all(np.diff(g.frame) == 1):
            # split discontinuous tracks conservatively: keep the longest continuous segment
            breaks = np.flatnonzero(np.diff(g.frame.to_numpy()) != 1) + 1
            segs = np.split(np.arange(len(g)), breaks)
            seg = max(segs, key=len)
            g = g.iloc[seg].reset_index(drop=True)
            if len(g) < CFG['window']:
                continue
        y = savgol_filter(g.lx.to_numpy() * FT, CFG['window'], CFG['order'])
        x = savgol_filter(g.ly.to_numpy() * FT, CFG['window'], CFG['order'])
        v = savgol_filter(np.gradient(x, 1 / FPS), CFG['window'], CFG['order'])
        tracks[int(vid)] = dict(frame=g.frame.to_numpy(), t=g.frame.to_numpy() / FPS, x=x, y=y, v=v,
                                length=float(np.median(g.length)) * FT, width=float(np.median(g.width)) * FT,
                                cls=int(np.median(g.cls)), lane=g.lane.to_numpy())
    return tracks


def lane_centres(tracks):
    ys = {}
    for tr in tracks.values():
        for lane in DRIVING:
            m = tr['lane'] == lane
            if m.any():
                ys.setdefault(lane, []).append(tr['y'][m])
    return {lane: float(np.median(np.concatenate(v))) for lane, v in ys.items()}


def crossings(t, p, level):
    idx = np.flatnonzero((p[:-1] < level) & (p[1:] >= level))
    return [float(t[i] + (t[i + 1] - t[i]) * (level - p[i]) / (p[i + 1] - p[i])) for i in idx]


def at(tr, time, key):
    return float(np.interp(time, tr['t'], tr[key]))


def extract_period(path, period):
    df = load(path)
    tracks = tracks_of(df)
    centres = lane_centres(tracks)
    by_frame = {}
    for vid, tr in tracks.items():
        for i, f in enumerate(tr['frame']):
            by_frame.setdefault(int(f), []).append((vid, i))
    events = []
    st = int(CFG['stable_s'] * FPS)
    for vid, tr in tracks.items():
        lanes = tr['lane']
        for k in np.flatnonzero(np.diff(lanes) != 0) + 1:
            old, new = int(lanes[k - 1]), int(lanes[k])
            flip = int(tr['frame'][k])
            row = dict(event_id=f'ngsim-us101-{period}:{vid}:{flip}', partition='ngsim_us101', recording=period,
                       vehicle=vid, flip_frame=flip, old_lane=old, new_lane=new)
            events.append(row)
            if old not in DRIVING or new not in DRIVING or abs(new - old) != 1:
                row.update(eligible=False, exclusion='nonadjacent_or_ramp_lane'); continue
            if k < st or k + st > len(lanes) or not (np.all(lanes[k - st:k] == old) and np.all(lanes[k:k + st] == new)):
                row.update(eligible=False, exclusion='unstable_lane_labels'); continue
            lo = max(0, k - int(CFG['search_s'] * FPS))
            hi = min(len(lanes), k + int(CFG['search_s'] * FPS) + 1)
            t = tr['t'][lo:hi]
            p = (tr['y'][lo:hi] - centres[old]) / (centres[new] - centres[old])
            mids = crossings(t, p, .5)
            if not mids:
                row.update(eligible=False, exclusion='no_50pct_crossing'); continue
            mid = min(mids, key=lambda x: abs(x - flip / FPS))
            if abs(mid - flip / FPS) > CFG['tol_s']:
                row.update(eligible=False, exclusion='50pct_crossing_far_from_flip'); continue
            starts = [x for x in crossings(t, p, .1) if x < mid]
            ends = [x for x in crossings(t, p, .9) if x > mid]
            if not starts or not ends:
                row.update(eligible=False, exclusion='unobserved_10_90_crossing'); continue
            times = [max(starts), mid, min(ends)]
            row['times'] = times
            f50 = int(np.ceil(mid * FPS - 1e-8))
            gx = at(tr, mid, 'x')
            cands = []
            for ovid, i in by_frame.get(f50, []):
                if ovid == vid:
                    continue
                o = tracks[ovid]
                if int(o['lane'][i]) != new:
                    continue
                front = float(o['x'][i])
                if front < gx - tr['length']:  # follower's front behind the generator's rear
                    cands.append((gx - tr['length'] - front, ovid))
            if not cands:
                row.update(eligible=False, exclusion='no_observed_destination_follower'); continue
            fol = min(cands)[1]
            row['follower'] = fol
            fo = tracks[fol]
            if tr['cls'] != 2 or fo['cls'] != 2:
                row.update(eligible=False, exclusion='non_car_pair'); continue
            f10, f90 = int(np.floor(times[0] * FPS)), int(np.ceil(times[2] * FPS))
            sel = (fo['frame'] >= f10) & (fo['frame'] <= f90)
            if sel.sum() != f90 - f10 + 1:
                row.update(eligible=False, exclusion='truncated_follower'); continue
            if not np.all(fo['lane'][sel] == new):
                row.update(eligible=False, exclusion='follower_changes_lane'); continue
            metrics = dict(duration_10_90=float(times[2] - times[0]))
            ok = True
            for name, tm in (('p10', times[0]), ('p50', times[1])):
                gv, tv = at(tr, tm, 'v'), at(fo, tm, 'v')
                gap = at(tr, tm, 'x') - tr['length'] - at(fo, tm, 'x')
                metrics[name] = dict(generator_speed=gv, target_speed=tv, gap=gap,
                                     headway=gap / tv if tv > 0 else None, closing_speed=tv - gv)
                ok &= gv >= CFG['min_speed'] and tv >= CFG['min_speed']
            if not ok:
                row.update(eligible=False, exclusion='below_speed_population', metrics=metrics); continue
            ts = fo['t'][sel]
            gaps = np.interp(ts, tr['t'], tr['x']) - tr['length'] - fo['x'][sel]
            if np.min(gaps) <= 0:
                row.update(eligible=False, exclusion='nonpositive_longitudinal_gap'); continue
            row.update(eligible=True, exclusion=None, metrics=metrics)
    return events, dict(period=period, rows=len(df), vehicles=len(tracks), centres=centres,
                        candidates=len(events), eligible=sum(1 for e in events if e.get('eligible')))


def main(data_dir, out):
    data_dir, out = Path(data_dir), Path(out)
    periods = [('0750am-0805am', 1), ('0805am-0820am', 2), ('0820am-0835am', 3)]
    all_events, records = [], []
    for name, pid in periods:
        path = data_dir / name / f'trajectories-{name}.txt'
        ev, rec = extract_period(path, pid)
        rec['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        all_events += ev
        records.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != 'centres'}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for e in all_events:
            f.write(json.dumps(e) + '\n')
    return all_events, records


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


SALT = 'acarl-b13-ngsim-v1:'


def select_cohort(events):
    """Extractor- and preflight-eligible events; per period, hash-ranked participant-disjoint greedy."""
    from fairtest import devboot  # noqa: F401
    from scripts.prepare_adversarial_b3_transfer import preflight, sanitize_event
    buckets, ledger = {}, []
    for e in events:
        row = dict(event_id=e['event_id'], recording=e['recording'], vehicle=e['vehicle'], follower=e.get('follower'),
                   extractor_eligible=bool(e.get('eligible')), exclusion=e.get('exclusion'), selected=False, reason=None,
                   rank_hash=hashlib.sha256((SALT + e['event_id']).encode('utf-8')).hexdigest())
        ledger.append(row)
        if not e.get('eligible'):
            row['reason'] = 'extractor_ineligible'
            continue
        meta = sanitize_event(e)
        check = preflight(meta)
        if not check['eligible']:
            row['reason'] = 'initial_domain:' + ','.join(check['reasons'])
            continue
        row['_meta'], row['_check'] = meta, check
        buckets.setdefault(e['recording'], []).append(row)
    cases = []
    for rec in sorted(buckets):
        used = set()
        for row in sorted(buckets[rec], key=lambda r: (r['rank_hash'], r['event_id'])):
            parts = {row['vehicle'], row['follower']}
            if parts & used:
                row['reason'] = 'participant_reuse'
                continue
            used |= parts
            row['selected'] = True
            cases.append(dict(case_id=len(cases), event=row['_meta'], preflight=row['_check']['checks'],
                              physical_initial_state=row['_check']['physical_initial_state'],
                              initial_state_source='local_preflight', follower=row['follower'], dev16=False))
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in ledger]
    return cases, clean
