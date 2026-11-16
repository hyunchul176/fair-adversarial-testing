"""HIGH-SIM cut-in extraction (B70: substitute prospective evaluation of the final agent v2s), the same event semantics as
the highD, NGSIM, TGSIM and MiTra extractors.

Source: HIGH-SIM (Shi, Zhao, Yao and Li 2021; CATS Lab, CC BY-SA 4.0) -- Interstate 75 in Florida, about 2.45 km of one
travel direction filmed by three helicopters and stitched; 38 minutes at 30 frames/s, 2,182 vehicles. One CSV without a
header line; columns as in the authors' data description (2026 version): 0 vehicle ID, 1 global time (s; stored at
float32 precision), 2 frame ID, 3 local X (ft, across the road), 4 local Y (ft, along the road, increasing with travel),
5-6 GPS, 7 width (ft), 8 length (ft), 9 class (2 for every vehicle in this file, trucks included), 10 speed (ft/s, 2 s
moving average), 11 acceleration, 12 lane (0 = ramp; 1, 2, 3 = mainline, 12 ft wide), 13 preceding ID, 14 space headway.
The file-level facts used here were read before the freeze from aggregates only (B70 data-access record, available on request).
Differences from MiTra (mitra.py), handled explicitly:
  * time = frame / 30 (the global time column is quantised to 1/128 s and not used);
  * one travel direction; road frame s = local Y, n = local X, in metres; no left/right sign is assumed (the lateral
    progress is measured between the two lane centres);
  * lane centres per mainline lane: median n of the lane's samples in 50 m bins of s (as MiTra), interpolated in s;
  * lateral position smoothed (Savitzky-Golay, 31 frames = 1 s, order 2); speed = column 10 * 0.3048;
  * passenger car = length < 7.0 m (the class column cannot tell cars from trucks here);
  * teleport check as MiTra (> 2 m along or > 0.5 m across between consecutive frames within [p10 - 1 s, p90 + 1 s],
    raw positions); it also excludes events at the joins of the helicopter views if the stitching jumps there;
  * recording = the 5-minute block of the flip frame (cluster intervals only; the selection ignores it).
Event rule (as highD / NGSIM / TGSIM / MiTra): adjacent mainline lane flip with 1 s stable labels either side; lateral
progress p10/p50/p90 crossings with the p50 crossing within 0.5 s of the flip; follower = nearest vehicle behind in the
new lane at the p50 frame, observed continuously in that lane from p10 to p90; speeds >= 15 m/s at p10 and p50; positive
bumper gap throughout. Merges from the ramp (lane 0) are not cut-ins here (as MiTra). No simulation, no outcomes.
Cohort (select_cohort): one hash-ranked (salt below) participant-disjoint greedy pass over all eligible events, at most
MAX_CASES = 60 cases (a compute bound fixed before the extraction: three refinements x 17 SUTs; the development set has 62).

  python -m fairtest.highsim HIGH_SIM_full.csv EVENTS.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 30.
FT = .3048
MAINLINE = (1, 2, 3)
COLS = ['vid', 'gtime', 'frame', 'x_ft', 'y_ft', 'gps5', 'gps6', 'width_ft', 'length_ft', 'cls', 'v_fts', 'a_fts2', 'lane',
        'preceding', 'headway_ft']
CFG = dict(stable_s=1., search_s=6., tol_s=.5, min_speed=15., window=31, order=2, min_track_s=2., bin_m=50.,
           jump_long_m=2., jump_lat_m=.5, jump_pad_s=1., car_max_length_m=7., block_s=300.)
SALT = 'acarl-b70-highsim-v2s:'
MAX_CASES = 60


def load(path):
    df = pd.read_csv(path, header=None, names=COLS, usecols=['vid', 'frame', 'x_ft', 'y_ft', 'width_ft', 'length_ft', 'v_fts', 'lane'],
                     skipinitialspace=True)
    df = df.dropna()
    df['vid'] = df['vid'].astype('int64')
    df['frame'] = df['frame'].astype('int64')
    df['lane'] = df['lane'].astype('int64')
    df['s'] = df['y_ft'].astype(float) * FT
    df['n'] = df['x_ft'].astype(float) * FT
    df['v'] = df['v_fts'].astype(float) * FT
    df['length'] = df['length_ft'].astype(float) * FT
    df['width'] = df['width_ft'].astype(float) * FT
    return df.sort_values(['vid', 'frame']).drop_duplicates(['vid', 'frame']).reset_index(drop=True)


def tracks_of(df):
    tracks = {}
    for vid, g in df.groupby('vid', sort=True):
        g = g.reset_index(drop=True)
        f = g['frame'].to_numpy()
        breaks = np.flatnonzero(np.diff(f) != 1) + 1
        seg = max(np.split(np.arange(len(g)), breaks), key=len)
        g = g.iloc[seg].reset_index(drop=True)
        if len(g) < max(CFG['window'], int(CFG['min_track_s'] * FPS)):
            continue
        n_raw = g['n'].to_numpy(float)
        tracks[int(vid)] = dict(vid=int(vid), frame=g['frame'].to_numpy(), t=g['frame'].to_numpy() / FPS, s=g['s'].to_numpy(float),
                                n=savgol_filter(n_raw, CFG['window'], CFG['order']), n_raw=n_raw, v=g['v'].to_numpy(float),
                                length=float(np.median(g['length'])), width=float(np.median(g['width'])),
                                lane=g['lane'].to_numpy())
    return tracks


def is_car(tr):
    return tr['length'] < CFG['car_max_length_m']


def lane_centres(tracks):
    """{lane: (bin centres s, median n)} from the mainline lane's samples in CFG['bin_m'] bins of s."""
    acc = {}
    for tr in tracks.values():
        for lane in set(int(x) for x in tr['lane']) & set(MAINLINE):
            m = tr['lane'] == lane
            acc.setdefault(lane, []).append((tr['s'][m], tr['n'][m]))
    out = {}
    for lane, parts in acc.items():
        s = np.concatenate([p[0] for p in parts])
        n = np.concatenate([p[1] for p in parts])
        edges = np.arange(np.floor(s.min() / CFG['bin_m']), np.ceil(s.max() / CFG['bin_m']) + 1) * CFG['bin_m']
        mids, meds = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (s >= a) & (s < b)
            if m.sum() >= 20:
                mids.append((a + b) / 2)
                meds.append(float(np.median(n[m])))
        if mids:
            out[lane] = (np.asarray(mids), np.asarray(meds))
    return out


def centre_at(centres, lane, s):
    mids, meds = centres[lane]
    return np.interp(s, mids, meds)


def crossings(t, p, level):
    idx = np.flatnonzero((p[:-1] < level) & (p[1:] >= level))
    return [float(t[i] + (t[i + 1] - t[i]) * (level - p[i]) / (p[i + 1] - p[i])) for i in idx]


def at(tr, time, key):
    return float(np.interp(time, tr['t'], tr[key]))


def jumps(tr, f_lo, f_hi):
    """Teleport check: consecutive-frame moves > 2 m along or > 0.5 m across (raw positions) within [f_lo, f_hi]."""
    sel = (tr['frame'] >= f_lo) & (tr['frame'] <= f_hi)
    s, n = tr['s'][sel], tr['n_raw'][sel]
    return bool(len(s) > 1 and (np.any(np.abs(np.diff(s)) > CFG['jump_long_m']) or np.any(np.abs(np.diff(n)) > CFG['jump_lat_m'])))


def extract(df):
    tracks = tracks_of(df)
    centres = lane_centres(tracks)
    frame0 = int(df['frame'].min())
    keys = list(tracks)
    all_f = np.concatenate([tracks[k]['frame'] for k in keys]) if keys else np.zeros(0, int)
    kid = np.concatenate([np.full(len(tracks[k]['frame']), j) for j, k in enumerate(keys)]) if keys else np.zeros(0, int)
    pos = np.concatenate([np.arange(len(tracks[k]['frame'])) for k in keys]) if keys else np.zeros(0, int)
    order = np.argsort(all_f, kind='stable')
    all_f, kid, pos = all_f[order], kid[order], pos[order]

    def at_frame(f):
        lo, hi = np.searchsorted(all_f, f, 'left'), np.searchsorted(all_f, f, 'right')
        return [(keys[kid[j]], int(pos[j])) for j in range(lo, hi)]

    events = []
    st = int(CFG['stable_s'] * FPS)
    pad = int(CFG['jump_pad_s'] * FPS)
    for key, tr in tracks.items():
        vid, lanes = tr['vid'], tr['lane']
        for k in np.flatnonzero(np.diff(lanes) != 0) + 1:
            old, new = int(lanes[k - 1]), int(lanes[k])
            flip = int(tr['frame'][k])
            block = int((flip - frame0) // int(CFG['block_s'] * FPS))
            row = dict(event_id=f'highsim-i75:{vid}:{flip}', partition='highsim', recording=f'i75-b{block:02d}', vehicle=vid,
                       flip_frame=flip, old_lane=old, new_lane=new, acc_generator=None)
            events.append(row)
            if old not in MAINLINE or new not in MAINLINE or abs(new - old) != 1:
                row.update(eligible=False, exclusion='nonadjacent_or_ramp_lane'); continue
            if old not in centres or new not in centres:
                row.update(eligible=False, exclusion='no_lane_centre'); continue
            if k < st or k + st > len(lanes) or not (np.all(lanes[k - st:k] == old) and np.all(lanes[k:k + st] == new)):
                row.update(eligible=False, exclusion='unstable_lane_labels'); continue
            lo = max(0, k - int(CFG['search_s'] * FPS))
            hi = min(len(lanes), k + int(CFG['search_s'] * FPS) + 1)
            t, s = tr['t'][lo:hi], tr['s'][lo:hi]
            c_old, c_new = centre_at(centres, old, s), centre_at(centres, new, s)
            p = (tr['n'][lo:hi] - c_old) / (c_new - c_old)
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
            g_rear = at(tr, mid, 's') - tr['length'] / 2
            cands = []
            for okey, i in at_frame(f50):
                if okey == key:
                    continue
                o = tracks[okey]
                if int(o['lane'][i]) != new:
                    continue
                front = float(o['s'][i]) + o['length'] / 2
                if front < g_rear:  # follower's front behind the generator's rear
                    cands.append((g_rear - front, okey))
            if not cands:
                row.update(eligible=False, exclusion='no_observed_destination_follower'); continue
            fkey = min(cands)[1]
            fo = tracks[fkey]
            row['follower'] = fo['vid']
            row['acc_follower'] = None
            if not (is_car(tr) and is_car(fo)):
                row.update(eligible=False, exclusion='non_car_pair'); continue
            f10, f90 = int(np.floor(times[0] * FPS)), int(np.ceil(times[2] * FPS))
            sel = (fo['frame'] >= f10) & (fo['frame'] <= f90)
            if sel.sum() != f90 - f10 + 1:
                row.update(eligible=False, exclusion='truncated_follower'); continue
            if not np.all(fo['lane'][sel] == new):
                row.update(eligible=False, exclusion='follower_changes_lane'); continue
            if jumps(tr, f10 - pad, f90 + pad) or jumps(fo, f10 - pad, f90 + pad):
                row.update(eligible=False, exclusion='teleport_jump'); continue
            metrics = dict(duration_10_90=float(times[2] - times[0]))
            ok = True
            for name, tm in (('p10', times[0]), ('p50', times[1])):
                gv, tv = at(tr, tm, 'v'), at(fo, tm, 'v')
                gap = at(tr, tm, 's') - tr['length'] / 2 - (at(fo, tm, 's') + fo['length'] / 2)
                metrics[name] = dict(generator_speed=gv, target_speed=tv, gap=gap,
                                     headway=gap / tv if tv > 0 else None, closing_speed=tv - gv)
                ok &= gv >= CFG['min_speed'] and tv >= CFG['min_speed']
            if not ok:
                row.update(eligible=False, exclusion='below_speed_population', metrics=metrics); continue
            ts = fo['t'][sel]
            gaps = np.interp(ts, tr['t'], tr['s']) - tr['length'] / 2 - (fo['s'][sel] + fo['length'] / 2)
            if np.min(gaps) <= 0:
                row.update(eligible=False, exclusion='nonpositive_longitudinal_gap'); continue
            row.update(eligible=True, exclusion=None, metrics=metrics)
    record = dict(dataset='i75', rows=len(df), vehicles=len(tracks), frames=[int(df['frame'].min()), int(df['frame'].max())],
                  lane_centre_keys=sorted(centres), candidates=len(events), eligible=sum(1 for e in events if e.get('eligible')))
    return events, record


def select_cohort(events):
    """Extractor- and preflight-eligible events; one hash-ranked participant-disjoint greedy pass, at most MAX_CASES."""
    from fairtest import devboot  # noqa: F401
    from scripts.prepare_adversarial_b3_transfer import preflight, sanitize_event
    pool, ledger = [], []
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
        pool.append(row)
    cases, used = [], set()
    for row in sorted(pool, key=lambda r: (r['rank_hash'], r['event_id'])):
        parts = {row['vehicle'], row['follower']}
        if parts & used:
            row['reason'] = 'participant_reuse'
            continue
        if len(cases) >= MAX_CASES:
            row['reason'] = 'cohort_cap'
            continue
        used |= parts
        row['selected'] = True
        cases.append(dict(case_id=len(cases), event=row['_meta'], preflight=row['_check']['checks'],
                          physical_initial_state=row['_check']['physical_initial_state'],
                          initial_state_source='local_preflight', follower=row['follower'], dev16=False))
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in ledger]
    return cases, clean


def main(path, out):
    path, out = Path(path), Path(out)
    events, record = extract(load(path))
    record['csv_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(record), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for e in events:
            f.write(json.dumps(e) + '\n')
    return events, record


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
