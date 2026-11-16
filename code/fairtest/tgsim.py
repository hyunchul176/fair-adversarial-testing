"""TGSIM I-294 cut-in extraction (B28 external dataset), same event semantics as the highD and NGSIM extractors.

Source: FHWA TGSIM I-294 L1 and L2 trajectories (data.transportation.gov 7zjf-a4zf, tpsq-zrwa), 10 Hz, positions in
metres in the per-run reference image (helicopter video), lanes -5..-2 (southbound) and 2..5 (northbound).
Differences from highD/NGSIM, handled explicitly:
  * the export writes every value quoted with thousands separators ("1,778.595"): parsed as numbers;
  * vehicles are keyed (dataset, run, id); a track is split at time gaps and only its longest continuous segment
    (>= 2 s) is kept;
  * road frame per (dataset, run, direction): the principal axis of all positions of that direction's vehicles
    gives s (along, oriented with the mean travel direction) and n (lateral, positive to the left of travel);
  * lane centres: for each (dataset, run, lane) the median n of the lane's samples in 50 m bins of s, linearly
    interpolated in s (curvature over the 4.8 km section; the official centre-line files are not used);
  * positions are the bounding-box centre; bumper gap = s_gen - l_gen/2 - (s_fol + l_fol/2);
  * speed = speed_kf; lateral position smoothed (Savitzky-Golay, 11 frames = 1 s, order 2);
  * vehicle type 'small-vehicle' for both generator and follower (the ACC flag is recorded, not filtered).
Event rule (as highD/NGSIM): adjacent-lane flip within one direction with 1 s stable labels either side; lateral
progress p10/p50/p90 crossings with the p50 crossing within 0.5 s of the flip; follower = nearest vehicle behind in
the new lane at the p50 frame, observed continuously in that lane from p10 to p90; speeds >= 15 m/s at p10 and p50;
positive bumper gap throughout. No simulation, no outcomes.

  python -m fairtest.tgsim TGSIM_DIR EVENTS.jsonl        (TGSIM_DIR holds I294_L1_final.csv, I294_L2_final.csv)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 10.
FILES = (('L1', 'I294_L1_final.csv'), ('L2', 'I294_L2_final.csv'))
LANES = {-5, -4, -3, -2, 2, 3, 4, 5}
CFG = dict(stable_s=1., search_s=6., tol_s=.5, min_speed=15., window=11, order=2, min_track_s=2., bin_m=50.)
SALT = 'acarl-b28-tgsim-v1:'


def _num(col):
    return pd.to_numeric(col.astype(str).str.replace(',', '', regex=False), errors='coerce')


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ('id', 'time', 'xloc_kf', 'yloc_kf', 'lane_kf', 'speed_kf', 'acceleration_kf', 'length_smoothed',
              'width_smoothed', 'run_index'):
        df[c] = _num(df[c])
    df['frame'] = np.round(df['time'] * FPS).astype('int64')
    df = df.dropna(subset=['id', 'time', 'xloc_kf', 'yloc_kf', 'lane_kf', 'speed_kf', 'run_index'])
    df['id'] = df['id'].astype('int64')
    df['run_index'] = df['run_index'].astype('int64')
    df['lane_kf'] = df['lane_kf'].astype('int64')
    return df.sort_values(['run_index', 'id', 'frame']).drop_duplicates(['run_index', 'id', 'frame'])


def direction(lane):
    return 1 if lane > 0 else -1  # northbound lanes 2..5, southbound -5..-2


def frames(df):
    """Road frame per (run, direction): origin, unit along-travel vector u, unit left normal w."""
    out = {}
    for (run, sign), g in df[df['lane_kf'].isin(LANES)].assign(sign=lambda d: np.sign(d['lane_kf'])).groupby(['run_index', 'sign']):
        xy = g[['xloc_kf', 'yloc_kf']].to_numpy(float)
        c = xy.mean(axis=0)
        u = np.linalg.svd(xy - c, full_matrices=False)[2][0]
        # orient u with the travel direction: successive positions of each vehicle
        d = []
        for _, v in g.groupby('id'):
            p = v[['xloc_kf', 'yloc_kf']].to_numpy(float)
            if len(p) > 1:
                d.append(p[-1] - p[0])
        if d and float(np.sum(np.asarray(d) @ u)) < 0:
            u = -u
        w = np.array([-u[1], u[0]])  # left of travel in a right-handed frame; image y points down, so flip
        w = -w  # image coordinates (origin top-left, y down) are left-handed
        out[(int(run), int(sign))] = (c, u, w)
    return out


def tracks_of(df, fr):
    tracks = {}
    for (run, vid), g in df.groupby(['run_index', 'id'], sort=True):
        g = g.reset_index(drop=True)
        f = g['frame'].to_numpy()
        breaks = np.flatnonzero(np.diff(f) != 1) + 1
        seg = max(np.split(np.arange(len(g)), breaks), key=len)
        g = g.iloc[seg].reset_index(drop=True)
        if len(g) < max(CFG['window'], int(CFG['min_track_s'] * FPS)):
            continue
        lanes = g['lane_kf'].to_numpy()
        signs = {direction(int(x)) for x in lanes if int(x) in LANES}
        if len(signs) != 1:
            continue
        sign = signs.pop()
        c, u, w = fr[(int(run), sign)]
        xy = g[['xloc_kf', 'yloc_kf']].to_numpy(float) - c
        s = xy @ u
        n = savgol_filter(xy @ w, CFG['window'], CFG['order'])
        tracks[(int(run), int(vid))] = dict(run=int(run), vid=int(vid), sign=sign, frame=g['frame'].to_numpy(),
                                            t=g['frame'].to_numpy() / FPS, s=s, n=n, v=g['speed_kf'].to_numpy(float),
                                            length=float(np.median(g['length_smoothed'])),
                                            width=float(np.median(g['width_smoothed'])),
                                            kind=str(g['type_most_common'].mode().iloc[0]),
                                            acc=bool((g['acc'].str.lower() == 'yes').any()), lane=lanes)
    return tracks


def lane_centres(tracks):
    """{(run, lane): (bin centres s, median n)} from the lane's samples in CFG['bin_m'] bins of s."""
    acc = {}
    for tr in tracks.values():
        for lane in set(int(x) for x in tr['lane']) & LANES:
            m = tr['lane'] == lane
            acc.setdefault((tr['run'], lane), []).append((tr['s'][m], tr['n'][m]))
    out = {}
    for key, parts in acc.items():
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
            out[key] = (np.asarray(mids), np.asarray(meds))
    return out


def centre_at(centres, run, lane, s):
    mids, meds = centres[(run, lane)]
    return np.interp(s, mids, meds)


def crossings(t, p, level):
    idx = np.flatnonzero((p[:-1] < level) & (p[1:] >= level))
    return [float(t[i] + (t[i + 1] - t[i]) * (level - p[i]) / (p[i + 1] - p[i])) for i in idx]


def at(tr, time, key):
    return float(np.interp(time, tr['t'], tr[key]))


def extract(df, dataset):
    fr = frames(df)
    tracks = tracks_of(df, fr)
    centres = lane_centres(tracks)
    by_frame = {}
    for key, tr in tracks.items():
        for i, f in enumerate(tr['frame']):
            by_frame.setdefault((tr['run'], int(f)), []).append((key, i))
    events = []
    st = int(CFG['stable_s'] * FPS)
    for key, tr in tracks.items():
        run, vid, lanes = tr['run'], tr['vid'], tr['lane']
        for k in np.flatnonzero(np.diff(lanes) != 0) + 1:
            old, new = int(lanes[k - 1]), int(lanes[k])
            flip = int(tr['frame'][k])
            row = dict(event_id=f'tgsim-i294-{dataset}-r{run}:{vid}:{flip}', partition='tgsim_i294', recording=f'{dataset}-r{run}',
                       vehicle=vid, flip_frame=flip, old_lane=old, new_lane=new, acc_generator=tr['acc'])
            events.append(row)
            if old not in LANES or new not in LANES or direction(old) != direction(new) or abs(new - old) != 1:
                row.update(eligible=False, exclusion='nonadjacent_or_ramp_lane'); continue
            if (run, old) not in centres or (run, new) not in centres:
                row.update(eligible=False, exclusion='no_lane_centre'); continue
            if k < st or k + st > len(lanes) or not (np.all(lanes[k - st:k] == old) and np.all(lanes[k:k + st] == new)):
                row.update(eligible=False, exclusion='unstable_lane_labels'); continue
            lo = max(0, k - int(CFG['search_s'] * FPS))
            hi = min(len(lanes), k + int(CFG['search_s'] * FPS) + 1)
            t, s = tr['t'][lo:hi], tr['s'][lo:hi]
            c_old, c_new = centre_at(centres, run, old, s), centre_at(centres, run, new, s)
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
            for okey, i in by_frame.get((run, f50), []):
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
            row['acc_follower'] = fo['acc']
            if tr['kind'] != 'small-vehicle' or fo['kind'] != 'small-vehicle':
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
    record = dict(dataset=dataset, rows=len(df), vehicles=len(tracks), runs=sorted({k[0] for k in tracks}),
                  lane_centre_keys=len(centres), candidates=len(events), eligible=sum(1 for e in events if e.get('eligible')))
    return events, record


def main(data_dir, out):
    data_dir, out = Path(data_dir), Path(out)
    all_events, records = [], []
    for dataset, name in FILES:
        path = data_dir / name
        ev, rec = extract(load(path), dataset)
        rec['file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        all_events += ev
        records.append(rec)
        print(json.dumps(rec), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for e in all_events:
            f.write(json.dumps(e) + '\n')
    return all_events, records


def select_cohort(events):
    """Extractor- and preflight-eligible events; per recording (dataset-run), hash-ranked participant-disjoint greedy
    (the B13 rule with the B28 salt)."""
    from fairtest import devboot  # noqa: F401
    from scripts.prepare_adversarial_b3_transfer import preflight, sanitize_event
    buckets, ledger = {}, []
    for e in events:
        row = dict(event_id=e['event_id'], recording=e['recording'], vehicle=e['vehicle'], follower=e.get('follower'),
                   extractor_eligible=bool(e.get('eligible')), exclusion=e.get('exclusion'), selected=False, reason=None,
                   acc_generator=e.get('acc_generator'), acc_follower=e.get('acc_follower'),
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
                              initial_state_source='local_preflight', follower=row['follower'], dev16=False,
                              acc_follower=row['acc_follower']))
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in ledger]
    return cases, clean


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
