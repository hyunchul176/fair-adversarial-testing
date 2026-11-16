"""MiTra cut-in extraction (B41 external dataset), the same event semantics as the highD, NGSIM and TGSIM extractors.

Source: MiTra (TU Dresden OPARA, DOI 10.25532/OPARA-881, CC BY 4.0) -- the A50 urban freeway in Milan, a 900 m
section filmed by six drones in nine flight campaigns T1..T9. Used: the stitched six-drone file T<k>_DAll.csv of each
campaign (complete tracks over the section), 30 frames/s, UTM positions (m), speed km/h, lanes 0-3 (mainline, one
direction) and 4-7 (the other direction); ramp lanes 10-11 / 20-21 are not cut-in lanes here.
Differences from TGSIM (tgsim.py), handled explicitly:
  * the campaign CSV is read from its archive Data_T<k>.zip; recording = campaign (T1..T9);
  * road frame per (campaign, direction): the principal axis of that direction's mainline positions gives s (along,
    oriented with the mean travel direction) and n (lateral, positive to the left of travel; UTM is right-handed);
  * lane centres per (campaign, lane): median n of the lane's samples in 50 m bins of s, linearly interpolated in s;
  * lateral position smoothed (Savitzky-Golay, 31 frames = 1 s, order 2); speed = Speed / 3.6;
  * vehicle type 'car' (case-insensitive) for generator and follower;
  * teleport check (the MiTra authors' own rule): an event is excluded when the generator or the follower moves more
    than 2 m along or 0.5 m across the road between consecutive frames within [p10 - 1 s, p90 + 1 s] (raw positions).
Event rule (as highD / NGSIM / TGSIM): adjacent-lane flip within one direction with 1 s stable labels either side;
lateral progress p10/p50/p90 crossings with the p50 crossing within 0.5 s of the flip; follower = nearest vehicle behind
in the new lane at the p50 frame, observed continuously in that lane from p10 to p90; speeds >= 15 m/s at p10 and p50;
positive bumper gap throughout. No simulation, no outcomes.
Cohort (select_cohort): per campaign, hash-ranked (salt below) participant-disjoint greedy, at most 20 per campaign.

  python -m fairtest.mitra MITRA_DIR EVENTS.jsonl          (MITRA_DIR holds Data_T1.zip .. Data_T9.zip)
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FPS = 30.
CAMPAIGNS = tuple(range(1, 10))
DIRS = {1: (0, 1, 2, 3), -1: (4, 5, 6, 7)}  # mainline lanes of each travel direction (Readme: left-to-right, right-to-left)
LANES = set(DIRS[1]) | set(DIRS[-1])
COL = dict(id='Vehicle_ID', kind='Vehicle_type', t='Time [s]', x='x [m]', y='y [m]', v='Speed [km/h]', lane='Lane',
           length='Vehicle_length [m]', width='Vehicle_width [m]')
CFG = dict(stable_s=1., search_s=6., tol_s=.5, min_speed=15., window=31, order=2, min_track_s=2., bin_m=50.,
           jump_long_m=2., jump_lat_m=.5, jump_pad_s=1.)
SALT = 'acarl-b41-mitra-v1:'
PER_RECORDING = 20


def direction(lane):
    return 1 if lane in DIRS[1] else (-1 if lane in DIRS[-1] else 0)


def member_of(z, campaign):
    name = f'T{campaign}_DAll.csv'
    hits = [m for m in z.namelist() if m.replace('\\', '/').split('/')[-1] == name]
    if len(hits) != 1:
        raise ValueError(f'{name}: {len(hits)} members')
    return hits[0]


def load(zip_path, campaign):
    with zipfile.ZipFile(zip_path) as z, z.open(member_of(z, campaign)) as f:
        df = pd.read_csv(f, usecols=list(COL.values()), encoding='utf-8-sig')
    df = df.rename(columns={v: k for k, v in COL.items()})
    df = df.dropna(subset=['id', 't', 'x', 'y', 'v', 'lane'])
    df['frame'] = np.round(df['t'].astype(float) * FPS).astype('int64')
    df['id'] = df['id'].astype('int64')
    df['lane'] = df['lane'].astype('int64')
    df['v'] = df['v'].astype(float) / 3.6
    return df.sort_values(['id', 'frame']).drop_duplicates(['id', 'frame']).reset_index(drop=True)


def frames(df):
    """Road frame per direction: origin, unit along-travel vector u, unit left normal w."""
    out = {}
    for sign, lanes in DIRS.items():
        g = df[df['lane'].isin(lanes)]
        if len(g) < 100:
            continue
        xy = g[['x', 'y']].to_numpy(float)
        c = xy.mean(axis=0)
        u = np.linalg.svd(xy - c, full_matrices=False)[2][0]
        d = [v[['x', 'y']].to_numpy(float)[[0, -1]] for _, v in g.groupby('id') if len(v) > 1]
        if d and float(np.sum([(p[1] - p[0]) @ u for p in d])) < 0:
            u = -u
        out[sign] = (c, u, np.array([-u[1], u[0]]))  # left of travel in right-handed UTM coordinates
    return out


def tracks_of(df, fr, campaign):
    tracks = {}
    for vid, g in df.groupby('id', sort=True):
        g = g.reset_index(drop=True)
        f = g['frame'].to_numpy()
        breaks = np.flatnonzero(np.diff(f) != 1) + 1
        seg = max(np.split(np.arange(len(g)), breaks), key=len)
        g = g.iloc[seg].reset_index(drop=True)
        if len(g) < max(CFG['window'], int(CFG['min_track_s'] * FPS)):
            continue
        lanes = g['lane'].to_numpy()
        signs = {direction(int(x)) for x in lanes if int(x) in LANES}
        if len(signs) != 1:
            continue
        sign = signs.pop()
        if sign not in fr:
            continue
        c, u, w = fr[sign]
        xy = g[['x', 'y']].to_numpy(float) - c
        n_raw = xy @ w
        tracks[(campaign, int(vid))] = dict(run=campaign, vid=int(vid), sign=sign, frame=g['frame'].to_numpy(),
                                            t=g['frame'].to_numpy() / FPS, s=xy @ u, n=savgol_filter(n_raw, CFG['window'], CFG['order']),
                                            n_raw=n_raw, v=g['v'].to_numpy(float), length=float(np.median(g['length'])),
                                            width=float(np.median(g['width'])), kind=str(g['kind'].mode().iloc[0]), lane=lanes)
    return tracks


def is_car(kind):
    return kind.strip().lower() == 'car'


def lane_centres(tracks):
    """{(campaign, lane): (bin centres s, median n)} from the lane's samples in CFG['bin_m'] bins of s."""
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


def jumps(tr, f_lo, f_hi):
    """Teleport check: consecutive-frame moves > 2 m along or > 0.5 m across (raw positions) within [f_lo, f_hi]."""
    sel = (tr['frame'] >= f_lo) & (tr['frame'] <= f_hi)
    s, n = tr['s'][sel], tr['n_raw'][sel]
    return bool(len(s) > 1 and (np.any(np.abs(np.diff(s)) > CFG['jump_long_m']) or np.any(np.abs(np.diff(n)) > CFG['jump_lat_m'])))


def extract(df, campaign):
    fr = frames(df)
    tracks = tracks_of(df, fr, campaign)
    centres = lane_centres(tracks)
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
        run, vid, lanes = tr['run'], tr['vid'], tr['lane']
        for k in np.flatnonzero(np.diff(lanes) != 0) + 1:
            old, new = int(lanes[k - 1]), int(lanes[k])
            flip = int(tr['frame'][k])
            row = dict(event_id=f'mitra-T{run}:{vid}:{flip}', partition='mitra', recording=f'T{run}', vehicle=vid,
                       flip_frame=flip, old_lane=old, new_lane=new, acc_generator=None)
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
            if not (is_car(tr['kind']) and is_car(fo['kind'])):
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
    record = dict(dataset=f'T{campaign}', rows=len(df), vehicles=len(tracks), directions=sorted(fr),
                  lane_centre_keys=len(centres), candidates=len(events), eligible=sum(1 for e in events if e.get('eligible')),
                  vehicle_types=sorted({str(x) for x in df['kind'].unique()}))
    return events, record


def main(data_dir, out):
    data_dir, out = Path(data_dir), Path(out)
    all_events, records = [], []
    for k in CAMPAIGNS:
        path = data_dir / f'Data_T{k}.zip'
        ev, rec = extract(load(path, k), k)
        rec['archive_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        all_events += ev
        records.append(rec)
        print(json.dumps(rec), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for e in all_events:
            f.write(json.dumps(e) + '\n')
    return all_events, records


def select_cohort(events):
    """Extractor- and preflight-eligible events; per campaign, hash-ranked participant-disjoint greedy, at most
    PER_RECORDING (the B13/B28 rule with the B41 salt and a per-campaign cap)."""
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
        used, taken = set(), 0
        for row in sorted(buckets[rec], key=lambda r: (r['rank_hash'], r['event_id'])):
            parts = {row['vehicle'], row['follower']}
            if parts & used:
                row['reason'] = 'participant_reuse'
                continue
            if taken >= PER_RECORDING:
                row['reason'] = 'recording_cap'
                continue
            used |= parts
            taken += 1
            row['selected'] = True
            cases.append(dict(case_id=len(cases), event=row['_meta'], preflight=row['_check']['checks'],
                              physical_initial_state=row['_check']['physical_initial_state'],
                              initial_state_source='local_preflight', follower=row['follower'], dev16=False))
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in ledger]
    return cases, clean


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
