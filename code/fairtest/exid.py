"""exiD cut-in extraction (B15 external dataset, exiD protocol draft available on request): the event semantics of the highD extractor
(scripts/analyze_highd_cutin_joint_support.py), with the lanes taken from the Lanelet2 maps.

Source: exiD (levelXData, 'Highway Dataset Format' 2.1): per recording XX_recordingMeta.csv, XX_tracksMeta.csv and
XX_tracks.csv with the map-based enrichment columns; one Lanelet2 map (.osm) per location. xCenter, yCenter = bounding-box
centre in the local UTM-like frame (m); frameRate from recordingMeta; no smoothing (as highD: the same provider's tracks).
exiD has no lane id. Lanes and lateral progress, handled explicitly:
  * map: every lanelet relation has a left and a right bound way; lanelets tagged lxd_virtual are left out (as by the
    provider). Each bound is put in the driving direction (the left bound lies to the left of it). Lanelet B continues
    lanelet A when the ends of both of A's bounds meet the starts of B's bounds (within 0.1 m). A lane runs through
    one-to-one continuations only (A has no other continuation, B continues no other lanelet), so merges and diverges
    end lanes and never join two lanes. Two lanes are adjacent when a lanelet of one shares a bound way with a lanelet
    of the other. Kind of a lanelet from the custom tags: onramp=yes -> onramp, offramp=yes -> offramp, subtype
    emergency_lane -> shoulder (the hard shoulder is no travel lane; added from the map check before the extraction),
    else mainline (interchange=yes lanelets are highway lanelets without ramp tags: mainline);
  * a vehicle's lanelet at a frame: of its listed lanelets (laneletId) that are in the map, the one with the smallest
    |latLaneCenterOffset| (the centroid lies in several overlapping lanelets near merges and diverges), with that
    lanelet's offset and laneWidth; its lane = that lanelet's lane (none when no listed lanelet is in the map);
  * a lane change = a change of lane between consecutive frames. Cohorts: mainline (both lanelets mainline; the primary
    cohort) and onramp_merge (onramp -> mainline; auxiliary, descriptive only); any other change is excluded: shoulder_lane
    when either lanelet is a shoulder, else ramp_lane;
  * lateral position relative to the old lane centre: sigma * offset in the old lane and d + sigma * offset in the new
    one, over d = (w_old + w_new) / 2 from laneWidth (the lane not occupied takes its width at the flip); sigma = sign of
    the offset jump at the flip (no side convention is assumed). The jump must equal d within 1 m, and the offset may
    not jump by more than 0.5 m between other consecutive frames of the window (map seams);
  * along the road: positions and velocities projected on the follower's direction of travel at that time (its unit
    velocity); speeds = these projections; bumper gap = projected centre distance - (l_gen + l_fol) / 2 (tracksMeta
    lengths); 'behind' at the p50 frame uses each candidate's own direction;
  * class 'car' (case-insensitive) for generator and follower (exiD lists vans separately).
Event rule (as highD): adjacent-lane flip with 1 s stable lane labels either side; search window +-6 s bounded by the
neighbouring lane changes; lateral progress p10/p50/p90 crossings with the p50 crossing within 0.5 s of the flip;
follower = the nearest vehicle in the new lane at the p50 frame whose centre is behind the generator's centre, present in
every frame from p10 to p90 and in that lane throughout; speeds >= 15 m/s at p10 and p50; positive bumper gap
throughout. No simulation, no outcomes.
Cohort (select_cohort): primary (mainline) events; per recording, hash-ranked (salt below: the string of the v2j draft),
participant-disjoint greedy, at most 4 per recording; then, over all recordings, the TOTAL_CAP selections with the smallest
hash (a compute bound fixed before the extraction: three refinements x 17 SUTs). The case keeps the event's locationId
(`location`, for the location-cluster interval).

  python -m fairtest.exid extract DATA_DIR MAPS_DIR EVENTS.jsonl      (refuses an existing EVENTS.jsonl)
  python -m fairtest.exid maps MAPS_DIR                              (per map: lanelets, lanes, adjacency, tags)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

CFG = dict(stable_s=1., search_s=6., tol_s=.5, min_speed=15., jump_tol_m=1., seam_m=.5, meet_m=.1)
SALT = 'acarl-b15-exid-v2j:'
PER_RECORDING = 4
TOTAL_CAP = 120
COLS = ('trackId', 'frame', 'xCenter', 'yCenter', 'heading', 'xVelocity', 'yVelocity', 'laneletId', 'latLaneCenterOffset',
        'laneWidth')
LISTS = ('laneletId', 'latLaneCenterOffset', 'laneWidth')


# ---------------------------------------------------------------- map
class LaneMap:
    """Lanelet2 map: lane (connected continuations) and kind of every lanelet, lane adjacency."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.path = str(path)
        nodes = {}
        latlon = []
        for n in root.findall('node'):
            tags = {t.get('k'): t.get('v') for t in n.findall('tag')}
            if 'local_x' in tags and 'local_y' in tags:
                nodes[int(n.get('id'))] = (float(tags['local_x']), float(tags['local_y']))
            else:
                latlon.append((int(n.get('id')), float(n.get('lat')), float(n.get('lon'))))
        if latlon:  # equirectangular around the mean latitude: only orientation and coincidence are used
            lat0 = math.radians(float(np.mean([a for _, a, _ in latlon])))
            for i, lat, lon in latlon:
                nodes[i] = (6371000. * math.cos(lat0) * math.radians(lon), 6371000. * math.radians(lat))
        ways = {int(w.get('id')): [int(x.get('ref')) for x in w.findall('nd')] for w in root.findall('way')}
        self.lanelets, self.malformed, self.virtual = {}, 0, 0
        for r in root.findall('relation'):
            tags = {t.get('k'): t.get('v') for t in r.findall('tag')}
            if tags.get('type') != 'lanelet':
                continue
            if 'lxd_virtual' in tags:  # ignored, as by the provider's surrounding-vehicle search
                self.virtual += 1
                continue
            left = [int(m.get('ref')) for m in r.findall('member') if m.get('role') == 'left' and m.get('type') == 'way']
            right = [int(m.get('ref')) for m in r.findall('member') if m.get('role') == 'right' and m.get('type') == 'way']
            if len(left) != 1 or len(right) != 1 or left[0] not in ways or right[0] not in ways:
                self.malformed += 1
                continue
            L = np.array([nodes[x] for x in ways[left[0]]], float)
            R = np.array([nodes[x] for x in ways[right[0]]], float)
            if len(L) < 2 or len(R) < 2:
                self.malformed += 1
                continue
            o = L.mean(0) - R.mean(0)  # from the right bound to the left bound

            def forward(P):
                c = P[-1] - P[0]
                return P if c[0] * o[1] - c[1] * o[0] > 0 else P[::-1]
            L, R = forward(L), forward(R)
            kind = ('onramp' if tags.get('onramp') == 'yes' else 'offramp' if tags.get('offramp') == 'yes'
                    else 'shoulder' if tags.get('subtype') == 'emergency_lane' else 'mainline')
            self.lanelets[int(r.get('id'))] = dict(left=left[0], right=right[0], start=(L[0], R[0]), end=(L[-1], R[-1]),
                                                    kind=kind, tags=tags)
        ids = sorted(self.lanelets)
        parent = {i: i for i in ids}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        if ids:
            E = np.array([np.concatenate(self.lanelets[i]['end']) for i in ids])      # (n, 4): left end, right end
            S = np.array([np.concatenate(self.lanelets[i]['start']) for i in ids])
            dl = np.hypot(E[:, None, 0] - S[None, :, 0], E[:, None, 1] - S[None, :, 1])
            dr = np.hypot(E[:, None, 2] - S[None, :, 2], E[:, None, 3] - S[None, :, 3])
            meet = (dl <= CFG['meet_m']) & (dr <= CFG['meet_m'])
            np.fill_diagonal(meet, False)
        else:
            meet = np.zeros((0, 0), bool)
        n_succ, n_pred = meet.sum(1), meet.sum(0)
        self.continuations, self.links = int(meet.sum()), 0
        self.merges, self.diverges = int((n_pred > 1).sum()), int((n_succ > 1).sum())
        for x, y in zip(*np.nonzero(meet)):  # a lane runs through one-to-one continuations only (merges and diverges end it)
            if n_succ[x] == 1 and n_pred[y] == 1:
                parent[find(ids[x])] = find(ids[y])
                self.links += 1
        roots = {}
        self.lane = {i: roots.setdefault(find(i), len(roots)) for i in ids}
        self._ids = np.array(ids, np.int64)
        self._lanes = np.array([self.lane[i] for i in ids], np.int64)
        by_way = collections.defaultdict(set)
        for i, ll in self.lanelets.items():
            by_way[ll['left']].add(i)
            by_way[ll['right']].add(i)
        self.adjacent = set()
        for members in by_way.values():
            lanes = {self.lane[i] for i in members}
            for x in lanes:
                for y in lanes:
                    if x != y:
                        self.adjacent.add((x, y))

    def summary(self):
        kinds = collections.Counter(v['kind'] for v in self.lanelets.values())
        lane_kinds = collections.defaultdict(set)
        for i, v in self.lanelets.items():
            lane_kinds[self.lane[i]].add(v['kind'])
        tag_keys = collections.Counter(k for v in self.lanelets.values() for k in v['tags'])
        return dict(map=Path(self.path).name, lanelets=len(self.lanelets), virtual=self.virtual, malformed=self.malformed,
                    lanes=len(set(self.lane.values())), continuations=self.continuations, lane_links=self.links,
                    merges=self.merges, diverges=self.diverges, adjacent_lane_pairs=len(self.adjacent) // 2,
                    lanelet_kinds=dict(kinds), lanes_by_kinds=dict(collections.Counter('+'.join(sorted(k)) for k in lane_kinds.values())),
                    tag_keys=dict(tag_keys.most_common()))

    def lanes_of(self, lanelet):
        """Lane of each lanelet id (float array, NaN = none) -> int array, -1 when not in the map."""
        x = np.nan_to_num(np.asarray(lanelet, float), nan=-1.).astype(np.int64)
        out = np.full(len(x), -1, np.int64)
        if len(self._ids):
            j = np.clip(np.searchsorted(self._ids, x), 0, len(self._ids) - 1)
            hit = self._ids[j] == x
            out[hit] = self._lanes[j[hit]]
        return out


def map_files(maps_dir):
    """{locationId: path} of the .osm files under maps_dir: 'location<k>' in the name, else its only integer."""
    out, bad = {}, []
    for p in sorted(Path(maps_dir).rglob('*.osm')):
        m = re.search(r'location[_-]?(\d+)', p.stem, re.I)
        nums = [int(x) for x in re.findall(r'\d+', p.stem)]
        loc = int(m.group(1)) if m else (nums[0] if len(nums) == 1 else None)
        if loc is None or loc in out:
            bad.append(str(p))
            continue
        out[loc] = p
    if bad:
        raise SystemExit(f'map files without a unique location id: {bad}')
    return out


# ---------------------------------------------------------------- tracks
def primary_lanelets(df, lmap):
    """Per row: the listed lanelet in the map with the smallest |offset| and its offset and width (NaN when none)."""
    lid = df['laneletId'].astype(str).str.strip()
    off = df['latLaneCenterOffset'].astype(str).str.strip()
    wid = df['laneWidth'].astype(str).str.strip()
    multi = lid.str.contains(';', regex=False).to_numpy()
    out_l = pd.to_numeric(lid.where(~multi), errors='coerce').to_numpy(dtype=float, copy=True)
    out_o = pd.to_numeric(off.where(~multi), errors='coerce').to_numpy(dtype=float, copy=True)
    out_w = pd.to_numeric(wid.where(~multi), errors='coerce').to_numpy(dtype=float, copy=True)
    unknown = (lmap.lanes_of(out_l) < 0) & ~multi
    out_l[unknown], out_o[unknown], out_w[unknown] = np.nan, np.nan, np.nan
    for i in np.flatnonzero(multi):
        best = None
        for a, b, c in zip(lid.iat[i].split(';'), off.iat[i].split(';'), wid.iat[i].split(';')):
            try:
                a, b, c = int(float(a)), float(b), float(c)
            except ValueError:
                continue
            if a in lmap.lanelets and (best is None or abs(b) < abs(best[1])):
                best = (a, b, c)
        out_l[i], out_o[i], out_w[i] = best if best else (np.nan, np.nan, np.nan)
    return out_l, out_o, out_w


def tracks_of(df, static, lmap, fps):
    lanelet, off, width = primary_lanelets(df, lmap)
    df = df.assign(_ll=lanelet, _off=off, _w=width)
    tracks = {}
    for tid, g in df.groupby('trackId', sort=True):
        g = g.sort_values('frame').reset_index(drop=True)
        f = g['frame'].to_numpy()
        breaks = np.flatnonzero(np.diff(f) != 1) + 1
        seg = max(np.split(np.arange(len(g)), breaks), key=len)
        g = g.iloc[seg].reset_index(drop=True)
        ll = g['_ll'].to_numpy(float)
        lane = lmap.lanes_of(ll)
        st = static.loc[tid] if tid in static.index else None
        tracks[int(tid)] = dict(vid=int(tid), frame=g['frame'].to_numpy(), t=g['frame'].to_numpy() / fps,
                                x=g['xCenter'].to_numpy(float), y=g['yCenter'].to_numpy(float),
                                vx=g['xVelocity'].to_numpy(float), vy=g['yVelocity'].to_numpy(float),
                                heading=g['heading'].to_numpy(float), lanelet=ll, lane=lane,
                                off=g['_off'].to_numpy(float), w=g['_w'].to_numpy(float),
                                length=float(st['length']) if st is not None else float('nan'),
                                kind=str(st['class']) if st is not None else '')
    return tracks


def is_car(kind):
    return kind.strip().lower() == 'car'


def crossings(t, p, level):
    idx = np.flatnonzero((p[:-1] < level) & (p[1:] >= level))
    return [float(t[i] + (t[i + 1] - t[i]) * (level - p[i]) / (p[i + 1] - p[i])) for i in idx]


def anchors(t, p, reference, tolerance):
    """highD rule: the p50 crossing nearest the flip (within tolerance), the last p10 crossing before it and the first p90
    crossing after it."""
    mids = crossings(t, p, .5)
    if not mids:
        return None
    mid = min(mids, key=lambda x: abs(x - reference))
    if abs(mid - reference) > tolerance:
        return None
    starts = [x for x in crossings(t, p, .1) if x < mid]
    ends = [x for x in crossings(t, p, .9) if x > mid]
    return [max(starts), mid, min(ends)] if starts and ends else None


def unit(vx, vy, heading_deg):
    s = math.hypot(vx, vy)
    if s >= .5:
        return np.array([vx / s, vy / s])
    h = math.radians(heading_deg)
    return np.array([math.cos(h), math.sin(h)])


def at(tr, time, key):
    return float(np.interp(time, tr['t'], tr[key]))


def extract(df, static, lmap, rec, loc, fps):
    tracks = tracks_of(df, static, lmap, fps)
    keys = sorted(tracks)
    all_f = np.concatenate([tracks[k]['frame'] for k in keys]) if keys else np.zeros(0, int)
    kid = np.concatenate([np.full(len(tracks[k]['frame']), j) for j, k in enumerate(keys)]) if keys else np.zeros(0, int)
    pos = np.concatenate([np.arange(len(tracks[k]['frame'])) for k in keys]) if keys else np.zeros(0, int)
    order = np.argsort(all_f, kind='stable')
    all_f, kid, pos = all_f[order], kid[order], pos[order]

    def at_frame(f):
        lo, hi = np.searchsorted(all_f, f, 'left'), np.searchsorted(all_f, f, 'right')
        return [(keys[kid[j]], int(pos[j])) for j in range(lo, hi)]

    st = int(round(CFG['stable_s'] * fps))
    radius = int(CFG['search_s'] * fps)
    events = []
    for vid in keys:
        tr = tracks[vid]
        lanes = tr['lane']
        changes = np.flatnonzero(np.diff(lanes) != 0) + 1
        for k in changes:
            old, new = int(lanes[k - 1]), int(lanes[k])
            flip = int(tr['frame'][k])
            a = None if np.isnan(tr['lanelet'][k - 1]) else int(tr['lanelet'][k - 1])
            b = None if np.isnan(tr['lanelet'][k]) else int(tr['lanelet'][k])
            row = dict(event_id=f'exid-{rec:02d}:{vid}:{flip}', partition='exid', recording=f'{rec:02d}', location=loc, vehicle=vid,
                       flip_frame=flip, old_lane=old, new_lane=new, old_lanelet=a, new_lanelet=b, cohort=None, acc_generator=None)
            events.append(row)
            if old < 0 or new < 0 or (old, new) not in lmap.adjacent:
                row.update(eligible=False, exclusion='nonadjacent_or_unmapped_lane'); continue
            ka, kb = lmap.lanelets[a]['kind'], lmap.lanelets[b]['kind']
            row['cohort'] = 'mainline' if (ka, kb) == ('mainline', 'mainline') else ('onramp_merge' if (ka, kb) == ('onramp', 'mainline') else None)
            if row['cohort'] is None:
                row.update(eligible=False, exclusion='shoulder_lane' if 'shoulder' in (ka, kb) else 'ramp_lane'); continue
            if k < st or k + st > len(lanes):
                row.update(eligible=False, exclusion='truncated_stable_lane_window'); continue
            if not np.all(np.diff(tr['frame'][k - st:k + st]) == 1):
                row.update(eligible=False, exclusion='discontinuous_stable_lane_window'); continue
            if not (np.all(lanes[k - st:k] == old) and np.all(lanes[k:k + st] == new)):
                row.update(eligible=False, exclusion='unstable_lane_labels'); continue
            prev, nxt = changes[changes < k], changes[changes > k]
            low = max(int(prev[-1]) if len(prev) else 0, k - radius)
            high = min(int(nxt[0]) if len(nxt) else len(lanes), k + radius + 1)
            if not np.all(np.diff(tr['frame'][low:high]) == 1):
                row.update(eligible=False, exclusion='discontinuous_generator_window'); continue
            off, w = tr['off'][low:high], tr['w'][low:high]
            if not (np.all(np.isfinite(off)) and np.all(np.isfinite(w)) and np.all(w > 0)):
                row.update(eligible=False, exclusion='missing_lane_offsets'); continue
            i = k - low  # first frame in the new lane, within the window
            jump = off[i - 1] - off[i]
            sigma = 1. if jump > 0 else -1.
            d0 = (w[i - 1] + w[i]) / 2
            if abs(abs(jump) - d0) > CFG['jump_tol_m']:
                row.update(eligible=False, exclusion='inconsistent_lane_change_offsets'); continue
            steps = np.abs(np.diff(off))
            steps[i - 1] = 0.
            if np.any(steps > CFG['seam_m']):
                row.update(eligible=False, exclusion='lane_offset_discontinuity'); continue
            d = np.where(np.arange(len(off)) < i, (w + w[i]) / 2, (w[i - 1] + w) / 2)
            n = np.where(np.arange(len(off)) < i, sigma * off, d + sigma * off)
            progress = n / d
            times = tr['t'][low:high]
            ev = anchors(times, progress, flip / fps, CFG['tol_s'])
            if ev is None:
                row.update(eligible=False, exclusion='unobserved_or_inconsistent_10_50_90_crossing'); continue
            row.update(times=ev, lane_center_distance=float(d0), lateral_backtrack_fraction=float(max(0., -np.min(np.diff(progress)))))
            f50 = int(np.ceil(ev[1] * fps - 1e-8))
            gp = np.array([at(tr, f50 / fps, 'x'), at(tr, f50 / fps, 'y')])
            cands = []
            for okey, j in at_frame(f50):
                if okey == vid:
                    continue
                o = tracks[okey]
                if int(o['lane'][j]) != new:
                    continue
                u = unit(o['vx'][j], o['vy'][j], o['heading'][j])
                behind = float((gp - np.array([o['x'][j], o['y'][j]])) @ u)
                if behind > 0:
                    cands.append((behind, okey))
            if not cands:
                row.update(eligible=False, exclusion='no_observed_destination_follower'); continue
            fid = min(cands)[1]
            fo = tracks[fid]
            row['follower'] = fid
            row['acc_follower'] = None
            if not (is_car(tr['kind']) and is_car(fo['kind'])):
                row.update(eligible=False, exclusion='non_car_pair'); continue
            f10, f90 = int(np.floor(ev[0] * fps)), int(np.ceil(ev[2] * fps))
            sel = (fo['frame'] >= f10) & (fo['frame'] <= f90)
            if sel.sum() != f90 - f10 + 1:
                row.update(eligible=False, exclusion='truncated_or_discontinuous_follower'); continue
            if not np.all(fo['lane'][sel] == new):
                row.update(eligible=False, exclusion='follower_changes_lane'); continue

            def along(time):
                uf = unit(at(fo, time, 'vx'), at(fo, time, 'vy'), at(fo, time, 'heading'))
                gx = np.array([at(tr, time, 'x'), at(tr, time, 'y')])
                fx = np.array([at(fo, time, 'x'), at(fo, time, 'y')])
                gv = float(np.array([at(tr, time, 'vx'), at(tr, time, 'vy')]) @ uf)
                tv = float(np.array([at(fo, time, 'vx'), at(fo, time, 'vy')]) @ uf)
                gap = float((gx - fx) @ uf) - (tr['length'] + fo['length']) / 2
                return gv, tv, gap
            metrics = dict(duration_10_90=float(ev[2] - ev[0]))
            ok = True
            for name, tm in (('p10', ev[0]), ('p50', ev[1])):
                gv, tv, gap = along(tm)
                metrics[name] = dict(generator_speed=gv, target_speed=tv, gap=gap, headway=gap / tv if tv > 0 else None,
                                     closing_speed=tv - gv)
                ok &= gv >= CFG['min_speed'] and tv >= CFG['min_speed']
            if not ok:
                row.update(eligible=False, exclusion='below_speed_population', metrics=metrics); continue
            gaps = [along(tm)[2] for tm in fo['t'][sel]]
            if min(gaps) <= 0:
                row.update(eligible=False, exclusion='nonpositive_longitudinal_gap'); continue
            row.update(eligible=True, exclusion=None, metrics=metrics)
    n_frames = sum(len(v['lane']) for v in tracks.values())
    record = dict(recording=f'{rec:02d}', location=loc, fps=fps, rows=len(df), vehicles=len(tracks),
                  mapped_frames=round(sum(int((v['lane'] >= 0).sum()) for v in tracks.values()) / max(1, n_frames), 4),
                  candidates=len(events), eligible=sum(1 for e in events if e.get('eligible')),
                  eligible_by_cohort=dict(collections.Counter(e['cohort'] for e in events if e.get('eligible'))),
                  exclusions=dict(collections.Counter(e['exclusion'] for e in events if not e.get('eligible'))),
                  classes=dict(collections.Counter(str(x) for x in static['class'])))
    return events, record


def load_recording(data_dir, rec):
    d = Path(data_dir)
    meta = pd.read_csv(d / f'{rec:02d}_recordingMeta.csv').iloc[0].to_dict()
    static = pd.read_csv(d / f'{rec:02d}_tracksMeta.csv').set_index('trackId')
    path = d / f'{rec:02d}_tracks.csv'
    head = pd.read_csv(path, nrows=0).columns
    missing = [c for c in COLS if c not in head]
    if missing:
        raise SystemExit(f'{path.name}: missing columns {missing} (the map-based enrichment is required)')
    df = pd.read_csv(path, usecols=list(COLS), dtype={c: str for c in LISTS})
    if df.duplicated(['trackId', 'frame']).any():
        raise SystemExit(f'{path.name}: duplicated (trackId, frame)')
    return meta, static, df, path


def recordings(data_dir):
    return sorted(int(m.group(1)) for p in Path(data_dir).glob('*_tracks.csv') if (m := re.match(r'^(\d+)_tracks\.csv$', p.name)))


def run_extract(data_dir, maps_dir, out):
    out = Path(out)
    maps = {loc: LaneMap(p) for loc, p in map_files(maps_dir).items()}
    all_events, records = [], []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:  # extraction happens once (protocol section 7)
        for rec in recordings(data_dir):
            meta, static, df, path = load_recording(data_dir, rec)
            loc = int(meta['locationId'])
            if loc not in maps:
                raise SystemExit(f'recording {rec:02d}: no map for location {loc}')
            ev, record = extract(df, static, maps[loc], rec, loc, float(meta['frameRate']))
            record['tracks_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            for e in ev:
                f.write(json.dumps(e) + '\n')
            all_events += ev
            records.append(record)
            print(json.dumps(record), flush=True)
    return all_events, records


def select_cohort(events):
    """Primary-cohort (mainline) extractor- and preflight-eligible events; per recording, hash-ranked participant-disjoint
    greedy, at most PER_RECORDING (the B13/B28/B41 rule with the exiD salt and cap); then the TOTAL_CAP of these with the
    smallest hash. Cases are numbered in recording order, hash order within a recording."""
    from fairtest import devboot  # noqa: F401
    from scripts.prepare_adversarial_b3_transfer import preflight, sanitize_event
    buckets, ledger = {}, []
    for e in events:
        row = dict(event_id=e['event_id'], recording=e['recording'], vehicle=e['vehicle'], follower=e.get('follower'),
                   cohort=e.get('cohort'), extractor_eligible=bool(e.get('eligible')), exclusion=e.get('exclusion'), selected=False,
                   reason=None, rank_hash=hashlib.sha256((SALT + e['event_id']).encode('utf-8')).hexdigest())
        ledger.append(row)
        if not e.get('eligible'):
            row['reason'] = 'extractor_ineligible'
            continue
        if e.get('cohort') != 'mainline':
            row['reason'] = 'auxiliary_cohort'
            continue
        meta = sanitize_event(e)
        meta['location'] = e.get('location')
        check = preflight(meta)
        if not check['eligible']:
            row['reason'] = 'initial_domain:' + ','.join(check['reasons'])
            continue
        row['_meta'], row['_check'] = meta, check
        buckets.setdefault(e['recording'], []).append(row)
    picked = []
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
            picked.append(row)
    keep = {r['event_id'] for r in sorted(picked, key=lambda r: (r['rank_hash'], r['event_id']))[:TOTAL_CAP]}
    cases = []
    for row in picked:
        if row['event_id'] not in keep:
            row['reason'] = 'total_cap'
            continue
        row['selected'] = True
        cases.append(dict(case_id=len(cases), event=row['_meta'], preflight=row['_check']['checks'],
                          physical_initial_state=row['_check']['physical_initial_state'],
                          initial_state_source='local_preflight', follower=row['follower'], dev16=False))
    clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in ledger]
    return cases, clean


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('extract')
    p.add_argument('data_dir')
    p.add_argument('maps_dir')
    p.add_argument('out')
    m = sub.add_parser('maps')
    m.add_argument('maps_dir')
    a = ap.parse_args()
    if a.cmd == 'extract':
        run_extract(a.data_dir, a.maps_dir, a.out)
    else:
        for loc, path in sorted(map_files(a.maps_dir).items()):
            print(json.dumps(dict(location=loc, **LaneMap(path).summary())))
