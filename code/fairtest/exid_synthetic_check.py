"""Synthetic check of the exiD extractor (no exiD data is read): a tilted straight road of three mainline lanes in six
200 m lanelets each, an acceleration lane (onramp) and a deceleration lane (offramp) on the right, a hard shoulder
(subtype emergency_lane) beside the left lane's third segment and a virtual lanelet over the left lane's last segment,
written as a Lanelet2 map; two recordings at 25 frames/s in the exiD CSV layout.
Planted (recording 00): a car cut-in 0 -> 1 in front of a follower (eligible, true p10 speeds, bumper gap and 10-90
duration), an onramp merge (eligible, auxiliary cohort), a change into the offramp (ramp_lane), a truck cut-in
(non_car_pair), a one-quarter-second excursion into the next lane (two unstable_lane_labels), a slow pair
(below_speed_population), a cut-in whose follower changes lane during it (follower_changes_lane) and that follower's own
change with nobody behind it (no_observed_destination_follower); the left-lane traffic crosses the virtual lanelet, which
the tracks list with a smaller offset (must not split the lane). Recording 01: a second eligible cut-in, a pair whose
'follower' overlaps the generator (nonpositive_longitudinal_gap) and a cut-in with a one-frame 0.8 m offset seam
(lane_offset_discontinuity) and a car leaving the shoulder in front of a follower (shoulder_lane). Every lanelet
transition along a lane is a continuation, never a lane change.
Variants that must give the same events: map in local_x/local_y with some bound ways stored backwards and offsets
positive to the left; map in lat/lon only with offsets positive to the right.

  python -m fairtest.exid_synthetic_check
"""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from fairtest import exid

FPS, W, SEG, L_CAR = 25., 3.5, 200., 4.6
THETA = math.radians(25.)
ORIGIN = np.array([300., -50.])
U = np.array([math.cos(THETA), math.sin(THETA)])
NV = np.array([-U[1], U[0]])  # left of travel
LAT0, LON0 = 50.8, 6.1
TRACK_COLS = ['recordingId', 'trackId', 'frame', 'trackLifetime', 'xCenter', 'yCenter', 'heading', 'width', 'length', 'xVelocity',
              'yVelocity', 'xAcceleration', 'yAcceleration', 'lonVelocity', 'latVelocity', 'lonAcceleration', 'latAcceleration',
              'traveledDistance', 'latLaneCenterOffset', 'laneWidth', 'laneletId', 'laneChange', 'lonLaneletPos', 'laneletLength',
              'leadDHW', 'leadDV', 'leadTHW', 'leadTTC', 'leadId', 'rearId', 'leftLeadId', 'leftRearId', 'leftAlongsideId',
              'rightLeadId', 'rightRearId', 'rightAlongsideId']
VIRTUAL = 400
SHOULDER = 500


def xy(s, n):
    p = ORIGIN + np.outer(np.atleast_1d(s), U) + np.outer(np.atleast_1d(n), NV)
    return p[:, 0], p[:, 1]


def to_latlon(x, y):
    lat = LAT0 + math.degrees(y / 6371000.)
    lon = LON0 + math.degrees(x / (6371000. * math.cos(math.radians(LAT0))))
    return lat, lon


# ------------------------------------------------------------------ map
def lanelet_at(s, n):
    """(lanelet id, centre n) of the drivable lanelet containing (s, n), or (None, None)."""
    if 0 <= n < 3 * W and 0 <= s < 6 * SEG:
        i, j = int(n // W), int(s // SEG)
        return 100 + 10 * i + j, (i + .5) * W
    if -W <= n < 0 and 200 <= s < 400:
        return 200, -W / 2
    if -W <= n < 0 and 800 <= s < 1000:
        return 300, -W / 2
    if 3 * W <= n < 4 * W and 400 <= s < 600:
        return SHOULDER, 3.5 * W
    return None, None


def write_map(path, local, reversed_ways):
    nodes, ways, rels = {}, {}, []

    def node(key, s, n):
        if key not in nodes:
            x, y = xy(s, n)
            nodes[key] = (len(nodes) + 1, float(x[0]), float(y[0]))
        return nodes[key][0]

    def way(wid, pts):
        ids = [node(k, s, n) for k, s, n in pts]
        ways[wid] = ids[::-1] if wid in reversed_ways else ids
        return wid

    def boundary(b, j):  # boundary b (n = b * W) over segment j, three nodes
        return way(1000 + 10 * b + j, [((b, m), 100. * m, b * W) for m in (2 * j, 2 * j + 1, 2 * j + 2)])
    for i in range(3):
        for j in range(6):
            rels.append((100 + 10 * i + j, boundary(i + 1, j), boundary(i, j), {}))
    acc = way(2000, [(('acc', m), 100. * m, -W) for m in (2, 3, 4)])
    rels.append((200, boundary(0, 1), acc, {'onramp': 'yes'}))
    off = way(3000, [(('off', m), 100. * m, -W) for m in (8, 9, 10)])
    rels.append((300, boundary(0, 4), off, {'offramp': 'yes'}))
    rels.append((VIRTUAL, boundary(3, 5), boundary(2, 5), {'lxd_virtual': 'yes'}))
    shoulder = way(4000, [(('sh', m), 100. * m, 4 * W) for m in (4, 5, 6)])
    rels.append((SHOULDER, shoulder, boundary(3, 2), {'subtype': 'emergency_lane'}))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="exid_synthetic_check">']
    for _, (nid, x, y) in sorted(nodes.items(), key=lambda kv: kv[1][0]):
        lat, lon = to_latlon(x, y)
        if local:
            lines.append(f'  <node id="{nid}" lat="{lat:.10f}" lon="{lon:.10f}"><tag k="local_x" v="{x:.4f}"/><tag k="local_y" v="{y:.4f}"/></node>')
        else:
            lines.append(f'  <node id="{nid}" lat="{lat:.10f}" lon="{lon:.10f}"/>')
    for wid, ids in ways.items():
        lines.append(f'  <way id="{wid}">' + ''.join(f'<nd ref="{i}"/>' for i in ids) + '<tag k="type" v="line_thin"/></way>')
    for rid, left, right, tags in rels:
        t = {'type': 'lanelet', 'subtype': 'highway', **tags}
        lines.append(f'  <relation id="{rid}"><member type="way" ref="{left}" role="left"/><member type="way" ref="{right}" role="right"/>'
                     + ''.join(f'<tag k="{k}" v="{v}"/>' for k, v in t.items()) + '</relation>')
    lines.append('</osm>')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ------------------------------------------------------------------ traffic
def quintic(u):
    u = np.clip(u, 0., 1.)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def lane_n(i):
    return (i + .5) * W


def change(t, n0, n1, t0, dur):
    return n0 + (n1 - n0) * quintic((t - t0) / dur)


def recording_00(t):
    v = {}
    v[1] = ('car', 50 + 27 * t, change(t, lane_n(0), lane_n(1), 8, 5), None)                    # A: eligible cut-in 0 -> 1
    v[2] = ('car', 241.4 + 29 * (t - 8), np.full_like(t, lane_n(1)), None)                     # A's follower
    v[3] = ('car', 205 + 24 * t, change(t, -W / 2, lane_n(0), 2, 5), (205, None))              # B: onramp merge
    v[4] = ('car', 223.4 + 26 * (t - 2), np.full_like(t, lane_n(0)), None)                     # B's follower
    v[5] = ('car', 600 + 25 * t, change(t, lane_n(0), -W / 2, 8, 4), (None, 1000))             # C: into the offramp
    v[6] = ('truck', 500 + 22 * t, change(t, lane_n(1), lane_n(2), 2, 4), None)                # D: truck
    v[7] = ('car', 520.7 + 24 * (t - 2), np.full_like(t, lane_n(2)), None)                     # D's follower
    v[8] = ('car', 400 + 30 * t, lane_n(2) - 2.2 * np.exp(-((t - 5.3) / .25) ** 2), None)       # E: excursion
    v[9] = ('car', 200 + 12 * (t - 20), change(t, lane_n(1), lane_n(0), 26, 4), None)          # F: slow
    v[10] = ('car', 252.4 + 13 * (t - 26), np.full_like(t, lane_n(0)), None)                   # F's follower
    v[11] = ('car', 300 + 25 * (t - 18), change(t, lane_n(0), lane_n(1), 18, 4), None)         # I: follower leaves
    v[12] = ('car', 277.4 + 27 * (t - 18), change(t, lane_n(1), lane_n(2), 18, 5), None)       # I's follower, then its own change
    return v


def recording_01(t):
    v = {}
    v[21] = ('car', 100 + 28 * t, change(t, lane_n(1), lane_n(2), 5, 5), None)                 # L: eligible cut-in 1 -> 2
    v[22] = ('car', 217.4 + 30 * (t - 5), np.full_like(t, lane_n(2)), None)                    # L's follower
    v[23] = ('car', 50 + 26 * (t - 14), change(t, lane_n(0), lane_n(1), 20, 4), None)          # H: overlapping 'follower'
    v[24] = ('car', 256 + 26 * (t - 22), np.full_like(t, lane_n(1)), None)
    v[25] = ('car', 200 + 28 * (t - 30), change(t, lane_n(2), lane_n(1), 30, 4), None)         # K: offset seam
    v[26] = ('car', 400 + 28 * (t - 33), change(t, 3.5 * W, lane_n(2), 34.5, 4), None)         # S: shoulder -> left lane
    v[27] = ('car', 380 + 30 * (t - 33), np.full_like(t, lane_n(2)), None)                     # S's follower
    return v


def write_recording(d, rec, vehicles, t, sign):
    rows, meta = [], []
    for vid, (kind, s, n, bounds) in vehicles.items():
        lo, hi = (bounds or (None, None))
        keep = (s >= (0 if lo is None else lo)) & (s <= (6 * SEG - 1e-6 if hi is None else hi))
        keep &= np.array([lanelet_at(a, b)[0] is not None for a, b in zip(s, n)])
        idx = np.flatnonzero(keep)
        idx = max(np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1), key=len)
        s, n, frames = s[idx], n[idx], idx
        ds, dn = np.gradient(s, 1 / FPS), np.gradient(n, 1 / FPS)
        x, y = xy(s, n)
        vxy = np.outer(ds, U) + np.outer(dn, NV)
        length = 12. if kind == 'truck' else L_CAR
        prev_lane = None
        for k, f in enumerate(frames):
            lid, centre = lanelet_at(s[k], n[k])
            lane = lid if lid in (200, 300, SHOULDER) else (lid // 10) % 10
            lc = int(prev_lane is not None and lane != prev_lane)
            prev_lane = lane
            off = sign * (n[k] - centre)
            ids, offs, widths = str(lid), f'{off:.4f}', f'{W}'
            if lid == 125:  # the virtual lanelet over the left lane's last segment is listed too, with offset 0
                ids, offs, widths = f'{VIRTUAL};125', f'0.0000;{off:.4f}', f'{W};{W}'
            if vid == 25 and abs(f / FPS - 31.) < 1e-9:  # K: a one-frame 0.8 m offset seam inside the window
                offs = f'{off + .8:.4f}'
            rows.append([rec, vid, int(f), k, f'{x[k]:.4f}', f'{y[k]:.4f}', f'{math.degrees(math.atan2(vxy[k, 1], vxy[k, 0])):.4f}',
                         1.9, length, f'{vxy[k, 0]:.4f}', f'{vxy[k, 1]:.4f}', 0, 0, f'{math.hypot(*vxy[k]):.4f}', f'{dn[k]:.4f}', 0, 0,
                         f'{s[k] - s[0]:.3f}', offs, widths, ids, lc, f'{s[k] % SEG:.3f}', SEG, -1, -1000, -1, -1, -1, -1, -1, -1, '',
                         -1, -1, ''])
        meta.append([rec, vid, int(frames[0]), int(frames[-1]), len(frames), 1.9, length, kind])
    rows.sort(key=lambda r: (r[1], r[2]))
    with open(Path(d) / f'{rec:02d}_tracks.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(TRACK_COLS)
        w.writerows(rows)
    with open(Path(d) / f'{rec:02d}_tracksMeta.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['recordingId', 'trackId', 'initialFrame', 'finalFrame', 'numFrames', 'width', 'length', 'class'])
        w.writerows(meta)
    with open(Path(d) / f'{rec:02d}_recordingMeta.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['recordingId', 'locationId', 'frameRate', 'speedLimit', 'weekday', 'startTime', 'duration', 'numTracks',
                    'numVehicles', 'numVrus', 'latLocation', 'lonLocation', 'xUtmOrigin', 'yUtmOrigin', 'orthoPxToMeter'])
        w.writerow([rec, 1, FPS, -1, 'Monday', 10, 40, len(vehicles), len(vehicles), 0, LAT0, LON0, 0, 0, .1])


def run(variant):
    t = np.arange(0., 40., 1 / FPS)
    with tempfile.TemporaryDirectory() as tmp:
        data, maps = Path(tmp) / 'data', Path(tmp) / 'maps'
        data.mkdir()
        maps.mkdir()
        local = variant == 'local'
        write_map(maps / 'location1_synthetic.osm', local, reversed_ways={1010, 1011, 1012, 1013, 1014, 1015, 2000} if local else set())
        sign = 1. if local else -1.
        write_recording(data, 0, recording_00(t), t, sign)
        write_recording(data, 1, recording_01(t), t, sign)
        summary = exid.LaneMap(maps / 'location1_synthetic.osm').summary()
        events, records = exid.run_extract(data, maps, Path(tmp) / 'events.jsonl')
    return summary, events, records


def u10():
    lo, hi = 0., .5
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if quintic(m) < .1 else (lo, m)
    return lo


def main():
    expected = {(0, 1): ('eligible', 2), (0, 3): ('eligible', 4), (0, 5): ('ramp_lane', None), (0, 6): ('non_car_pair', 7),
                (0, 8): ('unstable_lane_labels', None), (0, 9): ('below_speed_population', 10),
                (0, 11): ('follower_changes_lane', 12), (0, 12): ('no_observed_destination_follower', None),
                (1, 21): ('eligible', 22), (1, 23): ('nonpositive_longitudinal_gap', 24), (1, 25): ('lane_offset_discontinuity', None),
                (1, 26): ('shoulder_lane', None)}
    results = {}
    for variant in ('local', 'latlon'):
        summary, events, records = run(variant)
        print(variant, 'map', summary)
        got = {}
        for e in events:
            key = (int(e['recording']), e['vehicle'])
            got.setdefault(key, []).append(('eligible' if e['eligible'] else e['exclusion'], e.get('follower')))
        print(variant, 'events', sorted(got.items()))
        checks = dict(lanelets=summary['lanelets'] == 21 and summary['virtual'] == 1 and summary['lanes'] == 6,
                      adjacency=summary['adjacent_lane_pairs'] == 5,
                      kinds=summary['lanelet_kinds'] == {'mainline': 18, 'onramp': 1, 'offramp': 1, 'shoulder': 1},
                      planted_only=set(got) == set(expected), excursion_twice=got.get((0, 8)) == [('unstable_lane_labels', None)] * 2)
        for key, (outcome, follower) in expected.items():
            evs = got.get(key, [])
            if key == (0, 8):
                continue
            checks[f'{key}'] = len(evs) == 1 and evs[0][0] == outcome and (follower is None or evs[0][1] == follower)
        ok = {(int(e['recording']), e['vehicle']): e for e in events if e['eligible']}
        a, b, l_ = ok.get((0, 1)), ok.get((0, 3)), ok.get((1, 21))
        checks['cohorts'] = bool(a and b and l_ and a['cohort'] == 'mainline' and b['cohort'] == 'onramp_merge' and l_['cohort'] == 'mainline')
        q = u10()
        for name, e, (sg, vg, sf, vf, t0, dur) in (('A', a, (50., 27., 241.4 - 29 * 8, 29., 8., 5.)), ('L', l_, (100., 28., 217.4 - 30 * 5, 30., 5., 5.))):
            if not e:
                continue
            p10 = e['metrics']['p10']
            t10 = t0 + dur * q
            gap_true = (sg + vg * t10) - (sf + vf * t10) - L_CAR
            checks[f'{name}_metrics'] = (abs(p10['generator_speed'] - vg) < .05 and abs(p10['target_speed'] - vf) < .05
                                         and abs(p10['gap'] - gap_true) < .15 and abs(e['metrics']['duration_10_90'] - dur * (1 - 2 * q)) < .05
                                         and abs(e['times'][0] - t10) < .02)
            print(name, 'p10', {k: round(v, 3) for k, v in p10.items() if v is not None}, 'gap_true', round(gap_true, 3),
                  'duration', round(e['metrics']['duration_10_90'], 3), 'expected', round(dur * (1 - 2 * q), 3))
        print(variant, 'checks', checks)
        assert all(checks.values()), (variant, {k: v for k, v in checks.items() if not v})
        cases, ledger = exid.select_cohort(events)
        reasons = {(r['recording'], r['vehicle']): r['reason'] for r in ledger}
        print(variant, 'selected', [(c['event']['recording'], c['event']['vehicle']) for c in cases], 'onramp merge:', reasons[('00', 3)])
        assert [(c['event']['recording'], c['event']['vehicle']) for c in cases] == [('00', 1), ('01', 21)]
        assert reasons[('00', 3)] == 'auxiliary_cohort'
        assert all(c['event']['location'] == 1 for c in cases), 'the case must keep the locationId'
        cap, exid.TOTAL_CAP = exid.TOTAL_CAP, 1  # the total cap keeps the smallest hash over all recordings
        try:
            capped, ledger_capped = exid.select_cohort(events)
        finally:
            exid.TOTAL_CAP = cap
        rank = {(r['recording'], r['vehicle']): r['rank_hash'] for r in ledger}
        first = min((('00', 1), ('01', 21)), key=lambda k: rank[k])
        other = ('01', 21) if first == ('00', 1) else ('00', 1)
        why = {(r['recording'], r['vehicle']): r['reason'] for r in ledger_capped}
        assert [(c['event']['recording'], c['event']['vehicle']) for c in capped] == [first] and capped[0]['case_id'] == 0
        assert why[other] == 'total_cap' and why[first] is None
        print(variant, 'total cap 1 keeps', first, 'drops', other)
        results[variant] = sorted((e['event_id'], e['eligible'], e['exclusion'], e.get('follower'), e.get('cohort'),
                                   json_round(e.get('metrics'))) for e in events)
    assert results['local'] == results['latlon'], 'map coordinates, bound orientation or offset sign changed the events'
    diverge_merge_check()
    print('SYNTHETIC CHECK PASSED')


def diverge_merge_check():
    """A lane splits (P -> Q1, Q2 starting on P's end line) and two lanes join (R1, R2 -> T): lanes end at both, and a
    plain chain A -> B -> C stays one lane."""
    pts = {}

    def nd(s, n):
        return pts.setdefault((s, n), len(pts) + 1)
    ways, rels = {}, []

    def way(wid, seq):
        ways[wid] = [nd(s, n) for s, n in seq]
        return wid
    # chain A -> B -> C on n in [20, 23.5]
    for k, (a, b) in enumerate(((0, 100), (100, 200), (200, 300))):
        rels.append((10 + k, way(100 + k, [(a, 23.5), (b, 23.5)]), way(110 + k, [(a, 20.), (b, 20.)])))
    # diverge: P on [0, 100], Q1 straight, Q2 bending right; both start on P's end line
    rels.append((20, way(200, [(0, 3.5), (100, 3.5)]), way(201, [(0, 0.), (100, 0.)])))
    rels.append((21, way(202, [(100, 3.5), (200, 3.5)]), way(203, [(100, 0.), (200, 0.)])))
    rels.append((22, way(204, [(100, 3.5), (200, -3.)]), way(205, [(100, 0.), (200, -6.5)])))
    # merge: R1 and R2 end on T's start line
    rels.append((30, way(300, [(300, 43.5), (400, 43.5)]), way(301, [(300, 40.), (400, 40.)])))
    rels.append((31, way(302, [(300, 50.), (400, 43.5)]), way(303, [(300, 46.5), (400, 40.)])))
    rels.append((32, way(304, [(400, 43.5), (500, 43.5)]), way(305, [(400, 40.), (500, 40.)])))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">']
    for (s, n), i in pts.items():
        x, y = xy(s, n)
        lines.append(f'  <node id="{i}" lat="0" lon="0"><tag k="local_x" v="{x[0]:.4f}"/><tag k="local_y" v="{y[0]:.4f}"/></node>')
    for wid, ids in ways.items():
        lines.append(f'  <way id="{wid}">' + ''.join(f'<nd ref="{i}"/>' for i in ids) + '</way>')
    for rid, left, right in rels:
        lines.append(f'  <relation id="{rid}"><member type="way" ref="{left}" role="left"/><member type="way" ref="{right}" role="right"/>'
                     '<tag k="type" v="lanelet"/></relation>')
    lines.append('</osm>')
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'm.osm'
        p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        m = exid.LaneMap(p)
    lane = m.lane
    checks = dict(chain=lane[10] == lane[11] == lane[12], diverge=len({lane[20], lane[21], lane[22]}) == 3,
                  merge=len({lane[30], lane[31], lane[32]}) == 3, counts=(m.diverges, m.merges) == (1, 1))
    print('diverge/merge', checks, m.summary())
    assert all(checks.values()), checks


def json_round(m):
    if m is None:
        return None
    return tuple((k, tuple(sorted((kk, round(vv, 6)) for kk, vv in v.items() if vv is not None)) if isinstance(v, dict) else round(v, 6))
                 for k, v in sorted(m.items()))


if __name__ == '__main__':
    main()
