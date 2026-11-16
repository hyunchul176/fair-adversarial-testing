"""Synthetic check of the MiTra extractor (no MiTra data is read): a tilted, gently curved road in UTM-like coordinates,
two directions of four mainline lanes (0-3 and 4-7) at 30 frames/s, written in the MiTra CSV layout inside a campaign
archive. Planted: one car cut-in from lane 1 to lane 2 in front of a follower (must be found with the true p10 speeds,
bumper gap and 10-90 duration), a truck pair (non_car_pair), a car pair whose generator jumps 0.8 m sideways in one
frame (teleport_jump) and a ramp merge 10 -> 3 (nonadjacent_or_ramp_lane).

  python -m fairtest.mitra_synthetic_check
"""
from __future__ import annotations

import csv
import io
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from fairtest import mitra

FPS, LANE_W = 30., 3.5
ANGLE = math.radians(30.)          # road axis tilted in UTM
CURV = 2e-5                        # lateral bend n0(s) = CURV * s^2 (about 4 m over 450 m)
ORIGIN = np.array([514000., 5035000.])
COLS = ['Vehicle_ID', 'Vehicle_type', 'Time [s]', 'x [m]', 'y [m]', 'Speed [km/h]', 'Lon. Acc. [ms-2]', 'Lat. Acc. [ms-2]',
        'Angle [rad]', 'Vehicle_length [m]', 'Vehicle_width [m]', 'Lane', 'Leader_ID', 'Follower_ID', 'Left_Leader_ID',
        'Left_Follower_ID', 'Right_Leader_ID', 'Right_Follower_ID']


def to_utm(s, n, sign=1):
    """Road (s along the +1 direction, n left of it) -> UTM; the -1 direction travels towards decreasing s."""
    u = np.array([math.cos(ANGLE), math.sin(ANGLE)])
    left = np.array([-u[1], u[0]])
    p = ORIGIN + np.outer(s, u) + np.outer(n + CURV * s ** 2, left)
    return p[:, 0], p[:, 1]


def lane_n(lane):
    if lane in (0, 1, 2, 3):
        return (1.5 - lane) * LANE_W + 10.     # +1 direction, lane 0 leftmost
    if lane in (4, 5, 6, 7):
        return -(1.5 - (lane - 4)) * LANE_W - 10.
    return (lane - 13) * LANE_W                # ramps (10 -> right of lane 3)


def quintic(u):
    u = np.clip(u, 0., 1.)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def vehicle(vid, t, s, n, v, lane_of, kind='Car', length=4.6, width=1.9, sign=1):
    x, y = to_utm(s, n, sign)
    return [(vid, kind, float(tt), float(xx), float(yy), float(v[k]) * 3.6, int(lane_of(k)), length, width)
            for k, (tt, xx, yy) in enumerate(zip(t, x, y))]


def build():
    t = np.arange(0., 30., 1 / FPS)
    rows = []
    for j, lane in enumerate((0, 1, 2, 3)):  # background, +1 direction
        for m in range(3):
            v0 = 26. + j
            s = -250. + 150. * m + v0 * t
            rows += vehicle(100 + 10 * j + m, t, s, np.full_like(t, lane_n(lane)), np.full_like(t, v0), lambda k, L=lane: L)
    for j, lane in enumerate((4, 5, 6, 7)):  # background, -1 direction (decreasing s)
        for m in range(3):
            v0 = 25. + j
            s = 700. - 150. * m - v0 * t
            rows += vehicle(200 + 10 * j + m, t, s, np.full_like(t, lane_n(lane)), np.full_like(t, v0), lambda k, L=lane: L, sign=-1)
    # the cut-in: generator lane 1 -> 2 between t = 10 and 14 s
    vg = np.full_like(t, 27.)
    sg = 40. + 27. * t
    ug = quintic((t - 10.) / 4.)
    flip = int(np.argmax(ug >= .5))
    rows += vehicle(1, t, sg, lane_n(1) + (lane_n(2) - lane_n(1)) * ug, vg, lambda k: 1 if k < flip else 2)
    # follower in lane 2, 20 m bumper gap behind the generator at t = 10 s, 29 m/s
    sf = (40. + 27. * 10.) - 4.6 - 20. - 29. * 10. + 29. * t
    rows += vehicle(2, t, sf, np.full_like(t, lane_n(2)), np.full_like(t, 29.), lambda k: 2)
    # a truck pair doing the same in lanes 2 -> 3 (non_car_pair)
    st_ = -120. + 26. * t
    ut = quintic((t - 16.) / 5.)
    rows += vehicle(3, t, st_, lane_n(2) + (lane_n(3) - lane_n(2)) * ut, np.full_like(t, 26.), lambda k: 2 if ut[k] < .5 else 3,
                    kind='Heavy Vehicle', length=12., width=2.5)
    rows += vehicle(4, t, st_ - 12. - 25., np.full_like(t, lane_n(3)), np.full_like(t, 27.), lambda k: 3)
    # a car pair whose generator jumps 0.8 m sideways in one frame during the manoeuvre (teleport_jump)
    sj = 200. + 28. * t
    uj = quintic((t - 18.) / 4.)
    nj = lane_n(0) + (lane_n(1) - lane_n(0)) * uj
    nj[int(19.5 * FPS)] += .8  # one-frame tracking jump (the authors' teleport rule flags > 0.5 m across in one step)
    rows += vehicle(5, t, sj, nj, np.full_like(t, 28.), lambda k: 0 if uj[k] < .5 else 1)
    rows += vehicle(6, t, sj - 4.6 - 30., np.full_like(t, lane_n(1)), np.full_like(t, 29.), lambda k: 1)
    # a ramp merge 10 -> 3 (nonadjacent_or_ramp_lane)
    sr = 100. + 24. * t
    ur = quintic((t - 20.) / 4.)
    rows += vehicle(7, t, sr, lane_n(10) + (lane_n(3) - lane_n(10)) * ur, np.full_like(t, 24.), lambda k: 10 if ur[k] < .5 else 3)
    return rows


def write_zip(path, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(COLS)
    for vid, kind, tt, x, y, v, lane, length, width in rows:
        w.writerow([vid, kind, f'{tt:.4f}', f'{x:.3f}', f'{y:.3f}', f'{v:.2f}', '0', '0', '0', length, width, lane, '', '', '', '', '', ''])
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('Data_T1/T1_DAll.csv', buf.getvalue())


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'Data_T1.zip'
        write_zip(path, build())
        events, rec = mitra.extract(mitra.load(path, 1), 1)
    ok = [e for e in events if e.get('eligible')]
    print('record', rec)
    print('events', [(e['event_id'], e['exclusion']) for e in events])
    assert len(ok) == 1, 'exactly the car cut-in must be eligible'
    e = ok[0]
    p10 = e['metrics']['p10']
    lo, hi = 0., .5
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if quintic(m) < .1 else (lo, m)
    t10 = 10. + 4. * lo
    gap_true = (40. + 27. * t10 - 4.6 / 2) - ((40. + 27. * 10.) - 4.6 - 20. - 29. * 10. + 29. * t10 + 4.6 / 2)
    checks = dict(vehicle=e['vehicle'] == 1, follower=e['follower'] == 2,
                  generator_speed=abs(p10['generator_speed'] - 27.) < .05, target_speed=abs(p10['target_speed'] - 29.) < .05,
                  gap=abs(p10['gap'] - gap_true) < .15, duration=abs(e['metrics']['duration_10_90'] - 4. * (1 - 2 * lo)) < .1,
                  truck=any(x['vehicle'] == 3 and x['exclusion'] == 'non_car_pair' for x in events),
                  teleport=any(x['vehicle'] == 5 and x['exclusion'] == 'teleport_jump' for x in events),
                  ramp=any(x['vehicle'] == 7 and x['exclusion'] == 'nonadjacent_or_ramp_lane' for x in events),
                  directions=rec['directions'] == [-1, 1])
    print('p10', p10, 'gap_true', round(gap_true, 3), 'duration', e['metrics']['duration_10_90'])
    print('checks', checks)
    assert all(checks.values()), checks
    cases, ledger = mitra.select_cohort(events)
    print('selected cases', len(cases))
    assert len(cases) == 1
    print('SYNTHETIC CHECK PASSED')


if __name__ == '__main__':
    main()
