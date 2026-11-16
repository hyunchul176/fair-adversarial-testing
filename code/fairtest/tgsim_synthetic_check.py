"""Synthetic check of the TGSIM extractor (no TGSIM data is read): a tilted, gently curved 4-lane road in image
coordinates, written in the export format (quoted values, thousands separators), with one cut-in from lane 3 to lane 4
in front of a follower, a lane-keeping neighbour, a truck pair and a southbound vehicle. The extractor must find the
cut-in with the true p10 speeds, bumper gap and 10-90 duration, and reject the truck pair.

  python -m fairtest.tgsim_synthetic_check
"""
from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

from fairtest import tgsim

LANE_W = 3.6
ANGLE = math.radians(25.)          # road axis tilted in the image
CURV = 2e-5                        # lateral bend n0(s) = CURV * s^2 (about 8 m over 600 m)


def to_image(s, n):
    """Road (s along travel, n left of travel) -> image (x right, y down), origin offset."""
    u = np.array([math.cos(ANGLE), math.sin(ANGLE)])
    left = np.array([u[1], -u[0]])  # left of travel in y-down image coordinates
    p = np.array([2500., 900.]) + np.outer(s, u) + np.outer(n + CURV * s ** 2, left)
    return p[:, 0], p[:, 1]


def lane_n(lane):
    return (3.5 - lane) * LANE_W   # lanes 2..5 left to right (lane 2 leftmost)


def quintic(u):
    u = np.clip(u, 0., 1.)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def write(path, rows):
    cols = ['id', 'time', 'xloc_kf', 'yloc_kf', 'lane_kf', 'speed_kf', 'acceleration_kf', 'length_smoothed',
            'width_smoothed', 'type_most_common', 'acc', 'run_index']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(cols)
        for r in rows:
            w.writerow([f'{r[0]:,}', f'{r[1]:.1f}', f'{r[2]:,.3f}', f'{r[3]:,.3f}', str(r[4]), f'{r[5]:.2f}', '0',
                        f'{r[6]:.2f}', f'{r[7]:.2f}', r[8], r[9], str(r[10])])


def vehicle(vid, t, s, n, v, lane_of, length=4.6, width=1.9, kind='small-vehicle', acc='no', run=1):
    x, y = to_image(s, n)
    return [(vid, float(tt), float(xx), float(yy), int(lane_of(k)), float(v[k]), length, width, kind, acc, run)
            for k, (tt, xx, yy) in enumerate(zip(t, x, y))]


def build():
    t = np.arange(0., 40., .1)
    rows = []
    # background traffic so that every lane has centre samples along the whole section
    for j, lane in enumerate((2, 3, 4, 5)):
        for m in range(4):
            v0 = 26. + j
            s = -300. + 150. * m + v0 * t
            rows += vehicle(100 + 10 * j + m, t, s, np.full_like(t, lane_n(lane)), np.full_like(t, v0), lambda k, L=lane: L)
    # the cut-in: generator in lane 3 moves to lane 4 between t = 15 and t = 19 s (duration 4 s)
    vg = np.full_like(t, 27.)
    sg = 60. + 27. * t
    ug = quintic((t - 15.) / 4.)
    ng = lane_n(3) + (lane_n(4) - lane_n(3)) * ug
    flip = np.argmax(ug >= .5)
    rows += vehicle(1, t, sg, ng, vg, lambda k: 3 if k < flip else 4)
    # follower in lane 4, 20 m bumper gap behind the generator at t = 15 s, 29 m/s
    vf = np.full_like(t, 29.)
    sf = (60. + 27. * 15.) - 4.6 - 20. - (29. * 15.) + 29. * t
    rows += vehicle(2, t, sf, np.full_like(t, lane_n(4)), vf, lambda k: 4, acc='yes')
    # a truck pair doing the same manoeuvre in lanes 4 -> 5 (must be rejected as non_car_pair)
    st_ = -150. + 26. * t
    ut = quintic((t - 22.) / 5.)
    rows += vehicle(3, t, st_, lane_n(4) + (lane_n(5) - lane_n(4)) * ut, np.full_like(t, 26.),
                    lambda k: 4 if ut[k] < .5 else 5, length=12., width=2.5, kind='large-vehicle')
    rows += vehicle(4, t, st_ - 12. - 25., np.full_like(t, lane_n(5)), np.full_like(t, 27.), lambda k: 5)
    # a southbound vehicle (opposite direction, negative lane id), lane keeping
    ss = 1500. - 28. * t
    x, y = to_image(ss, np.full_like(t, -12.))
    rows += [(9, float(tt), float(xx), float(yy), -3, 28., 4.6, 1.9, 'small-vehicle', 'no', 1) for tt, xx, yy in zip(t, x, y)]
    return rows


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'synthetic.csv'
        write(path, build())
        events, rec = tgsim.extract(tgsim.load(path), 'SYN')
    ok = [e for e in events if e.get('eligible')]
    print('record', rec)
    print('eligible', len(ok), [(e['event_id'], e['exclusion']) for e in events])
    assert len(ok) == 1, 'exactly the car cut-in must be eligible'
    e = ok[0]
    p10 = e['metrics']['p10']
    # truth: p10 crossing at u = 0.1 of the quintic between 15 and 19 s
    lo, hi = 0., .5
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if quintic(mid) < .1 else (lo, mid)
    t10 = 15. + 4. * lo
    gap_true = (60. + 27. * t10 - 4.6 / 2) - ((60. + 27. * 15.) - 4.6 - 20. - 29. * 15. + 29. * t10 + 4.6 / 2)
    checks = dict(generator_speed=abs(p10['generator_speed'] - 27.) < .05, target_speed=abs(p10['target_speed'] - 29.) < .05,
                  gap=abs(p10['gap'] - gap_true) < .15, duration=abs(e['metrics']['duration_10_90'] - 4. * (1 - 2 * lo)) < .1,
                  follower=e['follower'] == 2, acc_follower=e['acc_follower'] is True,
                  truck_rejected=any(x['vehicle'] == 3 and x['exclusion'] == 'non_car_pair' for x in events))
    print('p10', p10, 'gap_true', round(gap_true, 3), 'duration', e['metrics']['duration_10_90'])
    print('checks', checks)
    assert all(checks.values()), checks
    cases, ledger = tgsim.select_cohort(events)
    print('selected cases', len(cases), 'initial state source', cases[0]['initial_state_source'] if cases else None)
    assert len(cases) == 1
    print('SYNTHETIC CHECK PASSED')


if __name__ == '__main__':
    main()
