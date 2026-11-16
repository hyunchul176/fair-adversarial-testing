"""Synthetic check of the HIGH-SIM extractor (no HIGH-SIM data is read): one travel direction with the ramp (lane 0) and
three 12 ft mainline lanes (1-3, centres at local X 30, 18 and 6 ft, drifting 1 ft per 1,000 ft along the road), 30
frames/s from frame 138000, written in the HIGH-SIM layout (15 columns, no header, feet, global time at float32
precision, class 2 for every vehicle). Planted:
  * a car cut-in from lane 2 to lane 1 in front of a follower (must be found with the true p10 speeds, bumper gap and
    10-90 duration, in the first 5-minute block);
  * a truck (70 ft) doing the same in lanes 2 -> 3 (non_car_pair);
  * a car pair whose generator jumps 0.8 m sideways in one frame (teleport_jump);
  * a ramp merge 0 -> 1 (nonadjacent_or_ramp_lane);
  * a slow car cut-in at 12 m/s, 310 s later (below_speed_population, recording block 1).

  python -m fairtest.highsim_synthetic_check
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from fairtest import highsim

FPS, FT = 30., .3048
F0 = 138000
CENTRE_FT = {0: 54., 1: 30., 2: 18., 3: 6.}
DRIFT = 1e-3  # ft of lateral drift per ft along the road


def x_ft(lane_pos_ft, y_ft):
    return lane_pos_ft + DRIFT * y_ft


def quintic(u):
    u = np.clip(u, 0., 1.)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def vehicle(vid, frames, s_m, lat_ft, v_ms, lane_of, length_ft=15., width_ft=6.):
    y = s_m / FT
    x = x_ft(lat_ft, y)
    gtime = np.float32(64000.) + (frames - F0).astype(np.float32) / np.float32(FPS)
    return [(vid, float(gtime[k]), int(frames[k]), float(x[k]), float(y[k]), 28.1, -82.4, width_ft, length_ft, 2,
             float(v_ms[k] / FT), 0., int(lane_of(k)), -1, -1.) for k in range(len(frames))]


def build():
    rows = []
    frames = F0 + np.arange(int(30 * FPS))
    t = (frames - F0) / FPS
    for j, lane in enumerate((1, 2, 3)):  # background traffic covering the road in every mainline lane (lane 1 kept
        for m in range(6):                 # clear of the planted follower: nearest background car 147 m behind)
            v0 = 26. + j
            s = (-500. + 200. * m if lane == 1 else -200. + 120. * m) + v0 * t
            rows += vehicle(100 + 10 * j + m, frames, s, np.full_like(t, CENTRE_FT[lane]), np.full_like(t, v0), lambda k, L=lane: L)
    # the cut-in: generator lane 2 -> 1 between t = 10 and 14 s
    vg = np.full_like(t, 27.)
    sg = 40. + 27. * t
    ug = quintic((t - 10.) / 4.)
    flip = int(np.argmax(ug >= .5))
    rows += vehicle(1, frames, sg, CENTRE_FT[2] + (CENTRE_FT[1] - CENTRE_FT[2]) * ug, vg, lambda k: 2 if k < flip else 1)
    # follower in lane 1, 20 m bumper gap behind the generator at t = 10 s, 29 m/s
    sf = (40. + 27. * 10.) - 15. * FT - 20. - 29. * 10. + 29. * t
    rows += vehicle(2, frames, sf, np.full_like(t, CENTRE_FT[1]), np.full_like(t, 29.), lambda k: 1)
    # a 70 ft truck doing a cut-in 2 -> 3 (non_car_pair)
    st = -120. + 26. * t
    ut = quintic((t - 16.) / 5.)
    rows += vehicle(3, frames, st, CENTRE_FT[2] + (CENTRE_FT[3] - CENTRE_FT[2]) * ut, np.full_like(t, 26.), lambda k: 2 if ut[k] < .5 else 3,
                    length_ft=70., width_ft=8.5)
    rows += vehicle(4, frames, st - 70. * FT / 2 - 15. * FT / 2 - 25., np.full_like(t, CENTRE_FT[3]), np.full_like(t, 27.), lambda k: 3)
    # a car pair whose generator jumps 0.8 m sideways in one frame during the manoeuvre (teleport_jump)
    sj = 200. + 28. * t
    uj = quintic((t - 18.) / 4.)
    lj = CENTRE_FT[3] + (CENTRE_FT[2] - CENTRE_FT[3]) * uj
    lj[int(19.5 * FPS)] += .8 / FT
    rows += vehicle(5, frames, sj, lj, np.full_like(t, 28.), lambda k: 3 if uj[k] < .5 else 2)
    rows += vehicle(6, frames, sj - 15. * FT - 30., np.full_like(t, CENTRE_FT[2]), np.full_like(t, 29.), lambda k: 2)
    # a ramp merge 0 -> 1 (nonadjacent_or_ramp_lane)
    sr = 100. + 24. * t
    ur = quintic((t - 20.) / 4.)
    rows += vehicle(7, frames, sr, CENTRE_FT[0] + (CENTRE_FT[1] - CENTRE_FT[0]) * ur, np.full_like(t, 24.), lambda k: 0 if ur[k] < .5 else 1)
    # a slow car cut-in 1 -> 2 at 12 m/s, 310 s later, with its own background (below_speed_population, block 1)
    frames2 = F0 + int(310 * FPS) + np.arange(int(30 * FPS))
    t2 = (frames2 - frames2[0]) / FPS
    for j, lane in enumerate((1, 2)):
        for m in range(4):
            s = -100. + 90. * m + 12. * t2
            rows += vehicle(300 + 10 * j + m, frames2, s, np.full_like(t2, CENTRE_FT[lane]), np.full_like(t2, 12.), lambda k, L=lane: L)
    s8 = 30. + 12. * t2
    u8 = quintic((t2 - 10.) / 4.)
    rows += vehicle(8, frames2, s8, CENTRE_FT[1] + (CENTRE_FT[2] - CENTRE_FT[1]) * u8, np.full_like(t2, 12.), lambda k: 1 if u8[k] < .5 else 2)
    rows += vehicle(9, frames2, 30. + 12. * 10. - 15. * FT - 15. - 12. * 10. + 12. * t2, np.full_like(t2, CENTRE_FT[2]), np.full_like(t2, 12.),
                    lambda k: 2)
    return rows


def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(','.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in r) + '\n')


def main():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'HIGH_SIM_synthetic.csv'
        write_csv(path, build())
        events, rec = highsim.extract(highsim.load(path))
    ok = [e for e in events if e.get('eligible')]
    print('record', rec)
    print('events', [(e['event_id'], e['recording'], e['exclusion']) for e in events])
    assert len(ok) == 1, 'exactly the car cut-in must be eligible'
    e = ok[0]
    p10 = e['metrics']['p10']
    lo, hi = 0., .5
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if quintic(m) < .1 else (lo, m)
    t10 = 10. + 4. * lo
    L = 15. * FT
    gap_true = (40. + 27. * t10 - L / 2) - ((40. + 27. * 10.) - L - 20. - 29. * 10. + 29. * t10 + L / 2)
    checks = dict(vehicle=e['vehicle'] == 1, follower=e['follower'] == 2, block0=e['recording'] == 'i75-b00',
                  generator_speed=abs(p10['generator_speed'] - 27.) < .05, target_speed=abs(p10['target_speed'] - 29.) < .05,
                  gap=abs(p10['gap'] - gap_true) < .15, duration=abs(e['metrics']['duration_10_90'] - 4. * (1 - 2 * lo)) < .1,
                  truck=any(x['vehicle'] == 3 and x['exclusion'] == 'non_car_pair' for x in events),
                  teleport=any(x['vehicle'] == 5 and x['exclusion'] == 'teleport_jump' for x in events),
                  ramp=any(x['vehicle'] == 7 and x['exclusion'] == 'nonadjacent_or_ramp_lane' for x in events),
                  slow=any(x['vehicle'] == 8 and x['exclusion'] == 'below_speed_population' and x['recording'] == 'i75-b01' for x in events),
                  lanes=rec['lane_centre_keys'] == [1, 2, 3])
    print('p10', p10, 'gap_true', round(gap_true, 3), 'duration', e['metrics']['duration_10_90'])
    print('checks', checks)
    assert all(checks.values()), checks
    cases, ledger = highsim.select_cohort(events)
    print('selected cases', len(cases))
    assert len(cases) == 1
    cap, highsim.MAX_CASES = highsim.MAX_CASES, 0
    try:
        cases0, ledger0 = highsim.select_cohort(events)
    finally:
        highsim.MAX_CASES = cap
    assert not cases0 and any(r['reason'] == 'cohort_cap' for r in ledger0), 'the cohort cap must bind'
    print('SYNTHETIC CHECK PASSED')


if __name__ == '__main__':
    main()
