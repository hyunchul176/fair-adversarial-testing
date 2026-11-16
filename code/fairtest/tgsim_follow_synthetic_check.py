"""Synthetic check of tgsim_follow (B35): three cars in one lane at known gaps and speeds, a fourth car in another lane
and a truck; the extractor must find the E->L1 pair and the E-L1-L2 triple with the constructed gaps, and nothing else.

  python -m fairtest.tgsim_follow_synthetic_check
"""
from __future__ import annotations

import numpy as np

from fairtest import tgsim_follow as F


def track(run, vid, s0, v, lane, kind='small-vehicle', n=60, length=4.8):
    f = np.arange(100, 100 + n)
    t = f / 10.
    return dict(run=run, vid=vid, sign=1, frame=f, t=t, s=s0 + v * (t - t[0]), n=np.zeros(n), v=np.full(n, float(v)),
                length=length, width=1.9, kind=kind, acc=False, lane=np.full(n, lane))


def main():
    tracks = {(1, 1): track(1, 1, 0., 25., 3),           # E
              (1, 2): track(1, 2, 40., 25., 3),          # L1: bumper gap 40 - 4.8 = 35.2 m (THW 1.41 s)
              (1, 3): track(1, 3, 85., 25.5, 3),         # L2: gap 45 - 4.8 = 40.2 m at t0 (THW 1.61 s)
              (1, 4): track(1, 4, 20., 25., 4),          # other lane: never a leader
              (1, 5): track(1, 5, -60., 25., 3, 'truck')}  # a truck behind E: not a follower
    pre = F.preceding(tracks)
    assert pre[(1, 1)][0][0] == (1, 2) and abs(pre[(1, 1)][0][2] - 35.2) < 1e-9
    assert pre[(1, 2)][0][0] == (1, 3) and abs(pre[(1, 2)][0][2] - 40.2) < 1e-9
    assert pre[(1, 4)][0] is None and pre[(1, 3)][0] is None
    pairs = F.snapshots(tracks, pre, 'SYN', False)
    trips = F.snapshots(tracks, pre, 'SYN', True)
    got = sorted((p['follower'], p['lead']) for p in pairs)
    assert got == [(1, 2), (2, 3)], got   # E->L1 and L1->L2 pairs (the truck is no follower, lane 4 has no leader)
    assert [(p['follower'], p['lead'], p['lead2']) for p in trips] == [(1, 2, 3)]
    p = next(p for p in pairs if p['follower'] == 1)
    assert abs(p['gap'] - 35.2) < 1e-9 and p['frame'] == 110 and abs(p['thw'] - 35.2 / 25.) < 1e-9  # 1 s into the window
    q = trips[0]
    assert abs(q['gap2'] - (40.2 + .5 * 1.)) < 1e-6 and abs(q['v_lead2'] - 25.5) < 1e-9  # L2 pulls away 0.5 m/s
    print('tgsim_follow synthetic check passed')


if __name__ == '__main__':
    main()
