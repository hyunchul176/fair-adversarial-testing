"""B35: steady car-following snapshots from TGSIM I-294 (external replication of B30 deceleration and B34 cut-out).

The TGSIM loading, road frame and track preparation of tgsim.py (B28, synthetic check passed) are reused unchanged.
Preceding car = the nearest car ahead in the same lane label at the same frame (bumper gap
s_lead - l_lead/2 - (s_fol + l_fol/2) > 0). Snapshot rules as the highD extractors:
  pairs (deceleration, dec_cases): both cars >= 15 m/s, THW 0.5-3.0 s, |dv| < 3 m/s, the same preceding car
     throughout >= 3 s; the state 1 s into the first qualifying window; one snapshot per pair.
  triples (cut-out, cutout_cases): as pairs plus the lead's preceding car L2 constant, >= 15 m/s,
     THW(L1->L2) 0.5-4.0 s, |v2 - v1| < 3 m/s.
Only 'small-vehicle' tracks. Selection per recording (dataset-run): sha256(SALT + id) ranking, participant-disjoint
greedy, at most ceil(60 / recordings) per recording. No simulation, no outcome.

  python -m fairtest.tgsim_follow TGSIM_DIR OUT_DEC.json OUT_CUTOUT.json
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from fairtest import tgsim as T

TARGET = 60
SALT = 'acarl-b35-tgsim-follow-v1:'
WINDOW_S, OFFSET_S = 3., 1.


def preceding(tracks):
    """{track key: per sample (preceding key, its sample index, bumper gap) or None}: the next car ahead along s in the
    same main-line lane label at the same frame (None if the bodies overlap)."""
    groups = {}
    for key, tr in tracks.items():
        for i, f in enumerate(tr['frame']):
            lane = int(tr['lane'][i])
            if lane in T.LANES:
                groups.setdefault((tr['run'], int(f), lane), []).append((float(tr['s'][i]), key, i))
    out = {key: [None] * len(tr['frame']) for key, tr in tracks.items()}
    for members in groups.values():
        members.sort()
        for (s0, k0, i0), (s1, k1, i1) in zip(members, members[1:]):
            gap = s1 - tracks[k1]['length'] / 2 - (s0 + tracks[k0]['length'] / 2)
            if gap > 0:
                out[k0][i0] = (k1, i1, gap)
    return out


def snapshots(tracks, pre, dataset, triples):
    win, off = int(WINDOW_S * T.FPS), int(OFFSET_S * T.FPS)
    small = lambda k: k is not None and tracks[k]['kind'] == 'small-vehicle'  # noqa: E731
    out = []
    for key, tr in tracks.items():
        if tr['kind'] != 'small-vehicle':
            continue
        run_len, ident, done = 0, None, set()
        for i in range(len(tr['frame'])):
            p = pre[key][i]
            good, this = False, None
            if p is not None and small(p[0]) and p[0] not in done:
                lk, j, gap = p
                ld = tracks[lk]
                v0, v1 = float(tr['v'][i]), float(ld['v'][j])
                good = v0 >= 15 and v1 >= 15 and .5 <= gap / v0 <= 3. and abs(v1 - v0) < 3
                this = (lk,)
                if good and triples:
                    q = pre[lk][j]
                    good = q is not None and small(q[0])
                    if good:
                        l2, k2, gap2 = q
                        v2 = float(tracks[l2]['v'][k2])
                        good = v2 >= 15 and .5 <= gap2 / v1 <= 4. and abs(v2 - v1) < 3
                        this = (lk, l2)
            run_len = run_len + 1 if good and this == ident else (1 if good else 0)
            ident = this if good else None
            if run_len >= win:
                s = i - run_len + 1 + off
                lk, j, gap = pre[key][s]
                ld = tracks[lk]
                snap = dict(snapshot_id=f'tgsim-{dataset}-r{tr["run"]}:{tr["vid"]}:{ld["vid"]}:{int(tr["frame"][s])}',
                            recording=f'{dataset}-r{tr["run"]}', follower=tr['vid'], lead=ld['vid'], frame=int(tr['frame'][s]),
                            gap=float(gap), v_follower=float(tr['v'][s]), v_lead=float(ld['v'][j]),
                            length_follower=tr['length'], length_lead=ld['length'], acc_follower=tr['acc'], acc_lead=ld['acc'])
                snap['thw'] = snap['gap'] / snap['v_follower']
                if triples:
                    l2, k2, gap2 = pre[lk][j]
                    snap.update(lead2=tracks[l2]['vid'], gap2=float(gap2), v_lead2=float(tracks[l2]['v'][k2]),
                                length_lead2=tracks[l2]['length'], snapshot_id=snap['snapshot_id'] + f':{tracks[l2]["vid"]}')
                out.append(snap)
                done.add(lk)
                run_len, ident = 0, None
    return out


def select(cands):
    recs = sorted({c['recording'] for c in cands})
    cap = math.ceil(TARGET / max(1, len(recs)))
    cases = []
    for rec in recs:
        used, n = set(), 0
        for c in sorted((c for c in cands if c['recording'] == rec),
                        key=lambda c: hashlib.sha256((SALT + c['snapshot_id']).encode()).hexdigest()):
            who = {c['follower'], c['lead']} | ({c['lead2']} if 'lead2' in c else set())
            if n >= cap or who & used:
                continue
            used |= who
            cases.append(dict(c, partition='tgsim_i294'))
            n += 1
    for i, c in enumerate(cases):
        c['case_id'] = i
    return cases


def main(src, out_dec, out_cut):
    src = Path(src)
    pairs, trips = [], []
    for dataset, name in T.FILES:
        df = T.load(src / name)
        tracks = T.tracks_of(df, T.frames(df))
        pre = preceding(tracks)
        pairs += snapshots(tracks, pre, dataset, False)
        trips += snapshots(tracks, pre, dataset, True)
        print(json.dumps(dict(dataset=dataset, tracks=len(tracks), pairs=len(pairs), triples=len(trips))), flush=True)
    for cands, out in ((pairs, out_dec), (trips, out_cut)):
        cases = select(cands)
        Path(out).write_text(json.dumps(dict(salt=SALT, candidates=len(cands), cases=cases), indent=0), encoding='utf-8')
        print(json.dumps(dict(out=str(out), candidates=len(cands), selected=len(cases),
                              recordings=len({c['recording'] for c in cases}))), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:4])
