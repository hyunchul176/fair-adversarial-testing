"""B54: Autoware Universe as a SUT. The client side (our environment) of fairtest/aw_bridge.py, which drives one Autoware
planning-simulator instance (perception = ground-truth dummy objects -> Autoware's tracker and map-based prediction;
planning and control = Autoware; ego = Autoware's simple planning simulator).

Frames: highway-env (x forward, y to the right; adversary lane y = 0, SUT lane y = 4) -> Autoware map (x forward, y left;
SUT lane y = 0, adversary lane y = 4): map_x = hw_x + shift, map_y = 4 - hw_y, map_yaw = -hw_heading. The lanes are 4 m
wide, as in highway-env, so lateral positions map without scaling. The SUT's centre is its box centre (base_link +
centre offset along the heading).

Episodes (one SUT query each), both with a PRE_S pre-roll at the case speed:
  replay(bridge, row)       a fixed adversary trajectory (a stored attack or a refinement plan's highway-env trajectory);
                            during the pre-roll the adversary holds its t = 0 lateral position and speed, placed relative
                            to the ego so that the case geometry holds exactly at t = 0 (shift = ego centre - sut0 x)
  policy_episode(bridge, case, model, shield)
                            the RL adversary closed-loop (carla_rl.episode with the SUT in Autoware): a highway-env twin
                            computes the adversary's motion from the policy's decisions; the window shield checks the
                            highway-env CC counterfactual; the policy observes the twin generator and the Autoware ego
Contact = oriented boxes overlap (ego: Autoware vehicle dimensions; adversary: 5 m x 2 m, the highway-env body), checked
after every 0.1 s tick. Fairness is judged afterwards in highway-env: the realised adversary actions (policy) or plan are
replayed against the R157 CC, exactly as for every other SUT.

  python -m fairtest.aw_sut replay OUT.jsonl TRAJ.jsonl --port 5481
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import time

X_START, X_GOAL, PRE_S, DT, HORIZON, LANE = 200., 2400., 3., .1, 120, 4.
ADV_L, ADV_W = 5., 2.


class Bridge:
    def __init__(self, port, host='127.0.0.1', timeout=120.):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.f = self.sock.makefile('rw')
        self.info = self.call('info')

    def send(self, cmd, **kw):
        self.f.write(json.dumps(dict(cmd=cmd, **kw)) + '\n')
        self.f.flush()

    def recv(self):
        r = json.loads(self.f.readline())
        if not r['ok']:
            raise RuntimeError(r['error'])
        return r['result']

    def call(self, cmd, **kw):
        self.send(cmd, **kw)
        return self.recv()


def _poly(cx, cy, yaw, length, width):
    from shapely.geometry import Polygon
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(cx + c * dx - s * dy, cy + s * dx + c * dy) for dx, dy in
           ((length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2))]
    return Polygon(pts)


class Episode:
    """Bookkeeping shared by both episode kinds: frames, contact, gap, trace."""

    def __init__(self, br, sut0, cruise):
        self.br, self.sut0, self.cruise = br, sut0, cruise
        info = br.info
        self.off, self.L, self.W = info['centre_ahead_of_base_link'], info['length'], info['width']
        self.v0 = sut0['vx']
        self.shift = None
        self.trace, self.late = [], []
        self.hit, self.t_hit, self.contact_dx, self.min_gap, self.min_clear = False, None, None, math.inf, math.inf

    def centre(self, e):
        return e['x'] + self.off * math.cos(e['yaw']), e['y'] + self.off * math.sin(e['yaw'])

    def reset(self):
        base_x = X_START - PRE_S * self.v0 - self.off
        r = self.br.call('reset', ego=[base_x, 0., 0., self.v0], goal=[X_GOAL, 0., 0.], vmax=self.cruise)
        self.ego = r['ego']
        return r

    def adv_obj(self, g):
        """highway-env generator state (dict x, y, vx, vy) -> bridge object [x, y, yaw, vx, vy, L, W] in the map."""
        x = g['x'] + self.shift
        y = LANE - g['y']
        vx, vy = g['vx'], -g['vy']
        return [x, y, math.atan2(vy, max(vx, .1)), vx, vy, ADV_L, ADV_W]

    def sut_hw(self, e):
        """Autoware ego -> highway-env SUT state (x, y, vx) for observations and the shield window."""
        cx, cy = self.centre(e)
        return cx - self.shift, LANE - cy, e['v'] * math.cos(e['yaw'])

    def check(self, t, e, g):
        cx, cy = self.centre(e)
        ax, ay = g['x'] + self.shift, LANE - g['y']
        ayaw = math.atan2(-g['vy'], max(g['vx'], .1))
        pe, pa = _poly(cx, cy, e['yaw'], self.L, self.W), _poly(ax, ay, ayaw, ADV_L, ADV_W)
        clear = float(pe.distance(pa))
        self.min_clear = min(self.min_clear, clear)
        same_lane = abs(ay - cy) < (ADV_W + self.W) / 2
        gap = (ax - cx) - (ADV_L + self.L) / 2 if same_lane and ax > cx else None
        if gap is not None:
            self.min_gap = min(self.min_gap, gap)
        self.trace.append(dict(t=round(t, 2), ego=[round(cx - self.shift, 3), round(LANE - cy, 3), round(e['v'], 3),
                                                   None if e['a'] is None else round(e['a'], 3),
                                                   None if e['acc_cmd'] is None else round(e['acc_cmd'], 3)],
                               adv=[round(g['x'], 3), round(g['y'], 3), round(g['vx'], 3), round(g['vy'], 3)],
                               gap=None if gap is None else round(gap, 3), clear=round(clear, 3), late=e.get('late')))
        if clear <= 0. and not self.hit:
            self.hit, self.t_hit, self.contact_dx = True, round(t, 1), round(ax - cx, 2)
        return self.hit

    def preroll(self, g0):
        """PRE_S seconds with the adversary at its t = 0 lateral position and speed, placed relative to the ego."""
        rel0 = g0['x'] - self.sut0['x']
        e = self.ego
        for k in range(int(round(PRE_S / DT))):
            t = -PRE_S + k * DT
            cx, _ = self.centre(e)
            self.shift = cx - self.sut0['x'] - self.v0 * t  # provisional: as if the ego ran exactly at v0
            g = dict(x=self.sut0['x'] + self.v0 * t + rel0 + (g0['vx'] - self.v0) * t, y=g0['y'], vx=g0['vx'], vy=0.)
            e = self.br.call('step', obj=self.adv_obj(g))
            self.late.append(e['late'])
        cx, _ = self.centre(e)
        self.shift = cx - self.sut0['x']  # t = 0: the recorded frame aligned to the ego's actual position
        self.ego = e
        return e

    def result(self, **extra):
        e0 = self.trace[0] if self.trace else None
        return dict(contact=self.hit, t_contact=self.t_hit, contact_dx=self.contact_dx,
                    min_gap=None if math.isinf(self.min_gap) else round(self.min_gap, 3), min_clearance=round(self.min_clear, 3),
                    sut_min_acc=min((r['ego'][3] for r in self.trace if r['ego'][3] is not None), default=None),
                    v0_error=None if e0 is None else round(e0['ego'][2] - self.v0, 3), late_max=round(max(self.late), 4) if self.late else None,
                    late_mean=round(sum(self.late) / len(self.late), 4) if self.late else None, trace=self.trace, **extra)


def replay(br, row, tail_s=.5):
    """A fixed adversary trajectory (row: sut0, cruise, trajectory with t = 0 .. 12 s at 0.1 s) against Autoware."""
    ep = Episode(br, row['sut0'], row['cruise'])
    tr = row['trajectory']
    r = ep.reset()
    ep.preroll(tr[0])
    after = 0
    for k in range(HORIZON):
        e = br.call('step', obj=ep.adv_obj(tr[k]))
        ep.late.append(e['late'])
        if ep.check((k + 1) * DT, e, tr[k + 1]):
            after += 1
            if after > int(tail_s / DT):
                break
    br.call('clear')
    return ep.result(reset=r['timings'])


def main(a):
    br = Bridge(a.port)
    rows = [json.loads(l) for l in open(a.traj, encoding='utf-8') if l.strip()]
    with open(a.out, 'a', encoding='utf-8') as f:
        for row in rows:
            t0 = time.time()
            try:
                res = replay(br, row)
            except Exception as e:  # recorded, never dropped
                res = dict(error=f'{type(e).__name__}: {e}')
            res.update(case_id=row.get('case_id'), sut=row.get('sut'), source=row.get('source'), hwenv=row.get('hwenv'),
                       seconds=round(time.time() - t0, 1))
            f.write(json.dumps(res) + '\n')
            f.flush()
            print(json.dumps({k: res.get(k) for k in ('case_id', 'sut', 'contact', 't_contact', 'min_gap', 'sut_min_acc', 'v0_error',
                                                      'late_max', 'seconds', 'error')}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('replay')
    p.add_argument('out')
    p.add_argument('traj')
    p.add_argument('--port', type=int, default=5481)
    main(ap.parse_args())
