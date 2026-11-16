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
STALL_S, AUTONOMOUS = .5, 2  # planning output older than 0.5 s = a stalled planner; OperationModeState.AUTONOMOUS


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
    """Bookkeeping shared by both episode kinds: frames, contact, gap, trace.

    B58 options (defaults = the B54/B55 harness): latency_s delays the adversary object Autoware perceives by that many
    seconds (the map-frame measurement of `lag` ticks earlier, extrapolated to now at its own velocity; contact and the
    attacker's own observation use the true state); v0_max makes a start whose ego speed at t = 0 is off by more than v0_max m/s an unhealthy episode
    (an infrastructure failure, restarted and run again by aw_run), after B55's slow starts.
    B60 options: noise = (sd along the lane m, sd across m, sd of the speed m/s) adds independent Gaussian errors to each
    published object; dropout = (p, ticks): at each tick a missed-detection burst of `ticks` starts with probability p,
    during which no object is published and Autoware's dummy perception coasts the last one at constant velocity (as a
    tracker coasting through missed detections). The noise generator is seeded per episode and the seed is recorded."""

    def __init__(self, br, sut0, cruise, latency_s=0., v0_max=None, noise=None, dropout=None):
        self.br, self.sut0, self.cruise = br, sut0, cruise
        info = br.info
        self.off, self.L, self.W = info['centre_ahead_of_base_link'], info['length'], info['width']
        self.v0 = sut0['vx']
        self.shift = None
        self.trace, self.late = [], []
        self.hit, self.t_hit, self.contact_dx, self.min_gap, self.min_clear = False, None, None, math.inf, math.inf
        self.lag, self.v0_max, self.published = int(round(latency_s / DT)), v0_max, []
        self.noise, self.dropout, self.gap_left, self.sent, self.missed = noise, dropout, 0, 0, 0
        self.seed = None
        if noise or dropout:
            import os
            import random
            self.seed = int.from_bytes(os.urandom(4), 'little')
            self.rng = random.Random(self.seed)

    def centre(self, e):
        return e['x'] + self.off * math.cos(e['yaw']), e['y'] + self.off * math.sin(e['yaw'])

    def reset(self):
        base_x = X_START - PRE_S * self.v0 - self.off
        r = self.br.call('reset', ego=[base_x, 0., 0., self.v0], goal=[X_GOAL, 0., 0.], vmax=self.cruise)
        self.ego = r['ego']
        return r

    def adv_obj(self, g):
        """highway-env generator state (dict x, y, vx, vy) -> bridge object [x, y, yaw, vx, vy, L, W] in the map.
        Called once per published tick; with a latency the object of `lag` ticks earlier is returned (B58)."""
        x = g['x'] + self.shift
        y = LANE - g['y']
        vx, vy = g['vx'], -g['vy']
        obj = [x, y, math.atan2(vy, max(vx, .1)), vx, vy, ADV_L, ADV_W]
        if self.lag:
            self.published.append(obj)
            # the measurement of `lag` ticks earlier, extrapolated to now at its own velocity (a latency-compensating
            # tracker): what stays late is any change of motion within the latency, e.g. the start of the lateral cut-in
            k = min(self.lag, len(self.published) - 1)
            d = self.published[-1 - k]
            obj = [d[0] + d[3] * k * DT, d[1] + d[4] * k * DT] + d[2:]
        return self.perceive(obj)

    def perceive(self, obj):
        """B60: measurement noise and missed-detection bursts on the object Autoware receives (None = not published)."""
        if not (self.noise or self.dropout):
            return obj
        if self.dropout and self.sent:  # the first object is always published (the dummy object must exist)
            p, ticks = self.dropout
            if self.gap_left == 0 and self.rng.random() < p:
                self.gap_left = int(ticks)
            if self.gap_left > 0:
                self.gap_left -= 1
                self.missed += 1
                return None
        if self.noise:
            s_long, s_lat, s_v = self.noise
            x, y, yaw, vx, vy = obj[:5]
            c, s = math.cos(yaw), math.sin(yaw)
            e_long, e_lat = self.rng.gauss(0., s_long), self.rng.gauss(0., s_lat)
            speed = max(0., math.hypot(vx, vy) + self.rng.gauss(0., s_v))
            obj = [x + c * e_long - s * e_lat, y + s * e_long + c * e_lat, yaw, speed * c, speed * s] + obj[5:]
        self.sent += 1
        return obj

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
                               gap=None if gap is None else round(gap, 3), clear=round(clear, 3), late=e.get('late'),
                               traj_age=e.get('traj_age'), mode=e.get('mode'), mrm=e.get('mrm')))
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

    def health(self):
        """Infrastructure health of the recorded ticks: planning-output age, operation modes, fail-safe (MRM) states.
        healthy = the planner published within STALL_S at every tick and the ego stayed in autonomous mode; an MRM is
        Autoware's own behaviour and is recorded, not excluded. Only the ticks up to the contact count."""
        rows = [r for r in self.trace if self.t_hit is None or r['t'] <= self.t_hit + 1e-6]
        ages = [r['traj_age'] for r in rows if r.get('traj_age') is not None]
        modes = sorted({r['mode'] for r in rows if r.get('mode') is not None})
        mrm = sorted({tuple(r['mrm']) for r in rows if r.get('mrm')})
        age_max = max(ages) if ages else None
        out = dict(traj_age_max=age_max, modes=modes, mrm_states=[list(m) for m in mrm],
                   healthy=bool(age_max is not None and age_max <= STALL_S and modes == [AUTONOMOUS]))
        if self.v0_max is not None:  # B58: a start off the case speed is an infrastructure failure
            err = self.trace[0]['ego'][2] - self.v0 if self.trace else None
            out['v0_ok'] = err is not None and abs(err) <= self.v0_max
            out['healthy'] = out['healthy'] and out['v0_ok']
        return out

    def result(self, **extra):
        e0 = self.trace[0] if self.trace else None
        if self.seed is not None:  # B60: the perception model's seed and its missed / published objects
            extra = dict(extra, perception=dict(seed=self.seed, missed=self.missed, sent=self.sent))
        return dict(contact=self.hit, t_contact=self.t_hit, contact_dx=self.contact_dx, health=self.health(),
                    min_gap=None if math.isinf(self.min_gap) else round(self.min_gap, 3), min_clearance=round(self.min_clear, 3),
                    sut_min_acc=min((r['ego'][3] for r in self.trace if r['ego'][3] is not None), default=None),
                    v0_error=None if e0 is None else round(e0['ego'][2] - self.v0, 3), late_max=round(max(self.late), 4) if self.late else None,
                    late_mean=round(sum(self.late) / len(self.late), 4) if self.late else None, trace=self.trace, **extra)


def replay(br, row, tail_s=.5, latency_s=0., v0_max=None, noise=None, dropout=None):
    """A fixed adversary trajectory (row: sut0, cruise, trajectory with t = 0 .. 12 s at 0.1 s) against Autoware."""
    ep = Episode(br, row['sut0'], row['cruise'], latency_s, v0_max, noise, dropout)
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
    br.call('end')
    return ep.result(reset=r['timings'])


def policy_episode(br, case, model, shield=True, commit=True, latency_s=0., v0_max=None, noise=None, dropout=None):
    """The RL adversary closed-loop against Autoware (carla_rl.episode with the SUT in Autoware).

    Per tick the bridge publishes the adversary state of time t_k and returns the ego at t_k+1. At a decision tick the
    request for that tick is written first and the (shielded) decision for [t_k, t_k+0.5 s] is computed while the
    bridge waits, from the ego state at t_k. After the episode the realised decisions are replayed against the
    highway-env CC (b23_crossjudge.replay), as for every other SUT: fair = SUT contact, valid manoeuvre, CC avoids.
    """
    from fairtest.b23_crossjudge import replay as cc_replay
    from fairtest.b23_margin import clearance
    from fairtest.b8_env import B8AdversaryEnv
    from fairtest.carla_rl import observation, shielded
    from fairtest.env import B7Env
    from fairtest.runner import _STATE
    t0 = case['physical_initial_state'][1]
    sut0 = dict(x=t0['x'], y=t0['y'], vx=t0['vx'], vy=t0['vy'])
    cruise = case['event']['metrics']['p10']['target_speed']
    twin = B7Env(_STATE['base'], case, _STATE['b1'], _STATE['b2'], case['physical_initial_state'], target='oracle_t0', target_h=0.,
                 durations=(2., 3., 4., 5., 6.))
    twin.reset()
    ep = Episode(br, sut0, cruise, latency_s, v0_max, noise, dropout)
    try:
        r = ep.reset()
        g_init = twin.initial[0]
        traj = [dict(x=g_init['x'], y=g_init['y'], vx=g_init['vx'], vy=g_init['vy'])]
        e = ep.preroll(traj[0])
        last_v, acc, entered, actions, k, done = e['v'], 0., False, [], 0, False
        last, sh_checks, sh_unsafe, decide_s = None, 0, 0, []
        while not done and not ep.hit and k < HORIZON:
            sut_hw = ep.sut_hw(e)
            obs = observation(twin, sut_hw, acc, entered)
            br.send('step', obj=ep.adv_obj(traj[-1]))  # tick k runs while the decision is computed
            td = time.time()
            if shield:
                (lane, request), n, via = shielded(model, obs, case, [tuple(x) for x in actions], entered, commit, last,
                                                   window=True, tick=k, sut_hw=sut_hw)
                sh_checks, sh_unsafe, last = sh_checks + n, sh_unsafe + (via is None), ((lane, request), via)
            else:
                a_, _ = model.predict(obs, deterministic=True)
                lane, request = B8AdversaryEnv.decode(int(a_))
                if commit and entered and lane != 4.:
                    lane = 4.
            decide_s.append(time.time() - td)
            entered = entered or lane == 4.
            actions.append((lane, request))
            before = len(twin.meter.trace)
            _, _, done, _, _ = twin.step((lane, request))
            rows = [row['generator'] for row in twin.meter.trace[before:]]
            for j, g in enumerate(rows):
                e = br.recv() if j == 0 else br.call('step', obj=ep.adv_obj(traj[-1]))
                ep.late.append(e['late'])
                traj.append(dict(x=g['x'], y=g['y'], vx=g['vx'], vy=g['vy']))
                acc, last_v, k = (e['v'] - last_v) / DT, e['v'], k + 1
                if ep.check(k * DT, e, traj[-1]):
                    break
            if not rows:  # the twin ended without a tick: drain the pending step
                e = br.recv()
        valid = bool(twin.maneuver_valid)
    finally:
        twin.close()
        br.call('end')
    out = ep.result(reset=r['timings'], valid=valid, decisions=len(actions), actions=[list(x) for x in actions],
                    decide_max_s=round(max(decide_s), 3) if decide_s else None,
                    adv_traj=[dict(t=round(i * DT, 1), **{k: round(g[k], 4) for k in ('x', 'y', 'vx', 'vy')}) for i, g in enumerate(traj)])
    if shield:
        out['shield'] = dict(mode='window', cc_rollouts=sh_checks, unshieldable=sh_unsafe)
    if ep.hit:
        cc = cc_replay(case, actions, 'cc_r157', 0.)
        out.update(cc_contact=bool(cc['pair_contact']), cc_min_clearance=round(float(clearance(cc['trace'])), 3))
        out['fair'] = bool(valid and not cc['pair_contact'])
    return out


def main(a):
    br = Bridge(a.port)
    if a.cmd == 'policy':
        from stable_baselines3 import PPO
        from fairtest import b17_rl, devboot  # noqa: F401
        from fairtest.runner import _STATE, _init
        _init()
        b17_rl.SHIELD_MARGIN = .5  # v2: the shield demands a 0.5 m CC clearance
        model = PPO.load(a.model, device='cpu')
        with open(a.out, 'a', encoding='utf-8') as f:
            for c in (int(x) for x in a.cases.split(',')):
                t0 = time.time()
                try:
                    res = policy_episode(br, _STATE['cases'][c], model, shield=not a.no_shield)
                except Exception as e:  # recorded, never dropped
                    res = dict(error=f'{type(e).__name__}: {e}')
                res.update(case_id=c, model=a.model.split('/')[-1], seconds=round(time.time() - t0, 1))
                f.write(json.dumps(res) + '\n')
                f.flush()
                print(json.dumps({k: res.get(k) for k in ('case_id', 'contact', 't_contact', 'fair', 'cc_contact', 'cc_min_clearance',
                                                          'contact_dx', 'decisions', 'decide_max_s', 'late_max', 'late_mean', 'shield',
                                                          'seconds', 'error')}), flush=True)
        return
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
    q = sub.add_parser('policy')
    q.add_argument('out')
    q.add_argument('model')
    q.add_argument('--cases', default='207')
    q.add_argument('--port', type=int, default=5481)
    q.add_argument('--no-shield', action='store_true')
    main(ap.parse_args())
