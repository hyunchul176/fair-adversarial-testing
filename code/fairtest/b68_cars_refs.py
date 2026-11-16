"""B68 (B68 design, available on request): the R157 CC and the FSM as read by an
independent implementation, the CARS code (github.com/RoboSafe-Lab/CARS-code @ 6e32a26, scripts/evaluate/replay.py,
Apache-2.0), ported to this simulator. Nothing else changes.

cc_jrc    CARS 'CC-JP-LaneAware' (stated to match JRC-FSM CC_react / CC_check_safety_lane_info): the reaction starts only
          when, at the same step, the cutting-in vehicle is ahead and >= 1.095 m from its lane centre (the lane-width form
          0.72 + 0.375 m of compute_dist_to_react) AND the gap closes (> 0.1 m/s) with TTC < 2 s; then 0.75 s at
          -0.4 m/s^2, then a ramp at 12.65 m/s^3 to 0.774 g held to the end (no release; CARS also stops braking once the
          cutting-in vehicle is no longer ahead by a positive bumper gap, which the port does not copy: with lateral
          overlap that is contact, so it matters only alongside). Our C (cc_target.CCDriver)
          starts the reaction at the lateral threshold alone and gates only the emergency braking with TTC <= 2 s.
fsm_cars  CARS 'FSM' (Mattas et al. 2022, sec. 3.4, Eqs. 5-16, Table 2): no risk unless the cutting-in vehicle is ahead
          (positive bumper gap); while laterally clear, risk only if it moves towards the ego lane, the ego closes
          (> 0.1 m/s) and T_lat <= T_lon + 0.1 s; PFS with d1 = 2 m, CFS from the ego's current acceleration, b_comf 4,
          b_max 6, b_cut-in max 7 m/s^2, tau 0.75 s, reaction force Eq. 16; after the first positive request a tau window
          without braking, then a ramp (12.65 m/s^3) to the request, capped at 6 m/s^2. Our ref_fsm (ref_models) uses
          b_comf 3, the CC's lateral trigger with a -0.4 m/s^2 reaction and a relative-motion CFS.
Both follow this simulator's nominal law (following_law) until their trigger, as every reference driver here does;
after the trigger cc_jrc never releases, and fsm_cars holds its speed when its request is zero (CARS leaves the ego on
its log). Lane keeping is unchanged. Activated per worker process by replacing env's constructors (target cc_r157 ->
cc_jrc, ref_fsm -> fsm_cars); the frozen judge code is not edited.

  python -m fairtest.b68_cars_refs JOBS.jsonl OUT.jsonl --variant cc_r157|cc_jrc|ref_fsm|fsm_cars --workers W
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time

from fairtest import devboot  # noqa: F401
from fairtest.cc_target import CCDriver, following_law
from fairtest.ref_models import _gap, _states, project
from src.environments.hierarchical_interaction_env import lane_controller

DT = .1
PERCEIVE_M = .375 + .72
TARGET = dict(cc_r157='cc_r157', cc_jrc='cc_r157', ref_fsm='ref_fsm', fsm_cars='ref_fsm')


def _act(env, t, request, previous_a, jerk, amax, phase, name, extra, following):
    applied = float(project(request, math.hypot(t.vx, t.vy), previous_a, jerk, amax))
    steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
    env.target.act(dict(acceleration=applied, steering=steer))
    return applied, dict(raw_action=[applied / 5, 0.], following_mps2=following,
                         requested_mps2=request, post_shield_mps2=request, post_stop_mps2=request,
                         previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                         viability=dict(constraint_infeasible=False), shield=dict(intervened=False),
                         recognition=dict(cc_phase=phase, model=name, **extra))


class CCJRC(CCDriver):
    """CARS / JRC reading of the R157 CC (constructed with CCDriver's arguments; only reaction and emergency are used)."""
    # 0.774 g is 7.5929 m/s^2 in CARS; the simulator's CC contract caps at C's 7.59 m/s^2 (a replay above it stops with a
    # contract error), so the port brakes at 7.59, the value C uses (fix after the first run, 2026-10-04)
    critical_ttc, release, emergency_mps2_cars, jerk_cars = 2., -.4, -7.59, 12.65

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triggered_at = None

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g, t = _states(env)
        displacement = abs(g.y)
        ahead = g.x > t.x
        d = _gap(g, t)
        closing = t.vx - g.vx
        ttc = d / closing if closing > .1 and d > 0 else None
        if self.triggered_at is None and ahead and displacement >= PERCEIVE_M and ttc is not None and ttc < self.critical_ttc:
            self.triggered_at = env.cc_clock
        if self.triggered_at is None:
            request, phase = following_law(g, t, cruise), 'free'
        elif env.cc_clock - self.triggered_at < self.reaction_s - 1e-9:
            request, phase = self.release, 'reaction'
        else:
            request, phase = self.emergency_mps2_cars, 'emergency'
        return _act(env, t, request, previous_a, self.jerk_cars, -self.emergency_mps2_cars, phase, 'cc_jrc',
                    dict(perceived_at_s=self.triggered_at, emergency=phase == 'emergency', lateral_displacement=displacement, ttc=ttc),
                    following_law(g, t, cruise))


class FSMCARS:
    """CARS port of the FSM (Mattas et al. 2022)."""
    tau, b2_comf, b2_max, b1_max, d1, jerk, cap = .75, 4., 6., 7., 2., 12.65, 6.

    def __init__(self):
        self.triggered, self.rt_counter, self.current = False, 0., 0.

    @staticmethod
    def mu(x, x_safe, x_unsafe):
        if x_safe <= x_unsafe:
            return 1. if x < x_safe else 0.
        return float(min(1., max(0., (x_safe - x) / (x_safe - x_unsafe))))

    def target_decel(self, g, t, a2):
        d = _gap(g, t)
        if not (g.x > t.x) or d <= 0:
            return 0.
        u2, u1 = t.vx, g.vx
        lat_body_gap = abs(g.y - t.y) - (t.width + g.width) / 2
        toward = g.vy if t.y > g.y else -g.vy
        if lat_body_gap > 0:
            if toward <= 0 or u2 - u1 <= .1:
                return 0.
            t_lat = lat_body_gap / toward
            t_lon = (d + t.length + g.length) / (u2 - u1)
            if t_lat > t_lon + .1:
                return 0.
        d_safe = u2 * self.tau + u2 ** 2 / (2 * self.b2_comf) - u1 ** 2 / (2 * self.b1_max) + self.d1
        d_unsafe = u2 * self.tau + u2 ** 2 / (2 * self.b2_max) - u1 ** 2 / (2 * self.b1_max)
        pfs = self.mu(d - self.d1, d_safe, d_unsafe)
        cfs = 0.
        if u2 > u1:
            a2p = max(a2, -self.b2_comf)
            u2s = u2 + a2p * self.tau
            if u2s <= u1:
                c_safe = c_unsafe = (u2 - u1) ** 2 / (2 * abs(a2p) if abs(a2p) > 1e-6 else 1e-6)
            else:
                d_new = ((u2 + u2s) / 2 - u1) * self.tau
                c_safe = d_new + (u2s - u1) ** 2 / (2 * self.b2_comf)
                c_unsafe = d_new + (u2s - u1) ** 2 / (2 * self.b2_max)
            cfs = self.mu(d, c_safe, c_unsafe)
        return cfs * (self.b2_max - self.b2_comf) + self.b2_comf if cfs > 0 else pfs * self.b2_comf

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g, t = _states(env)
        want = self.target_decel(g, t, previous_a)
        if not self.triggered:
            if want > 0:
                self.triggered, self.rt_counter, self.current = True, self.tau, 0.
            request, phase = following_law(g, t, cruise), 'free'
            if self.triggered:
                request, phase = 0., 'reaction'
        elif self.rt_counter > 1e-9:
            self.rt_counter -= DT
            request, phase = 0., 'reaction'
        else:
            self.current = min(self.current + self.jerk * DT, want, self.cap)
            request, phase = -self.current, 'fsm'
        return _act(env, t, request, previous_a, self.jerk, self.cap, phase, 'fsm_cars',
                    dict(perceived_at_s=None, emergency=self.triggered, fsm_request=want), following_law(g, t, cruise))


STATE = {}


def _init_worker(variant):
    from fairtest import env as E
    if variant == 'cc_jrc':
        E.CCDriver = CCJRC
    elif variant == 'fsm_cars':
        E.FSMDriver = FSMCARS
    STATE['variant'] = variant


def task(job):
    from fairtest import runner
    from fairtest.b23_margin import clearance
    from fairtest.b29_consensus import _run
    runner._init()
    case = runner._STATE['cases'][job['case_id']]
    out = {k: v for k, v in job.items() if k != 'payload'}
    out['variant'] = STATE['variant']
    try:
        raw = _run(case, job['how'], job['payload'], TARGET[STATE['variant']])
        out.update(contact=bool(raw['pair_contact']), clearance=round(float(clearance(raw['trace'])), 3),
                   models=sorted({str(((r.get('target_control') or {}).get('recognition') or {}).get('model', ''))
                                  for r in raw.get('trace') or []} - {''}))
    except Exception as e:  # recorded, never dropped
        out['error'] = f'{type(e).__name__}: {e}'[:300]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs')
    ap.add_argument('out')
    ap.add_argument('--variant', choices=sorted(TARGET), required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    jobs = [json.loads(line) for line in open(a.jobs, encoding='utf-8') if line.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)['job'] for line in open(a.out, encoding='utf-8') if line.strip()}
    todo = [j for j in jobs if j['job'] not in done]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(a.variant,)) as pool, open(a.out, 'a', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(task, todo, chunksize=4)):
            f.write(json.dumps(r) + '\n')
            if (i + 1) % 500 == 0:
                f.flush()
                print(json.dumps(dict(done=i + 1, of=len(todo), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(todo), variant=a.variant, elapsed_s=round(time.time() - t0, 1))), flush=True)
