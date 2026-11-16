"""B42: the R157 deceleration scenario (B30, 1-D) replayed in CARLA 0.9.16 (Town04) -- do the 1-D verdicts transfer
to physically simulated followers?

Plans: every reported plan of the B30 development evaluation (the agent's robust fair failures and the
collision-seeking baseline's collisions), each replayed against every follower.
Lead: kinematic (pose and velocity set every tick, as the cut-in adversary of carla_replay) in the follower's lane,
following the dec_sim lead model for its plan theta = (t_b, a_b, d_b, a_pre, a_post), i.e. the 1-D lead trajectory.
Followers, all with the same vehicle physics (FAIRTEST_CARLA_BP / _BRAKE / _PI as carla_replay):
  dec:<name>   a dec_sim follower (cc_dec, cc_dec_aeb, idm_d0/_d5/_d10, acc_prod, acc_prod_aeb, rss_dec) on CARLA
               states -- bumper gap from the road frame and the bounding boxes, the lead's speed and acceleration from
               its trajectory, exactly the 1-D inputs -- actuated as the carla_replay shim does (constant-velocity mode
               for requests above -0.3 m/s^2, the calibrated brake map below)
  ba:<behavior>, tm:<d>   CARLA's BehaviorAgent / Traffic Manager (third-party followers, carla_replay settings)
t = 0 at x_start(v0) (room for the agents' run-up at speeds up to 50 m/s); before t = 0 the lead drives FAR ahead at
the follower's speed (outside every agent's perception) and is then placed at the case's bumper gap with the case's
lead speed, as the 1-D episode starts. Contact = oriented bounding boxes overlap. Each dec kind also carries the 1-D
verdict of the same follower on the same plan.

  python -m fairtest.carla_dec OUT.jsonl CASES.json EVAL.jsonl [--kinds ...] [--slice lo:hi] [--port 2000] [--resume]
  python -m fairtest.carla_dec --selfcheck CASES.json EVAL.jsonl     (no CARLA: plumbing reproduces dec_sim exactly)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from fairtest import carla_replay as CR
from fairtest import dec_sim as D

FOLLOWERS = ('cc_dec', 'cc_dec_aeb') + D.SUTS
KINDS = ','.join(f'dec:{f}' for f in FOLLOWERS) + ',ba:normal,tm:default'
FAR = 200.  # m ahead of the follower before t = 0


def lead_trajectory(case, theta):
    """[(x, v, a)] of the lead for t = 0 .. STEPS * DT: the dec_sim lead model (x from its t = 0 position; a = the
    acceleration applied from that step on, which is what the 1-D followers see)."""
    lead = D.Lead(theta, 0.)
    x, v, out = 0., float(case['v_lead']), []
    for k in range(D.STEPS + 1):
        a = lead.step(k * D.DT, v) if k < D.STEPS else 0.
        out.append((x, v, a))
        x += v * D.DT + .5 * a * D.DT * D.DT
        v = max(0., v + a * D.DT)
    return out


def x_start(v0):
    """Road-frame x of the follower at t = 0: the agents' run-up (BA_CV_S + BA_SETTLE_S at v0) from x = 10 m."""
    return 10. + (CR.BA_CV_S + CR.BA_SETTLE_S) * v0


class DecShim:
    """A dec_sim follower driving a CARLA vehicle (the carla_replay ShimTarget actuation, same constants)."""

    def __init__(self, name, cruise):
        self.ctrl, self.cruise = D.follower(name), float(cruise)
        self.integ, self.v_cmd, self.braking, self._measured = 0., None, False, 0.
        self._v_prev, self.b_prev, self.d_hat, self.req_prev = None, 0., 0., 0.  # h2 actuation state

    def step(self, t, gap, v_l, a_l, sut_state, w, vehicle, carla):
        return self.actuate(float(self.ctrl(t, gap, sut_state[2], v_l, a_l, self.cruise)), sut_state, w, vehicle, carla)

    def actuate(self, req, sut_state, w, vehicle, carla):
        x, y, vx, vy, heading = sut_state
        steer = -heading + math.atan2(.8 * (w - y), max(vx, 1.))
        ctl = carla.VehicleControl(steer=float(np.clip(steer / CR.STEER_MAX_RAD, -1., 1.)))
        CR.actuate(self, req, vx, vehicle, ctl, carla)  # same actuation as the ShimTarget (FAIRTEST_CARLA_ACT)
        vehicle.apply_control(ctl)
        return req


def run(world, road, bp, case, theta, kind):
    import carla
    w = road.w
    v0, gap0 = float(case['v_follower']), float(case['gap'])
    lead = lead_trajectory(case, theta)
    agent = CR.is_agent(kind)
    xs_start = x_start(v0)
    x_spawn = xs_start - ((CR.BA_CV_S + CR.BA_SETTLE_S) if agent else CR.PREROLL_S) * v0
    sut = world.spawn_actor(bp, road.transform(x_spawn, w))
    adv = world.spawn_actor(bp, road.transform(x_spawn + FAR, w))
    out = dict(kind=kind)
    try:
        CR._brakes(sut)
        CR._brakes(adv)
        world.tick()
        e, ea = sut.bounding_box.extent, adv.bounding_box.extent
        half = e.x + ea.x  # bumper gap = centre distance - half

        def hold():
            xs = road.state(sut)[0]
            adv.set_transform(road.transform(xs + FAR, w))
            adv.set_target_velocity(road.velocity(v0, 0., xs + FAR))

        if agent:  # carla_replay.start_target, with the lead held FAR ahead
            for _ in range(int(CR.BA_CV_S / CR.DT)):
                sut.enable_constant_velocity(carla.Vector3D(v0, 0., 0.))
                hold()
                world.tick()
            sut.disable_constant_velocity()
            if kind.startswith('ba:'):
                target = CR.BATarget(kind.split(':')[1], sut, road, v0, CR.ba_plan(road, road.state(sut)[0]))
            else:
                target = CR.TMTarget(sut, v0, kind)
            for _ in range(int(20. / CR.DT)):
                target.step(0., None, None, (2 * e.x, 2 * e.y), sut, carla)
                hold()
                world.tick()
                x, y = road.state(sut)[:2]
                if x >= xs_start and abs(y - w) < 1.:
                    break
            else:
                raise RuntimeError(f'{kind} did not reach the start point')
        else:
            target = DecShim(kind.split(':', 1)[1], v0)
            sut.set_target_velocity(road.velocity(v0, 0., x_spawn))
            corr = 0.
            for i in range(int(CR.PREROLL_S / CR.DT)):
                x, y, vx, vy, h = road.state(sut)
                if CR.ACT != 'v1' and i > 0:  # h2.1: constant-velocity speed feedback (the mode runs ~0.5% slow)
                    corr += CR.CV_GAIN * (v0 - vx)
                sut.enable_constant_velocity(carla.Vector3D(v0 + corr, 0., 0.))
                sut.apply_control(carla.VehicleControl(steer=float(np.clip((-h + math.atan2(.8 * (w - y), max(vx, 1.))) / CR.STEER_MAX_RAD, -1, 1))))
                hold()
                world.tick()
        xs0, _, vs0 = road.state(sut)[:3]
        out['v0_error'] = round(vs0 - v0, 3)
        if not agent:
            target.v_cmd, target.cv_corr = vs0, corr
        x_lead0 = xs0 + gap0 + half  # the case's bumper gap at t = 0

        def place(k):
            xl, vl, al = lead[k]
            adv.set_transform(road.transform(x_lead0 + xl, w))
            adv.set_target_velocity(road.velocity(vl, 0., x_lead0 + xl))
            return x_lead0 + xl, vl, al

        gx, gv, ga = place(0)
        last_v, min_acc, min_req, min_gap, min_clear = vs0, 0., 0., math.inf, math.inf
        hit, t_hit, beyond = False, None, False
        for k in range(CR.HORIZON):
            st = road.state(sut)
            if agent:
                target.step(k * CR.DT, None, None, (2 * e.x, 2 * e.y), sut, carla)
            else:
                target._measured = (st[2] - last_v) / CR.DT
                min_req = min(min_req, target.step(k * CR.DT, gx - st[0] - half, gv, ga, st, w, sut, carla))
            world.tick()
            gx, gv, ga = place(k + 1)  # both vehicles now at t = (k + 1) * DT
            st = road.state(sut)
            acc = (st[2] - last_v) / CR.DT
            min_acc, last_v = min(min_acc, acc), st[2]
            min_gap = min(min_gap, gx - st[0] - half)
            beyond = beyond or st[0] > road.length - 5.
            min_clear = min(min_clear, CR.clearance(sut, adv))
            if min_clear <= 0.:
                hit, t_hit = True, round((k + 1) * CR.DT, 1)
                break
        out.update(contact=hit, t_contact=t_hit, sut_min_acc=round(min_acc, 2), min_gap=round(min_gap, 3),
                   min_clearance=round(min_clear, 3), x_end=round(st[0], 1), beyond_stretch=beyond)
        if not agent:
            out['min_request'] = round(min_req, 2)
    finally:
        CR.release(sut, kind, world)
        sut.destroy()
        adv.destroy()
        world.tick()
    return out


def jobs(cases_path, eval_path):
    """Every reported plan of the B30 evaluation: the agent's robust fair failures and the baseline's collisions."""
    cases = {c['case_id']: c for c in json.load(open(cases_path, encoding='utf-8'))['cases']}
    out = []
    for line in open(eval_path, encoding='utf-8'):
        r = json.loads(line)
        for key in sorted(r['runs']):
            method, sut = key.split('|')
            b = r['runs'][key].get('best')
            if b and b.get('success'):
                out.append(dict(case=cases[r['case_id']], case_id=r['case_id'], method=method, sut=sut, theta=b['theta']))
    return sorted(out, key=lambda j: (j['case_id'], j['method'], j['sut']))


def oned(case, theta, name):
    r = D.run(case, theta, name)
    return dict(contact=r['contact'], min_gap=round(r['min_gap'], 3), t_contact=r['t_contact'])


def selfcheck(cases_path, eval_path):
    """No CARLA. With perfect actuation (the 1-D integrator), the replay's inputs -- lead_trajectory, the bumper gap
    from positions, the call order -- must reproduce dec_sim.run for every plan and follower."""
    worst, n, mismatch = 0., 0, 0
    for j in jobs(cases_path, eval_path):
        case, th = j['case'], j['theta']
        lead = lead_trajectory(case, th)
        for name in FOLLOWERS:
            ref = D.run(case, th, name)
            ctl, v, xf, x_lead0 = D.follower(name), float(case['v_follower']), 0., float(case['gap'])
            min_gap, contact = x_lead0, False
            for k in range(D.STEPS):
                xl, vl, al = lead[k]
                a = ctl(k * D.DT, x_lead0 + xl - xf, v, vl, al, float(case['v_follower']))
                if v + a * D.DT < 0:
                    a = -v / D.DT
                xf += v * D.DT + .5 * a * D.DT * D.DT
                v = max(0., v + a * D.DT)
                g = x_lead0 + lead[k + 1][0] - xf
                min_gap = min(min_gap, g)
                if g <= 0:
                    contact = True
                    break
            n += 1
            mismatch += contact != ref['contact']
            worst = max(worst, abs(min_gap - ref['min_gap']))
    print(json.dumps(dict(episodes=n, contact_mismatch=mismatch, max_min_gap_error_m=worst)))
    return mismatch == 0 and worst < 1e-6


def main(a):
    todo = jobs(a.cases, a.eval)
    if a.slice:
        lo, hi = (int(x) for x in a.slice.split(':'))
        todo = todo[lo:hi]
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {(r['case_id'], r['method'], r['sut']) for r in (json.loads(l) for l in open(a.out, encoding='utf-8') if l.strip())}
    todo = [j for j in todo if (j['case_id'], j['method'], j['sut']) not in done]
    world = CR.connect(a.port)
    road = CR.Road(world)
    bp = world.get_blueprint_library().find(CR.BLUEPRINT)
    print(json.dumps(dict(todo=len(todo), road_length=road.length, lane_width=round(road.w, 2), blueprint=CR.BLUEPRINT,
                          brake=[CR.BRAKE_SCALE, CR.BRAKE_A0, CR.BRAKE_K], pi=[CR.BRAKE_KP, CR.BRAKE_KI])), flush=True)
    try:
        with open(a.out, 'a', encoding='utf-8') as f:
            for i, j in enumerate(todo):
                res = dict(case_id=j['case_id'], method=j['method'], sut=j['sut'], theta=j['theta'], carla={}, oned={})
                for kind in a.kinds.split(','):
                    if kind.startswith('dec:'):
                        res['oned'][kind] = oned(j['case'], j['theta'], kind[4:])
                    t0 = time.time()
                    try:
                        res['carla'][kind] = dict(run(world, road, bp, j['case'], j['theta'], kind), seconds=round(time.time() - t0, 1))
                    except Exception as e:  # recorded, never dropped
                        res['carla'][kind] = dict(error=f'{type(e).__name__}: {e}')
                f.write(json.dumps(res) + '\n')
                f.flush()
                if (i + 1) % 10 == 0:
                    print(json.dumps(dict(done=i + 1, of=len(todo))), flush=True)
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='OUT CASES EVAL, or CASES EVAL with --selfcheck')
    ap.add_argument('--kinds', default=KINDS)
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--slice', help='lo:hi of the sorted job list (parallel servers)')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        a.cases, a.eval = a.paths
        raise SystemExit(0 if selfcheck(a.cases, a.eval) else 1)
    a.out, a.cases, a.eval = a.paths
    main(a)
