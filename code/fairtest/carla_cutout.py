"""B43: the R157 cut-out scenario (B34, 1-D + the cut-out car's lateral position) replayed in CARLA 0.9.16 (Town04)
-- do the 1-D verdicts transfer to physically simulated followers and to real vehicle geometry?

Plans: every reported plan of the B34 development evaluation -- the agent (B34), the consensus agent (B34b) and the
collision-seeking baseline -- each replayed against every follower.
L1 (cuts out) and L2 (revealed): kinematic, on the cutout_sim trajectories. L1 keeps its speed and moves to the left
lane on the quintic (lateral progress scaled by w / 4, as carla_replay); L2 brakes per the plan. Bumper gaps at t = 0
from the case: follower -> L1 = gap, follower -> L2 = gap + length_lead + gap2 (the 1-D value; L1 is a CARLA car, so
the L1 -> L2 bumper gap differs from gap2 by the length difference).
Followers, all with the same vehicle physics:
  co:<name>   a cutout_sim follower (cc_co and its variants, idm_d0/_d5/_d10, acc_prod, acc_prod_aeb, rss_dec) on CARLA
              states with exactly the 1-D observation (L1's lateral offset from the plan in highway-env units, bumper
              gaps from positions, speeds and L2's acceleration from the trajectories), actuated as in carla_dec
  ba:<behavior>, tm:<d>   CARLA's BehaviorAgent / Traffic Manager with their own perception
Start as in carla_dec (x_start(v0); before t = 0, L1 and L2 drive FAR and FAR + 40 m ahead). Contact = the follower's
box overlaps L1's or L2's (L1 only while its body is still over the follower's lane in 1-D; CARLA uses the geometry).
min_gap applies the 1-D rule (the nearer of L1 while its body is in the lane, and L2) to CARLA positions.

  python -m fairtest.carla_cutout OUT.jsonl CASES.json EVAL.jsonl CONSENSUS.jsonl [--kinds ...] [--slice lo:hi] [--port P] [--resume]
  python -m fairtest.carla_cutout --selfcheck CASES.json EVAL.jsonl CONSENSUS.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from fairtest import carla_dec as CD
from fairtest import carla_replay as CR
from fairtest import cutout_sim as X
from fairtest import dec_sim as D

FOLLOWERS = ('cc_co',) + X.CC_VARIANTS + X.SUTS
KINDS = ','.join(f'co:{f}' for f in FOLLOWERS) + ',ba:normal,tm:default'
L2_BEHIND = 40.  # before t = 0, L2 drives this far beyond L1


def lat_rate(t, t_co, d_co):
    tau = (t - t_co) / d_co
    return 0. if tau <= 0. or tau >= 1. else X.LANE_W * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / d_co


def trajectories(case, theta):
    """[(x1, y1, vy1, v1, x2, v2, a2)] for t = 0 .. STEPS * DT, the cutout_sim world: L1 from its t = 0 position with its
    lateral offset y1 and rate (highway-env units, towards the left lane), L2 from its t = 0 position; a2 = L2's
    acceleration from that step on (what the 1-D followers see)."""
    t_co, d_co, t_b, a_b, d_b = (float(x) for x in theta)
    v1, v2 = float(case['v_lead']), float(case['v_lead2'])
    lead2 = D.Lead((t_b, a_b, d_b, 0., 0.))
    x1 = x2 = 0.
    out = []
    for k in range(X.STEPS + 1):
        t = k * X.DT
        a2 = lead2.step(t, v2) if k < X.STEPS else 0.
        out.append((x1, X.lateral(t, t_co, d_co), lat_rate(t, t_co, d_co), v1, x2, v2, a2))
        x1 += v1 * X.DT
        x2 += v2 * X.DT + .5 * a2 * X.DT * X.DT
        v2 = max(0., v2 + a2 * X.DT)
    return out


def gap2_0(case):
    return float(case['gap']) + float(case['length_lead']) + float(case['gap2'])


def near_gap(g1, g2, y1):
    """The 1-D rule: the car the follower must not hit (L1 while its body is in the lane and nearer, else L2)."""
    return g1 if y1 < X.BODY_OUT and g1 < g2 else g2


class CutoutShim(CD.DecShim):
    def __init__(self, name, cruise):
        super().__init__('cc_dec', cruise)
        self.ctrl = X.follower(name)

    def step(self, t, obs, sut_state, w, vehicle, carla):
        return self.actuate(float(self.ctrl(t, obs, sut_state[2], self.cruise)), sut_state, w, vehicle, carla)


def run(world, road, bp, case, theta, kind):
    import carla
    w, k_lat = road.w, road.w / CR.LANE_HW
    v0 = float(case['v_follower'])
    tr = trajectories(case, theta)
    agent = CR.is_agent(kind)
    xs_start = CD.x_start(v0)
    x_spawn = xs_start - ((CR.BA_CV_S + CR.BA_SETTLE_S) if agent else CR.PREROLL_S) * v0
    sut = world.spawn_actor(bp, road.transform(x_spawn, w))
    l1 = world.spawn_actor(bp, road.transform(x_spawn + CD.FAR, w))
    l2 = world.spawn_actor(bp, road.transform(x_spawn + CD.FAR + L2_BEHIND, w))
    out = dict(kind=kind)
    try:
        for v in (sut, l1, l2):
            CR._brakes(v)
        world.tick()
        e, e1, e2 = sut.bounding_box.extent, l1.bounding_box.extent, l2.bounding_box.extent
        half1, half2 = e.x + e1.x, e.x + e2.x

        def hold():
            xs = road.state(sut)[0]
            for car, d in ((l1, CD.FAR), (l2, CD.FAR + L2_BEHIND)):
                car.set_transform(road.transform(xs + d, w))
                car.set_target_velocity(road.velocity(v0, 0., xs + d))

        if agent:  # carla_replay.start_target, with L1 and L2 held far ahead
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
            target = CutoutShim(kind.split(':', 1)[1], v0)
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
        x10, x20 = xs0 + float(case['gap']) + half1, xs0 + gap2_0(case) + half2

        def place(k):
            x1, y1, vy1, v1, x2, v2, a2 = tr[k]
            yr, vyr = w - y1 * k_lat, -vy1 * k_lat
            l1.set_transform(road.transform(x10 + x1, yr, math.atan2(vyr, max(v1, .1))))
            l1.set_target_velocity(road.velocity(v1, vyr, x10 + x1))
            l2.set_transform(road.transform(x20 + x2, w))
            l2.set_target_velocity(road.velocity(v2, 0., x20 + x2))
            return x10 + x1, y1, v1, x20 + x2, v2, a2

        X1, y1, v1, X2, v2, a2 = place(0)
        last_v, min_acc, min_req, min_gap = vs0, 0., 0., math.inf
        min_c1 = min_c2 = math.inf
        hit, t_hit, with_, beyond = False, None, None, False
        for k in range(CR.HORIZON):
            st = road.state(sut)
            if agent:
                target.step(k * CR.DT, None, None, (2 * e.x, 2 * e.y), sut, carla)
            else:
                target._measured = (st[2] - last_v) / CR.DT
                obs = dict(y1=y1, l1=(X1 - st[0] - half1, v1, 0.), l2=(X2 - st[0] - half2, v2, a2))
                min_req = min(min_req, target.step(k * CR.DT, obs, st, w, sut, carla))
            world.tick()
            X1, y1, v1, X2, v2, a2 = place(k + 1)  # all vehicles now at t = (k + 1) * DT
            st = road.state(sut)
            acc = (st[2] - last_v) / CR.DT
            min_acc, last_v = min(min_acc, acc), st[2]
            min_gap = min(min_gap, near_gap(X1 - st[0] - half1, X2 - st[0] - half2, y1))
            beyond = beyond or st[0] > road.length - 5.
            c1, c2 = CR.clearance(sut, l1), CR.clearance(sut, l2)
            min_c1, min_c2 = min(min_c1, c1), min(min_c2, c2)
            if c1 <= 0. or c2 <= 0.:
                hit, t_hit, with_ = True, round((k + 1) * CR.DT, 1), 'l1' if c1 <= 0. else 'l2'
                break
        out.update(contact=hit, contact_with=with_, t_contact=t_hit, sut_min_acc=round(min_acc, 2),
                   min_gap=round(min_gap, 3), min_clearance_l1=round(min_c1, 3), min_clearance_l2=round(min_c2, 3),
                   x_end=round(st[0], 1), beyond_stretch=beyond)
        if not agent:
            out['min_request'] = round(min_req, 2)
    finally:
        CR.release(sut, kind, world)
        for v in (sut, l1, l2):
            v.destroy()
        world.tick()
    return out


def jobs(cases_path, eval_path, consensus_path):
    """Every reported plan of B34: the agent's robust fair failures (agent), the consensus agent's (consensus) and the
    baseline's collisions (collide)."""
    cases = {c['case_id']: c for c in json.load(open(cases_path, encoding='utf-8'))['cases']}
    out = []
    for path, rename in ((eval_path, {}), (consensus_path, {'agent': 'consensus'})):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key in sorted(r['runs']):
                method, sut = key.split('|')
                b = r['runs'][key].get('best')
                if b and b.get('success'):
                    out.append(dict(case=cases[r['case_id']], case_id=r['case_id'], method=rename.get(method, method), sut=sut,
                                    theta=b['theta']))
    return sorted(out, key=lambda j: (j['case_id'], j['method'], j['sut']))


def oned(case, theta, name):
    r = X.run(case, theta, name)
    return dict(valid=r['valid'], contact=r['contact'], min_gap=round(r['min_gap'], 3), t_contact=r['t_contact'])


def selfcheck(cases_path, eval_path, consensus_path):
    """No CARLA. With perfect actuation (the 1-D integrator), the replay's inputs -- the trajectories, the bumper gaps
    from positions, the call order, the 1-D contact rule -- must reproduce cutout_sim.run for every plan and follower."""
    worst, n, mismatch, invalid = 0., 0, 0, 0
    for j in jobs(cases_path, eval_path, consensus_path):
        case, th = j['case'], j['theta']
        tr = trajectories(case, th)
        v0, g10, g20 = float(case['v_follower']), float(case['gap']), gap2_0(case)
        for name in FOLLOWERS:
            ref = X.run(case, th, name)
            if not ref['valid']:
                invalid += 1
                continue
            ctl, v, xe, min_gap, contact = X.follower(name), v0, 0., math.inf, False
            for k in range(X.STEPS):
                x1, y1, _, v1, x2, v2, a2 = tr[k]
                a = ctl(k * X.DT, dict(y1=y1, l1=(g10 + x1 - xe, v1, 0.), l2=(g20 + x2 - xe, v2, a2)), v, v0)
                if v + a * X.DT < 0:
                    a = -v / X.DT
                xe += v * X.DT + .5 * a * X.DT * X.DT
                v = max(0., v + a * X.DT)
                x1, y1, _, _, x2, _, _ = tr[k + 1]
                g = near_gap(g10 + x1 - xe, g20 + x2 - xe, y1)
                min_gap = min(min_gap, g)
                if g <= 0:
                    contact = True
                    break
            n += 1
            mismatch += contact != ref['contact']
            worst = max(worst, abs(min_gap - ref['min_gap']))
    print(json.dumps(dict(episodes=n, invalid=invalid, contact_mismatch=mismatch, max_min_gap_error_m=worst)))
    return mismatch == 0 and invalid == 0 and worst < 1e-6


def main(a):
    todo = jobs(a.cases, a.eval, a.consensus)
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
                    if kind.startswith('co:'):
                        res['oned'][kind] = oned(j['case'], j['theta'], kind[3:])
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
    ap.add_argument('paths', nargs='+', help='OUT CASES EVAL CONSENSUS, or CASES EVAL CONSENSUS with --selfcheck')
    ap.add_argument('--kinds', default=KINDS)
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--slice', help='lo:hi of the sorted job list (parallel servers)')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        a.cases, a.eval, a.consensus = a.paths
        raise SystemExit(0 if selfcheck(a.cases, a.eval, a.consensus) else 1)
    a.out, a.cases, a.eval, a.consensus = a.paths
    main(a)
