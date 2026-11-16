"""E7: replay highway-env adversary trajectories in CARLA 0.9.16 (Town04 highway) against physically simulated SUTs.

Road frame: a straight Town04 stretch with a same-direction left lane. x = distance along the SUT lane, y = lateral
position with the adversary's lane centre at y = 0 and the SUT's lane centre at y = w (3.5 m) (highway-env: 4 m).
- Adversary: the recorded highway-env trajectory, lateral progress scaled by w/4 (lane centre to lane centre), driven
  kinematically (pose and velocity set every tick), i.e. the attack is held fixed as in the CC-replay semantics.
- SUT / reference kinds (all with the same vehicle physics: Tesla Model 3, wheel brake torque x2.5 so that full brake
  gives ~10.5 m/s^2; calibrated brake map decel = 0.56 + 9.97 * brake):
    cc                  the R157 CC driver (`cc_target.CCDriver`), fixed-distance perception in metres
    ladder:<name>:<H>   our SUT controllers (following / reference_idm / following_aeb / rss_sut) via a state shim
    ba:<behavior>       CARLA's BehaviorAgent (cautious / normal / aggressive), cruise set to the case's SUT speed
    tm:<d>              CARLA's Traffic Manager autopilot, no lane changes, desired speed = cruise, leading-vehicle
                        distance d m ('default' keeps the Traffic Manager's own)
  Shim-driven vehicles track the controller's acceleration request (`actuate`, FAIRTEST_CARLA_ACT): 'h2' (default,
  harness v2, 2026-10-29) tracks every request above CV_FLOOR (-1 m/s^2, i.e. cruising and the CC's 0.4 m/s^2
  reaction deceleration) exactly in CARLA's constant-velocity mode, and stronger braking with the calibrated brake map
  minus the disturbance seen over the previous braking tick (measured minus brake-map acceleration: the residual drive
  force that builds up in constant-velocity mode, which made coasting and the braking plateau depend on how long the
  vehicle had cruised). h2.1 (same day): CARLA's constant-velocity mode runs ~0.5% slower than commanded (31.84 m/s
  for 32), so cruising targets fell behind their highway-env twins by ~0.5-1 m before a cut-in; the commanded speed now
  gets an integral correction from the measured speed, in the pre-roll too (v0 error -0.08 -> 0.000 m/s). 'v1' is the earlier actuation (constant-velocity mode above -0.3 m/s^2, brake map plus a PI
  whose measured acceleration was always 0 in the harness loops). Lateral control is a Stanley lane keeper (the
  controllers' own steering is ignored).
- 3 s pre-roll at the case's SUT speed before t = 0; contact = oriented bounding boxes overlap (as in highway-env).
- Vehicle model: FAIRTEST_CARLA_BP (default vehicle.tesla.model3) with FAIRTEST_CARLA_BRAKE = 'scale,a0,k' from
  `carla_calib` (default 2.5,0.56,9.97, the Tesla calibration).

  python -m fairtest.carla_replay OUT.jsonl TRAJECTORIES.jsonl --kinds cc,ladder:following:0,ba:normal [--port 2000]
  python -m fairtest.carla_replay --selftest [--port 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

CARLA_ROOT = os.path.expanduser('~/carla_0916')
BLUEPRINT = os.environ.get('FAIRTEST_CARLA_BP', 'vehicle.tesla.model3')
BRAKE_SCALE, BRAKE_A0, BRAKE_K = (float(x) for x in os.environ.get('FAIRTEST_CARLA_BRAKE', '2.5,.56,9.97').split(','))
# PI correction of the brake map (kp, ki). E7 ran with 0.5,0.5, which overshoots the requested deceleration at
# braking onset (achieved ~9.9 vs requested 7.6 m/s^2 over 0.3 s); 0,0 = calibrated feed-forward only
BRAKE_KP, BRAKE_KI = (float(x) for x in os.environ.get('FAIRTEST_CARLA_PI', '.5,.5').split(','))
ACT = os.environ.get('FAIRTEST_CARLA_ACT', 'h2')  # 'h2' (harness v2) or 'v1' (the earlier actuation, see the docstring)
CV_GAIN = 1.  # h2.1: integral gain of the constant-velocity speed feedback (deadbeat)
CV_FLOOR = -1.  # h2: requests above this are tracked exactly in constant-velocity mode (coasting reaches ~0.8 m/s^2)
LANE_HW, DT, PREROLL_S, HORIZON = 4., .1, 3., 120  # trajectories carry HORIZON + 1 states (t = 0 .. 12 s)
START_X = 250.  # road-frame x of the SUT at t = 0
BA_CV_S, BA_SETTLE_S = 1., 4.  # BehaviorAgent: constant-velocity lift, then own driving before t = 0
STEER_MAX_RAD = math.radians(70.)
_CLIENT, TM_PORT, _TM = None, 8000, None  # set by connect(); one Traffic Manager per CARLA server (8000 + port - 2000)
WATCHDOG_S = float(os.environ.get('FAIRTEST_CARLA_WATCHDOG_S', 120.))
_BEAT = [time.time()]


def _polys(a, b):
    from shapely.geometry import Polygon
    polys = []
    for v in (a, b):
        tf, e = v.get_transform(), v.bounding_box.extent
        yaw = math.radians(tf.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        cx, cy = tf.location.x, tf.location.y
        pts = [(cx + dx * c - dy * s, cy + dx * s + dy * c) for dx, dy in ((e.x, e.y), (e.x, -e.y), (-e.x, -e.y), (-e.x, e.y))]
        polys.append(Polygon(pts))
    return polys


def contact(a, b):
    p, q = _polys(a, b)
    return p.intersects(q)


def clearance(a, b):
    """Smallest distance between the two oriented bounding boxes (0 at contact)."""
    p, q = _polys(a, b)
    return float(p.distance(q))


class Road:
    """Straight stretch: world <-> road frame."""

    def __init__(self, world, lengths=(900., 800., 700., 650., 600.), yaw_tol=5.):
        import carla
        self.carla = carla
        cmap = world.get_map()
        self.wp0 = None
        waypoints = cmap.generate_waypoints(5.)
        for length_m in lengths:
            self.wp0 = self._find(carla, waypoints, length_m, yaw_tol)
            if self.wp0 is not None:
                self.length = length_m
                break
        if self.wp0 is None:
            raise RuntimeError('no straight stretch found')
        self._frame()

    @staticmethod
    def _find(carla, waypoints, length_m, yaw_tol):
        for wp in waypoints:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            left = wp.get_left_lane()
            if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * wp.lane_id <= 0:
                continue
            ahead, ok, yaw0 = wp, True, wp.transform.rotation.yaw
            for _ in range(int(length_m / 5)):
                nxt = ahead.next(5.)
                if not nxt or nxt[0].lane_id != wp.lane_id or abs(((nxt[0].transform.rotation.yaw - yaw0) + 180) % 360 - 180) > yaw_tol:
                    ok = False
                    break
                ahead = nxt[0]
                lft = ahead.get_left_lane()
                if lft is None or lft.lane_type != carla.LaneType.Driving or lft.lane_id * ahead.lane_id <= 0:
                    ok = False
                    break
            if ok:
                return wp
        return None

    def _frame(self):
        """Curvilinear frame along the SUT lane centre, sampled every metre (the stretch may bend by up to 1 deg)."""
        wps = [self.wp0] + [self.wp0.next(float(d))[0] for d in range(1, int(self.length) + 1)]
        self.loc = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
        self.zs = np.array([w.transform.location.z for w in wps])
        self.yaws = np.unwrap(np.radians([w.transform.rotation.yaw for w in wps]))
        self.w = self.wp0.lane_width
        self.yaw = float(self.yaws[0])

    def _at(self, x):
        i = float(np.clip(x, 0., len(self.loc) - 1.001))
        k, f = int(i), i - int(i)
        loc = self.loc[k] + (self.loc[k + 1] - self.loc[k]) * f
        yaw = self.yaws[k] + (self.yaws[k + 1] - self.yaws[k]) * f
        z = self.zs[k] + (self.zs[k + 1] - self.zs[k]) * f
        extra = x - i  # beyond the sampled stretch: extrapolate along the end tangent
        fwd, right = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        return loc + fwd * extra, yaw, z, fwd, right

    def transform(self, x, y, heading=0.):
        loc, yaw, z, fwd, right = self._at(x)
        p = loc + right * (y - self.w)
        return self.carla.Transform(self.carla.Location(float(p[0]), float(p[1]), float(z) + .3),
                                    self.carla.Rotation(yaw=math.degrees(yaw + heading)))

    def velocity(self, vx, vy, x=0.):
        _, _, _, fwd, right = self._at(x)
        v = fwd * vx + right * vy
        return self.carla.Vector3D(float(v[0]), float(v[1]), 0.)

    def state(self, actor):
        """(x, y, vx, vy, heading) of an actor in the curvilinear road frame (metres)."""
        tf, vel = actor.get_transform(), actor.get_velocity()
        p = np.array([tf.location.x, tf.location.y])
        k = int(np.argmin(np.sum((self.loc - p) ** 2, axis=1)))
        yaw = self.yaws[k]
        fwd, right = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        d, v = p - self.loc[k], np.array([vel.x, vel.y])
        heading = math.atan2(math.sin(math.radians(tf.rotation.yaw) - yaw), math.cos(math.radians(tf.rotation.yaw) - yaw))
        return float(k + d @ fwd), float(self.w + d @ right), float(v @ fwd), float(v @ right), heading


class _View:
    """highway-env vehicle view used by our controllers (position, velocity, heading, LENGTH, WIDTH, act)."""

    def __init__(self, x, y, vx, vy, heading, length, width):
        self.position, self.velocity = np.array([x, y]), np.array([vx, vy])
        self.heading, self.LENGTH, self.WIDTH = heading, length, width
        self.speed = math.hypot(vx, vy)
        self.action = None

    def act(self, action):
        self.action = action


class ShimTarget:
    """Runs one of our controllers on CARLA states and converts its acceleration request into vehicle control."""

    def __init__(self, kind, cruise, base):
        from fairtest import cc_target, ref_models, sut_targets
        from src.evaluation.predictive_recognition import controller
        self.kind, self.cruise, self.prev_a, self.integ = kind, cruise, 0., 0.
        self.v_cmd, self.braking = None, False
        self._v_prev, self.b_prev, self.d_hat, self.req_prev = None, 0., 0., 0.  # h2 actuation state
        # ':hw' suffix: the controller perceives highway-env geometry (lateral scaled by 4/w, 5 m x 2 m vehicles) while
        # the vehicle is simulated in CARLA -- separates perception geometry from vehicle physics
        self.hw = kind.endswith(':hw')
        kind = kind[:-3] if self.hw else kind
        if kind == 'cc':
            self.ctrl, self.target_id = cc_target.CCDriver(), 'cc_r157'
        else:
            _, name, h = kind.split(':')
            h = float(h)
            self.target_id = name
            if name in ('following', 'reference_idm'):
                self.ctrl = controller(base['idm_reference'], h)
            elif name == 'following_aeb':
                self.ctrl = sut_targets.FollowingAEB(base['idm_reference'], h)
            elif name == 'rss_sut':
                self.ctrl = ref_models.RSSSut(base['idm_reference'])
            elif name == 'rss_lat_sut':
                self.ctrl = ref_models.RSSLatSut(base['idm_reference'])
            else:
                raise ValueError(kind)

    def step(self, t, gen_state, sut_state, dims, vehicle, carla):
        if self.hw:
            k = LANE_HW / self._w
            gx, gy, gvx, gvy, gh = gen_state
            sx, sy, svx, svy, sh = sut_state
            g, s = _View(gx, gy * k, gvx, gvy * k, gh, 5., 2.), _View(sx, sy * k, svx, svy * k, sh, 5., 2.)
        else:
            g = _View(*gen_state, *dims)
            s = _View(*sut_state, *dims)
        shim = SimpleNamespace(inner=SimpleNamespace(vehicle=g, road=SimpleNamespace(vehicles=[g, s])), target=s, cc_clock=t)
        applied, _ = self.ctrl(shim, self.target_id, None, self.cruise, self.prev_a, 'viable_v1')
        self.prev_a = float(applied)
        x, y, vx, vy, heading = sut_state
        # Stanley lane keeper towards the SUT lane centre (y = w)
        steer = -heading + math.atan2(.8 * (self._w - y), max(vx, 1.))
        ctl = carla.VehicleControl(steer=float(np.clip(steer / STEER_MAX_RAD, -1., 1.)))
        actuate(self, self.prev_a, vx, vehicle, ctl, carla)
        vehicle.apply_control(ctl)
        return self.prev_a


def actuate(tgt, req, vx, vehicle, ctl, carla):
    """Longitudinal actuation of a shim-driven vehicle for the request `req` (m/s^2) at the measured speed `vx`; sets
    the constant-velocity mode or ctl.brake. `tgt` keeps the state (v_cmd, braking, integ, _measured for 'v1';
    _v_prev, b_prev, d_hat, req_prev for 'h2')."""
    if ACT == 'v1':
        if req > -.3:
            if tgt.braking:  # resume cruising from the speed actually reached after braking
                tgt.v_cmd, tgt.braking = vx, False
            tgt.v_cmd = max(0., tgt.v_cmd + req * DT)
            vehicle.enable_constant_velocity(carla.Vector3D(tgt.v_cmd, 0., 0.))
            tgt.integ = 0.
        else:
            tgt.braking = True
            vehicle.disable_constant_velocity()
            err = req - tgt._measured
            tgt.integ = float(np.clip(tgt.integ + err * DT, -2., 2.))
            decel = -req - BRAKE_KP * err - BRAKE_KI * tgt.integ
            ctl.brake = float(np.clip((decel - BRAKE_A0) / BRAKE_K, 0., 1.))
        return
    meas = 0. if tgt._v_prev is None else (vx - tgt._v_prev) / DT  # acceleration over the last tick
    tgt._v_prev = vx
    if req > CV_FLOOR:
        if tgt.braking:  # resume cruising from the speed actually reached after braking
            tgt.v_cmd, tgt.braking = vx, False
        elif getattr(tgt, 'cv_goal', None) is not None:  # constant-velocity mode runs ~0.5% slow: feed the speed error back
            tgt.cv_corr = getattr(tgt, 'cv_corr', 0.) + CV_GAIN * (tgt.cv_goal - vx)
        tgt.v_cmd = max(0., tgt.v_cmd + req * DT)
        tgt.cv_goal = tgt.v_cmd
        vehicle.enable_constant_velocity(carla.Vector3D(tgt.v_cmd + getattr(tgt, 'cv_corr', 0.), 0., 0.))
        tgt.d_hat = 0.
    else:
        if tgt.braking:  # disturbance over the last braking tick: measured minus what the brake map should have given
            tgt.d_hat = meas + BRAKE_A0 + BRAKE_K * tgt.b_prev if tgt.b_prev > 0. else meas - tgt.req_prev
        else:
            tgt.d_hat = 0.
        tgt.braking, tgt.cv_goal = True, None
        vehicle.disable_constant_velocity()
        tgt.b_prev = float(np.clip((-req - BRAKE_A0 + tgt.d_hat) / BRAKE_K, 0., 1.))
        ctl.brake = tgt.b_prev
    tgt.req_prev = req


def ba_plan(road, x_from):
    """Lane-following waypoints from x_from to 150 m beyond the stretch (no lane changes)."""
    return [road.wp0.next(float(d))[0] for d in range(int(x_from) + 5, int(road.length) + 150, 2)]


class BATarget:
    def __init__(self, behavior, vehicle, road, cruise, plan):
        sys.path.append(os.path.join(CARLA_ROOT, 'PythonAPI', 'carla'))
        from agents.navigation.behavior_agent import BehaviorAgent
        from agents.navigation.local_planner import RoadOption
        self.agent = BehaviorAgent(vehicle, behavior=behavior)
        self.agent._behavior.max_speed = cruise * 3.6
        self.agent._behavior.speed_lim_dist = -1000.
        self.agent.set_global_plan([(w, RoadOption.LANEFOLLOW) for w in plan], stop_waypoint_creation=True, clean_queue=True)

    def step(self, t, gen_state, sut_state, dims, vehicle, carla):
        vehicle.apply_control(self.agent.run_step())
        return None


class TMTarget:
    """CARLA's Traffic Manager autopilot: lane following without lane changes, desired speed = the case's cruise
    speed, other vehicles never ignored; `tm:<d>` sets the distance to the leading vehicle (m)."""

    def __init__(self, vehicle, cruise, kind):
        global _TM
        if _TM is None:  # configure once per process: repeating the synchronous-mode setup per episode hung the client
            _TM = _CLIENT.get_trafficmanager(TM_PORT)
            _TM.set_synchronous_mode(True)
            _TM.set_random_device_seed(0)
        tm = self.tm = _TM
        vehicle.set_autopilot(True, TM_PORT)
        tm.auto_lane_change(vehicle, False)
        tm.random_left_lanechange_percentage(vehicle, 0.)
        tm.random_right_lanechange_percentage(vehicle, 0.)
        tm.keep_slow_lane_rule_percentage(vehicle, 0.)
        tm.ignore_vehicles_percentage(vehicle, 0.)
        tm.ignore_lights_percentage(vehicle, 100.)
        tm.ignore_signs_percentage(vehicle, 100.)
        tm.set_desired_speed(vehicle, cruise * 3.6)
        d = kind.split(':')[1]
        if d != 'default':
            tm.distance_to_leading_vehicle(vehicle, float(d))

    def step(self, t, gen_state, sut_state, dims, vehicle, carla):
        return None  # the Traffic Manager applies its control during world.tick()


def is_agent(kind):
    """CARLA's own agents (BehaviorAgent, Traffic Manager) drive themselves; our controllers run through the shim."""
    return kind.split(':')[0] in ('ba', 'tm')


def spawn_x(kind, v0):
    return START_X - ((BA_CV_S + BA_SETTLE_S) if is_agent(kind) else PREROLL_S) * v0


def start_target(world, road, sut, kind, sut0, cruise, base, dims):
    """Bring the SUT to the case speed at START_X (t = 0) and return its controller (see `run`)."""
    import carla
    w = road.w
    if is_agent(kind):
        for _ in range(int(BA_CV_S / DT)):
            sut.enable_constant_velocity(carla.Vector3D(sut0['vx'], 0., 0.))
            world.tick()
        sut.disable_constant_velocity()
        if kind.startswith('ba:'):
            target = BATarget(kind.split(':')[1], sut, road, cruise, ba_plan(road, road.state(sut)[0]))
        else:
            target = TMTarget(sut, cruise, kind)
        for _ in range(int(20. / DT)):
            target.step(0., None, None, dims, sut, carla)
            world.tick()
            x, y = road.state(sut)[:2]
            if x >= START_X and abs(y - w) < 1.:
                return target
        raise RuntimeError(f'{kind} did not reach the start point')
    target = ShimTarget(kind, cruise, base)
    target._w, target._measured = w, 0.
    sut.set_target_velocity(road.velocity(sut0['vx'], 0., spawn_x(kind, sut0['vx'])))
    corr = 0.
    for i in range(int(PREROLL_S / DT)):
        x, y, vx, vy, h = road.state(sut)
        if ACT != 'v1' and i > 0:  # h2.1: constant-velocity speed feedback (the mode runs ~0.5% slow)
            corr += CV_GAIN * (sut0['vx'] - vx)
        sut.enable_constant_velocity(carla.Vector3D(sut0['vx'] + corr, 0., 0.))
        sut.apply_control(carla.VehicleControl(steer=float(np.clip((-h + math.atan2(.8 * (w - y), max(vx, 1.))) / STEER_MAX_RAD, -1, 1))))
        world.tick()
    target.cv_corr = corr
    return target


def release(sut, kind, world):
    """Hand the SUT back before it is destroyed. A Traffic Manager vehicle is unregistered and the world ticked twice
    first: destroying it while the synchronous Traffic Manager still held it deadlocked a later tick."""
    try:
        if kind.startswith('tm:'):
            sut.set_autopilot(False, TM_PORT)
            world.tick()
            world.tick()
        elif not is_agent(kind):
            sut.disable_constant_velocity()
    except RuntimeError:
        pass


def _brakes(vehicle):
    pc = vehicle.get_physics_control()
    wheels = pc.wheels
    for wh in wheels:
        wh.max_brake_torque *= BRAKE_SCALE
    pc.wheels = wheels
    vehicle.apply_physics_control(pc)


def run(world, road, bp, traj_row, kind, base):
    """One CARLA episode of a fixed adversary trajectory against one target kind.

    Shim targets start PREROLL_S before START_X at the case speed (constant-velocity pre-roll). BehaviorAgent targets
    are lifted to the case speed in constant-velocity mode for BA_CV_S and then drive themselves along a
    lane-following plan for about BA_SETTLE_S before t = 0 (a velocity reset corrupts CARLA's drivetrain state for
    many seconds; after a constant-velocity lift the transient lasts < 1 s); t = 0 is when the SUT passes START_X.
    Traffic Manager targets are brought to t = 0 the same way as BehaviorAgent targets.
    """
    import carla
    w = road.w
    sut0, cruise = traj_row['sut0'], traj_row['cruise']
    tr = traj_row['trajectory']
    ba = is_agent(kind)
    sut = world.spawn_actor(bp, road.transform(spawn_x(kind, sut0['vx']), w))
    adv = world.spawn_actor(bp, road.transform(min(road.length - 20., START_X + 250.), 0.))
    out = dict(kind=kind)
    try:
        _brakes(sut)
        _brakes(adv)
        world.tick()
        e, ea = sut.bounding_box.extent, adv.bounding_box.extent
        dims = (2 * e.x, 2 * e.y)
        target = start_target(world, road, sut, kind, sut0, cruise, base, dims)
        xs0, _, v0 = road.state(sut)[:3]
        shift = xs0 - sut0['x']  # align the recorded x frame to the SUT's actual t = 0 position
        out['v0_error'] = round(v0 - sut0['vx'], 3)
        last_v, min_acc, spikes, min_gap, min_clear = v0, 0., 0, math.inf, math.inf
        hit, t_hit = False, None
        def place(p):
            gx, gy = p['x'] + shift, p['y'] * w / LANE_HW
            gvx, gvy = p['vx'], p['vy'] * w / LANE_HW
            gh = math.atan2(gvy, max(gvx, .1))
            adv.set_transform(road.transform(gx, gy, gh))
            adv.set_target_velocity(road.velocity(gvx, gvy, gx))
            return gx, gy, gvx, gvy, gh
        g_now = place(tr[0])  # tr[k] is the adversary at t = k * DT
        if not ba:
            target.v_cmd = v0
        for k in range(HORIZON):
            st = road.state(sut)
            if not ba:
                target._measured = (st[2] - last_v) / DT
            target.step(k * DT, g_now, st, dims, sut, carla)
            world.tick()
            gx, gy, gvx, gvy, gh = g_now = place(tr[k + 1])  # both vehicles now at t = (k + 1) * DT
            st = road.state(sut)
            acc = (st[2] - last_v) / DT
            min_acc, spikes, last_v = min(min_acc, acc), spikes + (acc < -11.), st[2]
            if abs(gy - st[1]) < ea.y + e.y and gx > st[0]:
                min_gap = min(min_gap, (gx - st[0]) - (ea.x + e.x))
            min_clear = min(min_clear, clearance(sut, adv))
            if min_clear <= 0.:
                hit, t_hit = True, round((k + 1) * DT, 1)
                out['contact_dx'] = round(gx - st[0], 2)  # adversary centre ahead of the SUT centre (>= 3.5 m: front contact)
                break
        out.update(contact=hit, t_contact=t_hit, sut_min_acc=round(min_acc, 2), decel_spikes=spikes,
                   min_gap=None if math.isinf(min_gap) else round(min_gap, 2), min_clearance=round(min_clear, 3))
    finally:
        release(sut, kind, world)
        sut.destroy()
        adv.destroy()
        world.tick()
    return out


class _World:
    """carla.World whose tick() refreshes the watchdog heartbeat; everything else is delegated."""

    def __init__(self, world):
        self._w = world

    def tick(self, *args):
        _BEAT[0] = time.time()
        frame = self._w.tick(*args)
        _BEAT[0] = time.time()
        return frame

    def __getattr__(self, name):
        return getattr(self._w, name)


def _watchdog():
    """Exit (code 3) when no world tick happened for WATCHDOG_S: the Traffic Manager occasionally deadlocks the client
    in synchronous mode. The supervisors (`run_carla_job.sh` etc.) then restart the server and resume."""
    import threading

    def loop():
        while True:
            time.sleep(5.)
            if time.time() - _BEAT[0] > WATCHDOG_S:
                print(json.dumps(dict(watchdog=f'no world tick for {WATCHDOG_S:g} s, exiting')), flush=True)
                os._exit(3)
    threading.Thread(target=loop, daemon=True).start()


def connect(port):
    import carla
    global _CLIENT, TM_PORT
    client = carla.Client('localhost', port)
    client.set_timeout(60.)
    _CLIENT, TM_PORT = client, 8000 + port - 2000
    _BEAT[0] = time.time() + 600.  # map loading and the road frame may take minutes; the first ticks re-arm it
    _watchdog()
    world = client.get_world()
    if not world.get_map().name.endswith('Town04'):
        world = client.load_world('Town04')
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    s.no_rendering_mode = True  # no cameras are used: skip rendering (GPU-bound otherwise); physics unchanged
    world.apply_settings(s)
    stale = world.get_actors().filter('vehicle.*')  # left behind by a killed client: the server keeps them
    for actor in stale:
        actor.destroy()
    if len(stale):
        world.tick()
    return _World(world)


def main(a):
    from fairtest.runner import _STATE, _init
    _init()
    base = _STATE['base']
    world = connect(a.port)
    road = Road(world)
    bp = world.get_blueprint_library().find(BLUEPRINT)
    try:
        if a.selftest:
            # synthetic attack: adversary 15 m ahead at the SUT's speed, cuts in over 3 s and brakes at 3 m/s^2
            tr, x, y, v = [], 25., 0., 27.
            for k in range(HORIZON + 1):
                t = k * DT
                y = min(LANE_HW, max(0., (t - 1.) / 3. * LANE_HW))
                v = max(15., v - (3. * DT if t > 2. else 0.))
                x += v * DT
                tr.append(dict(t=t, x=x, y=y, vx=v, vy=LANE_HW / 3. if 1. < t < 4. else 0., heading=0.))
            row = dict(sut0=dict(x=10., y=4., vx=27., vy=0.), cruise=27., trajectory=tr)
            for kind in a.kinds.split(','):
                t0 = time.time()
                print(json.dumps(dict(**run(world, road, bp, row, kind, base), seconds=round(time.time() - t0, 1))), flush=True)
            return
        rows = [json.loads(l) for l in open(a.trajectories, encoding='utf-8')]
        if a.resume and os.path.exists(a.out):  # restart after a CARLA server crash: skip replayed trajectories
            done = {(r['source'], r['case_id'], r['sut']) for r in (json.loads(l) for l in open(a.out, encoding='utf-8') if l.strip())}
            rows = [r for r in rows if (r['source'], r['case_id'], r['sut']) not in done]
        kinds = a.kinds.split(',')
        with open(a.out, 'a', encoding='utf-8') as f:
            for i, row in enumerate(rows):
                res = dict(source=row['source'], case_id=row['case_id'], sut=row['sut'], hwenv=row['hwenv'], carla={})
                for kind in kinds:
                    try:
                        res['carla'][kind] = run(world, road, bp, row, kind, base)
                    except Exception as e:  # recorded, never dropped
                        res['carla'][kind] = dict(error=f'{type(e).__name__}: {e}')
                f.write(json.dumps(res) + '\n')
                f.flush()
                if (i + 1) % 10 == 0:
                    print(json.dumps(dict(done=i + 1, of=len(rows))), flush=True)
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?')
    ap.add_argument('trajectories', nargs='?')
    ap.add_argument('--kinds', default='cc,ladder:following:0,ladder:following_aeb:0,ladder:rss_sut:0,ba:cautious,ba:normal,ba:aggressive')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--resume', action='store_true', help='skip trajectories already in OUT')
    main(ap.parse_args())
