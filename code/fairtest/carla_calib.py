"""E7: calibrate the CARLA vehicle's longitudinal response (acceleration vs throttle / brake at highway speed) and
check that the CARLA BehaviorAgent imports; --list prints the bounding boxes of the 4-wheel car blueprints (the
extent is only known for a spawned actor), closest to the highD passenger car (4.5 m x 1.92 m median) first.

  python -m fairtest.carla_calib [--port 2000] [--blueprint vehicle.tesla.model3] [--brake-scale 2.5]
  python -m fairtest.carla_calib --list [--port 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CARLA_ROOT = os.path.expanduser('~/carla_0916')
BLUEPRINT = 'vehicle.tesla.model3'


def straight_start(world, length_m=400.):
    import carla
    cmap = world.get_map()
    for wp in cmap.generate_waypoints(5.):
        if wp.lane_type != carla.LaneType.Driving:
            continue
        left = wp.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * wp.lane_id <= 0:
            continue
        ahead, ok, yaw0 = wp, True, wp.transform.rotation.yaw
        for _ in range(int(length_m / 5)):
            nxt = ahead.next(5.)
            if not nxt:
                ok = False
                break
            ahead = nxt[0]
            if abs(((ahead.transform.rotation.yaw - yaw0) + 180) % 360 - 180) > 1. or ahead.lane_id != wp.lane_id:
                ok = False
                break
        if ok:
            return wp
    return None


def blueprint_sizes(world, wp):
    """Length and width (m) of every 4-wheel car blueprint, spawned one at a time at wp."""
    out = []
    for bp in world.get_blueprint_library().filter('vehicle.*'):
        if bp.has_attribute('number_of_wheels') and int(bp.get_attribute('number_of_wheels')) != 4:
            continue
        base = bp.get_attribute('base_type').as_str() if bp.has_attribute('base_type') else ''
        tf = wp.transform
        tf.location.z += .5
        v = world.try_spawn_actor(bp, tf)
        if v is None:
            continue
        try:
            world.tick()
            e = v.bounding_box.extent
            out.append(dict(id=bp.id, base_type=base, length=round(2 * e.x, 3), width=round(2 * e.y, 3)))
        finally:
            v.destroy()
            world.tick()
    return sorted(out, key=lambda r: abs(r['width'] - 1.92) + .2 * abs(r['length'] - 4.5))


def main(a):
    import carla
    sys.path.append(os.path.join(CARLA_ROOT, 'PythonAPI', 'carla'))
    from agents.navigation.behavior_agent import BehaviorAgent  # noqa: F401  (import check)
    client = carla.Client('localhost', a.port)
    client.set_timeout(60.)
    world = client.get_world()
    if not world.get_map().name.endswith('Town04'):  # the server is started with Town04; reload only if needed
        world = client.load_world('Town04')
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, .1
    world.apply_settings(s)
    wp = straight_start(world, 600.)
    if a.list:
        for r in blueprint_sizes(world, wp):
            print(json.dumps(r))
        s.synchronous_mode = False
        world.apply_settings(s)
        return
    bp = world.get_blueprint_library().find(a.blueprint)
    out = dict(behavior_agent_import=True, points=[])
    for kind, level in [('throttle', x) for x in (0., .2, .4, .6, .8, 1.)] + [('brake', x) for x in (.1, .2, .3, .5, .7, 1.)]:
        tf = wp.transform
        tf.location.z += .3
        v = world.spawn_actor(bp, tf)
        try:
            if a.brake_scale != 1.:
                pc = v.get_physics_control()
                wheels = pc.wheels
                for w in wheels:
                    w.max_brake_torque *= a.brake_scale
                pc.wheels = wheels
                v.apply_physics_control(pc)
            fwd = tf.get_forward_vector()
            world.tick()
            v.set_target_velocity(carla.Vector3D(fwd.x * 28., fwd.y * 28., 0.))
            gear = dict(manual_gear_shift=True, gear=a.gear) if a.gear else {}
            for _ in range(5):  # settle at 28 m/s
                v.apply_control(carla.VehicleControl(throttle=.35, **gear))
                world.tick()
            speeds = []
            for _ in range(15):
                ctl = carla.VehicleControl(throttle=level, **gear) if kind == 'throttle' else carla.VehicleControl(brake=level, **gear)
                v.apply_control(ctl)
                world.tick()
                speeds.append(v.get_velocity().length())
            acc = (speeds[-1] - speeds[4]) / (10 * .1)  # mean over ticks 5..15 (after the actuation delay)
            out['points'].append(dict(kind=kind, level=level, v_start=round(speeds[4], 2), acc=round(acc, 3),
                                      acc_first_05s=round((speeds[4] - speeds[0]) / .4, 3)))
        finally:
            v.destroy()
            world.tick()
    s.synchronous_mode = False
    world.apply_settings(s)
    out['gear'], out['brake_scale'] = a.gear, a.brake_scale
    print(json.dumps(dict(blueprint=a.blueprint, gear=a.gear, brake_scale=a.brake_scale,
                          acc={f"{p['kind'][0]}{p['level']}": p['acc'] for p in out['points']})))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--gear', type=int, default=0, help='manual gear (0 = automatic)')
    ap.add_argument('--brake-scale', type=float, default=1.)
    ap.add_argument('--blueprint', default=BLUEPRINT)
    ap.add_argument('--list', action='store_true', help='print the car blueprints with their bounding boxes')
    main(ap.parse_args())
