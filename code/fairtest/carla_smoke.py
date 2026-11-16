"""E7 CARLA smoke test: connect, load a highway map, find a long straight multi-lane stretch, spawn two vehicles in
adjacent lanes and step the world synchronously at 10 Hz.

  python -m fairtest.carla_smoke [--port 2000] [--town Town06]
"""
from __future__ import annotations

import argparse
import json
import math
import time


def main(a):
    import carla
    client = carla.Client('localhost', a.port)
    client.set_timeout(60.)
    t0 = time.time()
    world = client.load_world(a.town)
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, .1
    world.apply_settings(settings)
    cmap = world.get_map()
    # straight driving lanes with a same-direction neighbour: scan waypoints every 5 m
    best = None
    for wp in cmap.generate_waypoints(5.):
        if wp.lane_type != carla.LaneType.Driving:
            continue
        left = wp.get_left_lane()
        if left is None or left.lane_type != carla.LaneType.Driving or left.lane_id * wp.lane_id <= 0:
            continue
        ahead, ok, yaw0 = wp, True, wp.transform.rotation.yaw
        for _ in range(60):  # 300 m straight
            nxt = ahead.next(5.)
            if not nxt:
                ok = False
                break
            ahead = nxt[0]
            if abs(((ahead.transform.rotation.yaw - yaw0) + 180) % 360 - 180) > 1. or ahead.lane_id != wp.lane_id:
                ok = False
                break
        if ok:
            best = wp
            break
    out = dict(town=a.town, load_s=round(time.time() - t0, 1), straight_found=best is not None)
    if best is not None:
        bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        sut_tf = best.transform
        sut_tf.location.z += .3
        adv_wp = best.get_left_lane().next(20.)[0]
        adv_tf = adv_wp.transform
        adv_tf.location.z += .3
        sut, adv = world.spawn_actor(bp, sut_tf), world.spawn_actor(bp, adv_tf)
        try:
            for v, speed in ((sut, 25.), (adv, 27.)):
                fwd = v.get_transform().get_forward_vector()
                v.set_target_velocity(carla.Vector3D(fwd.x * speed, fwd.y * speed, 0.))
            world.tick()
            for _ in range(30):
                sut.apply_control(carla.VehicleControl(throttle=.5))
                adv.apply_control(carla.VehicleControl(throttle=.5))
                world.tick()
            ext = sut.bounding_box.extent
            out.update(road_id=best.road_id, lane_id=best.lane_id, lane_width=round(best.lane_width, 2),
                       sut_speed=round(sut.get_velocity().length(), 2), adv_speed=round(adv.get_velocity().length(), 2),
                       gap_m=round(adv.get_location().distance(sut.get_location()), 1),
                       vehicle_extent=[round(ext.x * 2, 2), round(ext.y * 2, 2)])
        finally:
            sut.destroy()
            adv.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print(json.dumps(out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town06')
    main(ap.parse_args())
