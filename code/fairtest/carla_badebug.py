"""E7 debug: BehaviorAgent speed holding after a velocity reset (no adversary nearby), per-tick speed / acceleration /
control, for several pre-roll strategies.

  python -m fairtest.carla_badebug [--port 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from fairtest.carla_replay import BLUEPRINT, BRAKE_SCALE, CARLA_ROOT, DT, Road, connect


def main(a):
    import carla
    sys.path.append(os.path.join(CARLA_ROOT, 'PythonAPI', 'carla'))
    from agents.navigation.behavior_agent import BehaviorAgent
    world = connect(a.port)
    road = Road(world)
    bp = world.get_blueprint_library().find(BLUEPRINT)
    print(json.dumps(dict(road_length=road.length, lane_width=road.w)))
    for mode in ('constant_velocity', 'constant_then_velocity'):
        v = world.spawn_actor(bp, road.transform(20., road.w))
        try:
            pc = v.get_physics_control()
            wheels = pc.wheels
            for wh in wheels:
                wh.max_brake_torque *= BRAKE_SCALE
            pc.wheels = wheels
            v.apply_physics_control(pc)
            world.tick()
            if mode == 'constant_then_velocity':
                v.set_target_velocity(road.velocity(28., 0.))
            for _ in range(30):  # 3 s constant-velocity pre-roll
                v.enable_constant_velocity(carla.Vector3D(28., 0., 0.))
                world.tick()
            v.disable_constant_velocity()
            agent = BehaviorAgent(v, behavior='normal')
            agent._behavior.max_speed = 28. * 3.6
            agent._behavior.speed_lim_dist = -1000.
            from agents.navigation.local_planner import RoadOption
            agent.set_global_plan([(road.wp0.next(float(d))[0], RoadOption.LANEFOLLOW) for d in range(int(road.state(v)[0]) + 10, int(road.length) - 10, 2)], stop_waypoint_creation=True, clean_queue=True)
            trace, last = [], road.state(v)[2]
            for k in range(150):
                ctl = agent.run_step()
                v.apply_control(ctl)
                world.tick()
                s = road.state(v)[2]
                trace.append((round(k * DT, 1), round(s, 2), round((s - last) / DT, 2), round(ctl.throttle, 2), round(ctl.brake, 2)))
                last = s
            spikes = [x for x in trace if x[2] < -8.]
            print(json.dumps(dict(mode=mode, spikes=len(spikes), first_spikes=spikes[:5], speed_at=[x[1] for x in trace[::25]],
                                  min_acc=min(x[2] for x in trace))))
        finally:
            v.destroy()
            world.tick()
    s = world.get_settings()
    s.synchronous_mode = False
    world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=2000)
    main(ap.parse_args())
