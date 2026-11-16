"""B69 third-party SUT: the CommonRoad reactive planner (commonroad-reactive-planner 2025.1, BSD-3-Clause, TUM; sampled
quartic/quintic polynomials in a Frenet frame after Werling et al. 2010) decides the SUT's longitudinal acceleration in
this simulator; this simulator stays the physics and the judge.

Every 0.1 s tick the planner replans from the SUT's current state (rear-axle position, speed, heading and the acceleration
it is applying) along the centre line of the SUT's lane, with the adversary as a dynamic obstacle of its size predicted at
constant velocity over the planner's horizon. It returns the cheapest kinematically feasible, collision-free sampled
trajectory under its default comfort cost (acceleration, deviation from the set speed, lateral offset), and the SUT applies
that trajectory's speed change over the first step, a = (v(0.1 s) - v) / 0.1. The planner has no fallback above
standstill: when every sampled trajectory collides with the prediction the SUT brakes at the slot's 9 m/s^2 (phase
`fallback` in the telemetry). The set speed is the SUT's initial speed, as for every other SUT here.

Planner settings are the package defaults (dt 0.1 s, 60 steps = 6 s horizon, 4 sampling levels, t_min 0.4 s, deceleration
ratio 0.125, no safety margin around obstacles, default cost weights, the road boundary of the two-lane road) except:
the vehicle (5 x 2 m as in this simulator and a_max 9 m/s^2, the slot's contract; axle distances of the default vehicle);
lateral end states d = 0 only (the planner keeps its lane as every SUT here does, premise (iv) of Prop. 1; the current
offset stays a sample, as in the package); single-process checking; logging off. The lane's coordinate system gets
CommonRoad-CLCS's default parameters explicitly (the planner's wrapper passes none, which the installed commonroad-clcs
2025.2 no longer accepts). A stopped SUT passes 1e-6 m/s as its current speed to set_desired_velocity, whose zero-speed
branch reads a planning problem this setting does not have.
The planner keeps no headway of its own: with no safety margin and a constant-velocity prediction it slows only once a
predicted collision falls inside its 6 s horizon (probe 2026-10-04: 20 m behind a vehicle 1 m/s slower it holds speed).
The applied acceleration keeps this simulator's actuator rules (never reverse; at most +2 m/s^2) and the 9 m/s^2 contract
of the `acc_prod_aeb` slot it is run in; the planner's step-to-step changes are not jerk-limited.
"""
from __future__ import annotations

import math
import os

import numpy as np

from fairtest.ref_models import _states, project
from src.environments.hierarchical_interaction_env import lane_controller

DT = .1
LANE_Y = 4.  # centre of the SUT's lane; the adversary's origin lane is centred at y = 0 (4 m lanes)
ROAD = (-500., 6000.)
FALLBACK = -9.
_CR = {}


def _planner():
    """Planner, road boundary and lane coordinate system, built once per process (episodes replan from scratch)."""
    if _CR.get('pid') == os.getpid():
        return _CR
    import logging

    import commonroad_dc.pycrcc as pycrcc
    from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
    from commonroad.scenario.scenario import Scenario
    from commonroad_clcs.config import CLCSParams
    from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
    from commonroad_rp.reactive_planner import ReactivePlanner
    from commonroad_rp.state import ReactivePlannerState
    from commonroad_rp.utility.config import ReactivePlannerConfiguration
    from commonroad_rp.utility.utils_coordinate_system import CoordinateSystem

    x = np.linspace(ROAD[0], ROAD[1], int((ROAD[1] - ROAD[0]) / 10.) + 1)

    def lane(lid, yc, **adjacent):
        line = lambda y: np.column_stack([x, np.full_like(x, y)])  # noqa: E731
        return Lanelet(line(yc + 2.), line(yc), line(yc - 2.), lid, **adjacent)

    network = LaneletNetwork.create_from_lanelet_list([
        lane(1, LANE_Y, adjacent_right=2, adjacent_right_same_direction=True),
        lane(2, 0., adjacent_left=1, adjacent_left_same_direction=True)])
    scenario = Scenario(dt=DT)
    scenario.add_objects(network)
    config = ReactivePlannerConfiguration()
    config.vehicle.length, config.vehicle.width, config.vehicle.a_max = 5., 2., 9.
    config.vehicle._update_computed_params()
    config.sampling.d_min = config.sampling.d_max = 0.
    config.debug.multiproc = False
    config.debug.logging_level = 'CRITICAL'
    config.scenario = scenario
    planner = ReactivePlanner(config)
    logging.getLogger('RP_LOGGER').setLevel(logging.CRITICAL)
    _, boundary = create_road_boundary_obstacle(scenario)
    _CR.clear()
    _CR.update(pid=os.getpid(), planner=planner, config=config, boundary=boundary, pycrcc=pycrcc, state=ReactivePlannerState,
               co=CoordinateSystem(np.column_stack([x, np.full_like(x, LANE_Y)]), clcs_params=CLCSParams()))
    return _CR


class CommonRoadSUT:
    """Per-episode controller with the target-control interface of this simulator (see sut_acc.ProductionACC)."""

    def __init__(self):
        self.cr = _planner()
        self.cr['planner']._planning_times_list.clear()
        self.clock = 0.

    def _prediction(self, g, steps):
        """The adversary at constant velocity, one oriented box per planner step from now (time step 0)."""
        pycrcc = self.cr['pycrcc']
        tvo = pycrcc.TimeVariantCollisionObject(0)
        for k in range(steps):
            tvo.append_obstacle(pycrcc.RectOBB(g.length / 2, g.width / 2, g.heading, g.x + g.vx * k * DT, g.y + g.vy * k * DT))
        return tvo

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        cr = self.cr
        planner = cr['planner']
        g, t = _states(env)
        v = math.hypot(t.vx, t.vy)
        rear = cr['config'].vehicle.wb_rear_axle
        x0 = cr['state'](time_step=0, position=np.array([t.x - rear * math.cos(t.heading), t.y - rear * math.sin(t.heading)]),
                         velocity=v, orientation=t.heading, acceleration=float(previous_a), yaw_rate=0., steering_angle=0.)
        cc = cr['pycrcc'].CollisionChecker()
        cc.add_collision_object(cr['boundary'])
        cc.add_collision_object(self._prediction(g, planner.N + 2))
        planner.reset(initial_state_cart=x0, collision_checker=cc, coordinate_system=cr['co'])
        planner.set_desired_velocity(float(cruise), current_speed=max(v, 1e-6))
        result = planner.plan()
        rejected = int(planner.infeasible_count_collision)
        if result is None:
            phase, request = 'fallback', FALLBACK
        else:
            lon = result[1]
            phase, request = 'plan', (lon[1][1] - lon[0][1]) / DT
        applied = float(project(request, v, previous_a, 1e3, 9.))
        steer = .3 * lane_controller(LANE_Y - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        self.clock += DT
        constrained = phase == 'fallback' or rejected > 0
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=request, requested_mps2=request,
                             post_shield_mps2=request, post_stop_mps2=request, previous_mps2=previous_a,
                             applied_mps2=applied, steering=steer, viability=dict(constraint_infeasible=False),
                             shield=dict(intervened=False),
                             recognition=dict(phase=phase, leader=constrained, rejected_collision=rejected,
                                              model='commonroad_rp'))
