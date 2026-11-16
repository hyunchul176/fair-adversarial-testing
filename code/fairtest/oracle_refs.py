"""B16: solvability-style references (post-B13 development) to contrast with the human CC benchmark.

oracle_t0      omniscient longitudinal solvability: brake at full capability from t = 0. For a follower
               facing a fixed lead trajectory this maximises the gap at every future instant, so a
               collision it cannot avoid is longitudinally unsolvable (the FREA/STRIVE-type criterion
               restricted to the lane-keeping follower).
oracle_react0  ideal ADS without foresight: CC perception trigger (cut-in vehicle >= 1.095 m from its lane
               centre, ahead), zero reaction delay, full capability while closing.
Capability of both: 8 m/s^2 at 15 m/s^3 (the strongest actuator in the SUT panel), with the stopping
viability projection of ref_models.project. Lane keeping as for every target.
"""
from __future__ import annotations

import math

from fairtest.cc_target import following_law
from fairtest.ref_models import PERCEIVE_Y, _states, project
from src.environments.hierarchical_interaction_env import lane_controller

LIMITS = {'oracle_t0': dict(acceleration=8., jerk=15.), 'oracle_react0': dict(acceleration=8., jerk=15.)}


class OracleBrake:
    jerk, floor = 15., -8.

    def __init__(self, mode):
        if mode not in ('t0', 'react0'):
            raise ValueError(mode)
        self.mode = mode
        self.perceived_at = None

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g, t = _states(env)
        ahead, closing = g.x > t.x, t.vx > g.vx
        if self.mode == 't0':
            request, phase = self.floor, 'oracle_brake'
        else:
            if self.perceived_at is None and ahead and abs(g.y) >= PERCEIVE_Y:
                self.perceived_at = env.cc_clock
            if self.perceived_at is not None and ahead and closing:
                request, phase = self.floor, 'oracle_brake'
            else:
                request, phase = following_law(g, t, cruise), 'following'
        applied = float(project(request, math.hypot(t.vx, t.vy), previous_a, self.jerk, -self.floor))
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=following_law(g, t, cruise),
                             requested_mps2=request, post_shield_mps2=request, post_stop_mps2=request,
                             previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                             viability=dict(constraint_infeasible=False), shield=dict(intervened=False),
                             recognition=dict(cc_phase=phase, perceived_at_s=self.perceived_at,
                                              emergency=phase == 'oracle_brake', model=target_id))
