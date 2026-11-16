"""B20 (G3): the R157 CC driver with the regulation's time-based perception (post-B13 development).

UN R157 defines the careful-and-competent driver's risk perception as 0.4 s after the cutting-in
vehicle leaves its lateral wandering zone (0.375 m); the widely used fixed-distance implementation
(Mattas et al. 2022; JAMA) replaces this by 0.72 m = 1.8 m/s x 0.4 s. The two coincide only at a lateral
speed of 1.8 m/s: faster cut-ins are perceived earlier by the distance rule than R157 intends (the
reference becomes more capable, inflating 'fair' labels), slower ones later.

`cc_r157_time`: perception 0.4 s after the first sample at which the vehicle ahead is outside its
wandering zone, provided it is still outside; perception is released when the vehicle returns into the
wandering zone. If the scenario starts with the vehicle already outside the zone (our initial states are
the natural events' 10 % lateral-progress points), the exit time is taken as t = 0. Reaction 0.75 s,
emergency braking at TTC <= 2 s ramping to 7.59 m/s^2 at 12.65 m/s^3, as for cc_r157.
"""
from __future__ import annotations

import math

from fairtest.cc_target import (COMFORT_MPS2, DT, EMERGENCY_MPS2, JERK_MPS3, ORIGIN_LANE_Y, REACTION_MPS2,
                                 REACTION_S, TTC_EMERGENCY_S, WANDER_M, following_law)
from src.environments.hierarchical_interaction_env import lane_controller
from src.evaluation.safety_metrics import VehicleState

PERCEPTION_S = .4


class CCTimeDriver:
    def __init__(self, reaction_s=REACTION_S, perception_s=PERCEPTION_S):
        self.reaction_s, self.perception_s = float(reaction_s), float(perception_s)
        self.exit_at = None
        self.perceived_at = None
        self.emergency = False

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g = VehicleState.from_vehicle(env.inner.vehicle)
        t = VehicleState.from_vehicle(env.target)
        displacement = abs(g.y - ORIGIN_LANE_Y)
        ahead = g.x > t.x
        now = env.cc_clock
        if displacement < WANDER_M and not self.emergency:
            self.exit_at = None
            self.perceived_at = None
        elif ahead and self.exit_at is None:
            self.exit_at = now
        if self.perceived_at is None and self.exit_at is not None and now - self.exit_at >= self.perception_s - 1e-9:
            self.perceived_at = now
        gap = (g.x - t.x) - (g.length + t.length) / 2
        closing = t.vx - g.vx
        ttc = gap / closing if ahead and closing > 0 and gap > 0 else (0. if ahead and gap <= 0 else None)
        phase = 'free'
        if self.perceived_at is None:
            request = following_law(g, t, cruise)
        elif now - self.perceived_at < self.reaction_s - 1e-9:
            request, phase = REACTION_MPS2, 'reaction'
        else:
            if self.emergency or (ttc is not None and ttc <= TTC_EMERGENCY_S):
                self.emergency = closing > 0 or ttc is not None
            if self.emergency:
                request, phase = EMERGENCY_MPS2, 'emergency'
            else:
                request, phase = max(COMFORT_MPS2, following_law(g, t, cruise)), 'following'
        lower = max(EMERGENCY_MPS2, previous_a - JERK_MPS3 * DT)
        upper = min(2.0, previous_a + JERK_MPS3 * DT)
        applied = float(min(upper, max(lower, request)))
        speed = math.hypot(t.vx, t.vy)
        applied = max(applied, -speed / DT)
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        telemetry = dict(raw_action=[applied / 5, 0.], following_mps2=following_law(g, t, cruise),
                         requested_mps2=request, post_shield_mps2=request, post_stop_mps2=request,
                         previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                         viability=dict(constraint_infeasible=False), shield=dict(intervened=False),
                         recognition=dict(cc_phase=phase, perceived_at_s=self.perceived_at, exit_at_s=self.exit_at,
                                          lateral_displacement=displacement, ttc=ttc, emergency=self.emergency))
        return applied, telemetry
