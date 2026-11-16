"""B19: a tunable perception-threshold ADS (the design variable exposed by B17).

`ads@<d_p>`: the CC control structure with ADS-level parameters — the adjacent vehicle is treated as a
cut-in once its lateral displacement from its own lane centre reaches d_p (metres) while ahead;
perception is released below d_p (strict rule); 0.3 s latency with a 0.4 m/s^2 pre-brake; emergency
braking at 8 m/s^2 (jerk 15 m/s^3) when the longitudinal TTC is <= 2 s; otherwise the linear
following law capped at 3 m/s^2. Lower d_p reacts earlier (fewer fair failures) but perceives ordinary
lane-keeping wander of overtaken vehicles as cut-ins (phantom braking in natural traffic).
"""
from __future__ import annotations

import math

from fairtest.cc_target import following_law
from fairtest.ref_models import _states, project
from src.environments.hierarchical_interaction_env import lane_controller

LIMITS = dict(acceleration=8., jerk=15.)
LATENCY_S, PRE_MPS2, EMERGENCY_MPS2, TTC_S, COMFORT_MPS2 = .3, -.4, -8., 2., -3.


class PerceptionADS:
    jerk, floor = 15., -8.

    def __init__(self, perceive_m):
        self.perceive_m = float(perceive_m)
        self.perceived_at = None
        self.emergency = False

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g, t = _states(env)
        displacement = abs(g.y)
        ahead = g.x > t.x
        if self.perceived_at is None and ahead and displacement >= self.perceive_m:
            self.perceived_at = env.cc_clock
        elif self.perceived_at is not None and displacement < self.perceive_m and not self.emergency:
            self.perceived_at = None
        gap = (g.x - t.x) - (g.length + t.length) / 2
        closing = t.vx - g.vx
        ttc = gap / closing if ahead and closing > 0 and gap > 0 else (0. if ahead and gap <= 0 else None)
        if self.perceived_at is None:
            request, phase = following_law(g, t, cruise), 'free'
        elif env.cc_clock - self.perceived_at < LATENCY_S - 1e-9:
            request, phase = PRE_MPS2, 'latency'
        else:
            if self.emergency or (ttc is not None and ttc <= TTC_S):
                self.emergency = closing > 0 or ttc is not None
            if self.emergency:
                request, phase = EMERGENCY_MPS2, 'emergency'
            else:
                request, phase = max(COMFORT_MPS2, following_law(g, t, cruise)), 'following'
        applied = float(project(request, math.hypot(t.vx, t.vy), previous_a, self.jerk, -self.floor))
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=following_law(g, t, cruise),
                             requested_mps2=request, post_shield_mps2=request, post_stop_mps2=request,
                             previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                             viability=dict(constraint_infeasible=False), shield=dict(intervened=False),
                             recognition=dict(cc_phase=phase, perceived_at_s=self.perceived_at, emergency=self.emergency,
                                              model=f'ads@{self.perceive_m:g}'))
