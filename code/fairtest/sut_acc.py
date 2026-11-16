"""B29 (W3): production-like ACC SUTs, built from published parameters rather than tuned by us.

acc_prod      adaptive cruise control
  * target selection by lane assignment: the cut-in vehicle becomes the ACC target once it is ahead and its centre is
    within 2.0 m of the ego lane centre (displacement >= 2.0 m from its own lane centre: the centre has crossed the
    lane marking); released below that (strict);
  * control law of Milanes & Shladover (2014, TR-C 48) for production ACC: a = k1 (gap - t_h v - d0) + k2 (v_lead - v)
    with k1 = 0.23 s^-2, k2 = 0.07 s^-1, time gap t_h = 1.5 s; speed control k0 (v_set - v), k0 = 0.4 s^-1, the smaller
    of the two applies;
  * ISO 15622 comfort limits of ACC: deceleration <= 3.5 m/s^2, jerk <= 2.5 m/s^3, acceleration <= 2 m/s^2;
  * assumptions (not from the sources): standstill distance d0 = 2 m, first-order actuator lag tau = 0.4 s.
acc_prod_aeb  the same with an AEB layer on the assigned target: TTC <= 1.5 s triggers, after 0.3 s latency, braking at
              9 m/s^2 (jerk 30 m/s^3) until the closing speed is gone (Euro NCAP-type AEB, parameters assumed).
"""
from __future__ import annotations

import math

from fairtest.ref_models import _gap, _states, project
from src.environments.hierarchical_interaction_env import lane_controller

K1, K2, K0, T_H, D0, TAU, DT = .23, .07, .4, 1.5, 2., .4, .1
ASSIGN_M = 2.0
ACC_DECEL, ACC_JERK = 3.5, 2.5
AEB_TTC, AEB_LATENCY, AEB_DECEL, AEB_JERK = 1.5, .3, 9., 30.
LIMITS = {'acc_prod': dict(acceleration=ACC_DECEL, jerk=ACC_JERK), 'acc_prod_aeb': dict(acceleration=AEB_DECEL, jerk=AEB_JERK)}


class ProductionACC:
    def __init__(self, aeb=False):
        self.aeb = bool(aeb)
        self.a_filtered = 0.
        self.aeb_armed_at = None
        self.aeb_active = False
        self.clock = 0.

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g, t = _states(env)
        assigned = g.x > t.x and abs(g.y) >= ASSIGN_M
        a_speed = K0 * (cruise - t.vx)
        request, phase = a_speed, 'speed'
        gap = _gap(g, t)
        if assigned:
            a_follow = K1 * (gap - T_H * t.vx - D0) + K2 * (g.vx - t.vx)
            if a_follow < a_speed:
                request, phase = a_follow, 'follow'
        # first-order actuator lag on the ACC request
        self.a_filtered += (request - self.a_filtered) * DT / TAU
        request = self.a_filtered
        jerk, amax = ACC_JERK, ACC_DECEL
        if self.aeb:
            closing = t.vx - g.vx
            ttc = gap / closing if assigned and closing > 0 and gap > 0 else None
            if not self.aeb_active and self.aeb_armed_at is None and ttc is not None and ttc <= AEB_TTC:
                self.aeb_armed_at = self.clock
            if self.aeb_armed_at is not None and self.clock - self.aeb_armed_at >= AEB_LATENCY - 1e-9:
                self.aeb_active = closing > 0 and assigned
                if not self.aeb_active:
                    self.aeb_armed_at = None
            if self.aeb_active:
                request, phase, jerk, amax = -AEB_DECEL, 'aeb', AEB_JERK, AEB_DECEL
        speed = math.hypot(t.vx, t.vy)
        applied = float(project(request, speed, previous_a, jerk, amax))
        if phase != 'aeb':
            self.a_filtered = applied  # the lag state follows what the actuator delivered
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        self.clock += DT
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=request, requested_mps2=request,
                             post_shield_mps2=request, post_stop_mps2=request, previous_mps2=previous_a,
                             applied_mps2=applied, steering=steer, viability=dict(constraint_infeasible=False),
                             shield=dict(intervened=False),
                             recognition=dict(phase=phase, assigned=assigned, aeb=self.aeb_active, model=target_id))
