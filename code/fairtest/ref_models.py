"""B10: additional competent-driver reference models and a competent SUT (post-B9 development).

Parameters follow Mattas et al. (2022), Accident Analysis & Prevention 174:106743, Tables 1-2.
All reference drivers share the CC perception trigger (cut-in vehicle >= 1.095 m from its lane
centre, ahead of the ego) so that only the longitudinal response model differs.

ref_fsm  Fuzzy Safety Model: after tau=0.75 s, proactive (PFS) and critical (CFS) fuzzy
         surrogates; deceleration = CFS*(b_max-b_comf)+b_comf if CFS>0 else PFS*b_comf,
         b_comf=3, b_max=6, b_cutin_max=7, d1=2 m, s1=0.1 s. Jerk limit 12.65 m/s^3 (as CC).
         CFS here uses the relative-motion analogue of the PFS distances (documented approximation).
ref_rss  RSS: after rho=0.75 s, if the gap is below the RSS safe longitudinal distance
         d = v_r*rho + a_acc*rho^2/2 + (v_r+rho*a_acc)^2/(2 a_min_brake) - v_f^2/(2 a_max_brake)
         (a_acc=3, a_min_brake=a_max_brake=6), brake at a_min_brake=6 until safe or not closing.
rss_sut  competent SUT: proactive RSS controller with no extra reaction delay once the cut-in
         vehicle is perceived at lateral overlap-prediction (H=0.5 s CV lateral projection) and
         brakes up to 8 m/s^2 (jerk 15 m/s^3) whenever the RSS distance is violated.
rss_lat_sut  the same, but the cut-in vehicle is also relevant from the CC perception point on (>= 1.095 m
         from its lane centre, ahead). It perceives no later than the CC, reacts without the CC's 0.75 s and
         brakes harder (8 vs 7.59 m/s^2), so it dominates the reference by construction (B25 specificity SUT).
"""
from __future__ import annotations

import math

from fairtest.cc_target import CCDriver, following_law
from src.environments.hierarchical_interaction_env import lane_controller
from src.evaluation.predictive_recognition import predicted_entry
from src.evaluation.safety_metrics import VehicleState
from dataclasses import asdict

DT = .1
PERCEIVE_Y = .375 + .72


def viable_floor(speed, jerk, amax, dt=DT):
    """Most negative acceleration from which braking can be released at `jerk` before reversing.

    Generalises stopping_viability.project (fixed 5 m/s^2) to the reference drivers' capability,
    so that stopping never requires a jerk spike (B10 fix, 3/159,600 episodes affected).
    """
    def loss(a):
        if a >= 0:
            return 0.
        step = jerk * dt
        n = math.ceil(-a / step)
        return dt * (n * -a - step * n * (n - 1) / 2)
    lo, hi = -amax, 0.
    if loss(lo) <= max(0., speed - 1e-6):
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        if loss(mid) <= max(0., speed - 1e-6):
            hi = mid
        else:
            lo = mid
    return hi


def project(request, speed, previous, jerk, amax, dt=DT):
    floor = viable_floor(speed, jerk, amax, dt)
    lower = max(-amax, previous - jerk * dt, floor)
    upper = min(2., previous + jerk * dt)
    if lower > upper:  # already non-viable: preserve stopping, disclose via jerk check
        return max(floor, min(2., request))
    return min(upper, max(lower, request))


def _states(env):
    return VehicleState.from_vehicle(env.inner.vehicle), VehicleState.from_vehicle(env.target)


def _gap(g, t):
    return (g.x - t.x) - (g.length + t.length) / 2


def rss_distance(v_rear, v_front, rho=.75, a_acc=3., a_min_brake=6., a_max_brake=6.):
    return max(0., v_rear * rho + .5 * a_acc * rho * rho + (v_rear + rho * a_acc) ** 2 / (2 * a_min_brake)
               - v_front ** 2 / (2 * a_max_brake))


class _Reference:
    """Shared perception / reaction bookkeeping identical to the CC driver."""
    reaction_s = .75
    jerk = 12.65
    floor = -7.59
    name = 'ref'

    def __init__(self):
        self.perceived_at = None
        self.active = False

    def response(self, g, t, cruise):
        raise NotImplementedError

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g, t = _states(env)
        displacement = abs(g.y)
        ahead = g.x > t.x
        if self.perceived_at is None and ahead and displacement >= PERCEIVE_Y:
            self.perceived_at = env.cc_clock
        elif self.perceived_at is not None and displacement < .375 and not self.active:
            self.perceived_at = None
        if self.perceived_at is None:
            request, phase = following_law(g, t, cruise), 'free'
        elif env.cc_clock - self.perceived_at < self.reaction_s - 1e-9:
            request, phase = -.4, 'reaction'
        else:
            request, phase = self.response(g, t, cruise)
        applied = float(project(request, math.hypot(t.vx, t.vy), previous_a, self.jerk, -self.floor))
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=following_law(g, t, cruise),
                             requested_mps2=request, post_shield_mps2=request, post_stop_mps2=request,
                             previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                             viability=dict(constraint_infeasible=False), shield=dict(intervened=False),
                             recognition=dict(cc_phase=phase, perceived_at_s=self.perceived_at,
                                              emergency=self.active, model=self.name))


class FSMDriver(_Reference):
    name = 'ref_fsm'
    b_comf, b_max, b_cut, d1, s1 = 3., 6., 7., 2., .1
    floor = -6.

    def response(self, g, t, cruise):
        d = _gap(g, t)
        u_e, u_c = t.vx, g.vx
        d_safe = u_e * self.reaction_s + u_e ** 2 / (2 * self.b_comf) - u_c ** 2 / (2 * self.b_cut) + self.d1
        d_unsafe = u_e * self.reaction_s + u_e ** 2 / (2 * self.b_max) - u_c ** 2 / (2 * self.b_cut)
        pfs = 1. if d <= d_unsafe else (0. if d >= d_safe else (d_safe - d) / max(1e-9, d_safe - d_unsafe))
        cfs = 0.
        if u_e > u_c:
            rel = u_e - u_c
            c_safe = rel * self.reaction_s + rel ** 2 / (2 * self.b_comf) + self.d1
            c_unsafe = rel * self.s1 + rel ** 2 / (2 * self.b_max)
            cfs = 1. if d <= c_unsafe else (0. if d >= c_safe else (c_safe - d) / max(1e-9, c_safe - c_unsafe))
        self.active = cfs > 0 or pfs > 0
        if cfs > 0:
            return -(cfs * (self.b_max - self.b_comf) + self.b_comf), 'fsm_critical'
        if pfs > 0:
            return -(pfs * self.b_comf), 'fsm_proactive'
        return max(-self.b_comf, following_law(g, t, cruise)), 'following'


class RSSDriver(_Reference):
    name = 'ref_rss'
    a_brake = 6.
    floor = -6.

    def response(self, g, t, cruise):
        d = _gap(g, t)
        unsafe = d < rss_distance(t.vx, g.vx)
        closing = t.vx > g.vx
        self.active = unsafe and (closing or self.active)
        if self.active:
            return -self.a_brake, 'rss_brake'
        return max(-3., following_law(g, t, cruise)), 'following'


class RSSSut:
    """Competent SUT: RSS-proactive longitudinal controller (no reaction delay, 8 m/s^2)."""
    jerk, floor = 15., -8.
    lateral_aware = False

    def __init__(self, parameters, horizon=.5):
        self.parameters, self.horizon = parameters, float(horizon)

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        g, t = _states(env)
        gd, td = asdict(g), asdict(t)
        overlap = abs(g.y - t.y) < (g.width + t.width) / 2
        predicted = predicted_entry(gd, td, self.horizon, 'following')['predicted']
        perceived = self.lateral_aware and abs(g.y) >= PERCEIVE_Y  # the CC's perception point (origin lane at y = 0)
        relevant = g.x > t.x and (overlap or predicted or perceived)
        d = _gap(g, t)
        request = following_law(g, t, cruise)
        phase = 'base'
        if relevant and d < rss_distance(t.vx, g.vx, rho=0., a_acc=0., a_min_brake=6., a_max_brake=6.):
            request, phase = -8., 'rss_brake'
        applied = float(project(request, math.hypot(t.vx, t.vy), previous_a, self.jerk, -self.floor))
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=request, requested_mps2=request,
                             post_shield_mps2=request, post_stop_mps2=request, previous_mps2=previous_a,
                             applied_mps2=applied, steering=steer, viability=dict(constraint_infeasible=False),
                             shield=dict(intervened=False), recognition=dict(rss_phase=phase, predicted=predicted))


class RSSLatSut(RSSSut):
    """Lateral-aware competent SUT (see module docstring): dominates the R157 CC by construction."""
    lateral_aware = True


LIMITS = {'ref_fsm': dict(acceleration=6., jerk=12.65), 'ref_rss': dict(acceleration=6., jerk=12.65),
          'rss_sut': dict(acceleration=8., jerk=15.), 'rss_lat_sut': dict(acceleration=8., jerk=15.)}
