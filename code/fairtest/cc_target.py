"""UN R157 Competent-and-Careful (CC) human driver as a simulated target (reference SUT).

Parameters follow UN R157 Annex 4 Appendix 3 (original series; 'Performance model 1' of
Annex 3 in Rev.1 / 01 series, same values -- both checked against the text, B32) as
summarised and validated on highD by Mattas et al. (2022), Accident Analysis & Prevention
174:106743:
  * risk perception once the cutting-in vehicle leaves its 0.375 m wandering zone by a
    further 0.72 m (lateral displacement >= 1.095 m from its lane centre);
  * 0.75 s reaction time with a small 0.4 m/s^2 deceleration (foot off the pedal);
  * emergency braking only when TTC <= 2 s, ramping to 0.774 g = 7.59 m/s^2 within 0.6 s
    (jerk 12.65 m/s^3).
Outside an emergency the CC driver follows with the same linear law as the `following`
target, capped at the FSM comfortable deceleration of 3 m/s^2. The CC driver is a
preventability reference: a collision it cannot avoid is not attributed to the SUT.

The regulation's model is 'Reference Driver + AEB' (R157 original series, Annex 4 Appendix 3,
Figure 1 and Table 1): an AEB layer derived from UN R152, triggered at 'xx% wrap & TTC < TTC_AEB',
brakes 0.6 s to 0.85 g 'after full wrap'; xx%, TTC_AEB and the AEB delay are not specified.
The CC omits the layer. cc_r157_aeb (B32, post hoc) adds its strongest reading -- any body
overlap, TTC <= 2 s (the model's emergency-braking TTC), no delay -- so it bounds the effect.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from src.environments.hierarchical_interaction_env import lane_controller
from src.evaluation.safety_metrics import VehicleState

WANDER_M = 0.375
PERCEPTION_M = 0.72
REACTION_S = 0.75
REACTION_MPS2 = -0.4
EMERGENCY_MPS2 = -7.59
JERK_MPS3 = 12.65
TTC_EMERGENCY_S = 2.0
COMFORT_MPS2 = -3.0
ORIGIN_LANE_Y = 0.0
DT = 0.1

AEB_MPS2 = -0.85 * 9.81          # AEB layer (B32): 0.85 g ...
AEB_JERK_MPS3 = 0.85 * 9.81 / .6  # ... reached within 0.6 s
AEB_TTC_S = TTC_EMERGENCY_S

LIMITS = dict(acceleration=abs(EMERGENCY_MPS2), jerk=JERK_MPS3)
AEB_LIMITS = dict(acceleration=abs(AEB_MPS2), jerk=AEB_JERK_MPS3)


def following_law(g, t, cruise):
    c, s = math.cos(t.heading), math.sin(t.heading)
    v = t.vx * c + t.vy * s
    along = (g.x - t.x) * c + (g.y - t.y) * s
    delta = g.heading - t.heading
    radius = .5 * (g.length * abs(math.cos(delta)) + g.width * abs(math.sin(delta)))
    gap = max(0., along - t.length / 2 - radius)
    free = float(np.clip(.4 * (cruise - v), -5, 2))
    lateral_overlap = abs(g.y - t.y) < (g.width + t.width) / 2
    if along > 0 and lateral_overlap:
        return min(free, .18 * (gap - 5 - 1.8 * v) + .8 * ((g.vx - t.vx) * c + (g.vy - t.vy) * s))
    return free


class CCDriver:
    """Stateful per-episode controller with the predictive_recognition.controller interface."""

    def __init__(self, reaction_s=REACTION_S, unlatch_m=WANDER_M, emergency_mps2=EMERGENCY_MPS2, aeb=False,
                 perceive_m=WANDER_M + PERCEPTION_M, ttc_emergency_s=TTC_EMERGENCY_S, jerk_mps3=JERK_MPS3):
        self.reaction_s = float(reaction_s)
        self.emergency_mps2 = float(emergency_mps2)  # R157: -7.59; calibrated variants (cc_r157_cal) use less
        # B57 human-like references vary these too (defaults = the R157 values, so every other target is unchanged)
        self.perceive_m, self.ttc_emergency_s, self.jerk_mps3 = float(perceive_m), float(ttc_emergency_s), float(jerk_mps3)
        # Perception is released when the cutting-in vehicle returns below unlatch_m (default: its
        # wandering zone). The strict variant releases below the perception threshold itself.
        self.unlatch_m = float(unlatch_m)
        self.perceived_at = None
        self.emergency = False
        self.aeb, self.aeb_active = bool(aeb), False

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        if not target_id.startswith('cc_r157'):
            raise ValueError('CC driver controller requires a cc_r157 target id')
        if len(env.inner.road.vehicles) != 2:
            raise ValueError('Two-vehicle world required')
        g = VehicleState.from_vehicle(env.inner.vehicle)
        t = VehicleState.from_vehicle(env.target)
        displacement = abs(g.y - ORIGIN_LANE_Y)
        ahead = g.x > t.x
        if self.perceived_at is None and ahead and displacement >= self.perceive_m:
            self.perceived_at = env.cc_clock
        elif self.perceived_at is not None and displacement < self.unlatch_m and not self.emergency:
            self.perceived_at = None  # the cutting-in vehicle returned into its wandering zone
        # After perception the cutting-in vehicle is the lead: longitudinal bumper TTC (R157 model).
        gap = (g.x - t.x) - (g.length + t.length) / 2
        closing = t.vx - g.vx
        ttc = gap / closing if ahead and closing > 0 and gap > 0 else (0. if ahead and gap <= 0 else None)
        phase = 'free'
        if self.perceived_at is None:
            request = following_law(g, t, cruise)
        elif env.cc_clock - self.perceived_at < self.reaction_s - 1e-9:
            request, phase = REACTION_MPS2, 'reaction'
        else:
            if self.emergency or (ttc is not None and ttc <= self.ttc_emergency_s):
                self.emergency = closing > 0 or ttc is not None
            if self.emergency:
                request, phase = self.emergency_mps2, 'emergency'
            else:
                request, phase = max(COMFORT_MPS2, following_law(g, t, cruise)), 'following'
        floor, jerk = self.emergency_mps2, self.jerk_mps3
        if self.aeb:  # independent of the driver's perception and reaction; held while closing
            overlap = abs(g.y - t.y) < (g.width + t.width) / 2
            if not self.aeb_active and ahead and overlap and ttc is not None and ttc <= AEB_TTC_S:
                self.aeb_active = True
            elif self.aeb_active and not (closing > 0 or ttc is not None):
                self.aeb_active = False
            if self.aeb_active:
                request, phase, floor, jerk = AEB_MPS2, 'aeb', AEB_MPS2, AEB_JERK_MPS3
        lower = max(floor, previous_a - jerk * DT)
        upper = min(2.0, previous_a + jerk * DT)
        applied = float(min(upper, max(lower, request)))
        speed = math.hypot(t.vx, t.vy)
        applied = max(applied, -speed / DT)  # never reverse
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        telemetry = dict(raw_action=[applied / 5, 0.], following_mps2=following_law(g, t, cruise),
                         requested_mps2=request, post_shield_mps2=request, post_stop_mps2=request,
                         previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                         viability=dict(constraint_infeasible=False), shield=dict(intervened=False),
                         recognition=dict(cc_phase=phase, perceived_at_s=self.perceived_at,
                                          lateral_displacement=displacement, ttc=ttc,
                                          emergency=self.emergency, **({'aeb': True} if self.aeb_active else {})))
        return applied, telemetry
