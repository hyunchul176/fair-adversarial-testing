"""Additional systems under test for the B7 panel.

`following_aeb`: the unchanged `following` law (with its shield, stopping-viability projection
and optional predictive recognition H) plus an autonomous emergency braking stage. AEB engages
when the generator is ahead, laterally overlapping, closing, and the longitudinal bumper TTC is
below 1.5 s; it requests -8 m/s^2 with a 15 m/s^3 jerk limit and stays latched while closing.
After release it ramps back at 15 m/s^3 until the base law's |a|<=5 domain is re-entered.
This models an ADAS vehicle with late recognition but strong emergency braking.
"""
from __future__ import annotations

from dataclasses import asdict
import math

from src.environments.hierarchical_interaction_env import lane_controller
from src.evaluation.predictive_recognition import predicted_entry
from src.evaluation.recognition_attribution import commands
from src.evaluation.safety_metrics import VehicleState

AEB_TTC_S = 1.5
AEB_MPS2 = -8.0
AEB_JERK_MPS3 = 15.0
DT = .1
LIMITS = dict(acceleration=abs(AEB_MPS2), jerk=AEB_JERK_MPS3)


class FollowingAEB:
    def __init__(self, parameters, horizon, ttc_threshold=AEB_TTC_S):
        self.parameters, self.horizon = parameters, float(horizon)
        self.ttc_threshold = float(ttc_threshold)
        self.latched = False

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        if not target_id.startswith('following_aeb'):
            raise ValueError('FollowingAEB requires a following_aeb target id')
        g = asdict(VehicleState.from_vehicle(env.inner.vehicle))
        t = asdict(VehicleState.from_vehicle(env.target))
        gap = (g['x'] - t['x']) - (g['length'] + t['length']) / 2
        closing = t['vx'] - g['vx']
        overlap = abs(g['y'] - t['y']) < (g['width'] + t['width']) / 2
        ttc = gap / closing if overlap and gap > 0 and closing > 0 else None
        trigger = ttc is not None and ttc < self.ttc_threshold
        self.latched = (self.latched and closing > 0) or trigger
        base_previous = max(-5., min(5., previous_a))
        native = commands(g, t, cruise, self.parameters, base_previous)
        prediction = predicted_entry(g, t, self.horizon, 'following')
        early = bool(not native['following']['gate'] and prediction['predicted'])
        selected = commands(g, t, cruise, self.parameters, base_previous, True)['following'] if early else native['following']
        speed = math.hypot(t['vx'], t['vy'])
        if self.latched:
            request, phase = AEB_MPS2, 'aeb'
        elif previous_a < -5. - 1e-9:
            request, phase = selected['applied'], 'aeb_release'
        else:
            request, phase = None, 'base'
        if request is None:
            applied = selected['applied']
        else:
            lower = max(AEB_MPS2, previous_a - AEB_JERK_MPS3 * DT)
            upper = min(2., previous_a + AEB_JERK_MPS3 * DT)
            applied = min(upper, max(lower, request))
        applied = float(max(applied, -speed / DT))
        steer = .3 * lane_controller(4. - t['y'], t['heading'], preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        telemetry = dict(raw_action=[applied / 5, 0.], following_mps2=native['following']['request'],
                         requested_mps2=selected['request'] if request is None else request,
                         post_shield_mps2=selected['post_shield'], post_stop_mps2=selected['post_stop'],
                         previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                         viability=dict(constraint_infeasible=False), shield=dict(intervened=selected['shield_intervened']),
                         recognition=dict(native_gate=native['following']['gate'], early=early, aeb_phase=phase,
                                          aeb_ttc=ttc, aeb_latched=self.latched, **prediction))
        return applied, telemetry
