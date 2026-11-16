"""B4 pure causal entry-timing prototype; native performance is unmeasured.

Consumes completed public state and its own low-level planner. Never receives
the simulator, target H/intent, case identity, recorded future, or risk labels.
The overlap forecast is a heuristic using current headings/velocities, not TTC
measurement or a safety certificate. B2 physical auditing remains mandatory.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral

from src.evaluation.causal_lateral_reference import CommittedLateralPlanner
from src.evaluation.safety_metrics import VehicleState


@dataclass(frozen=True)
class EntryView:
    tick: int
    generator: VehicleState
    target: VehicleState
    backward_vy: float
    backward_ay: float

    def __post_init__(self):
        if isinstance(self.tick, bool) or not isinstance(self.tick, Integral) or self.tick not in range(0, 120, 5):
            raise ValueError('Completed 2Hz decision tick in the registered 12s horizon required')
        if not isinstance(self.generator, VehicleState) or not isinstance(self.target, VehicleState):
            raise ValueError('Public vehicle state required')
        if not all(math.isfinite(x) for x in (self.backward_vy, self.backward_ay)):
            raise ValueError('Finite completed lateral history required')
        if self.tick == 0 and (self.backward_vy != self.generator.vy or self.backward_ay != 0.):
            raise ValueError('Reset history must be explicit constant-velocity synthetic history')


def half_extent(vehicle, axis):
    c, s = abs(math.cos(vehicle.heading)), abs(math.sin(vehicle.heading))
    return (vehicle.length*c+vehicle.width*s)/2 if axis == 'x' else (vehicle.length*s+vehicle.width*c)/2


def entry_forecast(view, own_planner):
    """Clone own controller and ask for its exact B2 lane4 proposal, then CV scan."""
    if type(own_planner) is not CommittedLateralPlanner:
        raise ValueError('Own B2 planner state required')
    if own_planner.horizon != 12. or own_planner.durations != (2., 3., 4., 5., 6.):
        raise ValueError('Frozen B2 reference contract required')
    expected_previous = None if view.tick == 0 else (view.tick-5)*.1
    if own_planner.last_request_time != expected_previous:
        raise ValueError('Own planner must reflect exactly the preceding decision')
    p=deepcopy(own_planner)
    time=view.tick*.1
    try:
        receipt=p.request(time,4.,y=view.generator.y,vy=view.backward_vy,ay=view.backward_ay)
    except ValueError:
        return dict(eligible=False,reason='no_initial_reference',samples=0)
    if receipt['applied_lane_goal'] != 4.:
        return dict(eligible=False,reason='lane4_switch_rejected',samples=0)
    g,t=view.generator,view.target
    radius_y=half_extent(g,'y')+half_extent(t,'y')
    radius_x=half_extent(g,'x')+half_extent(t,'x')
    for offset in range(121-view.tick):
        tau=offset*.1
        if abs(p.evaluate(time+tau)-(t.y+t.vy*tau)) <= radius_y:
            gap=g.x-t.x-radius_x+(g.vx-t.vx)*tau
            closing=t.vx-g.vx
            return dict(eligible=True,reason='first_predicted_lateral_overlap',samples=offset+1,
                        overlap_delay_s=tau,predicted_x_body_gap_m=gap,closing_speed_mps=closing,
                        reference_duration_s=p.reference.duration,
                        request_entry=closing>0. and 0. <= gap <= 1.5*closing)
    return dict(eligible=True,reason='no_predicted_lateral_overlap',samples=121-view.tick,request_entry=False)


class CausalEntryAgent:
    """Brake in lane0 until the CV entry band is met; then commit to lane4.

    Composite actions: 1=lane0/brake, 4=lane4/brake. A failed/missed entry window
    keeps lane0; there is no deadline-forced merge or outcome-driven tuning.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_tick=None
        self.entered=False

    def act(self, view, own_planner):
        if type(view) is not EntryView:
            raise ValueError('Exact public observation DTO required')
        expected=0 if self.last_tick is None else self.last_tick+5
        if view.tick != expected:
            raise ValueError('Reset once, then provide every completed decision in order')
        if self.entered:
            # Check provenance even when the policy no longer scans candidates.
            if own_planner.goal != 4. or own_planner.last_request_time != (view.tick-5)*.1:
                raise ValueError('Committed entry was not applied by the own controller')
            receipt=dict(reason='continue_committed_entry',samples=0,request_entry=True)
        else:
            receipt=entry_forecast(view,own_planner)
        self.entered=self.entered or receipt.get('request_entry',False)
        self.last_tick=view.tick
        return (4 if self.entered else 1),dict(receipt,tick=view.tick,entered=self.entered,
                                              policy='B4-causal-entry-v1',native_validated=False)


def registered_fixed_plans():
    plans=[dict(method=f'constant_{a}',actions=[a]*24) for a in range(6)]
    for seconds in (2,4,6,8):
        plans.append(dict(method=f'brake_then_enter_{seconds}s',
                          actions=[1 if k*.5<seconds else 4 for k in range(24)]))
    return plans
