"""B5-L development prototype. Pure own-control arithmetic, NOT a simulator.

Keep B4 reference overlap time/current extents; change only own x/vx forecast.
No target response, contact or outcome model. Native performance is unmeasured.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np
from src.agents.causal_entry_agent import EntryView,half_extent
from src.evaluation.adversarial_native_action import acceleration_command,steering_command
from src.evaluation.causal_lateral_reference import CommittedLateralPlanner
from src.evaluation.safety_metrics import VehicleState


@dataclass(frozen=True)
class BrakeView:
    public: EntryView
    scalar_speed: float
    previous_measured_acceleration: float
    previous_applied_steering: float

    def __post_init__(self):
        if type(self.public) is not EntryView:raise ValueError('Exact public DTO required')
        values=(self.scalar_speed,self.previous_measured_acceleration,self.previous_applied_steering)
        if any(isinstance(x,bool) or not math.isfinite(x) for x in values):raise ValueError('Finite own telemetry required')
        if not 15.<=self.scalar_speed<=40. or abs(self.previous_measured_acceleration)>3.+1e-10 or abs(self.previous_applied_steering)>.3:
            raise ValueError('Own telemetry outside frozen authority')
        if abs(self.scalar_speed-math.hypot(self.public.generator.vx,self.public.generator.vy))>1e-9:
            raise ValueError('Own scalar speed must match completed public velocity')
        if self.public.tick==0 and (self.previous_measured_acceleration!=0. or self.previous_applied_steering!=0.):
            raise ValueError('Registered reset actuator state is zero')


def own_step(state,speed,previous_a,previous_steer,y_reference,request=-3.):
    """Existing B2 steering/projection plus isolated precontact bicycle equations."""
    if request not in (0.,-3.,3.):raise ValueError('Frozen longitudinal request required')
    desired_vy=(y_reference-state.y)/.1
    beta_request=math.asin(float(np.clip(desired_vy/speed,-.99,.99)))-state.heading
    steer_request=math.atan(2*math.tan(beta_request))
    _,steer=steering_command(steer_request,previous_steer)
    _,projection=acceleration_command(request,speed,previous_a)
    applied=projection['applied_mps2']
    beta=np.arctan(.5*np.tan(steer))
    x=state.x+speed*np.cos(state.heading+beta)*.1
    y=state.y+speed*np.sin(state.heading+beta)*.1
    heading=state.heading+speed*np.sin(beta)/(state.length/2)*.1
    next_speed=speed+applied*.1
    result=VehicleState(float(x),float(y),float(next_speed*np.cos(heading)),float(next_speed*np.sin(heading)),float(heading),state.length,state.width)
    # Native B2 feeds measured scalar acceleration, not requested/applied ideal,
    # into the next projection. Preserve exactly that update order.
    measured=(next_speed-speed)/.1
    return result,next_speed,measured,steer,dict(applied_acceleration=applied,applied_steering=steer,
        y_reference=y_reference,requested_acceleration=request)


def own_rollout(view,planner,ticks,request=-3.):
    if type(view) is not BrakeView or type(planner) is not CommittedLateralPlanner:
        raise ValueError('Pure DTO and own committed planner required')
    if isinstance(ticks,bool) or not isinstance(ticks,int) or not 0<=ticks<=120-view.public.tick:
        raise ValueError('Bounded in-horizon model ticks required')
    state=view.public.generator;speed=view.scalar_speed
    a=view.previous_measured_acceleration;steer=view.previous_applied_steering
    rows=[dict(state=state,scalar_speed=speed,measured_acceleration=a,applied_steering=steer)]
    for k in range(ticks):
        y=planner.evaluate((view.public.tick+k+1)*.1)
        state,speed,a,steer,control=own_step(state,speed,a,steer,y,request)
        rows.append(dict(state=state,scalar_speed=speed,measured_acceleration=a,applied_steering=steer,control=control))
    return rows


def braking_forecast(view,own_planner):
    if type(view) is not BrakeView or type(own_planner) is not CommittedLateralPlanner:
        raise ValueError('Exact pure inputs required')
    v=view.public
    if own_planner.horizon!=12. or own_planner.durations!=(2.,3.,4.,5.,6.):raise ValueError('Frozen B2 planner contract required')
    if own_planner.last_request_time!=(None if v.tick==0 else (v.tick-5)*.1):raise ValueError('Previous decision planner required')
    p=deepcopy(own_planner);time=v.tick*.1
    try:r=p.request(time,4.,y=v.generator.y,vy=v.backward_vy,ay=v.backward_ay)
    except ValueError:return dict(eligible=False,reason='no_initial_reference',samples=0,model_ticks=0,request_entry=False)
    if r['applied_lane_goal']!=4.:return dict(eligible=False,reason='lane4_switch_rejected',samples=0,model_ticks=0,request_entry=False)
    g,t=v.generator,v.target
    rx=half_extent(g,'x')+half_extent(t,'x');ry=half_extent(g,'y')+half_extent(t,'y')
    for offset in range(121-v.tick):
        tau=offset*.1
        # Exactly the B4 geometry: candidate reference and CURRENT headings.
        if abs(p.evaluate(time+tau)-(t.y+t.vy*tau))<=ry:
            own=own_rollout(view,p,offset)[-1]['state']
            gap=own.x-(t.x+t.vx*tau)-rx;closing=t.vx-own.vx
            return dict(eligible=True,reason='first_predicted_lateral_overlap',samples=offset+1,model_ticks=offset,
                overlap_delay_s=tau,predicted_x_body_gap_m=gap,closing_speed_mps=closing,
                reference_duration_s=p.reference.duration,request_entry=closing>0. and 0.<=gap<=1.5*closing,
                model_generator_x=own.x,model_generator_vx=own.vx)
    return dict(eligible=True,reason='no_predicted_lateral_overlap',samples=121-v.tick,model_ticks=0,request_entry=False)


class BrakingEntryAgent:
    def __init__(self):self.reset()
    def reset(self):self.last_tick=None;self.entered=False
    def act(self,view,own_planner):
        if type(view) is not BrakeView:raise ValueError('Exact own/public DTO required')
        if type(own_planner) is not CommittedLateralPlanner or own_planner.horizon!=12. or own_planner.durations!=(2.,3.,4.,5.,6.):
            raise ValueError('Exact frozen own planner required')
        tick=view.public.tick
        if tick!=(0 if self.last_tick is None else self.last_tick+5):raise ValueError('Ordered complete decision history required')
        if self.entered:
            if own_planner.goal!=4. or own_planner.last_request_time!=(tick-5)*.1:raise ValueError('Applied entry commitment required')
            receipt=dict(reason='continue_committed_entry',samples=0,model_ticks=0,request_entry=True)
        else:receipt=braking_forecast(view,own_planner)
        self.entered=self.entered or receipt['request_entry'];self.last_tick=tick
        return (4 if self.entered else 1),dict(receipt,tick=tick,entered=self.entered,policy='B5-L-braking-entry-v1',native_validated=False)
