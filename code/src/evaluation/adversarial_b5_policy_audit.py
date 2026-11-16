"""B5 raw-state/own-telemetry reconstruction and independent policy audit."""
from dataclasses import asdict
import math
import numpy as np
from src.evaluation.adversarial_b4_policy_audit import audit_policy as audit_b4,view_from_states,planner_snapshot
from src.evaluation.adversarial_b2_audit import candidate
from src.evaluation.braking_entry_audit import independent_step
from src.evaluation.safety_metrics import VehicleState


def own_telemetry(row,tick):
    if tick==0:return row['initial_generator_scalar_speed'],0.,0.
    prev=row['trace'][tick-1]
    speed=prev['generator_scalar_speed']
    before=row['initial_generator_scalar_speed'] if tick==1 else row['trace'][tick-2]['generator_scalar_speed']
    return speed,(speed-before)/.1,prev['generator_control']['native_steering']


def expected_forecast(view,telemetry):
    """Physical-seconds polynomial and independent closed-form actuator model."""
    g,t=view.generator,view.target;time=view.tick*.1;checks=0;chosen=None
    for duration in (2.,3.,4.,5.,6.):
        if time+duration>12.:continue
        checks+=1;p,eligible=candidate(g.y,view.backward_vy,view.backward_ay,4.,duration)
        if eligible:chosen=p,duration;break
    if chosen is None:
        return dict(action=1,eligible=False,reason='no_initial_reference' if view.tick==0 else 'lane4_switch_rejected',samples=0,model_ticks=0,checks=checks)
    p,duration=chosen
    def extent(v,axis):
        c,s=abs(math.cos(v.heading)),abs(math.sin(v.heading))
        return (v.length*c+v.width*s)/2 if axis=='x' else (v.length*s+v.width*c)/2
    rx=extent(g,'x')+extent(t,'x');ry=extent(g,'y')+extent(t,'y')
    def y(tau):return 4. if tau>=duration else float(p(tau))
    for offset in range(121-view.tick):
        tau=offset*.1
        if abs(y(tau)-(t.y+t.vy*tau))<=ry:
            state=g;speed,a,steer=telemetry
            for k in range(offset):state,speed,a,steer,_=independent_step(state,speed,a,steer,y((k+1)*.1),-3.)
            gap=state.x-(t.x+t.vx*tau)-rx;closing=t.vx-state.vx
            return dict(action=4 if closing>0 and 0<=gap<=1.5*closing else 1,eligible=True,
                reason='first_predicted_lateral_overlap',samples=offset+1,model_ticks=offset,checks=checks,
                overlap_delay_s=tau,predicted_x_body_gap_m=gap,closing_speed_mps=closing,
                reference_duration_s=duration,model_generator_x=state.x,model_generator_vx=state.vx)
    return dict(action=1,eligible=True,reason='no_predicted_lateral_overlap',samples=121-view.tick,model_ticks=0,checks=checks)


def audit_policy(row):
    if row['method']!='braking_entry':return dict(audit_b4(row),model_ticks=0)
    saved=row['policy_inputs'];decisions=row['decisions'];initial=row['physical_initial_state']
    if len(saved)!=len(decisions):raise ValueError('Missing B5 policy sidecar')
    ys=[initial[0]['y']]+[t['generator']['y'] for t in row['trace']]
    entered=False;entry_tick=None;checks=samples=model_ticks=0;forecasts=[]
    previous=dict(goal=None,start=None,last_request_time=None,duration=None,coefficients=None)
    for i,(s,d) in enumerate(zip(saved,decisions)):
        tick=i*5
        if d['tick']!=tick or s['tick']!=tick or s['action']!=d['action']:raise ValueError('B5 policy decision identity drift')
        if not math.isfinite(s['seconds']) or s['seconds']<0:raise ValueError('Invalid inference timing')
        states=initial if tick==0 else [row['trace'][tick-1][k] for k in ('generator','target')]
        v=view_from_states(tick,*(VehicleState(**x) for x in states),ys[:tick+1]);telemetry=own_telemetry(row,tick)
        expected_view=dict(public=asdict(v),scalar_speed=telemetry[0],previous_measured_acceleration=telemetry[1],previous_applied_steering=telemetry[2])
        if s['view']!=expected_view:raise ValueError('B5 policy consumed a different pre-step state or own telemetry')
        if s['planner']!=previous:raise ValueError('Own planner is not from preceding decision')
        r=s['forecast']
        if r['tick']!=tick or r['policy']!='B5-L-braking-entry-v1' or r['native_validated'] is not False:raise ValueError('B5 policy receipt identity drift')
        expected=dict(action=4,reason='continue_committed_entry',samples=0,model_ticks=0,checks=0) if entered else expected_forecast(v,telemetry)
        if entered and previous['goal']!=4.:raise ValueError('Entry commitment drift')
        if d['action']!=expected['action'] or s['candidate_checks']!=expected['checks']:raise ValueError('Independent B5 decision/cost mismatch')
        for key in ('reason','samples','model_ticks'):
            if r[key]!=expected[key]:raise ValueError('Independent B5 forecast cost/reason mismatch: '+key)
        if not entered and r['eligible']!=expected['eligible']:raise ValueError('B5 candidate eligibility drift')
        if 'overlap_delay_s' in expected:
            for key in ('overlap_delay_s','reference_duration_s','predicted_x_body_gap_m','closing_speed_mps','model_generator_x','model_generator_vx'):
                if not math.isfinite(r[key]) or abs(r[key]-expected[key])>1e-5:raise ValueError('Independent B5 forecast arithmetic mismatch: '+key)
            f=dict(tick=tick,predicted_overlap_delay_s=expected['overlap_delay_s'],predicted_x_body_gap_m=r['predicted_x_body_gap_m'],
                interpretation='Own/target prediction errors only when actual lane4 commitment and precontact trace support the predicted time; not risk labels.')
            end=tick+expected['model_ticks']
            # Prediction was for lane4. Never attach action1 future as its truth.
            contact=min([len(row['trace'])]+[max(0,c['step']-1) for c in row.get('confirmed_contacts',[])])
            future=row['trace'][tick:end]
            supported=(d['action']==4 and end<=len(row['trace']) and end<=contact and
                all(q.get('agent_action')==4 and not any(x is not None for x in q.get('consumed_impacts',[])) for q in future))
            f['censored']=not supported
            if supported:
                actual=states if end==tick else [row['trace'][end-1][k] for k in ('generator','target')]
                tau=expected['overlap_delay_s']
                f.update(own_x_error_m=actual[0]['x']-r['model_generator_x'],own_vx_error_mps=actual[0]['vx']-r['model_generator_vx'],
                    target_cv_x_error_m=actual[1]['x']-(v.target.x+v.target.vx*tau),target_cv_vx_error_mps=actual[1]['vx']-v.target.vx)
            forecasts.append(f)
        if d['action']==4 and not entered:entry_tick=tick
        entered=entered or d['action']==4
        if r['entered']!=entered or r['request_entry']!=(d['action']==4):raise ValueError('B5 entry latch drift')
        samples+=r['samples'];checks+=s['candidate_checks'];model_ticks+=r['model_ticks']
        p=d['lateral_plan'];previous=dict(goal=p['applied_lane_goal'],start=p['reference_start'],last_request_time=tick*.1,duration=p['duration'],coefficients=p['coefficients'])
    requested=[d['tick'] for d in decisions if d['action']//3==1];applied=[d['tick'] for d in decisions if d['lateral_plan']['applied_lane_goal']==4.]
    return dict(entry_tick=entry_tick,entered=entered,forecast_samples=samples,candidate_checks=checks,model_ticks=model_ticks,
        first_requested_lane4_tick=requested[0] if requested else None,first_applied_lane4_tick=applied[0] if applied else None,
        inference_seconds=sum(s['seconds'] for s in saved),forecasts=forecasts)
