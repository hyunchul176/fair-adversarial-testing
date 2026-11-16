"""Reconstruct policy inputs from completed raw states, independently check decisions."""
from dataclasses import asdict
import math
import numpy as np
from src.agents.causal_entry_agent import EntryView
from src.evaluation.safety_metrics import VehicleState
from src.evaluation.causal_entry_audit import expected_uncommitted_request
from src.evaluation.adversarial_b4_core import fixed_action


def view_from_states(tick, generator, target, ys):
    vy=generator.vy if tick==0 else (ys[-1]-ys[-2])/.1
    ay=0. if tick==0 else (ys[-1]-2*ys[-2]+ys[-3])/.01
    return EntryView(tick,generator,target,vy,ay)


def planner_snapshot(planner):
    return dict(goal=planner.goal,start=planner.start,last_request_time=planner.last_request_time,
                duration=planner.reference.duration if planner.reference is not None else None,
                coefficients=list(planner.reference.coefficients) if planner.reference is not None else None)


def audit_policy(row):
    from src.evaluation.adversarial_b2_audit import candidate
    saved=row['policy_inputs']; decisions=row['decisions']
    if len(saved)!=len(decisions): raise ValueError('Missing policy sidecar')
    initial=row['physical_initial_state']; ys=[initial[0]['y']]+[t['generator']['y'] for t in row['trace']]
    entered=False; samples=0; checks=0; entry_tick=None; forecasts=[]
    previous=dict(goal=None,start=None,last_request_time=None,duration=None,coefficients=None)
    for i,(s,d) in enumerate(zip(saved,decisions)):
        tick=i*5
        if d['tick']!=tick or s['tick']!=tick or s['action']!=d['action']: raise ValueError('Policy decision identity drift')
        if not math.isfinite(s['seconds']) or s['seconds']<0: raise ValueError('Invalid inference timing')
        if row['method']!='causal_entry':
            if d['action']!=fixed_action(row['method'],tick) or s['view'] is not None or s['forecast'] is not None or s['planner'] is not None or s['candidate_checks']!=0:
                raise ValueError('Fixed schedule drift')
        else:
            states=initial if tick==0 else [row['trace'][tick-1]['generator'],row['trace'][tick-1]['target']]
            view=view_from_states(tick,*(VehicleState(**v) for v in states),ys[:tick+1])
            if s['view']!=asdict(view): raise ValueError('Policy consumed a different pre-step state')
            if s['planner']!=previous: raise ValueError('Own planner is not from preceding decision')
            receipt=s['forecast']
            if receipt['tick']!=tick or receipt['policy']!='B4-causal-entry-v1' or receipt['native_validated'] is not False:
                raise ValueError('Policy receipt identity drift')
            expected=dict(action=4,samples=0,overlap_delay_s=None) if entered else expected_uncommitted_request(view)
            nchecks=0; chosen=None
            if not entered:
                for duration in (2.,3.,4.,5.,6.):
                    if tick*.1+duration>12.:continue
                    nchecks+=1
                    _,eligible=candidate(view.generator.y,view.backward_vy,view.backward_ay,4.,duration)
                    if eligible: chosen=duration;break
            if d['action']!=expected['action'] or receipt['samples']!=expected['samples'] or s['candidate_checks']!=nchecks:
                raise ValueError('Independent causal decision/cost mismatch')
            if entered:
                if receipt['reason']!='continue_committed_entry' or previous['goal']!=4.: raise ValueError('Entry commitment drift')
            elif expected['overlap_delay_s'] is not None:
                tau=expected['overlap_delay_s']; g,t=view.generator,view.target
                gx=(abs(math.cos(g.heading))*g.length+abs(math.sin(g.heading))*g.width)/2
                tx=(abs(math.cos(t.heading))*t.length+abs(math.sin(t.heading))*t.width)/2
                gap=g.x-t.x-gx-tx+(g.vx-t.vx)*tau
                np.testing.assert_allclose([receipt['overlap_delay_s'],receipt['predicted_x_body_gap_m'],receipt['closing_speed_mps'],receipt['reference_duration_s']],
                    [tau,gap,t.vx-g.vx,chosen],rtol=0,atol=1e-8)
                if receipt['reason']!='first_predicted_lateral_overlap': raise ValueError('Forecast reason drift')
                forecasts.append(dict(tick=tick,predicted_overlap_delay_s=tau,predicted_x_body_gap_m=gap))
            else:
                reason=('no_initial_reference' if tick==0 else 'lane4_switch_rejected') if chosen is None else 'no_predicted_lateral_overlap'
                if receipt['reason']!=reason: raise ValueError('Forecast rejection reason drift')
            if d['action']==4 and not entered:entry_tick=tick
            entered=entered or d['action']==4
            if receipt['entered']!=entered or bool(receipt.get('request_entry',False))!=(d['action']==4):raise ValueError('Entry latch drift')
            samples+=receipt['samples'];checks+=nchecks
        p=d['lateral_plan']
        previous=dict(goal=p['applied_lane_goal'],start=p['reference_start'],last_request_time=tick*.1,
                      duration=p['duration'],coefficients=p['coefficients'])
    # Descriptive prediction error, never a generated-risk label. Absence of a
    # realized overlap before termination remains censored rather than zero.
    def extent(v,axis):
        c,s=abs(math.cos(v['heading'])),abs(math.sin(v['heading']))
        return (v['length']*c+v['width']*s)/2 if axis=='x' else (v['length']*s+v['width']*c)/2
    states=[initial]+[[t['generator'],t['target']] for t in row['trace']]
    for f in forecasts:
        actual=None
        for tick in range(f['tick'],len(states)):
            g,t=states[tick]
            if abs(g['y']-t['y'])<=extent(g,'y')+extent(t,'y'):
                actual=dict(tick=tick,delay_s=(tick-f['tick'])/10,
                    x_body_gap_m=g['x']-t['x']-extent(g,'x')-extent(t,'x'))
                break
        f.update(realized_overlap=actual,censored=actual is None,
            delay_error_s=actual['delay_s']-f['predicted_overlap_delay_s'] if actual else None,
            gap_error_m=actual['x_body_gap_m']-f['predicted_x_body_gap_m'] if actual else None,
            interpretation='Counterfactual lane4 forecast versus realized policy rollout; subsequent actions and target response can differ.')
    requested=[d['tick'] for d in decisions if d['action']//3==1]
    applied=[d['tick'] for d in decisions if d['lateral_plan']['applied_lane_goal']==4.]
    return dict(entry_tick=entry_tick,entered=entered,forecast_samples=samples,candidate_checks=checks,
                first_requested_lane4_tick=requested[0] if requested else None,
                first_applied_lane4_tick=applied[0] if applied else None,
                inference_seconds=sum(s['seconds'] for s in saved),forecasts=forecasts)
