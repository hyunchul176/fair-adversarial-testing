"""Frozen observed-speed references, bounded synthetic lateral motion, paired targets."""
from dataclasses import asdict
import math
import numpy as np
from scripts.run_braking_response_feasibility import build_env
from src.evaluation.generator_measurement import GeneratorMeasurement,digest
from src.evaluation.geometry_target_cross import configure_geometry,target_action
from src.evaluation.highd_anchor_replay import profile,at_time
from src.evaluation.target_qualification import IDMReference
from src.evaluation.safety_metrics import VehicleState
from src.evaluation.stopping_viability import project


def lateral_reference(config,case,arm,time):
    if config.get('near_edge_abort') is not None:
        if arm!='abort':raise ValueError('Near-edge stop is an abort-only probe')
        from src.evaluation.near_edge_abort import reference
        return reference(case,config['near_edge_abort'],time)
    value=profile(case['event']['metrics']['duration_10_90'],arm,time,config['abort_release_s'])[0]
    if config.get('lateral_timing_mode')=='observed_midpoint' and arm=='continue':
        from src.evaluation.observed_lateral_timing import correction
        value+=correction(case,time)
    return value


def speed_reference(case,window,mode,t):
    if mode=='constant':return case['event']['metrics']['p10']['generator_speed']
    if mode=='counterfactual_delayed':
        from src.evaluation.counterfactual_speed_pulse import delayed_deficit
        return speed_reference(case,window,'observed_continuation',t)-delayed_deficit(t,case['pulse_duration_s'],case['pulse_amplitude_mps'],case['pulse_start_s'])
    if mode in ('counterfactual_pulse','counterfactual_shaped'):
        from src.evaluation.counterfactual_speed_pulse import deficit
        duration=case['pulse_duration_s'] if mode=='counterfactual_shaped' else window['times'][2]-window['times'][0]
        return speed_reference(case,window,'observed_continuation',t)-deficit(t,duration,case['pulse_amplitude_mps'])
    if mode not in ('observed','observed_continuation'):raise ValueError(mode)
    start,_,end=window['times'];g=window['generator']
    if mode=='observed_continuation':
        cutoff=case['continuation_end_time']
        if not end<=cutoff<=min(start+12,g['time'][-1]):raise ValueError('Invalid cached continuation cutoff')
        end=cutoff
    return float(np.interp(min(start+max(t,0.),end),g['time'],g['vx']))*window['sign']


def setup(config,case,arm,source=False):
    p=case['event']['metrics']['p10'];duration=case['event']['metrics']['duration_10_90']
    env,reset=build_env({**config,'seed':config['seed']+case['case_id']},p['target_speed'],p['headway'],0)
    configure_geometry(env,'adjacent','plain');g,t=env.inner.vehicle,env.target
    y,vy=profile(duration,arm,0,config['abort_release_s'])
    g.position[:]=[2000.,y];g.heading=math.atan2(vy,p['generator_speed']);g.speed=math.hypot(vy,p['generator_speed'])
    t.position[:]=[1000. if source else 2000.-p['gap']-5.,4.]
    for v in (g,t):v.on_state_update()
    env.previous_potential=env._potential();env.driving._initial_x=env.driving._previous_x=2000.
    reset.update(scenario_executed=config['version'],source_rehearsal=source,**env._metrics())
    return env,reset


def generator_command(config,case,window,mode,arm,tick,g,previous_a,previous_steer):
    dt=.1;b=config['generator_bounds'];duration=case['event']['metrics']['duration_10_90']
    y=lateral_reference(config,case,arm,(tick+1)*dt)
    desired_vy=(y-float(g.position[1]))/dt
    beta=math.asin(float(np.clip(desired_vy/float(g.speed),-.99,.99)))-float(g.heading)
    request=math.atan(2*math.tan(beta))
    steer=float(np.clip(request,-b['steering'],b['steering']))
    steer=float(np.clip(steer,previous_steer-b['steering_slew']*dt,previous_steer+b['steering_slew']*dt))
    # Same bicycle heading update as the pinned native Vehicle.step. Account
    # for heading when converting the next longitudinal velocity to scalar speed.
    slip=math.atan(.5*math.tan(steer))
    next_heading=float(g.heading)+float(g.speed)*math.sin(slip)/(g.LENGTH/2)*dt
    vx=speed_reference(case,window,mode,(tick+1)*dt)
    desired_scalar=vx/math.cos(next_heading)
    wanted=float(np.clip((desired_scalar-float(g.speed))/dt,-b['acceleration'],b['acceleration']))
    viability=project(wanted,float(g.speed)-b['speed_floor'],previous_a,dt,jerk=b['jerk'])
    if viability['constraint_infeasible']:raise RuntimeError('Generator floor/jerk intersection infeasible')
    action=np.array([viability['acceleration']/5,steer/.3],dtype=np.float32)
    return action,dict(vx_reference=vx,y_reference=y,requested_mps2=wanted,viability=viability)


def run(config,case,window,mode,arm,target='following',source=False,tape=None,control_fn=None):
    env,reset=setup(config,case,arm,source);g=env.inner.vehicle
    cruise=(case['event']['metrics']['p10']['target_speed'] if config.get('target_cruise_mode')=='initial' else config['cruise_mps'])
    model=IDMReference(env,cruise,config['idm_reference']) if target=='reference_idm' else None
    meter=GeneratorMeasurement(env,reset);physical=[asdict(VehicleState.from_vehicle(v)) for v in env.inner.road.vehicles]
    previous_a=previous_steer=previous_target=0.;info={}
    try:
        if not any(meter.reset_invalid.values()):
            for tick in range(config['environment']['duration']):
                previous_target,control=(control_fn or target_action)(env,target,model,cruise,previous_target,'viable_v1')
                if tape is None:action,gcontrol=generator_command(config,case,window,mode,arm,tick,g,previous_a,previous_steer)
                else:
                    action=np.array(tape[tick],dtype=np.float32)
                    gcontrol=dict(vx_reference=speed_reference(case,window,mode,(tick+1)*.1),
                        y_reference=lateral_reference(config,case,arm,(tick+1)*.1))
                previous_a=5*float(action[0]);previous_steer=.3*float(action[1])
                meter.before_step();_,_,term,trunc,info=env.step(action)
                info.update(high_level_action=action.tolist(),applied_action=action.tolist(),shield_intervened=False,low_speed_stop_guard_intervened=False)
                meter.after_step(info)
                meter.trace[-1].update(target_control=control,generator_control=gcontrol,generator_scalar_speed=float(g.speed))
                if term or trunc:break
        row=meter.finish(info);trace=row['trace'];tol=config['generator_bounds']['tolerance']
        contact=min((e['step'] for e in row['confirmed_contacts']),default=len(trace));safe=trace[:contact]
        scalar0=math.hypot(physical[0]['vx'],physical[0]['vy'])
        ys=[physical[0]['y']]+[r['generator']['y'] for r in safe]
        ay=np.diff(ys,n=2)/.01;jy=np.diff(ys,n=3)/.001
        speed_error=[abs(r['generator']['vx']-r['generator_control']['vx_reference']) for r in safe]
        path_error=[abs(r['generator']['y']-r['generator_control']['y_reference']) for r in safe]
        generator_checks=dict(valid_reset=row['reset_valid'],full_horizon=row['full_horizon'],
            physical=not row['physical_failure'],acceleration=all(abs(r['acceleration'][0])<=3+tol for r in safe),
            jerk=all(abs((r['acceleration'][0]-p)/.1)<=2+tol for r,p in zip(safe,[0.]+[r['acceleration'][0] for r in safe[:-1]])),
            speed_bounds=all(15-tol<=r['generator_scalar_speed']<=40+tol for r in safe),
            realization=all(abs(r['acceleration'][0]-5*r['applied_action'][0])<=tol for r in safe),
            lateral_acceleration=bool(np.all(abs(ay)<=2+tol)),lateral_jerk=bool(np.all(abs(jy)<=4+tol)),
            steering=all(abs(.3*r['applied_action'][1])<=.3+tol for r in safe),
            steering_slew=all(abs(.3*(r['applied_action'][1]-p)/.1)<=.15+tol for r,p in zip(safe,[0.]+[r['applied_action'][1] for r in safe[:-1]])),
            path=bool(path_error) and max(path_error)<=1e-5,speed_tracking=bool(speed_error) and max(speed_error)<=config['max_speed_tracking_error_mps'])
        target_checks=dict(acceleration=all(abs(r['acceleration'][1])<=5+tol for r in safe),
            jerk=all(abs((r['acceleration'][1]-p)/.1)<=2.5+tol for r,p in zip(safe,[0.]+[r['acceleration'][1] for r in safe[:-1]])),
            realization=all(abs(r['acceleration'][1]-r['target_control']['applied_mps2'])<=tol for r in safe),
            nonreverse=all(r['target']['vx']>=-tol for r in trace))
        row.update(case_id=case['case_id'],event_id=case['event']['event_id'],mode=mode,arm=arm,target_id=target,
            source_rehearsal=source,physical_initial_state=physical,physical_initial_hash=digest(physical),
            tape=[r['applied_action'] for r in trace],generator_checks=generator_checks,target_checks=target_checks,
            max_speed_error=max(speed_error,default=None),max_path_error=max(path_error,default=None),
            source_eligible=all(generator_checks.values()),
            final_target_speed=trace[-1]['target']['vx'] if trace else None,
            min_target_speed=min((r['target']['vx'] for r in trace),default=None),
            target_shield_steps=sum(r['target_control']['shield']['intervened'] for r in trace))
        states=[[physical[i]]+[r[k] for r in trace] for i,k in enumerate(('generator','target'))]
        row['anchor50']=at_time(states,case['event']['metrics']['duration_10_90']/2)
        return row
    finally:meter.close();env.close()
