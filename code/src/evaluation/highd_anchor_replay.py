"""Calibration longitudinal anchors with explicitly synthetic lateral profiles."""
from dataclasses import asdict
import hashlib
import math

import numpy as np

from scripts.run_braking_response_feasibility import build_env
from src.evaluation.generator_measurement import GeneratorMeasurement, digest
from src.evaluation.geometry_target_cross import configure_geometry, target_action
from src.evaluation.safety_metrics import VehicleState


def fraction(u):
    u=max(0.,min(1.,u))
    return 10*u**3-15*u**4+6*u**5


def tenth_fraction():
    lo,hi=0.,.5
    for _ in range(60):
        mid=(lo+hi)/2
        if fraction(mid)<.1:lo=mid
        else:hi=mid
    return (lo+hi)/2


U10=tenth_fraction()


class CaptureModel:
    def __init__(self,model):self.model=model;self.observation=None
    def predict(self,observation,**kwargs):
        self.observation=np.asarray(observation).copy()
        return self.model.predict(observation,**kwargs)


def profile(duration,arm,time,release):
    total=duration/(1-2*U10)
    velocity=4*30*U10**2*(1-U10)**2/total
    if time<=0:return .4,velocity
    if arm=='continue':
        y=4*fraction(U10+max(0.,time)/total)
    elif arm=='abort':
        u=max(0.,min(1.,time/release))
        y=.4+velocity*release*(u-u**3+.5*u**4)
    else:raise ValueError('Unknown arm')
    return y,velocity


def select_events(events,config):
    selected=[];ledger=[]
    for rec in config['recordings']:
        for stratum in ('closing','nonclosing'):
            candidates=[e for e in events if e.get('eligible') and e['partition']=='calibration'
                and e['recording']==rec and (e['metrics']['p10']['closing_speed']>0)==(stratum=='closing')]
            ordered=sorted(candidates,key=lambda e:hashlib.sha256(
                (config['selection_salt']+':'+e['event_id']).encode()).hexdigest())
            if not ordered:raise RuntimeError(f'Missing stratum {rec}/{stratum}; no replacement allowed')
            choice=ordered[0]
            selected.append(dict(case_id=len(selected),stratum=stratum,event=choice))
            for index,event in enumerate(ordered):
                ledger.append(dict(event_id=event['event_id'],recording=rec,stratum=stratum,rank=index,
                                   selected=index==0,selection_hash=hashlib.sha256(
                                       (config['selection_salt']+':'+event['event_id']).encode()).hexdigest()))
    assert len(selected)==config['reporting']['selected_events']
    return selected,ledger


def preflight(case,config):
    metric=case['event']['metrics'];duration=metric['duration_10_90'];p=metric['p10']
    _,vy=profile(duration,'continue',0,config['abort_release_s'])
    scalar=math.hypot(p['generator_speed'],vy)
    heading=math.atan2(vy,p['generator_speed'])
    end=profile(duration,'abort',100,config['abort_release_s'])[0]
    radius=(5*abs(math.sin(heading))+2*abs(math.cos(heading)))/2
    return dict(duration=config['duration_10_90_min_s']<=duration<=config['duration_10_90_max_s'],
        speed_floor=min(scalar,p['target_speed'])>=config['generator_bounds']['speed_floor'],
        speed_cap=max(scalar,p['target_speed'])<=config['max_initial_scalar_speed_mps'],
        abort_separated=end+radius<3.,positive_gap=p['gap']>0)


def at_time(states,time):
    times=np.arange(len(states[0]))*.1
    if time>times[-1]+1e-8:return None
    g,t=states
    def val(seq,key):return float(np.interp(time,times,[s[key] for s in seq]))
    vg,vt=val(g,'vx'),val(t,'vx')
    gap=val(g,'x')-val(t,'x')-(val(g,'length')+val(t,'length'))/2
    return dict(generator_speed=vg,target_speed=vt,closing_speed=vt-vg,gap=gap,
                headway=gap/vt if vt>0 else None)


def run(config,case,arm,target_id='idm',model=None,tape=None):
    event=case['event'];metric=event['metrics'];p=metric['p10'];duration=metric['duration_10_90']
    local={**config,'seed':config['seed']+case['case_id']}
    env,reset=build_env(local,p['target_speed'],p['headway'],0)
    configure_geometry(env,'adjacent',target_id)
    g=env.inner.vehicle
    y,vy=profile(duration,arm,0,config['abort_release_s'])
    g.position[1]=y;g.heading=math.atan2(vy,p['generator_speed']);g.speed=math.hypot(p['generator_speed'],vy)
    g.position[0]+=config.get('generator_offset_m',0.)
    g.on_state_update();env.previous_potential=env._potential()
    env.driving._initial_x=env.driving._previous_x=float(g.position[0])
    captured=CaptureModel(model) if model is not None and config.get('record_policy_observation',False) else None
    reset.update(scenario_executed=config['version'],event_id=event['event_id'],arm=arm,
                 target_controller=target_id,**env._metrics())
    meter=GeneratorMeasurement(env,reset)
    physical=[asdict(VehicleState.from_vehicle(v)) for v in env.inner.road.vehicles]
    previous_steer=previous_target=0.;info={}
    try:
        if not any(meter.reset_invalid.values()):
            for tick in range(config['environment']['duration']):
                control=None
                if target_id!='idm':
                    previous_target,control=target_action(env,target_id,captured or model,p['target_speed'],previous_target,config['target_actuator'])
                    if captured is not None:control['policy_observation']=captured.observation.tolist()
                reference=profile(duration,arm,(tick+1)*.1,config['abort_release_s'])[0]
                if tape is None:
                    desired_vy=(reference-float(g.position[1]))/.1
                    beta=math.asin(float(np.clip(desired_vy/float(g.speed),-.99,.99)))-float(g.heading)
                    request=math.atan(2*math.tan(beta));b=config['generator_bounds']
                    steer=float(np.clip(request,-b['steering'],b['steering']))
                    steer=float(np.clip(steer,previous_steer-b['steering_slew']*.1,previous_steer+b['steering_slew']*.1))
                    action=np.array([0.,steer/.3],dtype=np.float32)
                else:action=np.array(tape[tick],dtype=np.float32)
                previous_steer=float(action[1])*.3
                meter.before_step();_,_,term,trunc,info=env.step(action)
                info.update(high_level_action=action.tolist(),applied_action=action.tolist(),shield_intervened=False,
                            low_speed_stop_guard_intervened=False,merge_complete=abs(float(g.position[1])-4)<.5 and abs(g.heading)<.1)
                meter.after_step(info)
                meter.trace[-1].update(target_control=control,y_reference=reference,generator_scalar_speed=float(g.speed),
                                      target_scalar_speed=float(env.target.speed))
                if term or trunc:break
        row=meter.finish(info)
        row.update(physical_initial_state=physical,physical_initial_hash=digest(physical),
                   tape=[r['applied_action'] for r in row['trace']])
        contacts=[e['step'] for e in row['confirmed_contacts']]
        safe=row['trace'][:min(contacts,default=row['steps'])]
        b=config['generator_bounds'];lb=config['lateral_bounds'];tol=b['tolerance']
        ys=np.array([physical[0]['y']]+[r['generator']['y'] for r in safe])
        ay=np.diff(ys,n=2)/.1**2;jy=np.diff(ys,n=3)/.1**3
        def peak(x):return float(np.max(np.abs(x))) if len(x) else None
        row['position_peak_lateral_acceleration']=peak(ay);row['position_peak_lateral_jerk']=peak(jy)
        row['max_path_error']=max((abs(r['generator']['y']-r['y_reference']) for r in safe),default=None)
        all_y=np.array([physical[0]['y']]+[r['generator']['y'] for r in row['trace']])
        crossing=np.flatnonzero((all_y[:-1]<3.6)&(all_y[1:]>=3.6))
        achieved=None
        if len(crossing):
            i=int(crossing[0]);achieved=(i+(3.6-all_y[i])/(all_y[i+1]-all_y[i]))*.1
        row['measured_duration_10_90']=achieved
        row['motion_checks']=dict(valid_reset=row['reset_valid'],
            acceleration=all(abs(r['acceleration'][0])<=b['acceleration']+tol for r in safe),
            jerk=all(abs(r['jerk'][0])<=b['jerk']+tol for r in safe if r['jerk'] is not None),
            scalar_speed_floor=all(r['generator_scalar_speed']>=b['speed_floor']-tol for r in safe),
            acceleration_realized=all(abs(r['acceleration'][0]-5*r['applied_action'][0])<=tol for r in safe),
            steering=all(abs(r['applied_action'][1]*.3)<=b['steering']+tol for r in safe),
            steering_slew=all(abs((r['applied_action'][1]-prev)*.3/.1)<=b['steering_slew']+tol
                for r,prev in zip(safe,[0.]+[r['applied_action'][1] for r in safe[:-1]])),
            lateral_acceleration=bool(np.all(abs(ay)<=lb['acceleration']+tol)),
            lateral_jerk=bool(np.all(abs(jy)<=lb['jerk']+tol)),
            prescribed_path=row['max_path_error'] is not None and row['max_path_error']<=lb['path_tolerance_m'])
        row['motion_checks']['arm_completion']=(achieved is not None and abs(achieved-duration)<=lb['duration_tolerance_s']) if arm=='continue' else all(
            r['generator']['y']+(5*abs(math.sin(r['generator']['heading']))+2*abs(math.cos(r['generator']['heading'])))/2<3. for r in safe)
        row['source_eligible']=row['full_horizon'] and not row['physical_failure'] and all(row['motion_checks'].values())
        states=[[physical[i]]+[r[key] for r in row['trace']] for i,key in enumerate(('generator','target'))]
        row['anchor_measurements']=dict(p10=at_time(states,0),p50=at_time(states,duration/2),p90=at_time(states,duration))
        row['natural_p50_relative_time']=event['times'][1]-event['times'][0]
        row['synthetic_p50_relative_time']=duration/2
        speed=[physical[1]['vx']]+[r['target']['vx'] for r in row['trace']]
        row['response']=dict(progress_fraction=row['target_progress_m']/(p['target_speed']*row['horizon']*.1),
            speed_loss=p['target_speed']-min(speed),low_speed_seconds=.1*sum(v<config['nominal_quality']['low_speed_mps'] for v in speed[1:]),
            peak_braking=max([0.]+[-r['acceleration'][1] for r in safe]))
        n=config['nominal_quality'];resp=row['response']
        row['safe_and_progressing']=row['full_horizon'] and not row['physical_failure'] and row['target_progress_m']>=80
        row['nominal_quality_pass']=row['full_horizon'] and not (row['physical_failure'] or row['exclusive_ttc_event']) and (
            resp['progress_fraction']>=n['progress_fraction'] and resp['speed_loss']<=n['speed_loss_mps'] and
            resp['low_speed_seconds']/(row['horizon']*.1)<=n['low_speed_fraction'])
        row['target_actuator_checks']=None if target_id=='idm' else dict(
            acceleration=all(abs(r['acceleration'][1])<=5+tol for r in safe),
            jerk=all(abs(r['jerk'][1])<=2.5+tol for r in safe if r['jerk'] is not None),
            nonreverse=all(r['target_scalar_speed']>=-tol for r in safe),
            acceleration_realized=all(abs(r['acceleration'][1]-r['target_control']['applied_mps2'])<=tol for r in safe))
        return row
    finally:meter.close();env.close()
