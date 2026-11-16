"""Frozen purpose-specific synthetic nominal-driving qualification."""
from dataclasses import asdict
import math
import numpy as np

from scripts.run_braking_response_feasibility import build_env
from src.evaluation.braking_response import braking_pulse
from src.evaluation.generator_measurement import GeneratorMeasurement, digest
from src.evaluation.geometry_target_cross import configure_geometry, target_action
from src.evaluation.highd_anchor_replay import CaptureModel
from src.evaluation.safety_metrics import VehicleState


def cases(config):
    out=[]
    for family in config['families']:
        for index,speed in enumerate(config['initial_speeds']):
            spec=config['families'][family]
            out.append(dict(case_id=len(out),family=family,initial_speed=speed,
                # Identical physical RNG/launch for free_hold versus free_setpoint.
                seed=config['seed']+index,cruise=speed if family=='free_hold' else config['cruise_mps'],
                expected_late_speed=speed if family=='free_hold' else spec['expected_late_speed'],
                **{k:spec[k] for k in ('leader_speed','gap_m','geometry','primary')}))
    return out


def idm_acceleration(speed,cruise,gap=None,leader_speed=None,*,a=1.5,b=2.,headway=1.5,standstill=2.,delta=4):
    """Analytic IDM variant with nonnegative dynamic desired-gap increment.

    Parameters are engineering choices, not fit to these evaluation outcomes.
    Output clipping precedes the shared shield and stopping/jerk actuator.
    """
    command=a*(1-(speed/cruise)**delta)
    if gap is not None:
        desired=standstill+max(0.,headway*speed+speed*(speed-leader_speed)/(2*math.sqrt(a*b)))
        command-=a*(desired/max(gap,.01))**2
    return float(np.clip(command,-5.,2.))


class IDMReference:
    def __init__(self,env,cruise,parameters):self.env=env;self.cruise=cruise;self.parameters=parameters
    def predict(self,observation,**kwargs):
        g,t=self.env.inner.vehicle,self.env.target
        gap=float(g.position[0]-t.position[0])-(g.LENGTH+t.LENGTH)/2
        visible=abs(float(g.position[1]-t.position[1]))<(g.WIDTH+t.WIDTH)/2 and gap>0
        command=idm_acceleration(float(t.speed),self.cruise,gap if visible else None,
            float(g.speed) if visible else None,**self.parameters)
        return np.array([command/5.,0.],dtype=np.float32),None


def build_case(config,case):
    env,reset=build_env({**config,'seed':case['seed']},case['initial_speed'],case['gap_m']/case['initial_speed'],0)
    configure_geometry(env,case['geometry'],'plain')
    env.target.position[0]=config['target_initial_x_m']
    env.inner.vehicle.position[0]=config['target_initial_x_m']+case['gap_m']+(env.target.LENGTH+env.inner.vehicle.LENGTH)/2
    for vehicle in env.inner.road.vehicles:vehicle.on_state_update()
    env.inner.vehicle.speed=case['leader_speed']
    env.driving._initial_x=env.driving._previous_x=float(env.inner.vehicle.position[0])
    env.previous_potential=env._potential()
    reset.update(scenario_executed=config['version'],case=case,**env._metrics())
    return env,reset


def leader_tape(config,case):
    return [[float(np.float32(braking_pulse(t/10,1.,**config['pulse'])/5)) if case['family']=='gentle_slow' else 0.,0.]
            for t in range(config['environment']['duration'])]


def qualification(config,case,row):
    trace=row['trace'];q=config['qualification'];tol=q['numeric_tolerance']
    late=[r for r in trace if r['tick']>=q['late_start_tick']]
    errors=[abs(r['target']['vx']-case['expected_late_speed']) for r in late]
    contact=min((e['step'] for e in row['confirmed_contacts']),default=len(trace))
    safe=trace[:contact]
    previous=[0.]+[r['acceleration'][1] for r in safe[:-1]]
    actuator=dict(acceleration=all(abs(r['acceleration'][1])<=5+tol for r in safe),
        jerk=all(abs((r['acceleration'][1]-p)/.1)<=2.5+tol for r,p in zip(safe,previous)),
        realized=all(abs(r['acceleration'][1]-r['target_control']['applied_mps2'])<=tol for r in safe),
        nonreverse=all(r['target']['vx']>=-tol for r in trace))
    checks=dict(full_horizon=row['full_horizon'],reset_valid=row['reset_valid'],
        safety=not(row['physical_failure'] or row['exclusive_ttc_event']),actuator=all(actuator.values()),
        moving=bool(trace) and all(r['target']['vx']>=q['minimum_speed_mps'] for r in trace),
        late_complete=len(late)==config['environment']['duration']-q['late_start_tick'],
        late_mae=bool(errors) and bool(np.mean(errors)<=q['late_mae_mps']),
        late_max=bool(errors) and max(errors)<=q['late_max_error_mps'])
    return dict(checks=checks,passed=all(checks.values()),actuator=actuator,
        late_mae=float(np.mean(errors)) if errors else None,late_max_error=max(errors,default=None),
        final_speed=trace[-1]['target']['vx'] if trace else None,
        minimum_speed=min((r['target']['vx'] for r in trace),default=None),
        shield_steps=sum(r['target_control']['shield']['intervened'] for r in trace))


def run(config,case,target,model=None,control_fn=None):
    env,reset=build_case(config,case)
    if target=='reference_idm':model=IDMReference(env,case['cruise'],config['idm_reference'])
    captured=CaptureModel(model) if model is not None and control_fn is None else None
    meter=GeneratorMeasurement(env,reset)
    physical=[asdict(VehicleState.from_vehicle(v)) for v in env.inner.road.vehicles]
    tape=leader_tape(config,case);previous=0.;info={}
    try:
        if not any(meter.reset_invalid.values()):
            for tick,command in enumerate(tape):
                previous,control=(control_fn or target_action)(env,target,captured,case['cruise'],previous,'viable_v1')
                if captured is not None:control['policy_observation']=captured.observation.tolist()
                action=np.array(command,dtype=np.float32)
                meter.before_step();_,_,terminated,truncated,info=env.step(action)
                info.update(high_level_action=command,applied_action=command,shield_intervened=False,
                    low_speed_stop_guard_intervened=False)
                meter.after_step(info)
                meter.trace[-1]['target_control']=control
                if terminated or truncated:break
        row=meter.finish(info)
        row.update(physical_initial_state=physical,physical_initial_hash=digest(physical),
            case=case,target_id=target,tape_sha256=digest(tape))
        row['qualification']=qualification(config,case,row)
        return row
    finally:meter.close();env.close()
