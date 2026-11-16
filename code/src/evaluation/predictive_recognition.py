"""Bounded constant-velocity lateral recognition; no future tape or arm input."""
from dataclasses import asdict
import math
from src.evaluation.safety_metrics import VehicleState
from src.evaluation.recognition_attribution import commands
from src.environments.hierarchical_interaction_env import lane_controller


def predicted_entry(generator,target,horizon,kind):
    if not math.isfinite(horizon) or horizon<0:raise ValueError('Invalid recognition horizon')
    g,t=generator,target;c,s=math.cos(t['heading']),math.sin(t['heading'])
    dx,dy=g['x']-t['x'],g['y']-t['y'];dvx,dvy=g['vx']-t['vx'],g['vy']-t['vy']
    if kind=='following':
        along=dx*c+dy*s;side=-dx*s+dy*c;va=dvx*c+dvy*s;vs=-dvx*s+dvy*c
        delta=g['heading']-t['heading'];radius=t['width']/2+(g['length']*abs(math.sin(delta))+g['width']*abs(math.cos(delta)))/2
        front=along>0 and t['vx']*c+t['vy']*s>.1;front_boundary=0.
    elif kind=='reference_idm':
        along=dx;side=dy;va=dvx;vs=dvy;radius=(g['width']+t['width'])/2
        front_boundary=(g['length']+t['length'])/2;front=along>front_boundary
    else:raise ValueError('Unknown reference')
    # Only anticipates an approaching boundary, never invents lateral intent.
    distance=max(0.,abs(side)-radius)
    entering=side*vs<0
    entry=distance/abs(vs) if entering and abs(vs)>1e-12 else None
    predicted=bool(horizon>0 and front and entry is not None and entry<=horizon and along+va*entry>front_boundary)
    return dict(predicted=predicted,entry_s=entry,horizon_s=horizon,lateral_distance=distance,relative_lateral_speed=vs)


def controller(parameters,horizon):
    def action(env,target_id,model,cruise,previous_a,actuator='viable_v1'):
        if actuator!='viable_v1' or target_id not in ('following','reference_idm'):raise ValueError('Unsupported predictive pipeline')
        if len(env.inner.road.vehicles)!=2:raise ValueError('Predictive diagnostic requires a frozen two-vehicle world')
        g=asdict(VehicleState.from_vehicle(env.inner.vehicle));t=asdict(VehicleState.from_vehicle(env.target))
        native=commands(g,t,cruise,parameters,previous_a);prediction=predicted_entry(g,t,horizon,target_id)
        early=bool(not native[target_id]['gate'] and prediction['predicted'])
        selected=commands(g,t,cruise,parameters,previous_a,True)[target_id] if early else native[target_id]
        if selected['viability']['constraint_infeasible']:raise RuntimeError('Target stopping/jerk intersection infeasible')
        steer=.3*lane_controller(4.-t['y'],t['heading'],preview_distance=20.)
        env.target.act(dict(acceleration=selected['applied'],steering=steer))
        telemetry=dict(raw_action=[selected['request']/5,0.],following_mps2=native['following']['request'],
            requested_mps2=selected['request'],post_shield_mps2=selected['post_shield'],post_stop_mps2=selected['post_stop'],
            previous_mps2=previous_a,applied_mps2=selected['applied'],steering=steer,viability=selected['viability'],
            shield=dict(intervened=selected['shield_intervened']),
            recognition=dict(native_gate=native[target_id]['gate'],early=early,**prediction))
        return selected['applied'],telemetry
    return action
