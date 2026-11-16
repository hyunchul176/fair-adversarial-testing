"""Common bounded generator maneuvers and existing target pipelines."""
from dataclasses import asdict
import math
import numpy as np
from highway_env.vehicle.kinematics import Vehicle
from src.environments.hierarchical_interaction_env import lane_controller
from src.environments.physical_defender_shield import shield_action
from src.evaluation.anticipatory_nominal_control import following_command, bounded_acceleration
from src.evaluation.braking_response import braking_pulse
from src.evaluation.defender_observation import native_defender_observation
from src.evaluation.separation_projection_probe import separation_projected_acceleration
from src.evaluation.safety_metrics import VehicleState
from src.evaluation.stopping_viability import project


def quintic_position(time_s, duration_s, origin, destination):
    if duration_s<=0 or not all(math.isfinite(v) for v in (time_s,duration_s,origin,destination)):
        raise ValueError('Invalid lateral profile')
    u=max(0.,min(1.,time_s/duration_s))
    return origin+(destination-origin)*(10*u**3-15*u**4+6*u**5)


def configure_geometry(env, geometry, target_kind):
    if geometry not in ('aligned','adjacent'):
        raise ValueError('Unknown geometry')
    g, t = env.inner.vehicle, env.target
    g.position[1] = 4. if geometry=='aligned' else 0.
    t.position[1] = 4.
    g.heading = t.heading = 0.
    g.on_state_update()
    t.on_state_update()
    t.target_lane_index = t.lane_index
    if target_kind != 'idm':
        replacement = Vehicle.create_from(t)
        env.inner.road.vehicles[1] = replacement
        env.target = replacement
    return float(g.position[1]), 4.


def generator_action(env, config, maneuver, tick, origin_y, destination_y, previous_a, previous_steer):
    g, t = env.inner.vehicle, env.target
    bounds = config['generator_bounds']
    if maneuver in ('nominal','merge_only'):
        requested = 0.
    elif maneuver == 'pulse':
        requested = braking_pulse(tick/10, bounds['acceleration'], **config['pulse'])
    elif maneuver == 'feedback':
        # Use the same start as the pulse to preserve pre-action diagnostics.
        gap = max(0., float(g.position[0]-t.position[0])-(g.LENGTH+t.LENGTH)/2)
        closing = float(t.velocity[0]-g.velocity[0])
        overlap = abs(float(g.position[1]-t.position[1])) <= (g.WIDTH+t.WIDTH)/2
        f = config['feedback']
        requested = (-f['speed_gain']*(gap/f['target_ttc']-closing)
                     + max(closing,0.)/f['target_ttc']) if tick>=20 and overlap and gap>0 else 0.
    else:
        raise ValueError('Unknown maneuver')
    requested = float(np.clip(requested,-bounds['acceleration'],bounds['acceleration']))
    # Reserve enough speed to release braking at bounded jerk before speed floor.
    projection = project(requested, float(g.speed)-bounds['speed_floor'], previous_a,
                         .1, jerk=bounds['jerk'])
    if projection['constraint_infeasible']:
        raise RuntimeError('Generator floor/jerk intersection infeasible')
    destination = origin_y if maneuver=='nominal' else destination_y
    profile=config.get('lateral_profile')
    y_ref_next=None
    if profile is None:
        steer_request = bounds['steering']*lane_controller(destination-float(g.position[1]),
                                                           float(g.heading), preview_distance=30.)
    else:
        # Invert the actual highway-env bicycle position integration. Position
        # uses heading+slip beta, while vehicle.velocity uses heading alone.
        y_ref_next=quintic_position((tick+1)/10,profile['duration_s'],origin_y,destination)
        vy=(y_ref_next-float(g.position[1]))/.1
        beta=math.asin(float(np.clip(vy/max(float(g.speed),.1),-.99,.99)))-float(g.heading)
        steer_request=float(np.clip(math.atan(2*math.tan(beta)),-bounds['steering'],bounds['steering']))
    steer = float(np.clip(steer_request, previous_steer-bounds['steering_slew']*.1,
                         previous_steer+bounds['steering_slew']*.1))
    action = np.array([projection['acceleration']/5.,steer/.3],dtype=np.float32)
    telemetry={'requested_mps2':requested, 'projection':projection, 'steer_request':steer_request}
    if y_ref_next is not None:telemetry['y_ref_next']=y_ref_next
    return action, telemetry


def target_action(env, target_id, model, cruise, previous_a, actuator='legacy'):
    target, generator = env.target, env.inner.vehicle
    state = VehicleState.from_vehicle(target)
    traffic = [VehicleState.from_vehicle(v) for v in env.inner.road.vehicles if v is not target]
    follow = following_command(asdict(state), [asdict(v) for v in traffic], cruise)
    if target_id=='following':
        raw = [follow/5.,0.]
        requested, order = follow, None
    else:
        if model is None:
            raise ValueError('Missing policy')
        raw = np.asarray(model.predict(native_defender_observation(env.inner,target), deterministic=True)[0],dtype=np.float32).tolist()
        requested, order = 5.*raw[0], None
        if target_id.startswith('projected_'):
            requested, order = separation_projected_acceleration(requested,follow,
                                        float(target.position[0]),float(generator.position[0]))
    shielded, shield = shield_action(state,traffic,[requested/5.,raw[1]])
    post_stop = max(5.*float(shielded[0]),-float(target.speed)/.1)
    viability = None
    if actuator == 'legacy':
        acceleration = bounded_acceleration(post_stop,previous_a,.1)
    elif actuator == 'viable_v1':
        viability = project(post_stop,float(target.speed),previous_a,.1,jerk=2.5)
        if viability['constraint_infeasible']:
            raise RuntimeError('Target stopping/jerk intersection infeasible')
        acceleration = viability['acceleration']
    else:
        raise ValueError('Unknown target actuator')
    steering = .3*lane_controller(4.-float(target.position[1]),float(target.heading),preview_distance=20.)
    target.act({'acceleration':acceleration,'steering':steering})
    telemetry = dict(raw_action=raw,following_mps2=follow,requested_mps2=requested,
          projection_order=order,shield=shield,post_shield_mps2=5.*float(shielded[0]),
          post_stop_mps2=post_stop,previous_mps2=previous_a,applied_mps2=acceleration,steering=steering)
    # Leave legacy telemetry byte-for-byte unchanged for replay sentinels.
    if viability is not None:
        telemetry['viability']=viability
    return acceleration, telemetry
