"""Separate controlled variant of the frozen evaluator; default baseline audited on worker.
Adds a simple current-state following law before shield/jerk stages. New controller baseline, no safety guarantee.
The original multimode_feedback_probe.py remains byte-for-byte unchanged.
"""
from __future__ import annotations

from src.evaluation.nominal_interventions import brake_priority
from src.evaluation.ppo_retention_control import ARMS, flags, following_command, bounded_acceleration, target_jerk_limited

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Optional

import numpy as np

from src.environments.hierarchical_interaction_env import (
    HierarchicalInteractionEnv,
    apply_forward_stop_guard,
    lane_controller,
)
from src.environments.physical_defender_env import body_on_straight_road
from src.environments.physical_defender_shield import shield_action
from src.environments.physical_interaction_env import PhysicalInteractionEnv
from src.evaluation.counterfactual_probe import ContactRecorder, confirmed_contacts, risk_score
from src.evaluation.defender_observation import native_defender_observation
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc, following_headway

SCENE_FEATURE_VERSION = 'cutin-context-36d-v1'


@dataclass(frozen=True)
class FeedbackManeuver:
    scenario: str
    initial_gap: float
    generator_speed: float
    target_speed: float
    target_ttc: float
    speed_gain: float
    authority: float

    def __post_init__(self):
        if self.scenario not in {'cut_in', 'rear_end'}:
            raise ValueError('Unknown interaction mode')
        values = (self.initial_gap, self.generator_speed, self.target_speed,
                  self.target_ttc, self.speed_gain, self.authority)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError('Feedback parameters must be positive and finite')
        if not 0.6 <= self.target_ttc <= 3.0:
            raise ValueError('Target TTC outside registered mechanics range')
        if not 0.1 <= self.speed_gain <= 2.0 or not 0.5 <= self.authority <= 5.0:
            raise ValueError('Invalid feedback gain or authority')


def _bumper_gap(follower: VehicleState, leader: VehicleState) -> float:
    return max(0.0, leader.x - follower.x - (follower.length + leader.length) / 2.0)


def feedback_action(generator, target, maneuver, destination_y, previous_accel, previous_steer, dt):
    """Compute a bounded state-feedback attack action.

    Rear-end mode makes the generator approach a leading defender.  Cut-in
    mode merges ahead of the defender and regulates the defender's closing
    TTC by changing generator speed.  The controller observes current state;
    it has no access to future defender actions.
    """
    generator_state = VehicleState.from_vehicle(generator)
    target_state = VehicleState.from_vehicle(target)
    if maneuver.scenario == 'rear_end':
        gap = _bumper_gap(generator_state, target_state)
        closing = generator_state.vx - target_state.vx
        desired_closing = gap / maneuver.target_ttc
        # Feed-forward decay is required on the TTC manifold: closing speed
        # must fall as the remaining gap falls.  Proportional speed tracking
        # alone arrives with excessive momentum and contacts the leader.
        desired_accel = (maneuver.speed_gain * (desired_closing - closing)
                         - max(closing, 0.0) / maneuver.target_ttc)
        desired_accel = float(np.clip(desired_accel, -maneuver.authority, maneuver.authority))
        _, lateral = generator.lane.local_coordinates(generator.position)
        steering = 0.3 * lane_controller(-float(lateral), generator.heading, preview_distance=20.0)
    else:
        gap = _bumper_gap(target_state, generator_state)
        lateral_error = destination_y - float(generator.position[1])
        # Before lateral overlap, hold the registered lead while completing
        # the merge.  Once overlap is plausible, regulate closing TTC.
        lateral_overlap = abs(generator_state.y - target_state.y) <= (generator_state.width + target_state.width) / 2.0
        if lateral_overlap and gap > 0:
            closing = target_state.vx - generator_state.vx
            desired_closing = gap / maneuver.target_ttc
            desired_accel = (-maneuver.speed_gain * (desired_closing - closing)
                             + max(closing, 0.0) / maneuver.target_ttc)
            desired_accel = float(np.clip(desired_accel, -maneuver.authority, maneuver.authority))
        else:
            lead_error = maneuver.initial_gap - (generator_state.x - target_state.x)
            desired_accel = float(np.clip(0.12 * lead_error + 0.5 * (target_state.vx - generator_state.vx),
                                          -2.0, 2.0))
        steering = 0.3 * lane_controller(lateral_error, generator.heading, preview_distance=30.0)
    acceleration = float(np.clip(desired_accel, previous_accel - 2.5 * dt,
                                 previous_accel + 2.5 * dt))
    steering = float(np.clip(steering, previous_steer - 0.15 * dt,
                             previous_steer + 0.15 * dt))
    return np.asarray([acceleration / 5.0, steering / 0.3], dtype=np.float32)


def nominal_action(generator, origin_y, previous_accel, previous_steer, dt, cruise_speed=25.0):
    desired_accel = float(np.clip(0.4 * (cruise_speed - generator.speed), -2.0, 2.0))
    acceleration = float(np.clip(desired_accel, previous_accel - 2.5 * dt,
                                 previous_accel + 2.5 * dt))
    steering = 0.3 * lane_controller(origin_y - float(generator.position[1]), generator.heading,
                                     preview_distance=20.0)
    steering = float(np.clip(steering, previous_steer - 0.15 * dt,
                             previous_steer + 0.15 * dt))
    return np.asarray([acceleration / 5.0, steering / 0.3], dtype=np.float32)


def evasive_reference_action(target, generator, road, lane_y, scenario):
    """Documented privileged witness used only to test recoverability."""
    if scenario == 'cut_in':
        acceleration = min(2.0, 0.4 * (25.0 - target.speed))
        for other in road.vehicles:
            if other is target:
                continue
            dx, dy = other.position - target.position
            width = 4.5 if other is generator else 2.1
            if dx > 0 and abs(dy) < width:
                gap = dx - (other.LENGTH + target.LENGTH) / 2
                acceleration = min(acceleration, 0.18 * (gap - 5 - 1.8 * target.speed)
                                   + 0.8 * (other.velocity[0] - target.velocity[0]))
        steering = lane_controller(lane_y - target.position[1], target.heading)
    else:
        generator_state, target_state = VehicleState.from_vehicle(generator), VehicleState.from_vehicle(target)
        gap = _bumper_gap(generator_state, target_state)
        closing = generator_state.vx - target_state.vx
        threatened = closing > 0.1 and gap / closing < 4.0
        acceleration = 5.0 if threatened else min(2.0, 0.4 * (25.0 - target.speed))
        # The witness may use an adjacent lane when a rear impact is imminent.
        safe_lane_y = lane_y + 4.0 if lane_y < 6.0 else lane_y - 4.0
        destination = safe_lane_y if threatened else lane_y
        steering = lane_controller(destination - target.position[1], target.heading, preview_distance=12.0)
    return np.asarray([np.clip(acceleration / 5.0, -1.0, 1.0), steering], dtype=np.float32)


def _initial_hash(vehicles):
    states = [asdict(VehicleState.from_vehicle(vehicle)) for vehicle in vehicles]
    return hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest()


def _initial_overlaps(vehicles):
    overlaps = []
    for first, a in enumerate(vehicles):
        for second in range(first + 1, len(vehicles)):
            if a._is_colliding(vehicles[second], 0)[0]:
                overlaps.append([first, second])
    return overlaps


def initial_scene_features(generator, target, vehicles, nearest_count=6):
    """Fixed-size normalized context available before the maneuver starts."""
    generator_state, target_state = VehicleState.from_vehicle(generator), VehicleState.from_vehicle(target)
    pair = [
        generator_state.vx / 40.0, target_state.vx / 40.0,
        (target_state.x - generator_state.x) / 100.0,
        (target_state.y - generator_state.y) / 12.0,
        (target_state.vx - generator_state.vx) / 30.0,
        (target_state.vy - generator_state.vy) / 30.0,
    ]
    background = []
    for vehicle in vehicles:
        if vehicle is generator or vehicle is target:
            continue
        state = VehicleState.from_vehicle(vehicle)
        dx, dy = state.x - generator_state.x, state.y - generator_state.y
        key = (dx / 100.0) ** 2 + (dy / 12.0) ** 2
        background.append((key, [1.0, dx / 100.0, dy / 12.0,
                                 (state.vx - generator_state.vx) / 30.0,
                                 (state.vy - generator_state.vy) / 30.0]))
    rows = [row for _, row in sorted(background, key=lambda item: item[0])[:nearest_count]]
    rows.extend([[0.0] * 5 for _ in range(nearest_count - len(rows))])
    features = np.clip(np.asarray(pair + [value for row in rows for value in row], dtype=np.float32), -1.0, 1.0)
    if features.shape != (6 + 5 * nearest_count,) or not np.isfinite(features).all():
        raise ValueError('Invalid initial scene feature vector')
    return features


def rollout_feedback(maneuver: FeedbackManeuver, seed: int, defender='idm', model=None,
                     mode='attack', horizon=120, vehicles_count=15,
                     action_tape: Optional[list] = None, keep_trace=False, intervention="baseline"):
    """Run one paired world and retain actual contact actors."""
    from highway_env.vehicle.kinematics import Vehicle

    if intervention not in ARMS or mode != "nominal" or defender != "shielded_policy" or action_tape is not None:
        raise ValueError("Intervention study requires nominal shielded policy without tape")
    use_brake, use_lane = False, True
    use_generator_follow, use_target_follow = flags(intervention)
    use_target_jerk = target_jerk_limited(intervention)
    anticipation_rows = []
    intervention_rows = []

    if defender not in {'idm', 'mobil', 'policy', 'shielded_policy', 'reference'}:
        raise ValueError('Unknown defender')
    if mode not in {'attack', 'nominal'}:
        raise ValueError('Unknown rollout mode')
    if action_tape is not None:
        action_tape = np.asarray(action_tape, dtype=np.float32)
        if action_tape.shape != (horizon, 2) or not np.isfinite(action_tape).all() or np.any(np.abs(action_tape) > 1):
            raise ValueError('Replay tape must contain one bounded action per horizon tick')
    if maneuver.scenario == 'cut_in':
        env = HierarchicalInteractionEnv(vehicles_count=vehicles_count, initial_lead=maneuver.initial_gap,
                                         c_level=0.4, duration=max(200, horizon + 1))
    else:
        env = PhysicalInteractionEnv(scenario_type='rear_end', c_level=0.4,
                                     vehicles_count=vehicles_count, duration=max(200, horizon + 1))
    recorder = None
    try:
        _, reset_info = env.reset(seed=int(seed))
        inner, generator, target = env.inner, env.inner.vehicle, env.target
        if maneuver.scenario == 'rear_end':
            target.position = target.position.copy()
            target.position[0] = generator.position[0] + maneuver.initial_gap
            target.position[1] = generator.position[1]
            target.on_state_update()
        generator.speed = maneuver.generator_speed
        target.speed = maneuver.target_speed
        generator.on_state_update(); target.on_state_update()
        target.target_lane_index = target.lane_index
        origin_y = float(generator.position[1])
        destination_y = float(target.position[1])
        target_lane_y = destination_y
        initial_hash = _initial_hash(inner.road.vehicles)
        scene_features = initial_scene_features(generator, target, inner.road.vehicles)
        overlaps = _initial_overlaps(inner.road.vehicles)
        if overlaps:
            return {'valid_reset': False, 'initial_overlaps': overlaps, 'seed': int(seed),
                    'initial_hash': initial_hash, 'risk': 0.0, 'steps': 0,
                    'scene_feature_version': SCENE_FEATURE_VERSION,
                    'initial_scene_features': scene_features.tolist(),
                    'generator_controls': []}
        target_id = inner.road.vehicles.index(target)
        generator_id = inner.road.vehicles.index(generator)
        direct = defender in {'policy', 'shielded_policy', 'reference'}
        if direct:
            replacement = Vehicle.create_from(target)
            inner.road.vehicles[target_id] = replacement
            env.target = target = replacement
        elif hasattr(target, 'enable_lane_change'):
            target.enable_lane_change = defender == 'mobil'
        recorder = ContactRecorder(inner.road.vehicles)
        dt = 1.0 / env.driving._policy_freq
        previous_accel = previous_steer = 0.0
        previous_target_accel = 0.0
        min_ttc = None
        valid_frames = 0
        target_start_x = float(target.position[0])
        pair_ids = {generator_id, target_id}
        offroad = target_wrong_way = False
        merge_complete = maneuver.scenario == 'rear_end'
        trace, controls = [], []
        for step in range(horizon):
            anticipation = {'step': step, 'generator_follow': None, 'target_follow': None,
                            'previous_target_accel': previous_target_accel}
            def follow_input(vehicle, cruise):
                return {'ego': asdict(VehicleState.from_vehicle(vehicle)),
                        'traffic': [asdict(VehicleState.from_vehicle(v)) for v in inner.road.vehicles if v is not vehicle],
                        'cruise_speed': cruise}
            if direct:
                if defender == 'reference':
                    target_action = evasive_reference_action(target, generator, inner.road,
                                                               target_lane_y, maneuver.scenario)
                else:
                    if model is None:
                        raise ValueError('Missing defender model')
                    target_action = np.asarray(model.predict(native_defender_observation(inner, target),
                                                             deterministic=True)[0], dtype=np.float32)
                    anticipation['target_raw_policy'] = target_action.tolist()
                    if use_target_follow:
                        inputs = follow_input(target, maneuver.target_speed)
                        desired = following_command(**inputs)
                        anticipation['target_follow'] = {**inputs, 'desired_mps2': desired}
                        target_action = target_action.copy()
                        target_action[0] = desired / 5.
                    if intervention == 'zero_jerk_limited':
                        target_action = target_action.copy()
                        target_action[0] = 0.
                    anticipation['target_pre_shield'] = target_action.tolist()
                    if defender == 'shielded_policy':
                        target_action, _ = shield_action(VehicleState.from_vehicle(target),
                            [VehicleState.from_vehicle(v) for v in inner.road.vehicles if v is not target], target_action)
                if target_action.shape != (2,) or not np.isfinite(target_action).all():
                    raise ValueError('Bad defender action')
                original_target_action = target_action.copy()
                if use_target_jerk:
                    target_action = target_action.copy()
                    # Like the original generator pipeline, the jerk cap can
                    # weaken the instantaneous shield/stop request. No safety guarantee.
                    post_stop = max(5.*float(target_action[0]), -float(target.speed)/dt)
                    anticipation['target_post_stop'] = post_stop
                    target_action[0] = bounded_acceleration(post_stop, previous_target_accel, dt) / 5.
                anticipation['target_speed'] = float(target.speed)
                anticipation['target_jerk_mps3'] = (5.*float(target_action[0])-previous_target_accel)/dt
                previous_target_accel = 5.*float(target_action[0])
                if use_lane:
                    target_action = target_action.copy()
                    target_action[1] = lane_controller(target_lane_y - float(target.position[1]),
                                                      target.heading, preview_distance=20.0)
                target.act({'acceleration': 5.0 * float(np.clip(target_action[0], -1, 1)),
                            'steering': 0.3 * float(np.clip(target_action[1], -1, 1))})
            if action_tape is not None:
                action = action_tape[step].copy()
            elif mode == 'attack':
                action = feedback_action(generator, target, maneuver, destination_y,
                                         previous_accel, previous_steer, dt)
            else:
                action = nominal_action(generator, origin_y, previous_accel, previous_steer, dt,
                                        cruise_speed=maneuver.generator_speed)
            if use_generator_follow:
                inputs = follow_input(generator, maneuver.generator_speed)
                desired = following_command(**inputs)
                anticipation['generator_follow'] = {**inputs, 'desired_mps2': desired,
                                                    'prior_nominal_action': action.tolist()}
                action = action.copy()
                action[0] = min(float(action[0]), desired / 5.)
            anticipation['generator_pre_shield'] = action.tolist()
            anticipation_rows.append(anticipation)
            if action_tape is None:
                # Normal driving treats the designated vehicle as ordinary
                # traffic.  The attack leaves only that pair unshielded.
                guarded_traffic = [VehicleState.from_vehicle(v) for v in inner.road.vehicles
                                   if v is not generator and (mode == 'nominal' or v is not target)]
                action, generator_shield = shield_action(VehicleState.from_vehicle(generator), guarded_traffic, action)
                action, _, _ = apply_forward_stop_guard(action, float(generator.speed), dt)
                guarded_acceleration = float(action[0])
                action[0] = np.clip(action[0], (previous_accel - 2.5 * dt) / 5.0,
                                    (previous_accel + 2.5 * dt) / 5.0)
                jerk_limited_acceleration = float(action[0])
                action[0] = brake_priority(jerk_limited_acceleration, guarded_acceleration,
                                           generator_shield.get('intervened', False), use_brake)
                intervention_rows.append({'step':step, 'target_policy_shield_action':original_target_action.tolist(),
                    'target_y':float(target.position[1]),'target_heading':float(target.heading),
                    'target_initial_lane_y':target_lane_y,
                    'target_applied_action':np.clip(target_action,-1,1).tolist(),
                    'generator_shield_intervened':bool(generator_shield.get('intervened',False)),
                    'generator_post_stop':guarded_acceleration,'generator_after_jerk':jerk_limited_acceleration,
                    'generator_final':float(action[0]),
                    'generator_jerk_mps3':(float(action[0])*5.-previous_accel)/dt})
            # A witness replays the already guarded actuator tape exactly.
            previous_accel, previous_steer = float(action[0]) * 5.0, float(action[1]) * 0.3
            controls.append(action.tolist())
            recorder.step = step
            _, _, terminated, truncated, info = env.driving.step(action)
            generator_state, target_state = VehicleState.from_vehicle(generator), VehicleState.from_vehicle(target)
            ttc = constant_velocity_ttc(generator_state, target_state)
            if ttc is not None:
                min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
            follower, leader = ((generator_state, target_state) if maneuver.scenario == 'rear_end'
                                else (target_state, generator_state))
            valid_frames += int(following_headway(follower, leader) is not None)
            offroad |= not body_on_straight_road(generator) or not body_on_straight_road(target)
            target_wrong_way |= target_state.vx < -0.1
            if maneuver.scenario == 'cut_in':
                merge_complete |= abs(generator_state.y - target_lane_y) < 0.5 and abs(generator_state.heading) < 0.1
            if keep_trace:
                trace.append({'step': step, 'generator': asdict(generator_state),
                              'defender': asdict(target_state), 'action': action.tolist(),
                              'ttc': ttc})
            if terminated or truncated or generator.crashed or target.crashed or offroad or target_wrong_way:
                break
        involved = [event for event in confirmed_contacts(recorder.events, step)
                    if set(event['actors']) & pair_ids]
        pair_events = [event for event in involved if set(event['actors']) == pair_ids]
        background_events = [event for event in involved if set(event['actors']) != pair_ids]
        crash_flag = bool(generator.crashed or target.crashed)
        unattributed = crash_flag and not involved
        return {
            'intervention':intervention, 'intervention_controls':intervention_rows,
            'anticipation_controls':anticipation_rows,
            'valid_reset': True, 'seed': int(seed), 'traffic_seed': reset_info['traffic_seed'],
            'initial_hash': initial_hash, 'scenario': maneuver.scenario, 'defender': defender,
            'scene_feature_version': SCENE_FEATURE_VERSION,
            'initial_scene_features': scene_features.tolist(),
            'mode': mode, 'steps': step + 1, 'replayed_tape': action_tape is not None,
            'pair_contact': bool(pair_events), 'background_contact': bool(background_events),
            'unattributed_crash': bool(unattributed), 'crash_flag': crash_flag,
            'offroad': bool(offroad), 'wrong_way': bool(info.get('wrong_way', False) or target_wrong_way),
            'full_horizon': bool(step + 1 == horizon and not crash_flag and not offroad and not target_wrong_way),
            'merge_complete': bool(merge_complete), 'valid_fraction': valid_frames / horizon,
            'target_progress_m': float(target.position[0]) - target_start_x,
            'min_pair_ttc': min_ttc,
            'exclusive_near_miss': bool(not crash_flag and not involved and min_ttc is not None and min_ttc < 1.5),
            'risk': risk_score(bool(pair_events), min_ttc), 'contacts': involved,
            'generator_controls': controls, 'trace': trace, 'policy_dt': dt,
        }
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def paired_feedback_result(maneuver, seed, defender, model=None, horizon=120, vehicles_count=15):
    common = dict(maneuver=maneuver, seed=seed, horizon=horizon, vehicles_count=vehicles_count)
    attack = rollout_feedback(**common, defender=defender, model=model, mode='attack')
    nominal = rollout_feedback(**common, defender=defender, model=model, mode='nominal')
    witness = None
    if attack.get('full_horizon') and len(attack.get('generator_controls', [])) == horizon:
        witness = rollout_feedback(**common, defender='reference', mode='attack',
                                   action_tape=attack['generator_controls'])
    reasons = []
    rows = [nominal, attack] + ([witness] if witness is not None else [])
    if not all(row.get('valid_reset', False) for row in rows):
        reasons.append('invalid_initial_scene')
    hashes = {row.get('initial_hash') for row in rows}
    if len(hashes) != 1:
        reasons.append('paired_world_mismatch')
    for name, row in (('nominal', nominal), ('attack', attack)):
        if not row.get('full_horizon', False):
            reasons.append(name + '_not_full_horizon')
        if row.get('pair_contact') or row.get('background_contact') or row.get('unattributed_crash'):
            reasons.append(name + '_contact')
        if row.get('offroad') or row.get('wrong_way'):
            reasons.append(name + '_physical_failure')
    if maneuver.scenario == 'cut_in' and not attack.get('merge_complete', False):
        reasons.append('attack_merge_incomplete')
    if witness is None:
        reasons.append('missing_recoverability_witness')
    else:
        if (not witness.get('full_horizon', False) or witness.get('contacts')
                or witness.get('offroad') or witness.get('wrong_way')
                or witness.get('target_progress_m', 0.0) < 80.0):
            reasons.append('witness_not_safe_and_progressing')
    delta = float(attack.get('risk', 0.0) - nominal.get('risk', 0.0))
    if delta < -1e-9:
        reasons.append('negative_added_risk')
    return {
        'seed': int(seed), 'admissible': not reasons, 'rejection_reasons': sorted(set(reasons)),
        'paired_risk_delta': delta,
        'nominal': nominal, 'attack': attack, 'witness': witness,
    }
