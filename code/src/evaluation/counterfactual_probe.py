"""Development probe: paired maneuver search with explicit contact evidence.

This is a simulator experiment, not a new RL algorithm or a feasibility proof.
An identical state-only maneuver program runs against each defender. A safe
reference rollout witnesses recoverability only for that simulated scenario.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MethodType

import numpy as np

from src.environments.hierarchical_interaction_env import HierarchicalInteractionEnv, lane_controller
from src.environments.physical_defender_env import body_on_straight_road
from src.environments.physical_defender_shield import shield_action
from src.evaluation.defender_observation import native_defender_observation
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc, following_headway


@dataclass(frozen=True)
class Maneuver:
    initial_lead: float
    merge_start: float
    merge_duration: float
    brake_delay: float
    brake_magnitude: float
    brake_duration: float


def lower_tail_mean(values, fraction=0.5):
    """Uniform empirical lower-CVaR, including fractional boundary mass."""
    values = np.sort(np.asarray(values, dtype=float))
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < fraction <= 1:
        raise ValueError('Invalid empirical tail input')
    mass = fraction * len(values)
    whole = int(math.floor(mass))
    remainder = mass - whole
    total = float(values[:whole].sum())
    if remainder > 0:
        total += remainder * float(values[whole])
    return total / mass


def risk_score(pair_collision, min_ttc, threshold=3.0):
    if pair_collision:
        return 1.0
    return 0.0 if min_ttc is None else 0.8 * max(0.0, 1.0 - min_ttc / threshold)


def confirmed_contacts(events, last_step):
    # A scheduled impact is a simulator collision only after the following
    # integration step consumed it. Never count a future event beyond horizon.
    return [e for e in events if e['kind'] == 'overlap' or e['step'] < last_step]


class ContactRecorder:
    """Observe highway-env's native intersecting and scheduled-impact events.

    Native collision handling also marks crashes on the *next* integration step
    through `impact`; logging only current overlaps would miss those contacts.
    Instrument instance methods only; no global simulation code is changed.
    """
    def __init__(self, vehicles):
        self.events = []
        self.step = 0
        self.ids = {id(v): i for i, v in enumerate(vehicles)}
        self.originals = []
        for vehicle in vehicles:
            original = vehicle.handle_collisions
            self.originals.append((vehicle, original))

            def wrapped(this, other, dt=0, original=original):
                if other is not this and (this.check_collisions or other.check_collisions) and this.collidable and other.collidable:
                    intersecting, impending, _ = this._is_colliding(other, dt)
                    if this.solid and other.solid and (intersecting or impending):
                        self.events.append({
                            'step': self.step, 'actors': [self.ids[id(this)], self.ids.get(id(other), -1)],
                            'kind': 'overlap' if intersecting else 'scheduled_impact',
                            'states': [asdict(VehicleState.from_vehicle(this)), asdict(VehicleState.from_vehicle(other))],
                        })
                return original(other, dt)

            vehicle.handle_collisions = MethodType(wrapped, vehicle)

    def close(self):
        for vehicle, original in self.originals:
            vehicle.handle_collisions = original


def _smoothstep(x):
    x = float(np.clip(x, 0, 1))
    return x ** 3 * (10 + x * (-15 + 6 * x))


def maneuver_action(vehicle, maneuver, time_s, origin_y, destination_y, previous_accel, previous_steer, dt,
                    mode='attack'):
    """Bounded program depends only on generator state and clock, not defender."""
    merge = mode != 'nominal'
    phase = (time_s + 0.4 - maneuver.merge_start) / maneuver.merge_duration
    target_y = origin_y + (destination_y - origin_y) * _smoothstep(phase) if merge else origin_y
    steering = 0.3 * lane_controller(target_y - vehicle.position[1], vehicle.heading, preview_distance=10.0)
    steering = float(np.clip(steering, previous_steer - 0.15 * dt, previous_steer + 0.15 * dt))
    pulse_start = maneuver.merge_start + maneuver.merge_duration + maneuver.brake_delay
    pulse_time = time_s - pulse_start
    pulse = 0.0
    if mode == 'attack' and 0 <= pulse_time <= maneuver.brake_duration:
        edge = min(0.5, maneuver.brake_duration / 2)
        pulse = min(_smoothstep(pulse_time / edge), _smoothstep((maneuver.brake_duration - pulse_time) / edge))
    desired_accel = -maneuver.brake_magnitude * pulse if pulse > 0 else min(2.0, 0.4 * (25.0 - vehicle.speed))
    acceleration = float(np.clip(desired_accel, previous_accel - 2.5 * dt, previous_accel + 2.5 * dt))
    return np.asarray([acceleration / 5.0, steering / 0.3], dtype=np.float32)


def reference_action(target, generator, road, lane_y):
    """Conservative state-feedback follower, no access to future attack actions."""
    acceleration = min(2.0, 0.4 * (25.0 - target.speed))
    for other in road.vehicles:
        if other is target:
            continue
        dx, dy = other.position - target.position
        # Anticipate the designated adjacent cut-in vehicle; other leaders only
        # when close to the defender's lane. This privilege is documented.
        width = 4.5 if other is generator else 2.1
        if dx > 0 and abs(dy) < width:
            gap = dx - (other.LENGTH + target.LENGTH) / 2
            acceleration = min(acceleration, 0.18 * (gap - 5 - 1.8 * target.speed)
                               + 0.8 * (other.velocity[0] - target.velocity[0]))
    return np.asarray([np.clip(acceleration / 5, -1, 1),
                       lane_controller(lane_y - target.position[1], target.heading)], dtype=np.float32)


def rollout(maneuver, seed, defender='idm', model=None, mode='attack', horizon=120, vehicles_count=15,
            keep_trace=False, background_guard=False, action_tape=None, generator_speed=25.0,
            target_speed=25.0):
    from highway_env.vehicle.kinematics import Vehicle

    if mode not in {'attack', 'nominal', 'cutin_only'}:
        raise ValueError('Unknown maneuver mode')
    if defender not in {'idm', 'mobil', 'policy', 'shielded_policy', 'reference'}:
        raise ValueError('Unknown defender')
    if action_tape is not None:
        action_tape = np.asarray(action_tape, dtype=np.float32)
        if action_tape.shape != (horizon, 2) or not np.isfinite(action_tape).all() or np.any(np.abs(action_tape) > 1):
            raise ValueError('Replay tape must contain one bounded action per horizon tick')
    env = HierarchicalInteractionEnv(vehicles_count=vehicles_count, initial_lead=maneuver.initial_lead, c_level=0.4)
    recorder = None
    try:
        _, reset_info = env.reset(seed=seed)
        inner = env.inner
        generator, target = inner.vehicle, env.target
        origin_y, lane_y = float(generator.position[1]), float(target.position[1])
        if not (10.0 <= generator_speed <= 40.0 and 10.0 <= target_speed <= 40.0):
            raise ValueError('Initial speeds must stay within the registered 10--40 m/s range')
        generator.speed = float(generator_speed)
        target.speed = float(target_speed)
        # refresh lane bookkeeping after constructing the follower's position
        generator.on_state_update()
        target.on_state_update()
        target.target_lane_index = target.lane_index
        initial = [asdict(VehicleState.from_vehicle(v)) for v in inner.road.vehicles]
        initial_hash = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        initial_overlaps = []
        for i, a in enumerate(inner.road.vehicles):
            for j in range(i + 1, len(inner.road.vehicles)):
                if a._is_colliding(inner.road.vehicles[j], 0)[0]:
                    initial_overlaps.append([i, j])
        if initial_overlaps:
            return {'valid_reset': False, 'initial_overlaps': initial_overlaps, 'seed': seed,
                    'initial_hash': initial_hash, 'risk': 0.0, 'steps': 0}
        target_id = inner.road.vehicles.index(target)
        generator_id = inner.road.vehicles.index(generator)
        direct = defender in {'policy', 'shielded_policy', 'reference'}
        if direct:
            replacement = Vehicle.create_from(target)
            inner.road.vehicles[target_id] = replacement
            env.target = target = replacement
        else:
            target.enable_lane_change = defender == 'mobil'
        recorder = ContactRecorder(inner.road.vehicles)
        dt = 1.0 / env.driving._policy_freq
        previous_accel = previous_steer = 0.0
        min_ttc = None
        valid_frames = 0
        target_start_x = float(target.position[0])
        pair_ids = {generator_id, target_id}
        offroad = False
        target_wrong_way = False
        merge_complete = False
        trace, generator_controls = [], []
        for step in range(horizon):
            if direct:
                if defender == 'reference':
                    target_action = reference_action(target, generator, inner.road, lane_y)
                else:
                    if model is None:
                        raise ValueError('Missing defender model')
                    target_action = np.asarray(model.predict(native_defender_observation(inner, target), deterministic=True)[0])
                    if defender == 'shielded_policy':
                        target_action, _ = shield_action(VehicleState.from_vehicle(target),
                            [VehicleState.from_vehicle(v) for v in inner.road.vehicles if v is not target], target_action)
                if target_action.shape != (2,) or not np.isfinite(target_action).all():
                    raise ValueError('Bad defender action')
                target.act({'acceleration': 5 * float(np.clip(target_action[0], -1, 1)),
                            'steering': 0.3 * float(np.clip(target_action[1], -1, 1))})
            action = maneuver_action(generator, maneuver, step * dt, origin_y, lane_y,
                                     previous_accel, previous_steer, dt, mode)
            if action_tape is not None:
                action = action_tape[step].copy()
            elif background_guard:
                # Normal driving must respect background leaders. Apply the
                # same rule in every arm; the designated target is excluded.
                # If background feedback alters the attack between tested and
                # reference defenders, controls_match rejects its witness.
                cap, _ = shield_action(VehicleState.from_vehicle(generator),
                    [VehicleState.from_vehicle(v) for v in inner.road.vehicles if v not in (generator, target)], action)
                # Preserve the preregistered acceleration slew bound even when
                # the guard activates; collision rejection remains independent.
                action[0] = np.clip(cap[0], (previous_accel - 2.5 * dt) / 5,
                                    (previous_accel + 2.5 * dt) / 5)
            previous_accel, previous_steer = float(action[0]) * 5, float(action[1]) * .3
            generator_controls.append(action.tolist())
            recorder.step = step
            # Bypass generator-specific rewards/deadlines: defender escape is
            # a legitimate safe response, not an early interaction failure.
            _, _, term, trunc, info = env.driving.step(action)
            gs, ts = VehicleState.from_vehicle(generator), VehicleState.from_vehicle(target)
            ttc = constant_velocity_ttc(gs, ts)
            if ttc is not None:
                min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
            headway = following_headway(ts, gs)
            valid_frames += int(headway is not None)
            offroad |= not body_on_straight_road(generator) or not body_on_straight_road(target)
            target_wrong_way |= ts.vx < -0.1
            merge_complete |= abs(gs.y - lane_y) < 0.5 and abs(gs.heading) < .1
            if keep_trace:
                trace.append({'step': step, 'generator': asdict(gs), 'defender': asdict(ts),
                              'action': action.tolist(), 'ttc': ttc, 'headway': headway})
            if term or trunc or target.crashed or offroad or target_wrong_way:
                break
        involved = [e for e in confirmed_contacts(recorder.events, step) if set(e['actors']) & pair_ids]
        pair_events = [e for e in involved if set(e['actors']) == pair_ids]
        background_events = [e for e in involved if set(e['actors']) != pair_ids]
        flags = bool(generator.crashed or target.crashed)
        unattributed = flags and not involved
        pair_collision = bool(pair_events)
        return {
            'valid_reset': True, 'seed': seed, 'traffic_seed': reset_info['traffic_seed'],
            'initial_generator_speed': float(generator_speed),
            'initial_target_speed': float(target_speed),
            'initial_hash': initial_hash, 'defender': defender, 'mode': mode, 'steps': step + 1,
            'background_guard': background_guard,
            'replayed_tape': action_tape is not None,
            'pair_contact': pair_collision, 'background_contact': bool(background_events),
            'unattributed_crash': unattributed, 'crash_flag': flags,
            'offroad': bool(offroad), 'wrong_way': bool(info['wrong_way'] or target_wrong_way),
            'full_horizon': bool(step + 1 == horizon and not flags and not offroad and not target_wrong_way),
            'merge_complete': bool(merge_complete), 'valid_fraction': valid_frames / horizon,
            'target_progress_m': float(target.position[0]) - target_start_x,
            'min_pair_ttc': min_ttc,
            'exclusive_near_miss': bool(not flags and not involved and min_ttc is not None and min_ttc < 1.5),
            'risk': risk_score(pair_collision, min_ttc), 'contacts': involved,
            'pending_impacts_at_horizon': [e for e in recorder.events if e['step'] == step and e['kind'] == 'scheduled_impact'],
            'generator_controls': generator_controls, 'trace': trace,
        }
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def controls_match(first, second, atol=1e-6):
    a, b = first.get('generator_controls', []), second.get('generator_controls', [])
    length = min(len(a), len(b))
    # Scheduled impact can perturb states from the first recorded contact.
    for row in (first, second):
        if row.get('contacts'):
            length = min(length, min(e['step'] for e in row['contacts']) + 1)
    return bool(length > 0 and np.allclose(a[:length], b[:length], atol=atol, rtol=0))


def candidate_score(groups, fraction=.5):
    """Groups contain matched nominal/attack/witness for each scene/defender."""
    deltas, reasons = {}, []
    for group in groups:
        nominal, attack, witness = (group[k] for k in ('nominal', 'attack', 'witness'))
        if not all(r['valid_reset'] for r in (nominal, attack, witness)):
            reasons.append('invalid_initial_scene')
            continue
        if len({r['initial_hash'] for r in (nominal, attack, witness)}) != 1:
            raise RuntimeError('Paired worlds have different initial states')
        for key, row in (('nominal', nominal), ('witness', witness)):
            if (not row['full_horizon'] or row['contacts'] or row['wrong_way']
                    or row['target_progress_m'] < 100):
                reasons.append(key + '_not_safe_and_progressing')
        if attack['offroad'] or attack['wrong_way'] or attack['background_contact'] or attack['unattributed_crash']:
            reasons.append('invalid_attack')
        if not controls_match(attack, witness):
            reasons.append('witness_program_mismatch')
        # No encounter is assigned its observed risk (normally zero); never
        # drop safe MOBIL evasion from the denominator.
        deltas.setdefault(group['defender_id'], []).append(attack['risk'] - nominal['risk'])
    valid = not reasons and bool(deltas)
    by_defender = {key: float(np.mean(values)) for key, values in deltas.items()}
    score = lower_tail_mean(list(by_defender.values()), fraction) if valid else -1.0
    return {'admissible': valid, 'score': score, 'rejection_reasons': sorted(set(reasons)),
            'paired_risk_delta_by_defender': by_defender}
