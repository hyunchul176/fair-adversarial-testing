"""Observational wrapper for frozen nominal-rollout failure diagnosis.

Imports the simulator only when called on a worker. Hooks delegate every original
operation exactly once, consume no randomness and are restored even on failure.
Not thread safe: run one diagnostic rollout at a time in a dedicated worker.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
import importlib
import math


def _plain(value):
    """Copy numerical metadata without retaining mutable simulator objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if hasattr(value, 'item'):
        return _plain(value.item())
    return str(value)


def rollout_with_telemetry(**rollout_kwargs):
    """Return the unchanged nominal result plus ``diagnostic_telemetry``.

    ``keep_trace`` is forced true; its original pair trace remains untouched.
    Full actor snapshots retain only the final 20 policy ticks, while shield
    actions and observations retain the entire episode. This adds no controller
    calls, rollouts, candidate evaluations or changes to world/control state.
    """
    if rollout_kwargs.get('mode', 'attack') != 'nominal':
        raise ValueError('Nominal telemetry requires explicit mode=nominal')
    if rollout_kwargs.get('action_tape') is not None:
        raise ValueError('Nominal telemetry does not support actuator replay')
    module = importlib.import_module('src.evaluation.nominal_intervention_probe')
    names = ('ContactRecorder', 'native_defender_observation', 'shield_action',
             'apply_forward_stop_guard', '_initial_hash')
    original = {name: getattr(module, name) for name in names}
    shield_rows, observations, stop_rows = [], [], []
    snapshots = deque(maxlen=20)
    initial_before, initial_after = [], []
    role_indices = {'generator': None, 'target': None}
    vehicle_indices = {}
    generator_calls = 0
    expect_target_shield = False
    observation_config = None

    def world(vehicles):
        return [{
            'actor_index': index,
            'class': type(vehicle).__module__ + '.' + type(vehicle).__qualname__,
            'state': _plain(module.VehicleState.from_vehicle(vehicle)),
            'speed': _plain(getattr(vehicle, 'speed', None)),
            'lane_index': _plain(getattr(vehicle, 'lane_index', None)),
            'target_lane_index': _plain(getattr(vehicle, 'target_lane_index', None)),
            'target_speed': _plain(getattr(vehicle, 'target_speed', None)),
            'enable_lane_change': _plain(getattr(vehicle, 'enable_lane_change', None)),
            'action': _plain(getattr(vehicle, 'action', None)),
            'crashed': bool(getattr(vehicle, 'crashed', False)),
        } for index, vehicle in enumerate(vehicles)]

    def initial_hash(vehicles):
        nonlocal initial_before
        answer = original['_initial_hash'](vehicles)
        initial_before = world(vehicles)
        return answer

    class RecordingProxy:
        def __init__(self, vehicles):
            nonlocal initial_after, vehicle_indices
            self._vehicles = vehicles
            self._recorder = original['ContactRecorder'](vehicles)
            vehicle_indices = {id(v): i for i, v in enumerate(vehicles)}
            initial_after = world(vehicles)

        @property
        def events(self):
            return self._recorder.events

        @property
        def step(self):
            return self._recorder.step

        @step.setter
        def step(self, value):
            self._recorder.step = value
            snapshots.append({'step': int(value), 'actors_before_step': world(self._vehicles)})

        def close(self):
            return self._recorder.close()

    def observation(inner, observer_vehicle=None):
        nonlocal expect_target_shield, observation_config
        answer = original['native_defender_observation'](inner, observer_vehicle)
        role_indices['generator'] = vehicle_indices.get(id(inner.vehicle))
        role_indices['target'] = vehicle_indices.get(id(observer_vehicle))
        if observation_config is None:
            observation_config = _plain(inner.config.get('observation'))
        observations.append({'step': generator_calls, 'observation': _plain(answer)})
        expect_target_shield = rollout_kwargs.get('defender') == 'shielded_policy'
        return answer

    def shield(ego, traffic, raw_action, **kwargs):
        nonlocal generator_calls, expect_target_shield
        role = 'target' if expect_target_shield else 'generator'
        step = generator_calls
        raw = _plain(raw_action)
        answer = original['shield_action'](ego, traffic, raw_action, **kwargs)
        applied, details = answer
        shield_rows.append({'step': step, 'role': role, 'ego': _plain(ego),
                            'raw_action': raw, 'shielded_action': _plain(applied),
                            'shield': _plain(details)})
        if role == 'target':
            expect_target_shield = False
        else:
            generator_calls += 1
        return answer

    def stop_guard(action, speed, dt, **kwargs):
        raw = _plain(action)
        answer = original['apply_forward_stop_guard'](action, speed, dt, **kwargs)
        guarded, intervened, floor = answer
        stop_rows.append({'step': generator_calls - 1, 'speed': float(speed),
                          'dt': float(dt), 'before_action': raw,
                          'guarded_action': _plain(guarded),
                          'intervened': bool(intervened), 'floor': float(floor)})
        return answer

    replacements = dict(ContactRecorder=RecordingProxy,
                        native_defender_observation=observation,
                        shield_action=shield, apply_forward_stop_guard=stop_guard,
                        _initial_hash=initial_hash)
    try:
        for name, replacement in replacements.items():
            setattr(module, name, replacement)
        result = module.rollout_feedback(**{**rollout_kwargs, 'keep_trace': True})
    finally:
        for name, value in original.items():
            setattr(module, name, value)

    # Native observation supplies role identity for the intended policy runs.
    # No guessed mapping for invalid resets or controllers without observations.
    def actor_role(index):
        for role, actor in role_indices.items():
            if actor is not None and index == actor:
                return role
        return 'background' if all(v is not None for v in role_indices.values()) else 'unknown'

    contacts = [{**_plain(event), 'actor_roles': [actor_role(i) for i in event['actors']]}
                for event in result.get('contacts', [])]
    controls = result.get('generator_controls', [])
    overrides = []
    stops_by_step = {row['step']: row for row in stop_rows}
    for row in shield_rows:
        if row['role'] != 'generator' or row['step'] >= len(controls):
            continue
        applied = _plain(controls[row['step']])
        row['final_generator_action'] = applied
        # Separate the stop guard from the subsequent jerk cap; either can
        # relax a shield command, and their causal effects are not equivalent.
        difference = float(applied[0]) - float(row['shielded_action'][0])
        row['post_shield_acceleration_delta_mps2'] = 5.0 * difference
        stopped = stops_by_step.get(row['step'])
        post_stop = float(stopped['guarded_action'][0]) if stopped else None
        row['post_stop_acceleration_delta_mps2'] = (
            5.0 * (float(applied[0]) - post_stop) if post_stop is not None else None)
        if row['shield'].get('intervened') and difference > 1e-8:
            overrides.append({'step': row['step'], 'shield_acceleration_mps2':
                              5.0 * float(row['shielded_action'][0]),
                              'post_stop_acceleration_mps2': 5.0 * post_stop if post_stop is not None else None,
                              'final_acceleration_mps2': 5.0 * float(applied[0]),
                              'jerk_cap_relaxed_braking': bool(post_stop is not None
                                  and float(applied[0]) > post_stop + 1e-8)})
    for snapshot in snapshots:
        for actor in snapshot['actors_before_step']:
            actor['role'] = actor_role(actor['actor_index'])
    telemetry = {
        'version': 'nominal-intervention-telemetry-v1', 'observational_only': True,
        'scope': 'Telemetry wrapper is observational; evaluator intervention is explicit in result.',
        'extra_model_predictions': 0, 'extra_simulation_steps': 0,
        'role_indices': role_indices, 'observation_config': observation_config,
        'initial_actors_before_policy_replacement': initial_before,
        'initial_actors_after_policy_replacement': initial_after,
        'observations': observations, 'shield_actions': shield_rows,
        'forward_stop_guard_actions': stop_rows,
        'generator_braking_relaxed_after_shield': overrides,
        'contacts_with_roles': contacts, 'pre_step_actor_window': list(snapshots),
        'actor_window_policy_steps': 20,
        'limitation': 'Contact roles locate participants, not causal responsibility. '
                      'Pre-step background actions may change in native road.act. '
                      'Observation config is recorded as supplied, not resolved library defaults.'}
    return {**result, 'diagnostic_telemetry': telemetry}
