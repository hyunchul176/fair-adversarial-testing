#!/usr/bin/env python3
"""Frozen 108-episode mechanism screen; all simulations run on research CPU."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.environments.physical_interaction_env import PhysicalInteractionEnv
from src.evaluation.braking_response import DelayedIDMVehicle, braking_pulse, response_window
from src.evaluation.generator_measurement import GeneratorMeasurement, digest, json_value
from src.evaluation.safety_metrics import VehicleState


def build_env(config, speed, headway, delay):
    # cut_in selects the existing measurement roles (target follows generator).
    # The constructed same-lane launch is NOT an observed or executed cut-in.
    env = PhysicalInteractionEnv(scenario_type='cut_in', c_level=0.4, **config['environment'])
    _, info = env.reset(seed=config['seed'])
    leader = env.inner.vehicle
    old_target = env.target
    lane_index = leader.lane_index
    lane = env.inner.road.network.get_lane(lane_index)
    x, _ = lane.local_coordinates(leader.position)
    leader.position = lane.position(x, 0.)
    leader.heading = 0.
    leader.speed = speed
    leader.on_state_update()
    target = DelayedIDMVehicle(env.inner.road,
        lane.position(x - headway * speed - (leader.LENGTH + old_target.LENGTH) / 2, 0.),
        heading=0., speed=speed, target_lane_index=lane_index,
        target_speed=speed * config['target_desired_speed_factor'], enable_lane_change=False)
    target.initialize_delay(delay)
    env.target = target
    env.target_id = 1
    env.inner.road.vehicles = [leader, target]
    env.target_headway = headway
    env.previous_speed = None
    env.previous_potential = env._potential()
    env.driving._initial_x = float(leader.position[0])
    env.driving._previous_x = float(leader.position[0])
    info.update(target_id=1, target_headway=headway,
                launch_contract='synthetic-aligned-following-v1',
                initial_speed=speed, initial_headway=headway, delay_ticks=delay,
                scenario_executed='aligned_leader_braking', **env._metrics())
    return env, info


def run(config, speed, headway, delay, magnitude):
    env, reset_info = build_env(config, speed, headway, delay)
    meter = GeneratorMeasurement(env, reset_info)
    physical = [asdict(VehicleState.from_vehicle(v)) for v in env.inner.road.vehicles]
    info = {}
    try:
        if not any(meter.reset_invalid.values()):
            for tick in range(config['environment']['duration']):
                acceleration = braking_pulse(tick / 10, magnitude, **config['pulse'])
                action = np.array([acceleration / 5.0, 0.0], dtype=np.float32)
                meter.before_step()
                _, _, terminated, truncated, info = env.step(action)
                info.update(high_level_action=[float(magnitude), 0.],
                            applied_action=action.tolist(), shield_intervened=False,
                            low_speed_stop_guard_intervened=False)
                meter.after_step(info)
                meter.trace[-1].update(prescribed_acceleration=acceleration,
                                      target_act_calls=env.target.act_calls)
                if terminated or truncated:
                    break
        result = meter.finish(info)
        result.update(physical_initial_hash=digest(physical), physical_initial_state=physical,
                      target_act_calls=env.target.act_calls)
        result['windows'] = [response_window(result, a, b) for a, b in config['windows_ticks']]
        post = response_window(result, 20, 120)
        result['post_onset'] = post
        result['target_speed_loss_post'] = None if post['target_min_speed'] is None else speed - post['target_min_speed']
        rows = result['trace']
        bounds, tol = config['bounds'], config['bounds']['numeric_tolerance']
        # Only pre-contact dynamics are candidates for maneuver-validity checks;
        # collision response can itself create acceleration discontinuities.
        contact_ticks = [e['step'] for e in result['confirmed_contacts']]
        first_contact = min(contact_ticks, default=len(rows))
        before_contact = rows[:first_contact]
        actual = [r['acceleration'][0] for r in before_contact]
        jerk = [r['jerk'][0] for r in before_contact if r['jerk'] is not None]
        result['motion_checks'] = {
            'reset_valid': result['reset_valid'],
            'acceleration_bound': all(abs(a) <= bounds['generator_acceleration_abs_max'] + tol for a in actual),
            'jerk_bound': all(abs(j) <= bounds['generator_jerk_abs_max'] + tol for j in jerk),
            'speed_floor': all(r['generator']['vx'] >= bounds['generator_speed_min'] - tol for r in before_contact),
            'no_offroad_or_reverse': not result['offroad'] and not result['wrong_way'],
            'act_once_per_tick': result['target_act_calls'] == result['steps'],
            'prescribed_action_realized': all(abs(r['acceleration'][0] - r['prescribed_acceleration']) <= tol for r in before_contact),
        }
        result['generator_max_abs_acceleration_precontact'] = max(map(abs, actual), default=None)
        result['generator_max_abs_jerk_precontact'] = max(map(abs, jerk), default=None)
        return result
    finally:
        meter.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'protocol.json').write_bytes(args.config.read_bytes())
    import highway_env
    package = Path(highway_env.__file__).resolve().parent
    files = list((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve()]
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    external = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.rglob('*.py'))}
    provenance = dict(source_files=source, source_digest=digest(source), external_files=external,
                      external_digest=digest(external), config_digest=digest(config),
                      runtime={k: importlib.metadata.version(k) for k in ('numpy','gymnasium','highway-env')})
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    rows, start = [], time.monotonic()
    with (args.output/'episodes.jsonl').open('w') as stream:
        for speed, headway, delay, magnitude in itertools.product(config['initial_speeds_mps'],
                config['initial_headways_s'], config['target_delays_ticks'], config['braking_magnitudes_mps2']):
            row = dict(initial_speed=speed, initial_headway=headway, delay_ticks=delay, magnitude=magnitude)
            try:
                row.update(run(config, speed, headway, delay, magnitude))
            except Exception as exc:
                row['error'] = type(exc).__name__ + ': ' + str(exc)
            stream.write(json.dumps(row, default=json_value, allow_nan=False)+'\n')
            stream.flush()
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ('initial_speed','initial_headway','delay_ticks','magnitude','steps','min_pair_ttc','error')}), flush=True)
    errors = sum('error' in r for r in rows)
    matched = []
    for speed, headway, delay in itertools.product(config['initial_speeds_mps'], config['initial_headways_s'], config['target_delays_ticks']):
        group = [r for r in rows if (r['initial_speed'],r['initial_headway'],r['delay_ticks']) == (speed,headway,delay)]
        # Exclude intervention metadata when comparing identical prefix physics.
        prefixes = [[{k:t[k] for k in ('generator','target','acceleration','pair_ttc','following_drac')}
                     for t in r.get('trace',[])[:20]] for r in group]
        matched.append(not any('error' in r for r in group)
                       and len({r['initial_hash'] for r in group}) == 1
                       and len({digest(p) for p in prefixes}) == 1)
    summary = dict(registered=len(rows), errors=errors, steps=sum(r.get('steps',0) for r in rows),
                   all_prefixes_and_worlds_paired=all(matched),
                   all_motion_checks=not errors and all(all(r['motion_checks'].values()) for r in rows),
                   physical_failures=sum(r.get('physical_failure',False) for r in rows),
                   exclusive_ttc_events=sum(r.get('exclusive_ttc_event',False) for r in rows),
                   elapsed_seconds=time.monotonic()-start,
                   episodes_sha256=hashlib.sha256((args.output/'episodes.jsonl').read_bytes()).hexdigest())
    summary['integrity_passed'] = (not errors and summary['all_prefixes_and_worlds_paired']
                                  and summary['all_motion_checks'] and len(rows)==config['reporting']['registered_episodes'])
    (args.output/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not summary['integrity_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
