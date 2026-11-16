"""Separate fixed-speed tracking qualification from reactive interaction outcomes."""
from __future__ import annotations

from dataclasses import asdict
import math

from src.evaluation.safety_metrics import VehicleState, following_headway


def configure_pair_target(env, kind):
    if kind not in {'fixed_speed', 'idm'}:
        raise ValueError('Unknown target behavior')
    before = asdict(VehicleState.from_vehicle(env.target))
    if kind == 'fixed_speed':
        from highway_env.vehicle.kinematics import Vehicle
        if abs(before['heading']) > 1e-12 or abs(before['vy']) > 1e-12:
            raise ValueError('Fixed-speed bench requires a straight initial target')
        env.target = Vehicle.create_from(env.target)
        env.target.act({'acceleration': 0.0, 'steering': 0.0})
    env.inner.road.vehicles = [env.inner.vehicle, env.target]
    env.target_id = 1
    if asdict(VehicleState.from_vehicle(env.target)) != before:
        raise ValueError('Target behavior change modified physical initial state')
    return before


def window_metrics(episode, begin, end, hz=10):
    """Absolute half-open tick windows; never condition the horizon on survival."""
    if not 0 <= begin < end <= episode['horizon']:
        raise ValueError('Window outside planned horizon')
    rows = [r for r in episode['trace'] if begin <= r['tick'] < end]
    expected = end - begin
    errors = [r['headway_error'] for r in rows if r['headway_error'] is not None]
    ttc = [r['pair_ttc'] for r in rows if r['pair_ttc'] is not None]
    drac = [r['following_drac'] for r in rows if r['following_drac'] is not None]
    geometry_errors = []
    desired = episode['reset_info']['target_headway']
    for row in rows:
        g, t = VehicleState(**row['generator']), VehicleState(**row['target'])
        h = following_headway(t, g)
        valid_geometry = h is not None and abs(g.heading) <= math.radians(15) and abs(t.heading) <= math.radians(15)
        if valid_geometry and not row['generator_offroad'] and not row['target_offroad']:
            geometry_errors.append(abs(h - desired))
    total_collision = bool(episode['any_traffic_contact'] or episode['crashed_ids'])
    event = bool(ttc and min(ttc) < 1.5)
    speeds = [r['target']['vx'] for r in rows]
    return {
        'tick_range': [begin, end], 'planned_steps': expected, 'observed_steps': len(rows),
        'complete_window': len(rows) == expected, 'valid_steps': len(errors),
        'valid_fraction_planned': len(errors) / expected,
        'low_speed_steps': sum(v < 15 for v in speeds),
        'conditional_headway_mae': sum(errors) / len(errors) if errors else None,
        'geometry_headway_mae': sum(geometry_errors) / len(geometry_errors) if geometry_errors else None,
        'ttc_defined_steps': len(ttc), 'ttc_defined_fraction_planned': len(ttc) / expected,
        'min_pair_ttc': min(ttc) if ttc else None, 'ttc_event': event,
        'exclusive_ttc_event': event and not total_collision and episode['reset_valid'],
        'drac_defined_steps': len(drac), 'max_drac': max(drac) if drac else None,
        'target_speed_mean': sum(speeds) / len(speeds) if speeds else None,
        'target_speed_min': min(speeds) if speeds else None,
        'target_peak_braking': max([max(0., -r['acceleration'][1]) for r in rows], default=None),
        'shield_steps': sum(r['shield_intervened'] for r in rows),
        'sampled_duration_s': len(rows) / hz,
    }


def qualify_fixed_speed(episode, gate):
    if episode['target_kind'] != 'fixed_speed':
        raise ValueError('Tracking qualification is not an IDM safety gate')
    tail = episode['windows'][-1]
    return {
        'physical_valid': episode['reset_valid'] and not episode['physical_failure'],
        'full_horizon': episode['full_horizon'],
        'constant_target': episode['target_max_speed_drift'] <= gate['target_speed_drift_max'],
        'complete_tail': tail['complete_window'],
        'tail_all_valid': tail['valid_fraction_planned'] >= gate['tail_valid_fraction_min'],
        'tail_mae': tail['conditional_headway_mae'] is not None
                    and tail['conditional_headway_mae'] <= gate['tail_headway_mae_max'],
    }
