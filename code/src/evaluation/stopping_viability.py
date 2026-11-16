"""Discrete longitudinal stopping constraint; no collision safety guarantee.

For v_next=v+a*dt, reserve velocity for releasing negative acceleration at
maximum positive jerk. This is an engineering projection, not a novel method.
"""
import math
from contextlib import contextmanager

ARMS = ('following_jerk', 'following_viable', 'maximum_brake_jerk', 'maximum_brake_viable')


def release_velocity_loss(acceleration, dt, jerk=2.5):
    """Velocity consumed now and while releasing a negative acceleration."""
    if acceleration >= 0:
        return 0.0
    delta = jerk * dt
    n = math.ceil(-acceleration / delta)
    return dt * (n * -acceleration - delta * n * (n - 1) / 2)


def project(request, speed, previous, dt, jerk=2.5, reserve=1e-6):
    if not all(math.isfinite(x) for x in (request, speed, previous, dt, jerk, reserve)) or dt <= 0 or jerk <= 0 or reserve < 0:
        raise ValueError('Invalid actuator state')
    if abs(previous) > 5.000001 or speed < -1e-5:
        raise ValueError('Outside actuator operating domain')
    budget = max(0.0, speed - reserve)
    lo, hi = -5.0, 0.0
    # Upper endpoint is always feasible. Retain it to avoid unsafe root rounding.
    for _ in range(60):
        mid = (lo + hi) / 2
        if release_velocity_loss(mid, dt, jerk) <= budget:
            hi = mid
        else:
            lo = mid
    viable_floor = hi
    lower = max(-5.0, previous - jerk * dt, viable_floor)
    upper = min(5.0, previous + jerk * dt)
    # Float32 normalized commands have sub-micro acceleration roundoff.
    # Half the registered 1e-5 jerk tolerance is reserved for this boundary.
    infeasible = lower > upper + 5e-6 * dt
    # If an externally supplied state is already nonviable, preserve stopping
    # viability and disclose the jerk override. Do not silently claim both.
    applied = max(viable_floor, min(5.0, max(-5.0, request))) if infeasible else max(lower, min(upper, request))
    return dict(request=request, speed=speed, previous=previous, dt=dt,
                viable_floor=viable_floor, lower=lower, upper=upper,
                acceleration=applied, release_loss=release_velocity_loss(applied, dt, jerk),
                intersection_gap_mps2=max(0.0, lower-upper),
                constraint_infeasible=infeasible,
                jerk_override=abs(applied - previous) > jerk * dt + 1e-5,
                next_speed_prediction=speed + applied * dt)


@contextmanager
def intervention(module, arm):
    """Dedicated sequential worker: capture generator state, patch last stage."""
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    nominal, brake = module.nominal_action, module.brake_priority
    state, rows = {}, []

    def capture(generator, origin_y, previous_accel, previous_steer, dt, cruise_speed=25.0):
        state.update(speed=float(generator.speed), previous=previous_accel, dt=dt)
        action = nominal(generator, origin_y, previous_accel, previous_steer, dt, cruise_speed=cruise_speed).copy()
        if arm.startswith('maximum_'):
            action[0] = -1.0
        return action

    def viable(limited, guarded, shield_intervened, enabled):
        row = project(5 * guarded, **state)
        rows.append(dict(step=len(rows), **row))
        return row['acceleration'] / 5

    try:
        if arm != 'following_jerk':
            module.nominal_action = capture
        if arm.endswith('_viable'):
            module.brake_priority = viable
        yield rows
    finally:
        module.nominal_action, module.brake_priority = nominal, brake
