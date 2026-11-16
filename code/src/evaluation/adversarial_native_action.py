"""Quantization-aware commands for the pinned NumPy-2 native affine mapping."""
import math
import numpy as np
from src.evaluation.adversarial_agent_core import release_distance


def native_map(command, scale):
    s = np.float32(scale)
    return float(-s + (np.float32(command) + np.float32(1)) * np.float32(2 * scale) / np.float32(2))


def representable_command(request, lower, upper, scale):
    if not all(math.isfinite(x) for x in (request, lower, upper, scale)):
        raise ValueError('Nonfinite command interval')
    if scale <= 0 or lower > upper or lower < -scale or upper > scale:
        raise ValueError('Invalid command interval')
    desired = min(upper, max(lower, request))
    command = np.float32(desired / scale)
    applied = native_map(command, scale)
    if not lower <= applied <= upper:
        # Invert the monotone native mapping with an interior target. Native
        # float32 arithmetic has plateaus; stepping tiny normalized ULPs near
        # zero need not change the de-normalized value.
        interior = min(upper, max(lower, desired))
        margin = min(2e-6 * scale, (upper - lower) / 2)
        interior = min(upper - margin, max(lower + margin, interior))
        command = np.float32(interior / scale)
        applied = native_map(command, scale)
    if not lower <= applied <= upper:
        raise ValueError('No selected representable command inside feasible interval')
    return command, applied


def acceleration_command(request, speed, previous, dt=.1):
    """Actual limits 3 m/s² and 2 m/s³; conservative release reserve 1.99.

    Reserve slack is fixed across all online methods, not a relaxed audit bound.
    Float32 mapped commands, not ideal normalized*5, drive the next actuator state.
    """
    if not all(math.isfinite(x) for x in (request, speed, previous, dt)) or dt != .1:
        raise ValueError('Unsupported actuator input')
    if not 15 <= speed <= 40 or abs(previous) > 3 + 1e-10:
        raise ValueError('Outside physical actuator domain')

    def cap(distance):
        budget = max(0., distance - 1e-6)
        lo, hi = 0., 3. - 1e-6
        for _ in range(60):
            mid = (lo + hi) / 2
            if release_distance(mid, dt, 1.99) <= budget:
                lo = mid
            else:
                hi = mid
        return lo

    lower = max(-3. + 1e-6, previous - 2 * dt, -cap(speed - 15))
    upper = min(3. - 1e-6, previous + 2 * dt, cap(40 - speed))
    if lower > upper:
        raise ValueError('Native speed/acceleration/jerk intersection infeasible')
    command, applied = representable_command(request, lower, upper, 5.)
    return command, dict(requested_mps2=request, applied_mps2=applied,
                         lower_mps2=lower, upper_mps2=upper,
                         intervened=abs(applied - request) > 1e-5,
                         next_scalar_speed=speed + applied * dt,
                         release_jerk_mps3=1.99, speed_reserve_mps=1e-6)


def steering_command(request, previous, dt=.1):
    lower = max(-.3, previous - .15 * dt)
    upper = min(.3, previous + .15 * dt)
    return representable_command(request, lower, upper, .3)
