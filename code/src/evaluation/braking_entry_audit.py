"""Independent closed-form actuator envelope and own-motion audit, no policy calls."""
import math
import numpy as np
from src.evaluation.safety_metrics import VehicleState

def release_cap(distance):
    """Analytical inverse of the discrete release sum, independent of bisection."""
    budget = max(0., distance - 1e-6)
    decrement = .199
    for n in range(1, 17):
        endpoint = .1 * decrement * n * (n + 1) / 2
        if budget <= endpoint:
            return min(3. - 1e-6, budget / (.1 * n) + decrement * (n - 1) / 2)
    return 3. - 1e-6


def mapped(command, scale):
    """Pinned action's float32 affine formula; no production control helper."""
    a = np.float32(-scale)
    b = np.float32(np.float32(command) + np.float32(1))
    c = np.float32(b * np.float32(2 * scale))
    return float(np.float32(a + np.float32(c / np.float32(2))))


def mapped_projection(request, lower, upper, scale):
    desired = max(lower, min(upper, request))
    command = np.float32(desired / scale)
    value = mapped(command, scale)
    if value < lower or value > upper:
        reserve = min(2e-6 * scale, (upper - lower) / 2)
        command = np.float32(max(lower + reserve, min(upper - reserve, desired)) / scale)
        value = mapped(command, scale)
    if not lower <= value <= upper:
        raise ValueError('Unrepresentable independent interval')
    return value


def independent_step(state, speed, previous_a, previous_steer, y_reference, request):
    # Closed form release envelope rather than the production 60-step search.
    alo = max(-3. + 1e-6, previous_a - .2, -release_cap(speed - 15))
    ahi = min(3. - 1e-6, previous_a + .2, release_cap(40 - speed))
    if alo > ahi:
        raise ValueError('Infeasible independent acceleration interval')
    a = mapped_projection(request, alo, ahi, 5.)
    lateral_ratio = max(-.99, min(.99, (y_reference - state.y) / (.1 * speed)))
    command_angle = math.atan(2 * math.tan(math.asin(lateral_ratio) - state.heading))
    slo, shi = max(-.3, previous_steer - .015), min(.3, previous_steer + .015)
    steer = mapped_projection(command_angle, slo, shi, .3)
    beta = math.atan(math.tan(steer) / 2)
    heading = state.heading + 2 * speed * math.sin(beta) * .1 / state.length
    next_speed = speed + .1 * a
    result = VehicleState(state.x + .1 * speed * math.cos(state.heading + beta),
                          state.y + .1 * speed * math.sin(state.heading + beta),
                          next_speed * math.cos(heading), next_speed * math.sin(heading),
                          heading, state.length, state.width)
    return result, next_speed, (next_speed-speed)/.1, steer, a

