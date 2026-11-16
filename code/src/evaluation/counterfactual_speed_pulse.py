"""Bounded speed-deficit reference; a counterfactual, not observed braking."""
import math


def shape(duration, policy, curvature_cap=1.):
    """Limit pulse-component curvature, not total reference/vehicle jerk."""
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(curvature_cap) or curvature_cap <= 0:
        raise ValueError('Positive finite duration and curvature cap required')
    if policy == 'amplitude_cap':
        return min(2., curvature_cap*duration**2/24), duration
    if policy == 'duration_floor':
        return 2., max(duration, math.sqrt(48/curvature_cap))
    raise ValueError(policy)


def deficit(time, duration, amplitude):
    if not all(math.isfinite(x) for x in (time, duration, amplitude)) or duration <= 0 or not 0 <= amplitude <= 2:
        raise ValueError('Finite time, positive duration and amplitude in [0, 2] required')
    if time <= 0 or time >= duration:
        return 0.
    u = time / duration
    return amplitude * 64 * u**3 * (1-u)**3


def delayed_deficit(time, duration, amplitude, start):
    """Translate the complete C2 pulse without changing its shape or area."""
    if not math.isfinite(start) or start < 0:
        raise ValueError('Finite nonnegative pulse start required')
    return deficit(time-start, duration, amplitude)
