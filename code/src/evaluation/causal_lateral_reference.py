"""Pure lane-goal reference prototypes; NOT a validated native controller.

The only scene inputs are current lateral position/velocity/acceleration.
No recorded future path, target identity, outcome or future target state is read.
The committed planner uses a three-position history splice, not an instantaneous
C2 jet. Reference extrema cannot certify steering, body-on-road, or quantized
actuation. The older C2 constructor remains an explicit analytic comparison.
"""
from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial import Polynomial

PATH_ERROR_LIMIT = 1e-5
DT = .1
# For |e[n]|<=path tolerance, |Delta² e|/dt²<=4 eps/dt² and
# |Delta³ e|/dt³<=8 eps/dt³. Reserve these amounts from physical bounds.
REFERENCE_ACCELERATION_LIMIT = 2.-4*PATH_ERROR_LIMIT/DT**2
REFERENCE_JERK_LIMIT = 4.-8*PATH_ERROR_LIMIT/DT**3


def _finite(values):
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number))
           or not math.isfinite(x) for x in values):
        raise ValueError('Finite real scalars required')


@dataclass(frozen=True)
class LateralReference:
    coefficients: tuple
    duration: float
    history_dt: float = 0.

    def evaluate(self, elapsed, derivative=0):
        _finite((elapsed,))
        if derivative not in (0, 1, 2, 3) or isinstance(derivative, bool):
            raise ValueError('Only position through jerk are supported')
        if elapsed < 0:
            raise ValueError('Cannot query pre-branch reference')
        p = Polynomial(self.coefficients)
        if elapsed >= self.duration:
            return float(p(1.)) if derivative == 0 else 0.
        return float(p.deriv(derivative)(elapsed/self.duration)/self.duration**derivative)

    def extrema(self, derivative, *, include_history=False):
        if derivative not in (0, 1, 2, 3) or isinstance(derivative, bool):
            raise ValueError('Only position through jerk are supported')
        p = Polynomial(self.coefficients).deriv(derivative)/self.duration**derivative
        roots = p.deriv().roots()
        lower = -2*self.history_dt/self.duration if include_history else 0.
        points = [lower, 1.]+[float(x.real) for x in roots if abs(x.imag) <= 1e-9 and lower < x.real < 1]
        values = [float(p(x)) for x in points]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite polynomial extrema')
        return min(values), max(values)


def quintic_reference(y, vy, ay, lane_y, duration):
    """Preserve branch position/velocity/acceleration; stop at the lane center."""
    _finite((y, vy, ay, lane_y, duration))
    if duration <= 0:
        raise ValueError('Positive reference duration required')
    initial = np.asarray([y, vy*duration, .5*ay*duration**2], dtype=float)
    tail = np.linalg.solve([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]],
                          [lane_y-sum(initial), -initial[1]-2*initial[2], -2*initial[2]])
    coefficients = tuple(float(x) for x in np.concatenate((initial, tail)))
    if not all(math.isfinite(x) for x in coefficients):
        raise ValueError('Nonfinite reference coefficients')
    return LateralReference(coefficients, float(duration))


def history_reference(y, backward_vy, backward_ay, lane_y, duration, dt=.1):
    """Fit last three positions and terminal (y, vy=0, ay=0).

    backward_vy=(y[n]-y[n-1])/dt, backward_ay=(y[n]-2y[n-1]+y[n-2])/dt².
    At reset only, synthetic constant-velocity history requires explicit ay=0.
    This is a causal discrete-history splice, not continuous-derivative matching.
    """
    _finite((y, backward_vy, backward_ay, lane_y, duration, dt))
    if duration <= 0 or dt != .1:
        raise ValueError('Positive duration and frozen .1s history interval required')
    past = [y-2*backward_vy*dt+backward_ay*dt**2, y-backward_vy*dt, y]
    times = [-2*dt/duration, -dt/duration, 0., 1.]
    matrix = [[u**k for k in range(6)] for u in times]
    matrix += [[0., 1., 2., 3., 4., 5.], [0., 0., 2., 6., 12., 20.]]
    coefficients = tuple(float(x) for x in np.linalg.solve(matrix, [*past, lane_y, 0., 0.]))
    if not all(math.isfinite(x) for x in coefficients):
        raise ValueError('Nonfinite history reference')
    return LateralReference(coefficients, float(duration), dt)


def reference_audit(reference, *, corridor=(0., 4.), acceleration_limit=REFERENCE_ACCELERATION_LIMIT,
                    jerk_limit=REFERENCE_JERK_LIMIT, tolerance=1e-9):
    _finite((*corridor, acceleration_limit, jerk_limit, tolerance))
    if len(corridor) != 2 or corridor[0] >= corridor[1] or min(acceleration_limit, jerk_limit) <= 0 or tolerance < 0:
        raise ValueError('Invalid reference bounds')
    extrema = {name: reference.extrema(order, include_history=order >= 2)
               for order, name in enumerate(('y', 'vy', 'ay', 'jy'))}
    checks = dict(center_corridor=extrema['y'][0] >= corridor[0]-tolerance and extrema['y'][1] <= corridor[1]+tolerance,
                  lateral_acceleration=max(map(abs, extrema['ay'])) <= acceleration_limit+tolerance,
                  lateral_jerk=max(map(abs, extrema['jy'])) <= jerk_limit+tolerance)
    return dict(eligible=all(checks.values()), checks=checks, extrema=extrema,
                native_validated=False)


def choose_reference(y, vy, ay, lane_y, *, durations=(2., 3., 4., 5., 6.)):
    """Shortest eligible reference from a fixed grid, independent of target risk.

    The grid is an explicit engineering choice for this prototype. Infeasibility
    raises; do not replace it by a discontinuity, teleport or unrecorded fallback.
    """
    _finite((y, vy, ay, lane_y, *durations))
    if lane_y not in (0., 4.) or not durations or any(x <= 0 for x in durations) or list(durations) != sorted(set(durations)):
        raise ValueError('Lane goal 0/4 and strictly increasing positive duration grid required')
    for duration in durations:
        reference = quintic_reference(y, vy, ay, lane_y, duration)
        audit = reference_audit(reference)
        if audit['eligible']:
            return reference, audit
    raise ValueError('No reference in the frozen grid is eligible; native action is not authorized')


class CommittedLateralPlanner:
    """Retain a finite deadline until a feasible *different* lane goal is chosen.

    A rejected switch explicitly continues the certified existing tail. This is
    reference projection, shared by all future comparators, not a native safety
    guarantee. The adapter must separately check actual tracking and dynamics.
    """
    def __init__(self, episode_horizon=12., durations=(2., 3., 4., 5., 6.)):
        _finite((episode_horizon, *durations))
        if episode_horizon <= 0 or not durations or any(t <= 0 for t in durations) or list(durations) != sorted(set(durations)):
            raise ValueError('Positive horizon and increasing duration grid required')
        self.horizon, self.durations = float(episode_horizon), tuple(durations)
        self.reference = self.goal = self.start = self.last_request_time = None

    def request(self, time, lane_y, *, y, vy, ay):
        """vy/ay are completed-position backward differences, not body velocity."""
        _finite((time, lane_y, y, vy, ay))
        if (not 0 <= time < self.horizon or lane_y not in (0., 4.)
                or self.last_request_time is not None and time <= self.last_request_time):
            raise ValueError('Ordered in-horizon decision and lane goal 0/4 required')
        if self.reference is not None and lane_y == self.goal:
            self.last_request_time = time
            return self._receipt(time, lane_y, 'continue_committed_goal')
        eligible_durations = tuple(t for t in self.durations if time+t <= self.horizon)
        candidate = None
        for duration in eligible_durations:
            reference = history_reference(y, vy, ay, lane_y, duration)
            if reference_audit(reference)['eligible']:
                candidate = reference
                break
        if candidate is None and self.reference is None:
            raise ValueError('No initial eligible reference; no hidden fallback')
        self.last_request_time = time
        if candidate is None:
            return self._receipt(time, lane_y, 'switch_rejected_continue_existing_tail')
        self.reference, self.goal, self.start = candidate, float(lane_y), float(time)
        return self._receipt(time, lane_y, 'new_committed_goal')

    def _receipt(self, time, requested, status):
        return dict(time=time, requested_lane_goal=requested, applied_lane_goal=self.goal,
                    status=status, reference_start=self.start,
                    reference_deadline=self.start+self.reference.duration,
                    native_validated=False)

    def evaluate(self, time, derivative=0):
        _finite((time,))
        if self.reference is None:
            raise RuntimeError('Request an initial reference first')
        return self.reference.evaluate(time-self.start, derivative)
