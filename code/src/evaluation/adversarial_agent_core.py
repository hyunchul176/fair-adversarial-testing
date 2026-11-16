"""Pure primitives for the online adversary contract; no simulator or PPO imports.

These primitives do not certify a complete vehicle maneuver or ego recoverability.
The simulator adapter must independently validate pre-impact motion and contacts.
"""
from dataclasses import dataclass
import math
from numbers import Integral


VERSION = "adversarial-agent-core-v1"


def release_distance(rate, dt, slew):
    """Distance consumed by a nonnegative rate, including the current step.

    Later rates decrease by at most slew * dt until they reach zero.
    This is a discrete integrator constraint, not a continuous-time guarantee.
    """
    if not all(math.isfinite(x) for x in (rate, dt, slew)) or dt <= 0 or slew <= 0:
        raise ValueError("Finite rate and positive dt/slew required")
    if rate <= 0:
        return 0.0
    delta = slew * dt
    n = math.ceil(rate / delta)
    return dt * (n * rate - delta * n * (n - 1) / 2)


def project_bounded_rate(request, value, previous_rate, *, lower, upper,
                         rate_limit, slew_limit, dt):
    """Project rate for x_next=x+rate*dt, with reserve to release at BOTH bounds.

    Examples: deficit/rate/curvature, or vehicle scalar speed/acceleration/jerk.
    Infeasible external states raise instead of silently overriding slew limits.
    This function operates in float64; float32 vehicle commands require auditing.
    """
    values = (request, value, previous_rate, lower, upper, rate_limit, slew_limit, dt)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite bounded integrator input")
    if lower >= upper or rate_limit <= 0 or slew_limit <= 0 or dt <= 0:
        raise ValueError("Invalid bounded integrator limits")
    if not lower <= value <= upper or abs(previous_rate) > rate_limit + 1e-12:
        raise ValueError("State outside bounded integrator domain")

    def feasible_magnitude(distance):
        lo, hi = 0.0, rate_limit
        if release_distance(hi, dt, slew_limit) <= distance:
            return hi
        for _ in range(64):
            mid = (lo + hi) / 2
            if release_distance(mid, dt, slew_limit) <= distance:
                lo = mid
            else:
                hi = mid
        return lo

    lo = max(-rate_limit, previous_rate - slew_limit * dt,
             -feasible_magnitude(value - lower))
    hi = min(rate_limit, previous_rate + slew_limit * dt,
             feasible_magnitude(upper - value))
    if lo > hi + 1e-12:
        raise ValueError("No joint position/rate/slew feasible command")
    # At a numerically shared boundary retain the position-feasible endpoint;
    # the 1e-12 tolerance is numerical, not permission for physical overrides.
    applied = min(hi, max(lo, request))
    return dict(rate=applied, next_value=value + applied * dt,
                lower_rate=lo, upper_rate=hi,
                intervened=abs(applied - request) > 1e-12)


@dataclass(frozen=True)
class PulseState:
    phase: str = "armed"
    deficit: float = 0.0
    rate: float = 0.0
    goal: float = 0.0


class OneShotDeficitPulse:
    """Four decisions: hold/wait, deepen to 1, deepen to 2, release.

    A release is irreversible. Deepening is monotone until release. Actual
    reference deficit/rate remain continuous under the bounded integrator.
    With the default rate/slew settings, release at 8 s leaves recovery time in
    the 12 s prototype episode. Other settings require a new recovery check.
    Hold means retain the existing goal, not freeze the current deficit.
    """
    def __init__(self, dt=0.1, rate_limit=1.0, slew_limit=1.0, release_at=8.0):
        if not all(math.isfinite(x) and x > 0 for x in (dt, rate_limit, slew_limit, release_at)):
            raise ValueError("Positive finite pulse parameters required")
        self.dt, self.rate_limit, self.slew_limit = dt, rate_limit, slew_limit
        self.release_at = release_at
        self.state = PulseState()
        self.ticks = 0

    def request(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in range(4):
            raise ValueError("Action must be an integer in [0, 3]")
        action = int(action)
        s = self.state
        if s.phase in ("release", "done"):
            return False
        if action == 3:
            self.state = PulseState("release" if s.phase == "active" else "done",
                                    s.deficit, s.rate, 0.0)
        elif action in (1, 2):
            self.state = PulseState("active", s.deficit, s.rate, max(s.goal, float(action)))
        return self.state != s

    def advance(self):
        s = self.state
        if self.ticks * self.dt >= self.release_at - 1e-12:
            self.request(3)
            s = self.state
        if s.phase in ("armed", "done"):
            self.ticks += 1
            return self.state
        # During active deepening the current goal is itself a no-overshoot
        # bound. During release retain the outer bound while positive rate
        # is smoothly removed; deficit may briefly increase after release.
        upper = s.goal if s.phase == "active" else 2.0
        update = project_bounded_rate((s.goal - s.deficit) / self.dt,
            s.deficit, s.rate, lower=0.0, upper=upper,
            rate_limit=self.rate_limit, slew_limit=self.slew_limit, dt=self.dt)
        deficit, rate = update["next_value"], update["rate"]
        if abs(deficit) < 1e-14:
            deficit = 0.0
        phase = "done" if s.phase == "release" and deficit == 0.0 and abs(rate) < 1e-12 else s.phase
        self.state = PulseState(phase, deficit, rate, s.goal)
        self.ticks += 1
        return self.state


def ttc_risk(ttc, dense_horizon_s=3.0):
    """Current 2D predicted-contact score, NOT a physical safety classification.

    None must mean a successful geometry calculation predicts no contact.
    Calculation failures must raise upstream, never become None.
    """
    if not math.isfinite(dense_horizon_s) or dense_horizon_s <= 0:
        raise ValueError("Positive finite dense horizon required")
    if ttc is None:
        return 0.0
    if not math.isfinite(ttc) or ttc < 0:
        raise ValueError("TTC must be finite nonnegative or explicit None")
    return max(0.0, min(1.0, 1.0 - ttc / dense_horizon_s))


class RiskAccumulator:
    """Undiscounted, finite-horizon maximum-risk increments.

    The caller may select intended_contact ONLY after contact attribution and
    maneuver-validity checks. An environment failure is an exception, not an
    invalid-maneuver training sample. gamma=1 and zero terminal bootstrap are
    required. Reward clipping/time-varying scaling must be disabled. The
    simulator's old headway reward must not be added. maximum is the shaping
    accumulator (preterminal for contact/invalid), not an audited outcome metric.
    """
    def __init__(self, initial_ttc, dense_horizon_s=3.0):
        self.dense_horizon_s = dense_horizon_s
        self.initial = self.maximum = ttc_risk(initial_ttc, dense_horizon_s)
        self.total = 0.0
        self.closed = False

    def advance(self, ttc, outcome="ongoing"):
        if self.closed:
            raise RuntimeError("Episode already ended")
        if outcome not in ("ongoing", "horizon", "intended_contact", "maneuver_invalid"):
            raise ValueError("Unknown outcome; execution errors are not rewards")
        q = ttc_risk(ttc, self.dense_horizon_s)
        old = self.maximum
        if outcome == "maneuver_invalid":
            reward = -1.0 - old
        elif outcome == "intended_contact":
            reward = 2.0 - old
        else:
            self.maximum = max(old, q)
            reward = self.maximum - old
        self.total += reward
        self.closed = outcome != "ongoing"
        return reward
