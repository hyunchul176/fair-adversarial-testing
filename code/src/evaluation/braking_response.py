"""Bounded leader pulse and an explicitly synthetic IDM command-delay probe."""
from __future__ import annotations

import math
from highway_env.vehicle.behavior import IDMVehicle


def braking_pulse(time_s, magnitude, start_s=2.0, hold_s=1.0, jerk_mps3=2.0):
    """Trapezoidal acceleration, with magnitude controlling duration and impulse.

    Equal ramps at a fixed jerk surround a fixed hold. This is not an equal
    impulse intervention: both peak acceleration and speed loss change.
    """
    if not all(math.isfinite(x) for x in (time_s, magnitude, start_s, hold_s, jerk_mps3)):
        raise ValueError('Non-finite pulse parameter')
    if magnitude < 0 or start_s < 0 or hold_s < 0 or jerk_mps3 <= 0:
        raise ValueError('Invalid pulse parameter')
    elapsed = time_s - start_s
    ramp = magnitude / jerk_mps3
    if magnitude == 0 or elapsed <= 0 or elapsed >= 2 * ramp + hold_s:
        return 0.0
    return -min(magnitude, jerk_mps3 * elapsed,
                jerk_mps3 * (2 * ramp + hold_s - elapsed))


class DelayedIDMVehicle(IDMVehicle):
    """Delay only the IDM acceleration command, preserving its initial value.

    This is an actuator-command delay stress model, not delayed perception.
    Highway-env invokes act exactly once per physics tick in this protocol.
    """
    def initialize_delay(self, ticks):
        if not isinstance(ticks, int) or ticks < 0:
            raise ValueError('Delay must be a nonnegative integer tick count')
        self.delay_ticks = ticks
        self.acceleration_queue = None
        self.act_calls = 0

    def act(self, action=None):
        super().act(action)
        candidate = float(self.action['acceleration'])
        if self.acceleration_queue is None:
            self.acceleration_queue = [candidate] * self.delay_ticks
        self.acceleration_queue.append(candidate)
        self.action['acceleration'] = self.acceleration_queue.pop(0)
        self.act_calls += 1


def response_window(episode, begin, end):
    rows = [r for r in episode['trace'] if begin <= r['tick'] < end]
    finite = [r['pair_ttc'] for r in rows if r['pair_ttc'] is not None]
    drac = [r['following_drac'] for r in rows if r['following_drac'] is not None]
    return {
        'ticks': [begin, end], 'planned_steps': end - begin, 'observed_steps': len(rows),
        'ttc_defined_steps': len(finite), 'min_ttc': min(finite) if finite else None,
        'ttc_below_3_steps': sum(t < 3.0 for t in finite),
        'ttc_below_1_5_steps': sum(t < 1.5 for t in finite),
        'drac_defined_steps': len(drac), 'max_drac': max(drac) if drac else None,
        'target_peak_braking': max((max(0., -r['acceleration'][1]) for r in rows), default=None),
        'target_min_speed': min((r['target']['vx'] for r in rows), default=None),
    }
