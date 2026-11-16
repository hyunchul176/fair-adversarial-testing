"""Harness fix (H2) diagnostic: how closely does the CARLA actuation follow the requests? Varying the cruise time (T), the speed and the actuation mode, it issues
the R157 C's emergency request sequence (cruise -> -0.4 for 0.75 s -> to -7.59 with jerk 12.65 -> hold). Speed and gear are recorded per tick.
  V0: current harness (requests below -0.3 brake, feed-forward, PI off)
  V1: down to -1.0 exactly in constant-velocity mode, stronger requests feed-forward + PI (0.5, 0.5, measured acceleration computed correctly)
  V2: V1 + neutral while braking (manual gear, gear 0)
  V3: V1's constant-velocity mode + braking as feed-forward with a disturbance observer (measured minus expected over the last tick, low-pass 0.5)
  V4: V1 + PI(0.2, 3.0)
  V5: as V3, but the disturbance estimate is applied directly (no low-pass)
The vehicle model is chosen with the FAIRTEST_CARLA_BP/BRAKE environment variables.

  python h2_probe.py OUT.json --port 2030
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / 'code'))
from fairtest import carla_replay as R  # noqa: E402

LOG = []
_init = R.ShimTarget.__init__


def script(t, t_react):
    if t < t_react - 1e-9:
        return 0.
    if t < t_react + .75 - 1e-9:
        return -.4
    k = int(round((t - t_react - .75) / .1)) + 1
    return max(-7.59, -.4 - 1.265 * k)


def _shim_init(self, kind, cruise, base):
    if not kind.startswith('h2probe'):
        return _init(self, kind, cruise, base)
    _, self.variant, t_react = kind.split(':')
    self.t_react = float(t_react)
    self.kind, self.cruise, self.prev_a, self.integ = kind, cruise, 0., 0.
    self.v_cmd, self.braking, self.hw, self._v_prev = None, False, False, None
    self.b_prev, self.d_hat = 0., 0.
    self.target_id = 'h2probe'


def _probe_step(self, t, gen_state, sut_state, dims, vehicle, carla):
    if not self.kind.startswith('h2probe'):
        return _step(self, t, gen_state, sut_state, dims, vehicle, carla)
    x, y, vx, vy, heading = sut_state
    req = script(t, self.t_react)
    meas = 0. if self._v_prev is None else (vx - self._v_prev) / R.DT
    self._v_prev = vx
    steer = -heading + math.atan2(.8 * (self._w - y), max(vx, 1.))
    ctl = carla.VehicleControl(steer=float(np.clip(steer / R.STEER_MAX_RAD, -1., 1.)))
    thr = -.3 if self.variant == 'V0' else -1.
    if req > thr:
        if self.braking:
            self.v_cmd, self.braking = vx, False
        self.v_cmd = max(0., self.v_cmd + req * R.DT)
        vehicle.enable_constant_velocity(carla.Vector3D(self.v_cmd, 0., 0.))
        self.integ = 0.
    elif self.variant in ('V3', 'V5'):
        # disturbance observer: estimate the difference between the measured acceleration of the last tick and the brake map's expected acceleration (residual drive force etc.) and subtract it
        if not self.braking:
            self.d_hat = 0.
        else:
            d = meas + R.BRAKE_A0 + R.BRAKE_K * self.b_prev if self.b_prev > 0. else meas - self.prev_a
            g = .5 if self.variant == 'V3' else 1.
            self.d_hat = (1. - g) * self.d_hat + g * d
        self.braking = True
        vehicle.disable_constant_velocity()
        b = (-req - R.BRAKE_A0 + self.d_hat) / R.BRAKE_K
        ctl.brake = self.b_prev = float(np.clip(b, 0., 1.))
    else:
        self.braking = True
        vehicle.disable_constant_velocity()
        kp, ki = {'V0': (0., 0.), 'V1': (.5, .5), 'V2': (.5, .5), 'V4': (.2, 3.)}[self.variant]
        err = req - meas
        self.integ = float(np.clip(self.integ + err * R.DT, -3., 3.))
        decel = -req - kp * err - ki * self.integ
        ctl.brake = float(np.clip((decel - R.BRAKE_A0) / R.BRAKE_K, 0., 1.))
        if self.variant == 'V2':
            ctl.manual_gear_shift, ctl.gear = True, 0
    vehicle.apply_control(ctl)
    self.prev_a = req
    LOG.append(dict(t=round(t, 1), req=round(req, 3), v=round(float(vx), 4), gear=int(vehicle.get_control().gear), braking=self.braking))
    return req


_step = R.ShimTarget.step
R.ShimTarget.__init__, R.ShimTarget.step = _shim_init, _probe_step


def main(a):
    from fairtest.runner import _STATE, _init as init
    init()
    world = R.connect(a.port)
    road = R.Road(world)
    bp = world.get_blueprint_library().find(R.BLUEPRINT)
    out = {}
    try:
        for v0 in (20., 32.):
            for t_react in (2., 6., 10.):
                for variant in a.variants.split(','):
                    tr = [dict(t=round(k * .1, 1), x=400. + v0 * k * .1, y=0., vx=v0, vy=0., heading=0.) for k in range(R.HORIZON + 1)]
                    row = dict(sut0=dict(x=0., y=4., vx=v0, vy=0.), cruise=v0, trajectory=tr)
                    LOG.clear()
                    R.run(world, road, bp, row, f'h2probe:{variant}:{t_react}', _STATE['base'])
                    out[f'{v0}|{t_react}|{variant}'] = [x for x in LOG if x['t'] <= t_react + 4.]
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
    json.dump(dict(blueprint=R.BLUEPRINT, brake=[R.BRAKE_SCALE, R.BRAKE_A0, R.BRAKE_K], runs=out), open(a.out, 'w'))
    print(json.dumps(dict(done=len(out))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--port', type=int, default=2030)
    ap.add_argument('--variants', default='V0,V1,V2')
    main(ap.parse_args())
