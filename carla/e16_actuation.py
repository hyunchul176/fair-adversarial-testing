"""E16c preparation (addendum 11): measure the actuation of C alone in CARLA. Independently of the situation, it replays the request sequence that the R157 C issues in an emergency.
  t < 2 s: 0 (cruise) -> 2.0–2.75 s: -0.4 (reaction phase) -> from 2.75 s, jerk 12.65 m/s^3 down to -7.59 -> hold (until 5.5 s)
The vehicle model is chosen with the FAIRTEST_CARLA_BP/BRAKE/PI environment variables. The adversary vehicle is placed 300 m ahead in its own lane, so it does not cut in.

  python e16_actuation.py OUT.json --speeds 15,20,25,30,35 --port 2030
"""
import argparse
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / 'code'))
from fairtest import carla_replay as R  # noqa: E402

LOG = []
_init, _step = R.ShimTarget.__init__, R.ShimTarget.step


def script(t):
    if t < 2. - 1e-9:
        return 0.
    if t < 2.75 - 1e-9:
        return -.4
    k = int(round((t - 2.75) / .1)) + 1
    return max(-7.59, -.4 - 1.265 * k)


def _shim_init(self, kind, cruise, base):
    if not kind.startswith('script'):
        return _init(self, kind, cruise, base)
    self.kind, self.cruise, self.prev_a, self.integ = kind, cruise, 0., 0.
    self.v_cmd, self.braking, self.hw = None, False, False
    self.target_id = 'script'
    self.ctrl = lambda shim, tid, model, cruise, prev, act: (script(shim.cc_clock), None)


def _logged(self, t, gen_state, sut_state, dims, vehicle, carla):
    a = _step(self, t, gen_state, sut_state, dims, vehicle, carla)
    if self.kind.startswith('script'):
        LOG.append(dict(t=round(t, 1), req=round(float(self.prev_a), 3), v=round(float(sut_state[2]), 4), braking=bool(self.braking)))
    return a


R.ShimTarget.__init__, R.ShimTarget.step = _shim_init, _logged


def main(a):
    from fairtest.runner import _STATE, _init as init
    init()
    world = R.connect(a.port)
    road = R.Road(world)
    bp = world.get_blueprint_library().find(R.BLUEPRINT)
    out = {}
    try:
        for v0 in (float(x) for x in a.speeds.split(',')):
            tr = [dict(t=round(k * .1, 1), x=300. + v0 * k * .1, y=0., vx=v0, vy=0., heading=0.) for k in range(R.HORIZON + 1)]
            row = dict(sut0=dict(x=0., y=4., vx=v0, vy=0.), cruise=v0, trajectory=tr)
            LOG.clear()
            res = R.run(world, road, bp, row, 'script', _STATE['base'])
            out[str(v0)] = dict(result=res, log=[x for x in LOG if x['t'] <= 6.])
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
    json.dump(dict(blueprint=R.BLUEPRINT, brake=[R.BRAKE_SCALE, R.BRAKE_A0, R.BRAKE_K], runs=out), open(a.out, 'w'))
    print(json.dumps(dict(done=len(out))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--speeds', default='15,20,25,30,35')
    ap.add_argument('--port', type=int, default=2030)
    main(ap.parse_args())
