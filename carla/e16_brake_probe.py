"""E16 diagnostic (post hoc): when and how hard does CARLA's C brake on the same trajectory for each vehicle model? It wraps ShimTarget.step and records per tick
(t, C requested acceleration, achieved acceleration, speed, C phase). The vehicle model is chosen with the FAIRTEST_CARLA_BP/BRAKE/PI environment variables (one per process).

  python e16_brake_probe.py OUT.json KEYS.json --port 2030
"""
import argparse
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / 'code'))
from fairtest import carla_replay as R  # noqa: E402

LOG = []
_step = R.ShimTarget.step


def _logged(self, t, gen_state, sut_state, dims, vehicle, carla):
    a = _step(self, t, gen_state, sut_state, dims, vehicle, carla)
    rec = getattr(self.ctrl, 'perceived_at', None)
    LOG.append(dict(t=round(t, 1), req=round(float(self.prev_a), 3), meas=round(float(self._measured), 3), v=round(float(sut_state[2]), 3),
                    perceived=rec is not None, emergency=bool(getattr(self.ctrl, 'emergency', False)), braking=bool(self.braking)))
    return a


R.ShimTarget.step = _logged


def main(a):
    from fairtest.runner import _STATE, _init
    _init()
    keys = set(json.load(open(a.keys)))
    rows = [json.loads(l) for l in open(W / 'runs/E1_trajectories.jsonl', encoding='utf-8')]
    rows = [r for r in rows if f"{r['source']}|{r['case_id']}|{r['sut']}" in keys]
    world = R.connect(a.port)
    road = R.Road(world)
    bp = world.get_blueprint_library().find(R.BLUEPRINT)
    out = {}
    try:
        for r in rows:
            LOG.clear()
            res = R.run(world, road, bp, r, 'cc', _STATE['base'])
            out[f"{r['source']}|{r['case_id']}|{r['sut']}"] = dict(result=res, log=list(LOG))
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
    json.dump(dict(blueprint=R.BLUEPRINT, brake=[R.BRAKE_SCALE, R.BRAKE_A0, R.BRAKE_K], runs=out), open(a.out, 'w'))
    print(json.dumps(dict(done=len(out))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('keys')
    ap.add_argument('--port', type=int, default=2030)
    main(ap.parse_args())
