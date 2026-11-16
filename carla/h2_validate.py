"""Harness v2 (H2) validation: how closely does CARLA follow the requested accelerations of C and the SUT on real attack trajectories? It wraps ShimTarget.step
and records (request, speed) per tick, and measures the difference between the achieved acceleration (speed difference to the next tick) and the request. Vehicle model via FAIRTEST_CARLA_BP/BRAKE, actuation via FAIRTEST_CARLA_ACT.

  python h2_validate.py OUT.json KEYS.json --kinds cc,ladder:following:0 --port 2030
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
    LOG.append((round(t, 1), round(float(self.prev_a), 3), round(float(sut_state[2]), 4), bool(self.braking)))
    return a


R.ShimTarget.step = _logged


def stats(log, contact_t):
    """Request versus achieved (speed difference to the next tick): overall, constant-velocity phase, braking phase and emergency hold (request <= -7). Up to the collision only."""
    rows = []
    for (t, req, v, br), (_, _, v2, _) in zip(log, log[1:]):
        if contact_t is not None and t >= contact_t - .1:
            break
        rows.append((req, (v2 - v) / .1, br))
    def s(sel):
        if not sel:
            return None
        e = [a - r for r, a, _ in sel]
        return dict(n=len(e), mean=round(sum(e) / len(e), 3), rms=round((sum(x * x for x in e) / len(e)) ** .5, 3),
                    worst=round(max(e, key=abs), 3))
    return dict(all=s(rows), cv=s([x for x in rows if not x[2]]), brake=s([x for x in rows if x[2]]),
                plateau=s([x for x in rows if x[0] <= -7.]))


def main(a):
    from fairtest.runner import _STATE, _init
    _init()
    keys = set(json.load(open(a.keys)))
    rows = [json.loads(l) for l in open(a.trajectories, encoding='utf-8')]
    rows = [r for r in rows if f"{r['source']}|{r['case_id']}|{r['sut']}" in keys]
    world = R.connect(a.port)
    road = R.Road(world)
    bp = world.get_blueprint_library().find(R.BLUEPRINT)
    out = {}
    try:
        for r in rows:
            key = f"{r['source']}|{r['case_id']}|{r['sut']}"
            for kind in a.kinds.split(','):
                LOG.clear()
                res = R.run(world, road, bp, r, kind, _STATE['base'])
                out[f'{key}|{kind}'] = dict(result=res, stats=stats(LOG, res.get('t_contact')))
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)
    json.dump(dict(blueprint=R.BLUEPRINT, act=R.ACT, runs=out), open(a.out, 'w'))
    print(json.dumps(dict(done=len(out))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('keys')
    ap.add_argument('--trajectories', default=str(W / 'runs/E1_trajectories.jsonl'))
    ap.add_argument('--kinds', default='cc,ladder:following:0')
    ap.add_argument('--port', type=int, default=2030)
    main(ap.parse_args())
