"""E7 debug: tick-by-tick comparison of one attack in highway-env and CARLA (SUT speed, requested and achieved
acceleration, bumper gap).

  python -m fairtest.carla_trace TRAJECTORIES.jsonl INDEX [--kind ladder:following:0] [--port 2020]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


def main(a):
    import carla
    from fairtest import carla_replay as CR
    from fairtest.b23_crossjudge import replay
    from fairtest.runner import _STATE, _init
    _init()
    row = [json.loads(l) for l in open(a.trajectories, encoding='utf-8')][a.index]
    case = _STATE['cases'][row['case_id']]
    ev = None
    for line in open(a.eval, encoding='utf-8'):
        r = json.loads(line)
        if r['case_id'] == row['case_id'] and r['sut'] == row['sut']:
            ev = r
    name, h = row['sut'].rsplit('_H', 1)
    hw = replay(case, ev['actions'], name, float(h))
    hw_tr = [(round(k * .1, 1), s['target']['vx'], s['target_control']['applied_mps2'],
              (s['generator']['x'] - s['target']['x']) - 5., s['generator']['y']) for k, s in enumerate(hw['trace'])]
    # CARLA with per-tick logging: wrap ShimTarget.step
    log = []
    orig = CR.ShimTarget.step

    def step(self, t, gen_state, sut_state, dims, vehicle, carla_mod):
        req = orig(self, t, gen_state, sut_state, dims, vehicle, carla_mod)
        log.append((round(t, 1), sut_state[2], req, gen_state[0] - sut_state[0] - (dims[0] + 4.79) / 2, gen_state[1]))
        return req
    CR.ShimTarget.step = step
    world = CR.connect(a.port)
    road = CR.Road(world)
    bp = world.get_blueprint_library().find(CR.BLUEPRINT)
    res = CR.run(world, road, bp, row, a.kind, _STATE['base'])
    s = world.get_settings()
    s.synchronous_mode = False
    world.apply_settings(s)
    print(json.dumps(dict(case=row['case_id'], hw_contact=bool(hw['pair_contact']), hw_ticks=len(hw_tr), carla=res)))
    print(' t    | hw: v_sut  a_req  gap   gen_y | carla: v_sut  a_req  gap   gen_y(m)')
    for k in range(0, min(len(hw_tr), len(log)), 3):
        h_, c_ = hw_tr[k], log[k]
        print(f'{h_[0]:4.1f} | {h_[1]:6.2f} {h_[2]:6.2f} {h_[3]:6.2f} {h_[4]:5.2f} | {c_[1]:6.2f} {c_[2]:6.2f} {c_[3]:6.2f} {c_[4]:5.2f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('trajectories')
    ap.add_argument('index', type=int)
    ap.add_argument('--eval', required=True)
    ap.add_argument('--kind', default='ladder:following:0')
    ap.add_argument('--port', type=int, default=2020)
    main(ap.parse_args())
