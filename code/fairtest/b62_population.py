"""B62 training SUT population (B62 design, available on request).

The v2j policy was trained against the 7-SUT panel (b17_rl.PANEL), rule-based longitudinal followers only. B62 trains the
same policy (same mode, warm start, steps, envs) against the panel plus a population of drivers with the R157 CC structure
(cc_target.CCDriver via env target cc_r157@h<k>) in which each factor is weakened independently with probability 1/2:
  perceive_m  lateral displacement at which the cut-in is perceived   C 1.095 m   weakened U[1.2, 2.5]
  reaction_s  perception -> emergency-braking decision                  C 0.75 s    weakened U[0.9, 2.0]
  decel_mps2  braking limit (jerk = decel / buildup_s, buildup 0.6 s)   C 7.59      weakened U[4.0, 7.0]
  ttc_s       emergency TTC threshold                                   C 2.0 s     fixed
The fairness label, the shield and the judge stay the CC (C) itself. Held out from training (evaluation only): BC human,
the learned SUTs (drl_sut family), the ADS family, lateral RSS, Autoware, CARLA agents.

train_suts = the panel repeated PANEL_REPEAT times + one entry per driver (uniform sampling per episode over the list,
so about half of the episodes use the panel).

  python -m fairtest.b62_population fairtest/b62_train_population.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from fairtest.b17_rl import PANEL

SEED, N, PANEL_REPEAT, P_WEAK = 62, 60, 8, .5
C = dict(perceive_m=1.095, reaction_s=.75, ttc_s=2., decel_mps2=7.59, buildup_s=.6)
WEAK = dict(perceive_m=(1.2, 2.5), reaction_s=(.9, 2.), decel_mps2=(4., 7.))


def build():
    rng = np.random.default_rng(SEED)
    drivers = []
    for k in range(N):
        d = dict(k=k, pop='b62_train', **C)
        weakened = []
        for name, (lo, hi) in WEAK.items():
            u, x = rng.random(), rng.uniform(lo, hi)  # both drawn for every driver: a fixed stream
            if u < P_WEAK:
                d[name] = round(float(x), 4)
                weakened.append(name)
        d['jerk_mps3'] = round(d['decel_mps2'] / d['buildup_s'], 4)
        d['weakened'] = weakened
        drivers.append(d)
    train = [[s, h] for s, h in PANEL] * PANEL_REPEAT + [[f'cc_r157@h{d["k"]}', 0.] for d in drivers]
    spec = dict(source='B62 training population (b62_population.py)', seed=SEED, n=N, panel_repeat=PANEL_REPEAT, p_weak=P_WEAK,
                reference=C, weakened_ranges=WEAK)
    return dict(spec=spec, drivers=drivers, train_suts=train)


if __name__ == '__main__':
    out = build()
    with open(sys.argv[1], 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1)
        f.write('\n')
    w = [len(d['weakened']) for d in out['drivers']]
    print(json.dumps(dict(drivers=len(w), by_count={i: w.count(i) for i in range(4)}, train_suts=len(out['train_suts']))))
