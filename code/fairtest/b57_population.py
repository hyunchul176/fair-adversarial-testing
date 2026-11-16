"""B57: sample the human-like reference populations from the distribution spec fixed in the B57 design.

Each driver is the R157 CC model (perception -> reaction with -0.4 m/s^2 -> emergency braking when TTC <= threshold, else
the following law with 3 m/s^2 comfort) with its own
  perceive_m   lateral displacement of the cutting-in vehicle from its lane centre at which the driver perceives the risk
  reaction_s   time from perception to the emergency-braking decision
  ttc_s        TTC at or below which the driver brakes at its maximum
  decel_mps2   maximum (emergency) deceleration
  buildup_s    time to build the deceleration up to its maximum; jerk_mps3 = decel_mps2 / buildup_s (R157: 7.59 / 0.6)
Parameters are drawn independently (their correlations are not known); a truncated family is sampled by rejection.

spec: {"populations": {label: {"seed": int, "n": int, "params": {name: dist}}}, "brackets": {label: {"base": label,
       "n": int, "set": {name: value} | "scale": {name: [factor, lo, hi]}}}}
  dist = {"dist": "lognormal", "median": m, "sigma": s} | {"dist": "normal", "mean": m, "sd": s} |
         {"dist": "uniform", "lo": a, "hi": b} | {"dist": "fixed", "value": v}, with optional "lo"/"hi" truncation
A bracket population is the first n drivers of its base population with one parameter replaced (paired sensitivity).
The output lists every driver once, with its population label; the env target cc_r157@h<k> is driver k of that list.

  python -m fairtest.b57_population SPEC.json OUT.json
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

NAMES = ('perceive_m', 'reaction_s', 'ttc_s', 'decel_mps2', 'buildup_s')


def draw(rng, p):
    lo, hi = p.get('lo', -np.inf), p.get('hi', np.inf)
    for _ in range(10000):
        if p['dist'] == 'lognormal':
            x = float(rng.lognormal(math.log(p['median']), p['sigma']))
        elif p['dist'] == 'normal':
            x = float(rng.normal(p['mean'], p['sd']))
        elif p['dist'] == 'uniform':
            x = float(rng.uniform(p['lo'], p['hi']))
        elif p['dist'] == 'fixed':
            return float(p['value'])
        else:
            raise ValueError(p['dist'])
        if lo <= x <= hi:
            return x
    raise RuntimeError(f'truncation too tight: {p}')


def finish(d):
    d = {k: round(float(v), 4) for k, v in d.items()}
    d['jerk_mps3'] = round(d['decel_mps2'] / d['buildup_s'], 4)
    return d


def main(spec_path, out_path):
    spec = json.load(open(spec_path, encoding='utf-8'))
    pops = {}
    for label, p in spec['populations'].items():
        rng = np.random.default_rng(p['seed'])
        pops[label] = [finish({name: draw(rng, p['params'][name]) for name in NAMES}) for _ in range(p['n'])]
    for label, b in spec.get('brackets', {}).items():
        rows = []
        for d in pops[b['base']][:b['n']]:
            d = {k: v for k, v in d.items() if k in NAMES}
            for name, v in b.get('set', {}).items():
                d[name] = v
            for name, (factor, lo, hi) in b.get('scale', {}).items():
                d[name] = min(hi, max(lo, d[name] * factor))
            rows.append(finish(d))
        pops[label] = rows
    drivers = []
    for label, rows in pops.items():
        for d in rows:
            drivers.append(dict(k=len(drivers), pop=label, **d))
    summary = {label: {name: dict(zip(('p5', 'p25', 'median', 'p75', 'p95'),
                                      [round(float(np.quantile([d[name] for d in rows], q)), 3) for q in (.05, .25, .5, .75, .95)]))
                       for name in NAMES + ('jerk_mps3',)} for label, rows in pops.items()}
    json.dump(dict(spec=spec, summary=summary, drivers=drivers), open(out_path, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(dict(sizes={k: len(v) for k, v in pops.items()}, attentive=summary.get('attentive')), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
