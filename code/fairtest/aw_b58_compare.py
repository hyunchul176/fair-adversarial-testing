"""B58 (B58 design, available on request): does the fair agent detect a perception latency in
an otherwise competent Autoware (AW-r157nc), and does its guarantee hold? Paired by case within the B58 run.

  rate(v)        confirmed detection of the v2j policy stage (no refinement in B58), mean over seeds (aw_analysis.py)
  L1             attacker-caused share of v2j policy contacts <= 5% in each variant
  L2             rate(l600) - rate(l0) > 0: paired case bootstrap (10,000, rng 0), 95% lower bound > 0
  L3             dose-response: rate(l0) < rate(l300) < rate(l600) (point estimates)
  L4 (descr.)    braking onset (reaching -1 m/s^2 after the cut-in is visible) and peak deceleration of the confirmed
                 failures per variant
  check (descr.) the l0 policy-stage reported rate next to B55's AW-r157nc policy stage (6 of 186 episodes)

  python -m fairtest.aw_b58_compare B58_SUMMARY.json OUT.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

V = ('aw_r157nc_l0', 'aw_r157nc_l300', 'aw_r157nc_l600')


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, size=(n, len(x)), replace=True).mean(1)
    return [round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]


def main(summary, out):
    s = json.load(open(summary, encoding='utf-8'))
    pc = {v: s['per_case_confirmed'][v] for v in V}
    ids = sorted(pc[V[0]], key=int)
    assert all(sorted(pc[v], key=int) == ids for v in V), 'the variants must cover the same cases'
    rate = {v: float(np.mean([pc[v][c] for c in ids])) for v in V}
    diff = {}
    for a, b in ((V[2], V[0]), (V[1], V[0]), (V[2], V[1])):
        d = [pc[a][c] - pc[b][c] for c in ids]
        diff[f'{a} - {b}'] = dict(mean=round(float(np.mean(d)), 4), ci=boot(d))
    S = s['variants']
    R = dict(cases=len(ids), rate={v: round(r, 4) for v, r in rate.items()}, difference=diff)
    R['L1'] = {v: dict(S[v]['A1'], supported=S[v]['A1']['share'] is None or S[v]['A1']['share'] <= .05) for v in V}
    R['L2'] = dict(ci=diff[f'{V[2]} - {V[0]}']['ci'], supported=diff[f'{V[2]} - {V[0]}']['ci'][0] > 0)
    R['L3'] = dict(rates=[round(rate[v], 4) for v in V], supported=rate[V[0]] < rate[V[1]] < rate[V[2]])
    R['L4'] = {v: {k: S[v]['failure_mode'].get(k) for k in ('n', 'brake_delay_median', 'min_acc_median', 'cc_brake_delay_median',
                                                           'cc_min_acc_median', 'mrm_share')} for v in V}
    R['reported'] = {v: S[v]['detected'] for v in V}
    R['reproduction'] = {v: {k: S[v]['A5'][k] for k in ('failures', 'reruns', 'colliding', 'rate')} for v in V}
    R['errors'] = {v: S[v]['errors'] for v in V}
    R['fidelity'] = {v: S[v]['fidelity'] for v in V}
    json.dump(R, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
