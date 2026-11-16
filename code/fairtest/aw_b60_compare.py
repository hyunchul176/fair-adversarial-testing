"""B60 (B60 design, available on request): the fair agent against the braking-competent
Autoware (AW-r157nc) with perception noise or missed-detection bursts; paired by case within the B60 run.

  rate(v)        confirmed detection of the v2j policy stage (no refinement), mean over seeds (aw_analysis.py)
  N1             attacker-caused share of v2j policy contacts <= 5% in each variant
  N2             missed detections are found: rate(dp) - rate(l0) > 0, paired case bootstrap (10,000, rng 0) 95% lower bound > 0
  N3 (descr.)    measurement noise: rate(nz) - rate(l0) with its interval (no direction fixed in advance)
  also: failure mode, reproduction, fidelity, the share of missed ticks

  python -m fairtest.aw_b60_compare B60_SUMMARY.json OUT.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

V = ('aw_r157nc_l0', 'aw_r157nc_nz', 'aw_r157nc_dp')


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
    for a, b in ((V[2], V[0]), (V[1], V[0])):
        d = [pc[a][c] - pc[b][c] for c in ids]
        diff[f'{a} - {b}'] = dict(mean=round(float(np.mean(d)), 4), ci=boot(d))
    S = s['variants']
    R = dict(cases=len(ids), rate={v: round(r, 4) for v, r in rate.items()}, difference=diff)
    R['N1'] = {v: dict(S[v]['A1'], supported=S[v]['A1']['share'] is None or S[v]['A1']['share'] <= .05) for v in V}
    R['N2'] = dict(ci=diff[f'{V[2]} - {V[0]}']['ci'], supported=diff[f'{V[2]} - {V[0]}']['ci'][0] > 0)
    R['N3'] = diff[f'{V[1]} - {V[0]}']
    R['failure_mode'] = {v: {k: S[v]['failure_mode'].get(k) for k in ('n', 'brake_delay_median', 'min_acc_median', 'cc_brake_delay_median',
                                                                     'cc_min_acc_median', 'mrm_share')} for v in V}
    R['reported'] = {v: S[v]['detected'] for v in V}
    R['reproduction'] = {v: {k: S[v]['A5'][k] for k in ('failures', 'reruns', 'colliding', 'rate')} for v in V}
    R['errors'] = {v: S[v]['errors'] for v in V}
    R['fidelity'] = {v: S[v]['fidelity'] for v in V}
    json.dump(R, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
