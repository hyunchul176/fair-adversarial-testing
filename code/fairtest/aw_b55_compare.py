"""B55 (B55 design, available on request): does the B54 failure mode come from Autoware's
planning comfort limits? Paired by case against B54's AW-r157 (same frozen harness, cases and seeds).

  rate(v)        confirmed detection, mean over seeds (aw_analysis.py definition)
  C1             rate(AW-r157n) < rate(AW-r157): paired case bootstrap (10,000, rng 0), 95% upper bound of the difference < 0
  C2             the comfort limits are the main cause: rate(AW-r157n) <= 0.5 x rate(AW-r157)
  C3 (descr.)    rate(AW-r157nc) - rate(AW-r157n), the controller limits on top
  C4             the guarantee still holds: attacker-caused share of v2j policy contacts <= 5% in AW-r157n and AW-r157nc
Failure mode (descriptive): peak deceleration before contact and braking onset of the confirmed failures.

  python -m fairtest.aw_b55_compare B54_SUMMARY.json B55_SUMMARY.json OUT.json
"""
from __future__ import annotations

import json
import sys

import numpy as np


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, size=(n, len(x)), replace=True).mean(1)
    return [round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4)]


def main(b54, b55, out):
    s4, s5 = json.load(open(b54, encoding='utf-8')), json.load(open(b55, encoding='utf-8'))
    pc = dict(aw_r157=s4['per_case_confirmed']['aw_r157'], **{v: s5['per_case_confirmed'][v] for v in ('aw_r157n', 'aw_r157nc')})
    ids = sorted(pc['aw_r157'], key=int)
    assert all(sorted(pc[v], key=int) == ids for v in pc), 'the three variants must cover the same cases'
    rate = {v: float(np.mean([pc[v][c] for c in ids])) for v in pc}
    diff = {}
    for a, b in (('aw_r157n', 'aw_r157'), ('aw_r157nc', 'aw_r157'), ('aw_r157nc', 'aw_r157n')):
        d = [pc[a][c] - pc[b][c] for c in ids]
        diff[f'{a} - {b}'] = dict(mean=round(float(np.mean(d)), 4), ci=boot(d))
    V = dict(s4['variants'], **s5['variants'])
    R = dict(cases=len(ids), rate={v: round(r, 4) for v, r in rate.items()}, difference=diff)
    R['C1'] = dict(ci=diff['aw_r157n - aw_r157']['ci'], supported=diff['aw_r157n - aw_r157']['ci'][1] < 0)
    R['C2'] = dict(ratio=round(rate['aw_r157n'] / rate['aw_r157'], 4) if rate['aw_r157'] else None,
                   supported=rate['aw_r157n'] <= .5 * rate['aw_r157'])
    R['C3'] = diff['aw_r157nc - aw_r157n']
    R['C4'] = {v: dict(V[v]['A1'], supported=V[v]['A1']['share'] is None or V[v]['A1']['share'] <= .05) for v in ('aw_r157n', 'aw_r157nc')}
    R['failure_mode'] = {v: {k: V[v]['failure_mode'].get(k) for k in ('n', 'brake_delay_median', 'min_acc_median', 'cc_min_acc_median',
                                                                     'mrm_share')} for v in ('aw_r157', 'aw_r157n', 'aw_r157nc')}
    R['reproduction'] = {v: {k: V[v]['A5'][k] for k in ('failures', 'reruns', 'colliding', 'rate')} for v in ('aw_r157', 'aw_r157n', 'aw_r157nc')}
    json.dump(R, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
