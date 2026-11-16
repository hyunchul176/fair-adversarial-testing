"""B64 analysis (B64 design, available on request): the final agent v2s, v2s counting
only severe failures (closing >= 3 m/s at contact), and v2s-S (severity floor in the screened refinement, bo_warm_g4ss).

Per seed x, case c, SUT s (the b41_analysis / exid_analysis composition, k SUT queries):
  v2s       policy robust and realistic                    or  bo_warm_g4s first success q with 1 + q <= k
  v2s sev   the same policy failure with closing >= v_min  or  the bo_warm_g4s success with closing >= v_min
  v2s-S     the same policy failure with closing >= v_min  or  bo_warm_g4ss first success (closing >= v_min by its contract)
Closing speeds: policy contacts from b51_severity `closing` (closing_*.jsonl of B51, same policy episodes); bo_warm_g4s
successes from b51_severity `closing --method bo_warm_g4s` (B64).

  python -m fairtest.b64_analysis OUT.json --dir DIR --policy-closing CLOSING.jsonl --g4s B56.jsonl \
         --g4s-closing CLOSING_G4S.jsonl --g4ss B64.jsonl [--name mitra]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b41_analysis import K_FULL, boot_diff, load, rate
from fairtest.b51_severity import V_MIN
from fairtest.exid_analysis import COMPETENT8, load_refine

WEAK4 = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')


def closings(path, kind):
    out = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r.get('kind') == kind and 'error' not in r:
            out[(r['seed'], r['case_id'], r['sut']) if kind == 'policy' else (r['case_id'], r['sut'])] = r['closing']
    return out


def main(a):
    evals, realistic, _, pool, _ = load(Path(a.dir))
    cases = sorted(c['case_id'] for c in pool['cases'])
    seeds = sorted(evals['final'])
    first, recs, err, conf = load_refine([a.g4s, a.g4ss])
    pcl = closings(a.policy_closing, 'policy')
    gcl = closings(a.g4s_closing, 'bo_warm_g4s')
    suts = sorted({s for (_, s) in first['bo_warm_g4ss']})
    missing = dict(policy=0, g4s=0)

    def pol(x, c, s, sev):
        m = evals['final'][x].get((c, s), {})
        if not (m.get('robust') and realistic.get((x, c, s), False)):
            return False
        if not sev:
            return True
        v = pcl.get((x, c, s))
        if v is None:
            missing['policy'] += 1
            return False
        return v >= V_MIN

    def ref(c, s, variant, k):
        if k < 2:
            return False
        if variant == 'v2s-S':
            q = first['bo_warm_g4ss'].get((c, s))
            return q is not None and 1 + q <= k
        q = first['bo_warm_g4s'].get((c, s))
        if q is None or 1 + q > k:
            return False
        if variant == 'v2s':
            return True
        v = gcl.get((c, s))
        if v is None:
            missing['g4s'] += 1
            return False
        return v >= V_MIN

    def one(x, c, s, variant, k):
        if variant == 'v2s sev':
            # the failure v2s reports (policy if it succeeded, else the refinement), dropped if it closes below v_min:
            # v2s stops at a light policy failure (exid_extra S7, B63)
            if pol(x, c, s, False):
                return pol(x, c, s, True)
            return ref(c, s, variant, k)
        sev = variant == 'v2s-S'  # v2s-S: a light policy failure is no success, so the refinement (g4ss) runs
        return pol(x, c, s, sev) or ref(c, s, variant, k)

    def comp(variant, k):
        return {s: {c: float(np.mean([float(one(x, c, s, variant, k)) for x in seeds])) for c in cases} for s in suts}

    rep = dict(name=a.name, n_cases=len(cases), v_min=V_MIN, refine_errors=err, refine_conflicts=len(conf), variants={})
    for variant in ('v2s', 'v2s sev', 'v2s-S'):
        for k in (1, K_FULL):
            c = comp(variant, k)
            p1, ci = boot_diff(c['following_H0'], c['rss_sut_H0'])
            rep['variants'][f'{variant} k{k}'] = dict(
                suts={s: round(rate(c[s]), 4) for s in suts}, P1=round(p1, 4), P1_ci=[round(x, 4) for x in ci],
                weak4=round(float(np.mean([rate(c[s]) for s in WEAK4])), 4),
                competent8=round(float(np.mean([rate(c[s]) for s in COMPETENT8])), 4))
    # SS2: weak-4 severe detection, v2s-S - v2s sev (per-case mean over the four SUTs), case bootstrap
    sev37, s37 = comp('v2s sev', K_FULL), comp('v2s-S', K_FULL)
    w_sev = {c: float(np.mean([sev37[s][c] for s in WEAK4])) for c in cases}
    w_s = {c: float(np.mean([s37[s][c] for s in WEAK4])) for c in cases}
    d, ci = boot_diff(w_s, w_sev)
    rep['SS2'] = dict(diff=round(d, 4), ci=[round(x, 4) for x in ci], supported=bool(d >= 0))
    rep['SS1'] = dict(P1_v2sS=rep['variants'][f'v2s-S k{K_FULL}']['P1'], P1_v2s_sev=rep['variants'][f'v2s sev k{K_FULL}']['P1'],
                      supported=bool(rep['variants'][f'v2s-S k{K_FULL}']['P1'] >= .5))
    g4ss = recs['bo_warm_g4ss']
    cl = [v['best'].get('closing') for v in g4ss.values() if v.get('first_fair') is not None]
    rep['g4ss'] = dict(pairs=len(g4ss), successes=len(cl), closing_min=min(cl) if cl else None,
                       closing_median=float(np.median(cl)) if cl else None,
                       stops={k: sum(1 for v in g4ss.values() if v.get('stop') == k) for k in ('success', 'no_admissible_initial', None)})
    rep['missing_closing'] = missing
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for key, v in rep['variants'].items():
        print(f"{key:12s} P1 {v['P1'] * 100:5.1f} {[round(x * 100, 1) for x in v['P1_ci']]} weak4 {v['weak4'] * 100:5.1f} "
              f"competent8 {v['competent8'] * 100:4.1f} | " + ' '.join(f"{s.split('_H')[0][:12]} {r * 100:.1f}" for s, r in v['suts'].items()))
    print('SS1', rep['SS1'], '| SS2', rep['SS2'], '| g4ss', rep['g4ss'], '| missing', missing)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--dir', required=True)
    ap.add_argument('--policy-closing', required=True)
    ap.add_argument('--g4s', required=True)
    ap.add_argument('--g4s-closing', required=True)
    ap.add_argument('--g4ss', required=True)
    ap.add_argument('--name', default='')
    main(ap.parse_args())
