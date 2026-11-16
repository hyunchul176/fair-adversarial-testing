"""B54 post-hoc sensitivity: count v2j policy failures whose closed-loop episode had a shield-decision stall as NOT
detected (no refinement substitute), and recompute the confirmed detection rates and the paired difference.

Same definitions as fairtest.aw_analysis (detected / confirmed / paired case bootstrap). The stall is the recorded
per-episode maximum lateness `late_max` (true lateness, bridge fix 5b4cd9b), so this applies to runs made with that
bridge (the B54 MiTra post-hoc run); the development run's counter hid stalls and was estimated from ego motion instead.

  python -m fairtest.aw_stall_sensitivity OUT_DIR --cases 172 [--threshold 0.2] [--variants aw_default,aw_r157]
The paired difference is first variant minus second (B55: --variants aw_r157n,aw_r157nc --cases 62).
"""
import argparse
import json

import numpy as np

from fairtest.aw_analysis import CONFIRM, _load, boot_ci

SEEDS = (1, 2, 3)


def main(a):
    pol, ref, rep = _load(a.out, 'policy'), _load(a.out, 'refine'), _load(a.out, 'repro')
    P = {(r['variant'], r['case_id'], r['attacker'], r['seed']): r['result'] for r in pol}
    R = {(r['variant'], r['case_id']): r['result'] for r in ref}
    X = {}
    for r in rep:
        runs = r['result'].get('runs') or []
        X[(r['variant'], r['case_id'], r['source'])] = sum(bool(x['contact']) for x in runs)
    ids = sorted({c for (v, c, at, s) in P})
    assert len(ids) == a.cases, (len(ids), a.cases)
    variants = tuple(a.variants.split(','))
    out = dict(threshold_s=a.threshold, cases=len(ids), variants={})
    per = {}
    for v in variants:
        base, sens, stalled_eps, stalled_fail = {}, {}, 0, []
        for c in ids:
            rf = R.get((v, c))
            for s in SEEDS:
                p = P.get((v, c, 'v2j', s)) or {}
                late = p.get('late_max')
                stall = late is not None and late > a.threshold
                stalled_eps += bool(stall)
                src = None
                if p.get('reportable'):
                    src = f'policy_s{s}'
                elif rf is not None and rf.get('found'):
                    src = 'refine'
                ok = bool(src and X.get((v, c, src), 0) >= CONFIRM)
                base[(c, s)] = ok
                if src and src.startswith('policy') and stall:
                    stalled_fail.append(ok)
                    sens[(c, s)] = False
                else:
                    sens[(c, s)] = ok
        per[v] = (base, sens)
        rate = lambda d: round(float(np.mean([np.mean([d[(c, s)] for c in ids]) for s in SEEDS])), 4)
        case_mean = lambda d: [np.mean([d[(c, s)] for s in SEEDS]) for c in ids]
        out['variants'][v] = dict(
            confirmed=rate(base), confirmed_ci=boot_ci(case_mean(base)),
            confirmed_without_stalled_policy=rate(sens), confirmed_without_stalled_policy_ci=boot_ci(case_mean(sens)),
            stalled_policy_episodes=stalled_eps, v2j_policy_episodes=len(ids) * len(SEEDS),
            stalled_reported_failures=len(stalled_fail), stalled_failures_confirmed=int(sum(stalled_fail)))
    for key, idx in (('difference', 0), ('difference_without_stalled_policy', 1)):
        v1, v2 = variants[:2]
        d = [np.mean([per[v1][idx][(c, s)] for s in SEEDS]) - np.mean([per[v2][idx][(c, s)] for s in SEEDS]) for c in ids]
        out[key] = dict(pair=f'{v1} - {v2}', mean=round(float(np.mean(d)), 4), ci=boot_ci(d))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cases', type=int, required=True)
    ap.add_argument('--threshold', type=float, default=.2)
    ap.add_argument('--variants', default='aw_default,aw_r157', help='comma-separated; the difference is first - second')
    main(ap.parse_args())
