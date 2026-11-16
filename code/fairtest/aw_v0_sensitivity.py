"""B55 post-hoc check: episodes whose Autoware ego started far from the case's initial speed.

Each episode aligns the ego at t = 0 after a pre-roll; `v0_error` is the ego speed at t = 0 minus the case's initial
speed. A start is 'slow' when |v0_error| > --max (default 0.5 m/s). The script counts slow starts per variant, phase and
instance, then recomputes the confirmed detection (definitions of fairtest.aw_analysis) (a) per seed, (b) on the seeds
without slow policy starts, and (c) with every (case, seed) whose policy episode started slow left out of that case's
seed mean, and gives the paired case bootstrap of the first variant minus the second for (b) and (c).

  python -m fairtest.aw_v0_sensitivity OUT_DIR --cases 62 --variants aw_r157nc,aw_r157n
"""
import argparse
import collections
import json

import numpy as np

from fairtest.aw_analysis import CONFIRM, _load, boot_ci

SEEDS = (1, 2, 3)


def main(a):
    variants = tuple(a.variants.split(','))
    pol, ref, rep = _load(a.out, 'policy'), _load(a.out, 'refine'), _load(a.out, 'repro')
    slow = collections.Counter()
    total = collections.Counter()
    for r in pol:
        if r['attacker'] != 'v2j':
            continue
        e = r['result'].get('v0_error')
        total[(r['variant'], 'policy', r['instance'])] += 1
        slow[(r['variant'], 'policy', r['instance'])] += e is not None and abs(e) > a.max
    for r in ref:
        for q in r['result'].get('history') or []:
            e = (q.get('aw') or {}).get('v0_error')
            total[(r['variant'], 'refine', r['instance'])] += 1
            slow[(r['variant'], 'refine', r['instance'])] += e is not None and abs(e) > a.max
    for r in rep:
        for x in r['result'].get('runs') or []:
            e = x.get('v0_error')
            total[(r['variant'], 'repro', r['instance'])] += 1
            slow[(r['variant'], 'repro', r['instance'])] += e is not None and abs(e) > a.max
    P = {(r['variant'], r['case_id'], r['seed']): r['result'] for r in pol if r['attacker'] == 'v2j'}
    R = {(r['variant'], r['case_id']): r['result'] for r in ref}
    X = {(r['variant'], r['case_id'], r['source']): sum(bool(x['contact']) for x in (r['result'].get('runs') or [])) for r in rep}
    ids = sorted({c for (v, c, s) in P})
    assert len(ids) == a.cases, (len(ids), a.cases)
    conf, bad = {}, {}
    for v in variants:
        for c in ids:
            for s in SEEDS:
                p = P.get((v, c, s)) or {}
                src = f'policy_s{s}' if p.get('reportable') else ('refine' if (R.get((v, c)) or {}).get('found') else None)
                conf[(v, c, s)] = bool(src and X.get((v, c, src), 0) >= CONFIRM)
                e = p.get('v0_error')
                bad[(v, c, s)] = e is not None and abs(e) > a.max
    clean_seeds = [s for s in SEEDS if not any(bad[(v, c, s)] for v in variants for c in ids)]
    out = dict(max_abs_v0_error=a.max, slow_starts={f'{v}|{ph}|{i}': f'{slow[(v, ph, i)]}/{n}'
                                                     for (v, ph, i), n in sorted(total.items()) if v in variants},
               clean_seeds=clean_seeds, variants={})
    for v in variants:
        per_seed = {s: round(float(np.mean([conf[(v, c, s)] for c in ids])), 4) for s in SEEDS}
        keep = lambda c: [s for s in SEEDS if not any(bad[(w, c, s)] for w in variants)]  # noqa: E731
        drop = [np.mean([conf[(v, c, s)] for s in keep(c)]) for c in ids if keep(c)]
        out['variants'][v] = dict(per_seed=per_seed,
                                  clean_seeds_mean=round(float(np.mean([per_seed[s] for s in clean_seeds])), 4) if clean_seeds else None,
                                  slow_pairs_left_out=round(float(np.mean(drop)), 4), cases_kept=len(drop))
    v1, v2 = variants[:2]
    if clean_seeds:
        d = [np.mean([conf[(v1, c, s)] for s in clean_seeds]) - np.mean([conf[(v2, c, s)] for s in clean_seeds]) for c in ids]
        out['difference_clean_seeds'] = dict(pair=f'{v1} - {v2}', mean=round(float(np.mean(d)), 4), ci=boot_ci(d))
    d = []
    for c in ids:
        ks = [s for s in SEEDS if not any(bad[(w, c, s)] for w in variants)]
        if ks:
            d.append(np.mean([conf[(v1, c, s)] for s in ks]) - np.mean([conf[(v2, c, s)] for s in ks]))
    out['difference_slow_pairs_left_out'] = dict(pair=f'{v1} - {v2}', mean=round(float(np.mean(d)), 4), ci=boot_ci(d), cases=len(d))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--cases', type=int, required=True)
    ap.add_argument('--variants', required=True, help='comma-separated; the difference is first - second')
    ap.add_argument('--max', type=float, default=.5, help='|v0_error| above this (m/s) is a slow start')
    main(ap.parse_args())
