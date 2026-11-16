"""B67 analysis (B67 design, fixed before the run, available on request).

The final agent v2s per random stream of its refinement, development cases: policy stage fixed (b41_analysis.load of the
development stand-in), refinement bo_warm_g4s first success q (exid_analysis.composed, k = 37). Stream 0 = the stored B56
run, streams 1 and 2 = B67.
  ST1  P1 (following H0 - longitudinal RSS) of streams 1 and 2 each inside the case-bootstrap 95% interval of stream 0
  ST2  weak-4 mean detection of streams 1 and 2 each inside stream 0's case-bootstrap 95% interval
Descriptive: per-SUT detection per stream and its range over the streams (SUTs with range > 5 pp), refinement successes
and median first-success query per stream, pairwise agreement of the (case, SUT) success indicator (share, Cohen's kappa),
the SD of P1 over the streams beside stream 0's bootstrap SE (interval width / 3.92).

  python -m fairtest.b67_analysis OUT.json --dir DEV_STANDIN --stream0 B56_DEV.jsonl --streams B67_S1.jsonl,B67_S2.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b41_analysis import K_FULL, boot_diff, boot_rate, load, rate
from fairtest.exid_analysis import COMPETENT8, composed, load_refine

METHOD = 'bo_warm_g4s'
WEAK4 = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
RANGE_PP = 5.


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def main(a):
    evals, realistic, _, pool, _ = load(Path(a.dir))
    cases = sorted(c['case_id'] for c in pool['cases'])
    files = [a.stream0] + a.streams.split(',')
    rep = dict(n_cases=len(cases), streams={})
    succ = {}
    for i, path in enumerate(files):
        first, recs, err, conf = load_refine([path])
        suts = sorted({s for (_, s) in first[METHOD]})
        comp = composed(evals, realistic, first, METHOD, suts, cases, K_FULL)
        p1, ci = boot_diff(comp['following_H0'], comp['rss_sut_H0'])
        w = {c: float(np.mean([comp[s][c] for s in WEAK4])) for c in cases}
        wm, wci = boot_rate(w)
        qs = [q for q in first[METHOD].values() if q is not None]
        succ[i] = {k: first[METHOD].get(k) is not None for k in first[METHOD]}
        rep['streams'][str(i)] = dict(
            file=Path(path).name, pairs=len(first[METHOD]), errors=err.get(METHOD, 0), conflicts=len(conf),
            P1=round(p1, 4), P1_ci=[round(x, 4) for x in ci], weak4=round(wm, 4), weak4_ci=[round(x, 4) for x in wci],
            competent8=round(float(np.mean([rate(comp[s]) for s in COMPETENT8])), 4),
            suts={s: round(rate(comp[s]), 4) for s in suts},
            refine_successes=len(qs), refine_success_share=round(len(qs) / max(1, len(first[METHOD])), 4),
            median_first_query=float(np.median(qs)) if qs else None)
    s0 = rep['streams']['0']
    rep['ST1'] = {i: dict(P1=v['P1'], inside=bool(s0['P1_ci'][0] <= v['P1'] <= s0['P1_ci'][1])) for i, v in rep['streams'].items() if i != '0'}
    rep['ST1']['supported'] = all(v['inside'] for k, v in rep['ST1'].items() if k != 'supported')
    rep['ST2'] = {i: dict(weak4=v['weak4'], inside=bool(s0['weak4_ci'][0] <= v['weak4'] <= s0['weak4_ci'][1]))
                  for i, v in rep['streams'].items() if i != '0'}
    rep['ST2']['supported'] = all(v['inside'] for k, v in rep['ST2'].items() if k != 'supported')
    suts = list(s0['suts'])
    ranges = {s: round((max(v['suts'][s] for v in rep['streams'].values()) - min(v['suts'][s] for v in rep['streams'].values())) * 100, 2)
              for s in suts}
    rep['sut_range_pp'] = ranges
    rep['suts_range_over_5pp'] = sorted(s for s, r in ranges.items() if r > RANGE_PP)
    p1s = [v['P1'] for v in rep['streams'].values()]
    rep['P1_stream_sd'] = round(float(np.std(p1s, ddof=1)), 4)
    rep['P1_stream0_boot_se'] = round((s0['P1_ci'][1] - s0['P1_ci'][0]) / 3.92, 4)
    rep['agreement'] = {}
    for i, j in itertools.combinations(sorted(succ), 2):
        keys = sorted(set(succ[i]) & set(succ[j]))
        x, y = [succ[i][k] for k in keys], [succ[j][k] for k in keys]
        rep['agreement'][f'{i}-{j}'] = dict(pairs=len(keys), same=round(float(np.mean([p == q for p, q in zip(x, y)])), 4), kappa=kappa(x, y))
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for i, v in rep['streams'].items():
        print(f"stream {i}: P1 {v['P1'] * 100:5.1f} {[round(x * 100, 1) for x in v['P1_ci']]} weak4 {v['weak4'] * 100:5.1f} "
              f"competent8 {v['competent8'] * 100:4.1f} refine {v['refine_successes']}/{v['pairs']} q50 {v['median_first_query']} errors {v['errors']}")
    print('ST1', rep['ST1'], '| ST2', rep['ST2'])
    print('range > 5 pp', rep['suts_range_over_5pp'], '| P1 SD', rep['P1_stream_sd'], 'boot SE', rep['P1_stream0_boot_se'])
    print('agreement', rep['agreement'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--dir', required=True)
    ap.add_argument('--stream0', required=True)
    ap.add_argument('--streams', required=True)
    main(ap.parse_args())
