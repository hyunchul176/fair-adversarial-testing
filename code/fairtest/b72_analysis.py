"""B72 analysis (B72 design, fixed before the run, available on request).

  R1  reproduction: every output row of the first 60 cases (case_id < 60; the B70 cases, same ids) equals the stored B70
      row, with the wall-clock field 'seconds' dropped; per output file
  E   on the new cases alone (case_id >= 60): P1 (following H0 - longitudinal RSS, v2s, 37 queries) and following H0
      detection with case-bootstrap 95% intervals (10,000 resamples, seed 20260928). The extended estimates on all cases
      (P1-P3, K1, K2, v2s-S) come from the frozen analyses (exid_analysis, exid_extra, b64_analysis) of the run.
  D1  case mix, out of sample on the new cases: following H0 detection in cases with initial headway < 1.5 s minus cases
      with initial headway >= 4 s is at least 40 pp; not judged if either bin has fewer than 5 cases
  D2  descriptive: following H0 detection by initial headway bin [0, 1, 1.5, 2.5, 4, inf) s for the B70 cases, the new
      cases and all cases, and all cases reweighted to the headway mix of the exiD cohort (B71)
Detection per case = mean over policy seeds of (policy-stage reported failure or first bo_warm_g4s success within 37
queries), the composition of exid_analysis.

  python -m fairtest.b72_analysis OUT.json B72_DIR B70_DIR B71_CASE_POOL.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b41_analysis import K_FULL, load
from fairtest.exid_analysis import load_refine

N_B70 = 60
FILES = ('grid_cc.jsonl', 'b11.jsonl', 'final_s1.jsonl', 'final_s2.jsonl', 'final_s3.jsonl', 'collide_commit_s1.jsonl',
         'collide_commit_s2.jsonl', 'collide_commit_s3.jsonl', 'cars_s1.jsonl', 'cars_s2.jsonl', 'cars_s3.jsonl',
         'dynasto_s1.jsonl', 'dynasto_s2.jsonl', 'dynasto_s3.jsonl', 'margin.jsonl', 'policy_replay.jsonl', 'refine.jsonl',
         'refine_g4ss.jsonl', 'closing.jsonl', 'consensus.jsonl')
BINS = (0., 1., 1.5, 2.5, 4., float('inf'))
B, SEED = 10000, 20260928


def rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != 'seconds'}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def reproduction(d72, d70):
    out = {}
    for name in FILES:
        a = [json.dumps(strip(r), sort_keys=True) for r in rows(Path(d70) / name)]
        b = [json.dumps(strip(r), sort_keys=True) for r in rows(Path(d72) / name) if r.get('case_id', -1) < N_B70]
        ca, cb = collections.Counter(a), collections.Counter(b)
        out[name] = dict(b70_rows=len(a), b72_rows_first60=len(b), identical=ca == cb,
                         only_b70=sum((ca - cb).values()), only_b72=sum((cb - ca).values()))
    return dict(files=out, all_identical=all(v['identical'] for v in out.values()))


def per_case(d):
    evals, realistic, _, pool, _ = load(Path(d))
    first = load_refine([str(Path(d) / 'refine.jsonl')])[0]['bo_warm_g4s']
    seeds = sorted(evals['final'])
    det = {}
    for s in ('following_H0', 'rss_sut_H0'):
        for c in pool['cases']:
            cid = c['case_id']
            q = first.get((cid, s))
            det[(s, cid)] = float(np.mean([float(bool(evals['final'][x].get((cid, s), {}).get('robust') and realistic.get((x, cid, s), False))
                                                 or (q is not None and 1 + q <= K_FULL)) for x in seeds]))
    head = {c['case_id']: c['event']['metrics']['p10']['headway'] for c in pool['cases']}
    return det, head


def boot(vals, rng):
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def bins_of(ids, det, head):
    out = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        v = [det[('following_H0', c)] for c in ids if lo <= head[c] < hi]
        out.append(dict(bin=[lo, hi if hi != float('inf') else None], n=len(v), detection=round(float(np.mean(v)), 4) if v else None))
    return out


def main(out_path, d72, d70, exid_pool):
    rep = dict(R1=reproduction(d72, d70))
    det, head = per_case(d72)
    ids = sorted(head)
    new = [c for c in ids if c >= N_B70]
    old = [c for c in ids if c < N_B70]
    rng = np.random.default_rng(SEED)
    diff = [det[('following_H0', c)] - det[('rss_sut_H0', c)] for c in new]
    fh0 = [det[('following_H0', c)] for c in new]
    rep['E_new'] = dict(cases=len(new), P1=round(float(np.mean(diff)), 4), P1_ci=boot(diff, rng),
                        following_H0=round(float(np.mean(fh0)), 4), following_H0_ci=boot(fh0, rng),
                        rss=round(float(np.mean([det[('rss_sut_H0', c)] for c in new])), 4))
    lo = [det[('following_H0', c)] for c in new if head[c] < 1.5]
    hi = [det[('following_H0', c)] for c in new if head[c] >= 4.]
    judged = len(lo) >= 5 and len(hi) >= 5
    d1 = float(np.mean(lo) - np.mean(hi)) if lo and hi else None
    rep['D1'] = dict(n_short=len(lo), n_wide=len(hi), short=round(float(np.mean(lo)), 4) if lo else None,
                     wide=round(float(np.mean(hi)), 4) if hi else None, diff=round(d1, 4) if d1 is not None else None,
                     threshold=.4, judged=judged, supported=bool(judged and d1 >= .4))
    rep['D2'] = dict(b70=bins_of(old, det, head), new=bins_of(new, det, head), all=bins_of(ids, det, head))
    ex = [c['event']['metrics']['p10']['headway'] for c in json.load(open(exid_pool, encoding='utf-8'))['cases']]
    w = [sum(1 for h in ex if lo_ <= h < hi_) for lo_, hi_ in zip(BINS[:-1], BINS[1:])]
    cells = [(n, b['detection']) for n, b in zip(w, rep['D2']['all']) if b['detection'] is not None]
    rep['D2']['all_reweighted_to_exid'] = round(sum(n * v for n, v in cells) / sum(n for n, _ in cells), 4)
    rep['D2']['exid_bin_counts'] = w
    json.dump(rep, open(out_path, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(dict(R1=rep['R1']['all_identical'], E_new=rep['E_new'], D1=rep['D1'],
                          reweighted=rep['D2']['all_reweighted_to_exid']), indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
