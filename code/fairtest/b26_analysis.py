"""B26 (A): the final agent's robust fair-failure discovery, composed exactly per case from its two stages.

Stage 1 = the shielded policy (fair_guarded2, window shield + CC margin 0.5 m): one SUT query per case and seed.
Stage 2 = the refinement (GP-BO from the R157-boundary plan, G4-robust objective, bo_warm_g4): up to 36 queries,
run only if stage 1 did not produce a robust fair failure. Stage 2 never uses stage 1's outcome (it starts from the
CC-only boundary plan), so the composition is exact: for a case, the agent succeeds within k queries iff
  stage 1 is robust, or (k >= 2 and stage 2's first robust query q satisfies 1 + q <= k).
Robust (G4) = SUT contact with contact_dx >= 3.5 m while the CC avoids with clearance >= 0.5 m.

  python -m fairtest.b26_analysis final --dir fairtest_jobs/b26 --out OUT.json [--ks 1,2,6,37]
  (v2) ... --method bo_warm_g4g --refine fairtest_jobs/b29/guarded_{ds}.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

DATASETS = ('dev', 'b9', 'ngsim')
DELTA, FRONT = .5, 3.5


def _rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def policy_stage(d, ds):
    """{(seed, case, sut): robust} for the shielded policy, plus {(seed, case, sut): attacker-at-fault}."""
    margin = {}
    mp = d / 'final' / f'margin_{ds}.jsonl'
    if mp.exists():
        for r in _rows(mp):
            margin[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= FRONT
                                                                 and r['cc_clearance'] >= DELTA)
    robust, unfair = {}, {}
    for path in sorted((d / 'final').glob(f'final_s*_{ds}.jsonl')):
        seed = path.name.split('_')[1]
        for r in _rows(path):
            if 'valid' not in r:
                continue
            key = (seed, r['case_id'], r['sut'])
            col = bool(r['valid'] and r['contact'])
            fair = col and not r['refs']['cc_r157']['contact']
            robust[key] = bool(fair and margin.get((path.name, r['case_id'], r['sut']), False))
            unfair[key] = col and not fair
    return robust, unfair


def refine_stage(d, ds, method='bo_warm_g4', paths=None):
    """{(case, sut): first robust query or None} from the B26 E3-G4 runs (ladder) and the extra-SUT runs, or from
    explicit files (paths: '{ds}' is replaced by the dataset)."""
    out = {}
    files = [Path(p.format(ds=ds)) for p in paths] if paths else [d / f'e3g4_{ds}.jsonl', d / 'final' / f'e3g4x_{ds}.jsonl']
    for path in files:
        if not path.exists():
            continue
        for r in _rows(path):
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == method and 'error' not in rec:
                    out[(r['case_id'], s)] = rec.get('first_fair')
    return out


def final(a):
    d = Path(a.dir)
    ks = [int(x) for x in a.ks.split(',')]
    rng = np.random.default_rng(20260927)
    rep = {}
    for ds in DATASETS:
        robust, unfair = policy_stage(d, ds)
        ref = refine_stage(d, ds, a.method, a.refine)
        if not robust:
            continue
        seeds = sorted({k[0] for k in robust})
        suts = sorted({k[2] for k in robust})
        rep[ds] = {}
        for s in suts:
            cases = sorted({k[1] for k in robust if k[2] == s})
            has_ref = all((c, s) in ref for c in cases)
            per_seed = collections.defaultdict(dict)
            for seed in seeds:
                for k in ks:
                    hits = []
                    for c in cases:
                        p1 = robust.get((seed, c, s), False)
                        q = ref.get((c, s)) if has_ref else None
                        hits.append(p1 or (k >= 2 and q is not None and 1 + q <= k))
                    per_seed[seed][k] = np.array(hits, float)
            entry = dict(n_cases=len(cases), refinement_available=has_ref,
                         policy_robust=[float(per_seed[x][1].mean()) for x in seeds],
                         attacker_at_fault=[float(np.mean([unfair.get((x, c, s), False) for c in cases])) for x in seeds])
            for k in ks:
                vals = np.mean([per_seed[x][k] for x in seeds], axis=0)  # seed-averaged per case
                boot = vals[rng.integers(0, len(vals), (4000, len(vals)))].mean(1)
                entry[f'k{k}'] = dict(mean=float(vals.mean()), sd_seeds=float(np.std([per_seed[x][k].mean() for x in seeds])),
                                      ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
            rep[ds][s] = entry
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for ds, t in rep.items():
        print(f'== {ds}  (robust fair failure within k queries, % of cases; seed mean, [95% CI]; policy attacker-at-fault)')
        for s, e in t.items():
            cells = '  '.join(f"k={k}: {100 * e[f'k{k}']['mean']:5.1f} [{100 * e[f'k{k}']['ci'][0]:4.1f},{100 * e[f'k{k}']['ci'][1]:5.1f}]"
                              for k in ks if f'k{k}' in e)
            flag = '' if e['refinement_available'] else '  (policy only)'
            print(f"   {s:22} n={e['n_cases']:3d}  {cells}  unfair {100 * float(np.mean(e['attacker_at_fault'])):4.1f}{flag}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('final')
    f.add_argument('--dir', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--ks', default='1,2,6,37')
    f.add_argument('--method', default='bo_warm_g4', help='refinement method (v2: bo_warm_g4g)')
    f.add_argument('--refine', nargs='*', help="refinement files, '{ds}' = dataset (default: the B26 E3-G4 files)")
    a = ap.parse_args()
    final(a)
