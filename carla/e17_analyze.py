"""E17 analysis (addendum 12): Tesla closed loop, matched twin (A) versus the E2 settings as they are (B). Verdicts as in e2_analyze.

  python e17_analyze.py WORK_DIR      -> runs/E17_RESULT.json
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_analyze import COMPETENT, ORDER, load  # noqa: E402


def boot(cases, n=2000, seed=0):
    """Case bootstrap 95% interval: cases = {case: [0/1 per seed]}."""
    keys = sorted(cases)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = [np.mean(cases[k]) for k in keys]
    m = np.asarray(vals)[rng.integers(0, len(keys), (n, len(keys)))].mean(axis=1)
    return [round(100 * float(np.percentile(m, 2.5)), 1), round(100 * float(np.percentile(m, 97.5)), 1)]


def summary(work, cond):
    runs = {s: load(work / f'runs/E17{cond}_rl_{s}.jsonl') for s in ('s1', 's2', 's3') if (work / f'runs/E17{cond}_rl_{s}.jsonl').exists()}
    out = {}
    for kind in ORDER:
        c = collections.Counter(v for d in runs.values() for (case, k), v in d.items() if k == kind)
        n = sum(c.values()) - c['error']
        per_case = collections.defaultdict(list)
        for d in runs.values():
            for (case, k), v in d.items():
                if k == kind and v != 'error':
                    per_case[case].append(v == 'robust')
        out[kind] = dict(n=n, **{k: c[k] for k in ('robust', 'fair', 'attacker', 'other', 'none', 'error')},
                         robust_rate=round(100 * c['robust'] / n, 1) if n else None, ci=boot(per_case))
    return dict(seeds=sorted(runs), by_kind=out, attacker=sum(v['attacker'] for v in out.values()),
                runs=sum(v['n'] for v in out.values()))


def main(work):
    res = {c: summary(work, c) for c in ('A', 'B')}
    a, b = res['A'], res['B']
    res['E17_H1'] = dict(A=a['attacker'], A_share=round(100 * a['attacker'] / a['runs'], 2) if a['runs'] else None, B=b['attacker'],
                         supported=bool(a['runs'] and a['attacker'] / a['runs'] <= .01 and a['attacker'] < b['attacker']))
    h0 = a['by_kind']['ladder:following:0']['robust_rate']
    res['E17_H2'] = dict(following_H0=h0, supported=bool(h0 is not None and h0 >= 75.))
    comp = {k: a['by_kind'][k]['robust_rate'] for k in COMPETENT}
    res['E17_H3'] = dict(rates=comp, supported=all(v is not None and v <= 3. for v in comp.values()))
    json.dump(res, open(work / 'runs/E17_RESULT.json', 'w'), indent=1)
    for c in ('A', 'B'):
        r = res[c]
        print(f"== E17-{c} seeds {r['seeds']} runs {r['runs']} attacker at fault {r['attacker']}")
        for k, v in r['by_kind'].items():
            print(f"   {k:24} n {v['n']:3d} robust {v['robust']:3d} ({v['robust_rate']}%, {v['ci']})  fair {v['fair']:3d}  at fault {v['attacker']:3d}"
                  f"  other {v['other']:2d}  none {v['none']:3d}  error {v['error']}")
    for h in ('E17_H1', 'E17_H2', 'E17_H3'):
        print(h, res[h])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
