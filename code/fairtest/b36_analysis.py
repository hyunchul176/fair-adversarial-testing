"""B36 analysis: reaction-time margins of the cut-in verdicts (b36_reaction_margin outputs, one file per dataset).

  agent / baseline CC-fair   r_up (largest reaction the CC still avoids at): median, shares >= 1.0 / 1.25 / 1.5 / 2.0 s
  baseline attacker-at-fault r_need: share avoidable only with a faster-than-norm reaction (<= 0.5 s), share unavoidable
                             even with an instantaneous reaction at the perception point

  python -m fairtest.b36_analysis B36_DIR OUT.json
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

LEVELS = (1., 1.25, 1.5, 2.)


def family(source):
    return next((f for f in ('collide_commit', 'cars', 'dynasto', 'naive_fair') if source.startswith(f)), 'agent')


def up(rs):
    v = [r['r_up'] for r in rs]
    return dict(n=len(v), median=statistics.median(v) if v else None,
                **{f'ge_{x:g}': (sum(u >= x for u in v) / len(v)) if v else None for x in LEVELS})


def need(rs):
    v = [r['r_need'] for r in rs]
    return dict(n=len(v), unavoidable_even_at_0=(sum(x is None for x in v) / len(v)) if v else None,
                needs_le_0_5=(sum(x is not None for x in v) / len(v)) if v else None,
                dist=dict(collections.Counter('none' if x is None else f'{x:g}' for x in v)))


def main(d, out):
    rep = {}
    pooled = collections.defaultdict(list)
    for p in sorted(Path(d).glob('*.jsonl')):
        rows = [json.loads(line) for line in open(p, encoding='utf-8')]
        ok = [r for r in rows if 'error' not in r]
        g = collections.defaultdict(list)
        for r in ok:
            fam = 'agent' if r['stage'] in ('policy', 'refine') else family(r['source'])
            key = fam if fam == 'agent' else fam + ('|attacker' if r['stored_cc_contact'] else '|ccfair')
            g[key].append(r)
            pooled[key].append(r)
        rep[p.stem] = dict(rows=len(rows), errors=len(rows) - len(ok),
                           **{k: (need(v) if k.endswith('|attacker') else up(v)) for k, v in sorted(g.items())})
    rep['pooled'] = {k: (need(v) if k.endswith('|attacker') else up(v)) for k, v in sorted(pooled.items())}
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep['pooled'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
