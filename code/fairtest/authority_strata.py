"""Post hoc: v2j success within 37 queries by the case's attack authority, the relative deceleration headroom
r = SUT initial speed - 15 m/s (the generator's speed floor), in the bins < 2, 2-5, 5-10, >= 10 m/s (B41 section 6; exiD S10).
Each dataset is an evaluation directory in the B41 layout (b41_analysis.load).

  python -m fairtest.authority_strata OUT.json NAME DIR [NAME DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fairtest.b41_analysis import K_FULL, composed, load

BINS = ((-99, 2), (2, 5), (5, 10), (10, 99))
SUTS = ('following_H0', 'reference_idm_H0', 'following_aeb_H0', 'rss_sut_H0', 'bc_human_H0')


def main(out, pairs):
    rep = {}
    for name, d in zip(pairs[::2], pairs[1::2]):
        evals, realistic, refine, pool, _ = load(Path(d))
        cases = sorted(c['case_id'] for c in pool['cases'])
        head = {c['case_id']: c['physical_initial_state'][1]['vx'] - 15. for c in pool['cases']}
        full = composed(evals, realistic, refine, list(SUTS), cases, K_FULL)
        rep[name] = dict(headroom_median=float(np.median(list(head.values()))), bins={})
        for lo, hi in BINS:
            cs = [c for c in cases if lo <= head[c] < hi]
            rep[name]['bins'][f'{lo},{hi}'] = dict(n=len(cs), **{s: round(float(np.mean([full[s][c] for c in cs])), 4) if cs else None
                                                                   for s in SUTS})
        print(name, round(rep[name]['headroom_median'], 1), {k: (v['n'], v['following_H0']) for k, v in rep[name]['bins'].items()})
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
