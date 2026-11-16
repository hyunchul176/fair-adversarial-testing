"""R157-guided search: success@K SUT queries, computed offline from the landscape.

Per case, candidate plans are those whose CC-driver run is valid and contact-free (fair by
construction). Guided order: descending CC braking demand (hardest-preventable first), ties
by lower realism exceedance. Baseline: uniformly random order of the same candidates
(expected success@K computed exactly by the hypergeometric formula, not sampled).
success@K = fraction of eligible cases where one of the first K queried plans makes the SUT
collide (fair failure). K=1 is the zero-feedback R157-boundary planner; K=all is the oracle.
"""
from __future__ import annotations

import math
from math import comb

from fairtest.boundary_planner import CC, NOMINAL, SUTS, cc_demand, index
from fairtest.features import Realism


def curves(rows, ks=(1, 2, 3, 5, 10, 20, 50, 200), max_exceedance=math.inf):
    realism = Realism()
    by = index(rows)
    cases = sorted({c for c, _ in by})
    per_case = {}
    for (c, p), v in by.items():
        per_case.setdefault(c, []).append((p, v))
    scores = {}
    for (c, p), v in by.items():
        if CC in v and v[CC]['features'].get('cut_in'):
            sc = realism.score(v[CC]['features'])
            scores[(c, p)] = sc['max_exceedance'] if sc['max_exceedance'] is not None else math.inf
    out = {}
    for s in SUTS:
        name = f'{s[0]}_H{s[1]:g}'
        guided = {k: 0 for k in ks}
        random_expect = {k: 0. for k in ks}
        eligible = 0
        present = False
        for c in cases:
            nom = by.get((c, NOMINAL), {}).get(s)
            if nom is None:
                continue
            present = True
            if nom['contact'] or nom['ttc_event'] or nom['initial_event']:
                continue
            eligible += 1
            cand = []
            for p, v in per_case[c]:
                if not p.startswith('plan_') or p == NOMINAL or CC not in v or s not in v:
                    continue
                cc = v[CC]
                if not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
                    continue
                exc = scores[(c, p)]
                if exc > max_exceedance:
                    continue
                hit = v[s]['valid'] and v[s]['contact']
                cand.append((-cc_demand(cc), exc, p, hit))
            cand.sort()
            hits = [x[3] for x in cand]
            n, h = len(hits), sum(hits)
            for k in ks:
                guided[k] += any(hits[:k])
                kk = min(k, n)
                random_expect[k] += 0. if h == 0 else 1. - (comb(n - h, kk) / comb(n, kk) if n - h >= kk else 0.)
        if present:
            out[name] = dict(eligible=eligible,
                             guided={k: guided[k] / max(1, eligible) for k in ks},
                             random={k: random_expect[k] / max(1, eligible) for k in ks})
    return out
