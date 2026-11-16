"""Analyse B11 continuous search against the grid methods on the same exposed cases."""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from fairtest.b9_analysis import per_case_outcomes
from fairtest.features import Realism


def main(b11_path, rows, cases, models, out=None):
    realism = Realism()
    recs = [json.loads(l) for l in open(b11_path, encoding='utf-8')]
    errors = [r for r in recs if 'error' in r]
    recs = {r['case_id']: r for r in recs if 'error' not in r}
    grid = per_case_outcomes(rows, cases, models, 'cc_r157', math.inf)
    report = dict(cases=len(recs), errors=len(errors), suts={})
    for sut in ('following_H1', 'following_aeb_H0', 'following_H0.5'):
        g = grid[sut]
        tally = dict(eligible=0, grid_oracle=0, grid_boundary=0, grid_guided5=0, grid_learned1=0,
                     cont_boundary=0, cont_search=0, cont_search_new=0, union=0)
        queries, exc_found = [], []
        for c, r in recs.items():
            m = g.get(c)
            if m is None:
                continue
            tally['eligible'] += 1
            for k, key in (('grid_oracle', 'oracle'), ('grid_boundary', 'boundary'), ('grid_guided5', 'guided_5'),
                           ('grid_learned1', 'learned_1')):
                tally[k] += m[key][0]
            cb = r.get('cc_boundary', {})
            ev = cb.get('sut_eval', {}).get(sut)
            tally['cont_boundary'] += bool(ev and ev['contact'] and not cb['best']['cc_contact'])
            ss = r['sut_search'].get(sut, {})
            found = bool(ss.get('best') and ss['best']['fair'])
            tally['cont_search'] += found
            if found:
                queries.append(ss['sut_queries'])
                sc = realism.score(ss['best']['features']) if ss['best']['features'].get('cut_in') else None
                exc_found.append(sc['max_exceedance'] if sc else None)
            new = found and not m['oracle'][0]
            tally['cont_search_new'] += new
            tally['union'] += bool(found or m['oracle'][0])
        n = max(1, tally['eligible'])
        report['suts'][sut] = dict({k: (v if k == 'eligible' else round(100 * v / n, 1)) for k, v in tally.items()},
                                   median_queries_when_found=float(np.median(queries)) if queries else None,
                                   median_exceedance_found=float(np.median([e for e in exc_found if e is not None])) if exc_found else None)
    text = json.dumps(report, indent=1)
    print(text)
    if out:
        open(out, 'w', encoding='utf-8').write(text)
    return report
