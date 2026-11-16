"""B36 (post hoc): the reaction-time margin of the 1-D verdicts (deceleration dec_sim, cut-out cutout_sim).

For every failure the agent reports (best plan of a robust success) and every collision of the collision-seeking
baseline, the scenario's careful driver is replayed with other reaction times (0.4 s risk perception unchanged):
avoided at 0.75 s -> r_up over 1.0 ... 3.0 s (largest r avoided at every grid value up to r); collided at 0.75 s ->
r_need over 0, 0.25, 0.5 s (largest avoiding r, None if even 0 s collides). Same grid as b36_reaction_margin.

  python -m fairtest.b36_margin_1d dec|cut CASES.json EVAL.jsonl OUT.json
"""
from __future__ import annotations

import collections
import json
import statistics
import sys

from fairtest import cutout_sim as CO
from fairtest import dec_sim as D
from fairtest.b36_reaction_margin import DOWN, UP


def replay(kind, case, theta, r):
    """The scenario's careful driver with reaction r (the simulators parse cc_dec_rNNN / cc_co_rNNN, NNN = r x 100)."""
    tag = f'{int(round(r * 100)):03d}'
    return D.run(case, theta, f'cc_dec_r{tag}') if kind == 'dec' else CO.run(case, theta, f'cc_co_r{tag}')


def margin(kind, case, theta, avoided):
    if avoided:
        r_up = .75
        for r in UP:
            if replay(kind, case, theta, r)['contact']:
                break
            r_up = r
        return dict(r_up=r_up)
    ok = [r for r in DOWN if not replay(kind, case, theta, r)['contact']]
    return dict(r_need=max(ok) if ok else None)


def main(kind, cases_path, eval_path, out):
    cases = {c['case_id']: c for c in json.load(open(cases_path, encoding='utf-8'))['cases']}
    rows = []
    for line in open(eval_path, encoding='utf-8'):
        r = json.loads(line)
        case = cases[r['case_id']]
        for key, rec in r['runs'].items():
            method, sut = key.split('|')
            b = rec.get('best')
            if not (b and b.get('success')):
                continue
            avoided = True if method == 'agent' else not b.get('cc_contact')
            rows.append(dict(case_id=r['case_id'], method=method, sut=sut, cc_avoided=avoided,
                             **margin(kind, case, b['theta'], avoided)))
    summ = collections.defaultdict(dict)
    for method in ('agent', 'collide'):
        up = [x['r_up'] for x in rows if x['method'] == method and x['cc_avoided']]
        need = [x['r_need'] for x in rows if x['method'] == method and not x['cc_avoided']]
        summ[method] = dict(avoided=len(up), r_up_median=statistics.median(up) if up else None,
                            r_up_ge_1_25=sum(u >= 1.25 for u in up) / len(up) if up else None,
                            r_up_ge_1_5=sum(u >= 1.5 for u in up) / len(up) if up else None,
                            collided=len(need), r_need_none=sum(n is None for n in need),
                            r_need=dict(collections.Counter('none' if n is None else f'{n:g}' for n in need)))
    rep = dict(kind=kind, rows=rows, summary=summ)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
