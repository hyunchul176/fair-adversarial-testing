"""B32 (post hoc): the AEB layer of the R157 reference driver model, deceleration scenario (B30 development results).

R157 (original series) Annex 4 Appendix 3, Figure 1 and Table 1: the reference model is the careful driver plus an AEB
layer (0.6 s to 0.85 g, derived from UN R152) whose trigger TTC and delay are not specified. B30's careful driver
(cc_dec) omits the layer; cc_dec_aeb adds its strongest reading (dec_sim). Every failure B30 reported is replayed:
  agent    the agent's robust fair failures: still avoided by cc_dec_aeb, still with the 0.5 m margin?
  collide  the collision-seeking baseline's collisions: attacker-at-fault (the reference also collides) under
           cc_dec and under cc_dec_aeb
The stored cc_dec verdicts are re-computed first (the edited CCDec must reproduce them exactly).

  python -m fairtest.b32_dec_aeb DEC_CASES.json DEC_EVAL.jsonl OUT.json
"""
from __future__ import annotations

import collections
import json
import sys

from fairtest import dec_sim as D


def main(cases_path, eval_path, out_path):
    cases = {c['case_id']: c for c in json.load(open(cases_path, encoding='utf-8'))['cases']}
    agent, collide = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    mismatch, rows, gap_drop = 0, [], 0.
    for line in open(eval_path, encoding='utf-8'):
        r = json.loads(line)
        case = cases[r['case_id']]
        for key, rec in r['runs'].items():
            method, sut = key.split('|')
            b = rec.get('best')
            if not (b and b.get('success')):
                continue
            th = b['theta']
            s, cc, aeb = D.run(case, th, sut), D.run(case, th, 'cc_dec'), D.run(case, th, 'cc_dec_aeb')
            if not s['contact']:
                mismatch += 1
            if method == 'agent':
                mismatch += int(cc['contact'] != b['cc_contact'] or abs(cc['min_gap'] - b['cc_min_gap']) > 1e-9)
                c = agent[sut]
                c['n'] += 1
                c['fair_aeb'] += not aeb['contact']
                c['robust_aeb'] += (not aeb['contact']) and aeb['min_gap'] >= D.MARGIN
                gap_drop = max(gap_drop, cc['min_gap'] - aeb['min_gap'])
            else:
                if 'cc_contact' in b:
                    mismatch += int(cc['contact'] != b['cc_contact'])
                c = collide[sut]
                c['n'] += 1
                c['attacker_cc'] += cc['contact']
                c['attacker_aeb'] += aeb['contact']
            rows.append(dict(case_id=r['case_id'], method=method, sut=sut, a_b=round(th[1], 3), cc_contact=cc['contact'],
                             aeb_contact=aeb['contact'], cc_min_gap=round(cc['min_gap'], 3), aeb_min_gap=round(aeb['min_gap'], 3)))
    rep = dict(reproduction_mismatches=mismatch, max_margin_loss_m=round(gap_drop, 6),
               agent={s: dict(c, fair_aeb_share=c['fair_aeb'] / c['n'], robust_aeb_share=c['robust_aeb'] / c['n'])
                      for s, c in agent.items()},
               collide={s: dict(c, attacker_cc_share=c['attacker_cc'] / c['n'], attacker_aeb_share=c['attacker_aeb'] / c['n'])
                        for s, c in collide.items()},
               rows=rows)
    json.dump(rep, open(out_path, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
