"""B37 analysis (B37 design, available on request).

  H-I1  share of cases with an attacker-at-fault collision (7-SUT ladder pooled) non-increasing in h_min, and at
        h_min = 1.0 s <= 1/4 of h_min = 0
  H-I2  at h_min = 1.0 s the collision-rate range over the ladder >= 30 pp
  H-I3  at h_min = 0 the range <= 20 pp
  descriptive: per h_min and SUT the collision rate, the CC-fair rate (collision the CC avoids), the attacker share

  python -m fairtest.b37_analysis EVAL.jsonl OUT.json
"""
from __future__ import annotations

import collections
import json
import sys

LADDER = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')


def main(path, out):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    n = len(rows)
    hmins = sorted({float(k.split('|')[0]) for r in rows for k in r['runs']})
    rep, errors = dict(cases=n, hmins=hmins, by_h={}), 0
    for hm in hmins:
        per = {}
        unfair_cases = 0
        comp_col = comp_att = 0
        for s in LADDER:
            col = fair = att = 0
            for r in rows:
                rec = r['runs'].get(f'{hm:g}|{s}')
                if rec is None or 'error' in rec:
                    errors += rec is not None
                    continue
                if rec.get('first_collision') is not None:
                    col += 1
                    if rec['cc_contact']:
                        att += 1
                    else:
                        fair += 1
            per[s] = dict(collision=col / n, cc_fair=fair / n, attacker=att / n, attacker_share=att / col if col else None)
            unfair_cases += att
            if s in COMPETENT:
                comp_col += col
                comp_att += att
        rates = [per[s]['collision'] for s in LADDER]
        rep['by_h'][f'{hm:g}'] = dict(sut=per, range=max(rates) - min(rates),
                                      unfair_case_share=unfair_cases / (n * len(LADDER)),
                                      competent_collisions=comp_col, competent_attacker_share=comp_att / comp_col if comp_col else None)
    u = [rep['by_h'][f'{hm:g}']['unfair_case_share'] for hm in hmins]
    base, one = rep['by_h'].get('0'), rep['by_h'].get('1')
    rep['H-I1'] = dict(unfair_case_share=u, monotone=all(b <= a + 1e-12 for a, b in zip(u, u[1:])),
                       supported=bool(base and one and all(b <= a + 1e-12 for a, b in zip(u, u[1:]))
                                      and one['unfair_case_share'] <= base['unfair_case_share'] / 4))
    rep['H-I2'] = dict(range=one['range'] if one else None, supported=bool(one and one['range'] >= .30))
    rep['H-I3'] = dict(range=base['range'] if base else None, supported=bool(base and base['range'] <= .20))
    rep['errors'] = errors
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('H-I1', 'H-I2', 'H-I3', 'errors')}, indent=1))
    for hm in hmins:
        b = rep['by_h'][f'{hm:g}']
        print(f"h_min {hm:g}: range {b['range']:.2f}, unfair-case share {b['unfair_case_share']:.3f}, competent collisions "
              f"{b['competent_collisions']} (attacker share {b['competent_attacker_share']})")
        for s in LADDER:
            x = b['sut'][s]
            print(f"   {s:20s} collision {x['collision']:.2f}  cc_fair {x['cc_fair']:.2f}  attacker {x['attacker']:.2f}")


if __name__ == '__main__':
    main(*sys.argv[1:3])
