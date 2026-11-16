"""B39 analysis (B39 design, available on request).

Per shield reference (cc_r157, oracle_react0, oracle_t0), the v2 policy stage (seeds 1-3, averaged) judged by the R157
CC: per SUT the collision rate, the CC-fair rate (collision the CC avoids) and the attacker-at-fault rate (the CC
collides too); plus the shield reference's own verdict.
  F1  cc_r157 shield: attacker-at-fault 0% for every SUT
  F2  oracle_react0 shield: pooled attacker-at-fault case share > 5%, and > 50% of the competent SUTs' collisions
      attacker-at-fault by the CC
  F3  oracle_t0 shield: ladder collision-rate range <= 20 pp and >= 50% of its collisions attacker-at-fault by the CC

  python -m fairtest.b39_analysis B39_DIR OUT.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REFS = ('cc_r157', 'oracle_react0', 'oracle_t0')
LADDER = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'acc_prod_aeb_H0')


def load(d, ref):
    per = collections.defaultdict(list)  # sut -> [(collision, cc_fair, cc_unfair, own_fair)] per (seed, case)
    errors = 0
    for p in sorted(Path(d).glob(f'final_{ref}_s[0-9].jsonl')):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if 'valid' not in r:
                errors += 1
                continue
            col = bool(r['valid'] and r['contact'])
            refs = r.get('refs') or {}
            cc = refs.get('cc_r157', {}).get('contact')
            own = refs.get(ref, {}).get('contact')
            per[r['sut']].append((col, col and cc is False, col and cc is True, col and own is False))
    return per, errors


def main(d, out):
    rep = dict(by_ref={})
    for ref in REFS:
        per, errors = load(d, ref)
        if not per:
            continue
        sut = {s: dict(n=len(v), collision=sum(x[0] for x in v) / len(v), cc_fair=sum(x[1] for x in v) / len(v),
                       attacker=sum(x[2] for x in v) / len(v), own_fair=sum(x[3] for x in v) / len(v))
               for s, v in sorted(per.items())}
        rows = [x for v in per.values() for x in v]
        comp = [x for s, v in per.items() if s in COMPETENT for x in v if x[0]]
        lad = [sut[s]['collision'] for s in LADDER if s in sut]
        cols = [x for x in rows if x[0]]
        rep['by_ref'][ref] = dict(errors=errors, sut=sut, attacker_case_share=sum(x[2] for x in rows) / len(rows),
                                  competent_collisions=len(comp),
                                  competent_attacker_share=(sum(x[2] for x in comp) / len(comp)) if comp else None,
                                  ladder_range=max(lad) - min(lad) if lad else None,
                                  attacker_share_of_collisions=(sum(x[2] for x in cols) / len(cols)) if cols else None)
    b = rep['by_ref']
    if 'cc_r157' in b:
        rep['F1'] = dict(max_attacker=max(v['attacker'] for v in b['cc_r157']['sut'].values()),
                         supported=all(v['attacker'] == 0. for v in b['cc_r157']['sut'].values()))
    if 'oracle_react0' in b:
        x = b['oracle_react0']
        rep['F2'] = dict(attacker_case_share=x['attacker_case_share'], competent_attacker_share=x['competent_attacker_share'],
                         supported=x['attacker_case_share'] > .05 and (x['competent_attacker_share'] or 0.) > .5)
    if 'oracle_t0' in b:
        x = b['oracle_t0']
        rep['F3'] = dict(ladder_range=x['ladder_range'], attacker_share_of_collisions=x['attacker_share_of_collisions'],
                         supported=(x['ladder_range'] or 1.) <= .20 and (x['attacker_share_of_collisions'] or 0.) >= .5)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep.get(k) for k in ('F1', 'F2', 'F3')}, indent=1))
    for ref, x in b.items():
        print(f"== shield {ref}: attacker case share {x['attacker_case_share']:.3f}, ladder range {x['ladder_range']:.2f}, "
              f"competent collisions {x['competent_collisions']} (attacker share {x['competent_attacker_share']})")
        for s, v in x['sut'].items():
            print(f"   {s:20s} collision {v['collision']:.2f}  cc_fair {v['cc_fair']:.2f}  attacker {v['attacker']:.2f}  "
                  f"own-reference fair {v['own_fair']:.2f}")


if __name__ == '__main__':
    main(*sys.argv[1:3])
