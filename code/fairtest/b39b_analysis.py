"""B39b analysis (B39b design, available on request).

Refinement (bo_warm_g4g, dev) with the reference replaced by oracle_react0 / oracle_t0, its successes re-judged by the
R157 CC; compared with the CC-reference refinement (B29 guarded_dev).
  F2b  oracle_react0: >= 20% of its successes attacker-at-fault by the CC (pooled), and the competent SUTs'
       (following_aeb_H0/H0.5, rss_sut_H0) mean detection rate >= 20%
  F3b  oracle_t0: competent mean detection >= 50%, ladder (7 SUTs) detection range <= 30 pp

  python -m fairtest.b39b_analysis OUT.json CC_E3.jsonl REACT0_E3.jsonl REACT0_REJ.jsonl T0_E3.jsonl T0_REJ.jsonl
"""
from __future__ import annotations

import json
import sys

LADDER = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')


def detection(path):
    per = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == 'bo_warm_g4g':
                q = rec.get('first_fair')
                per.setdefault(s, []).append(q is not None and q <= 36)
    return {s: sum(v) / len(v) for s, v in per.items()}


def rejudge(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    ok = [r for r in rows if 'error' not in r]
    by = {}
    for r in ok:
        by.setdefault(r['sut'], []).append(r)
    return dict(n=len(ok), errors=len(rows) - len(ok),
                attacker_share=sum(r['cc_contact'] for r in ok) / len(ok) if ok else None,
                cc_robust_share=sum((not r['cc_contact']) and r['cc_clearance'] >= .5 for r in ok) / len(ok) if ok else None,
                by_sut={s: dict(n=len(v), attacker_share=sum(x['cc_contact'] for x in v) / len(v)) for s, v in sorted(by.items())})


def main(out, cc_e3, r0_e3, r0_rej, t0_e3, t0_rej):
    rep = dict(detection=dict(cc_r157=detection(cc_e3), oracle_react0=detection(r0_e3), oracle_t0=detection(t0_e3)),
               rejudge=dict(oracle_react0=rejudge(r0_rej), oracle_t0=rejudge(t0_rej)))
    d = rep['detection']
    comp = {k: sum(v.get(s, 0.) for s in COMPETENT) / len(COMPETENT) for k, v in d.items()}
    rng = {k: max(v.get(s, 0.) for s in LADDER) - min(v.get(s, 0.) for s in LADDER) for k, v in d.items()}
    rep['competent_mean'], rep['ladder_range'] = comp, rng
    rep['F2b'] = dict(attacker_share=rep['rejudge']['oracle_react0']['attacker_share'], competent_mean=comp['oracle_react0'],
                      supported=(rep['rejudge']['oracle_react0']['attacker_share'] or 0.) >= .20 and comp['oracle_react0'] >= .20)
    rep['F3b'] = dict(competent_mean=comp['oracle_t0'], ladder_range=rng['oracle_t0'],
                      supported=comp['oracle_t0'] >= .50 and rng['oracle_t0'] <= .30)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('F2b', 'F3b', 'competent_mean', 'ladder_range')}, indent=1))
    print('re-judged by the CC:', {k: (v['n'], v['errors'], round(v['attacker_share'] or 0, 3)) for k, v in rep['rejudge'].items()})
    print('SUT                    CC     react0  t0')
    for s in sorted(d['cc_r157']):
        print(f"{s:22s} {d['cc_r157'].get(s, 0):.2f}   {d['oracle_react0'].get(s, 0):.2f}   {d['oracle_t0'].get(s, 0):.2f}")


if __name__ == '__main__':
    main(*sys.argv[1:7])
