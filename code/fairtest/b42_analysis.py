"""B42 analysis (B42 design, available on request).

  K1  the agent's robust fair failures (1-D): the CARLA careful driver (dec:cc_dec) avoids contact in >= 90%
  K2  the same plans: the CARLA SUT (the controller the plan was found for) collides in >= 80%
  K3  every dec follower (CC, CC + AEB layer, the six SUTs): 1-D and CARLA contact verdicts agree on >= 90% of all plans
  K4  collision-seeking plans in the human braking range (a_b >= -6 m/s^2) with a CARLA SUT collision: the CARLA CC
      also collides in <= 5% (the principle with vehicle physics)
  K5  competent followers (idm_d0, rss_dec): CARLA fair failures (contact while the CARLA CC avoids) on <= 5% of plans
An episode with an error or beyond the straight stretch is excluded from the tests that use it (counted).

  python -m fairtest.b42_analysis OUT.json B42_a.jsonl [B42_b.jsonl ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter

from fairtest import dec_sim as D

CC, CCA = 'dec:cc_dec', 'dec:cc_dec_aeb'
COMPETENT = ('idm_d0', 'rss_dec')


def ok(ep):
    return ep is not None and 'error' not in ep and not ep.get('beyond_stretch')


def share(num, den):
    return None if not den else round(num / den, 4)


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 4)


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                rows[(r['case_id'], r['method'], r['sut'])] = r
    return [rows[k] for k in sorted(rows)]


def main(out, *paths):
    rows = load(paths)
    kinds = sorted({k for r in rows for k in r['carla']})
    rep = dict(rows=len(rows), plans=dict(Counter(f"{r['method']}|{r['sut']}" for r in rows)),
               errors={k: sum('error' in r['carla'].get(k, {}) for r in rows) for k in kinds},
               beyond={k: sum(bool(r['carla'].get(k, {}).get('beyond_stretch')) for r in rows) for k in kinds})
    agent = [r for r in rows if r['method'] == 'agent']
    collide = [r for r in rows if r['method'] == 'collide']

    # K1 / K2 / joint transfer of the agent's fair failures
    k1 = [r for r in agent if ok(r['carla'].get(CC))]
    avoid = [r for r in k1 if not r['carla'][CC]['contact']]
    k1a = [r for r in agent if ok(r['carla'].get(CCA))]
    k2 = [r for r in agent if ok(r['carla'].get(f"dec:{r['sut']}"))]
    hit = [r for r in k2 if r['carla'][f"dec:{r['sut']}"]['contact']]
    both = [r for r in agent if ok(r['carla'].get(CC)) and ok(r['carla'].get(f"dec:{r['sut']}"))]
    fair = [r for r in both if r['carla'][f"dec:{r['sut']}"]['contact'] and not r['carla'][CC]['contact']]
    robust = [r for r in fair if r['carla'][CC]['min_gap'] >= D.MARGIN]
    flips = [dict(case_id=r['case_id'], sut=r['sut'], oned_cc_min_gap=r['oned'][CC]['min_gap'],
                  carla_cc_min_gap=r['carla'][CC]['min_gap'], a_b=round(r['theta'][1], 2)) for r in k1 if r['carla'][CC]['contact']]
    rep['K1'] = dict(n=len(k1), cc_avoids=share(len(avoid), len(k1)),
                     cc_keeps_margin=share(sum(r['carla'][CC]['min_gap'] >= D.MARGIN for r in k1), len(k1)),
                     cc_aeb_avoids=share(sum(not r['carla'][CCA]['contact'] for r in k1a), len(k1a)),
                     flips=flips, supported=bool(k1) and len(avoid) / len(k1) >= .90)
    rep['K2'] = dict(n=len(k2), sut_collides=share(len(hit), len(k2)), supported=bool(k2) and len(hit) / len(k2) >= .80)
    per = {}
    for s in sorted({r['sut'] for r in agent}):
        a = [r for r in both if r['sut'] == s]
        f = [r for r in fair if r['sut'] == s]
        per[s] = dict(plans=len(a), fair_in_carla=len(f), robust_in_carla=sum(r in robust for r in f),
                      of_60_cases=share(len(f), 60), robust_of_60=share(sum(r in robust for r in f), 60))
    rep['fair_failure_transfer'] = dict(n=len(both), fair=share(len(fair), len(both)), robust=share(len(robust), len(both)), per_sut=per)

    # K3 verdict agreement, all plans, every dec follower
    k3 = {}
    for k in kinds:
        if not k.startswith('dec:'):
            continue
        pairs = [(r['oned'][k]['contact'], r['carla'][k]['contact']) for r in rows if ok(r['carla'].get(k)) and k in r['oned']]
        k3[k] = dict(n=len(pairs), agree=share(sum(a == b for a, b in pairs), len(pairs)), kappa=kappa(pairs),
                     oned_contact=sum(a for a, _ in pairs), carla_contact=sum(b for _, b in pairs),
                     lost=sum(a and not b for a, b in pairs), gained=sum(b and not a for a, b in pairs))
    rep['K3'] = dict(per_follower=k3, supported=bool(k3) and all(v['agree'] is not None and v['agree'] >= .90 for v in k3.values()))

    # K4 principle with physics: human-range collision-seeking plans
    def attacker_share(plans):
        c = [r for r in plans if ok(r['carla'].get(CC)) and ok(r['carla'].get(f"dec:{r['sut']}"))
             and r['carla'][f"dec:{r['sut']}"]['contact']]
        return len(c), sum(r['carla'][CC]['contact'] for r in c)
    human = [r for r in collide if r['theta'][1] >= D.G2_BRAKE - 1e-9]
    n4, u4 = attacker_share(human)
    nf, uf = attacker_share(collide)
    rep['K4'] = dict(plans=len(human), sut_collisions=n4, cc_also=u4, attacker_share=share(u4, n4),
                     full_box=dict(plans=len(collide), sut_collisions=nf, cc_also=uf, attacker_share=share(uf, nf)),
                     supported=bool(n4) and u4 / n4 <= .05)

    # K5 specificity with physics
    k5 = {}
    for s in COMPETENT:
        k = f'dec:{s}'
        v = [r for r in rows if ok(r['carla'].get(k)) and ok(r['carla'].get(CC))]
        k5[s] = dict(n=len(v), contact=share(sum(r['carla'][k]['contact'] for r in v), len(v)),
                     fair=share(sum(r['carla'][k]['contact'] and not r['carla'][CC]['contact'] for r in v), len(v)))
    rep['K5'] = dict(per_sut=k5, supported=all(x['fair'] is not None and x['fair'] <= .05 for x in k5.values()))

    # descriptive: CARLA's own agents as unseen followers, actuation
    third = {}
    for k in kinds:
        if k.startswith('dec:'):
            continue
        v = [r for r in rows if ok(r['carla'].get(k)) and ok(r['carla'].get(CC))]
        c = [r for r in v if r['carla'][k]['contact']]
        third[k] = dict(n=len(v), contact=share(len(c), len(v)),
                        fair_of_contacts=share(sum(not r['carla'][CC]['contact'] for r in c), len(c)),
                        robust_of_contacts=share(sum(r['carla'][CC]['min_gap'] >= D.MARGIN for r in c), len(c)),
                        by_method={m: share(sum(r['carla'][k]['contact'] for r in v if r['method'] == m),
                                            sum(r['method'] == m for r in v)) for m in ('agent', 'collide')},
                        v0_error_abs_max=max((abs(r['carla'][k]['v0_error']) for r in v), default=None))
    rep['third_party'] = third
    em = [r['carla'][CC]['sut_min_acc'] for r in rows if ok(r['carla'].get(CC)) and r['carla'][CC].get('min_request', 0) <= -7.5]
    rep['actuation'] = dict(cc_emergency_episodes=len(em), cc_min_acc_median=statistics.median(em) if em else None,
                            shim_v0_error_abs_max=max((abs(r['carla'][k]['v0_error']) for r in rows for k in kinds
                                                       if k.startswith('dec:') and ok(r['carla'].get(k))), default=None))
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('rows', 'errors', 'beyond')}))
    for k in ('K1', 'K2', 'K4'):
        print(k, json.dumps({x: y for x, y in rep[k].items() if x != 'flips'}))
    print('K3', rep['K3']['supported'], json.dumps(rep['K3']['per_follower']))
    print('K5', json.dumps(rep['K5']))
    print('fair transfer', json.dumps(rep['fair_failure_transfer']))
    print('third party', json.dumps(third))
    print('actuation', json.dumps(rep['actuation']))


if __name__ == '__main__':
    main(*sys.argv[1:])
