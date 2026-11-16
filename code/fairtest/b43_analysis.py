"""B43 analysis (B43 design, available on request).

  K1  the agent's and the consensus agent's robust fair failures (1-D): the CARLA careful driver (co:cc_co) avoids >= 90%
  K2  the same plans: the CARLA SUT (the controller the plan was found for) collides in >= 80%
  K3  every co follower (CC, its three variants, the six SUTs): 1-D and CARLA contact verdicts agree on >= 90% of plans
  K4  collision-seeking plans in the human braking range (a_b >= -6 m/s^2) with a CARLA SUT collision: the CARLA CC
      also collides in <= 5% (the principle with vehicle physics and geometry)
  K5  RSS: CARLA fair failures (contact while the CARLA CC avoids) on <= 5% of plans
  K6  the consensus agent's failures: every CARLA careful-driver variant avoids in >= 90%
An episode with an error or beyond the straight stretch is excluded from the tests that use it (counted).

  python -m fairtest.b43_analysis OUT.json B43_a.jsonl [B43_b.jsonl ...]
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter

from fairtest import cutout_sim as X
from fairtest.b42_analysis import kappa, load, ok, share

CC = 'co:cc_co'
VARIANTS = tuple(f'co:{v}' for v in X.CC_VARIANTS)


def main(out, *paths):
    rows = load(paths)
    kinds = sorted({k for r in rows for k in r['carla']})
    rep = dict(rows=len(rows), plans=dict(Counter(f"{r['method']}|{r['sut']}" for r in rows)),
               errors={k: sum('error' in r['carla'].get(k, {}) for r in rows) for k in kinds},
               beyond={k: sum(bool(r['carla'].get(k, {}).get('beyond_stretch')) for r in rows) for k in kinds})
    agents = [r for r in rows if r['method'] in ('agent', 'consensus')]
    own = lambda r: f"co:{r['sut']}"  # noqa: E731

    k1 = [r for r in agents if ok(r['carla'].get(CC))]
    avoid = [r for r in k1 if not r['carla'][CC]['contact']]
    k2 = [r for r in agents if ok(r['carla'].get(own(r)))]
    hit = [r for r in k2 if r['carla'][own(r)]['contact']]
    rep['K1'] = dict(n=len(k1), cc_avoids=share(len(avoid), len(k1)),
                     cc_keeps_margin=share(sum(r['carla'][CC]['min_gap'] >= X.MARGIN for r in k1), len(k1)),
                     by_method={m: share(sum(not r['carla'][CC]['contact'] for r in k1 if r['method'] == m),
                                         sum(r['method'] == m for r in k1)) for m in ('agent', 'consensus')},
                     flips=[dict(case_id=r['case_id'], method=r['method'], sut=r['sut'], oned_cc_min_gap=r['oned'][CC]['min_gap'],
                                 carla_contact_with=r['carla'][CC]['contact_with']) for r in k1 if r['carla'][CC]['contact']],
                     supported=bool(k1) and len(avoid) / len(k1) >= .90)
    rep['K2'] = dict(n=len(k2), sut_collides=share(len(hit), len(k2)),
                     contact_with=dict(Counter(r['carla'][own(r)]['contact_with'] for r in hit)),
                     supported=bool(k2) and len(hit) / len(k2) >= .80)
    per = {}
    for m in ('agent', 'consensus'):
        for s in sorted({r['sut'] for r in agents if r['method'] == m}):
            v = [r for r in agents if r['method'] == m and r['sut'] == s and ok(r['carla'].get(CC)) and ok(r['carla'].get(own(r)))]
            f = [r for r in v if r['carla'][own(r)]['contact'] and not r['carla'][CC]['contact']]
            per[f'{m}|{s}'] = dict(plans=len(v), fair_in_carla=len(f), of_60_cases=share(len(f), 60),
                                   robust_of_60=share(sum(r['carla'][CC]['min_gap'] >= X.MARGIN for r in f), 60))
    rep['fair_failure_transfer'] = per

    k3 = {}
    for k in kinds:
        if not k.startswith('co:'):
            continue
        pairs = [(r['oned'][k]['contact'], r['carla'][k]['contact']) for r in rows if ok(r['carla'].get(k)) and k in r['oned']]
        k3[k] = dict(n=len(pairs), agree=share(sum(a == b for a, b in pairs), len(pairs)), kappa=kappa(pairs),
                     oned_contact=sum(a for a, _ in pairs), carla_contact=sum(b for _, b in pairs),
                     lost=sum(a and not b for a, b in pairs), gained=sum(b and not a for a, b in pairs))
    rep['K3'] = dict(per_follower=k3, supported=bool(k3) and all(v['agree'] is not None and v['agree'] >= .90 for v in k3.values()))

    def attacker_share(plans):
        c = [r for r in plans if ok(r['carla'].get(CC)) and ok(r['carla'].get(own(r))) and r['carla'][own(r)]['contact']]
        return len(c), sum(r['carla'][CC]['contact'] for r in c)
    collide = [r for r in rows if r['method'] == 'collide']
    human = [r for r in collide if r['theta'][3] >= X.G2_BRAKE - 1e-9]
    n4, u4 = attacker_share(human)
    nf, uf = attacker_share(collide)
    rep['K4'] = dict(plans=len(human), sut_collisions=n4, cc_also=u4, attacker_share=share(u4, n4),
                     full_box=dict(plans=len(collide), sut_collisions=nf, cc_also=uf, attacker_share=share(uf, nf)),
                     supported=bool(n4) and u4 / n4 <= .05)

    k = 'co:rss_dec'
    v = [r for r in rows if ok(r['carla'].get(k)) and ok(r['carla'].get(CC))]
    rep['K5'] = dict(n=len(v), contact=share(sum(r['carla'][k]['contact'] for r in v), len(v)),
                     fair=share(sum(r['carla'][k]['contact'] and not r['carla'][CC]['contact'] for r in v), len(v)))
    rep['K5']['supported'] = rep['K5']['fair'] is not None and rep['K5']['fair'] <= .05

    cons = [r for r in rows if r['method'] == 'consensus' and all(ok(r['carla'].get(x)) for x in (CC,) + VARIANTS)]
    allv = [r for r in cons if not any(r['carla'][x]['contact'] for x in (CC,) + VARIANTS)]
    rep['K6'] = dict(n=len(cons), all_variants_avoid=share(len(allv), len(cons)),
                     per_variant={x: share(sum(not r['carla'][x]['contact'] for r in cons), len(cons)) for x in (CC,) + VARIANTS},
                     supported=bool(cons) and len(allv) / len(cons) >= .90)

    third = {}
    for k in kinds:
        if k.startswith('co:'):
            continue
        v = [r for r in rows if ok(r['carla'].get(k)) and ok(r['carla'].get(CC))]
        c = [r for r in v if r['carla'][k]['contact']]
        third[k] = dict(n=len(v), contact=share(len(c), len(v)),
                        fair_of_contacts=share(sum(not r['carla'][CC]['contact'] for r in c), len(c)),
                        robust_of_contacts=share(sum(r['carla'][CC]['min_gap'] >= X.MARGIN for r in c), len(c)),
                        contact_with=dict(Counter(r['carla'][k]['contact_with'] for r in c)),
                        by_method={m: share(sum(r['carla'][k]['contact'] for r in v if r['method'] == m),
                                            sum(r['method'] == m for r in v)) for m in ('agent', 'consensus', 'collide')},
                        v0_error_abs_max=max((abs(r['carla'][k]['v0_error']) for r in v), default=None))
    rep['third_party'] = third
    em = [r['carla'][CC]['sut_min_acc'] for r in rows if ok(r['carla'].get(CC)) and r['carla'][CC].get('min_request', 0) <= -7.5]
    rep['actuation'] = dict(cc_emergency_episodes=len(em), cc_min_acc_median=statistics.median(em) if em else None,
                            shim_v0_error_abs_max=max((abs(r['carla'][k]['v0_error']) for r in rows for k in kinds
                                                       if k.startswith('co:') and ok(r['carla'].get(k))), default=None))
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('rows', 'errors', 'beyond')}))
    for k in ('K1', 'K2', 'K4', 'K5', 'K6'):
        print(k, json.dumps({x: y for x, y in rep[k].items() if x != 'flips'}))
    print('K3', rep['K3']['supported'], json.dumps(rep['K3']['per_follower']))
    print('fair transfer', json.dumps(per))
    print('third party', json.dumps(third))
    print('actuation', json.dumps(rep['actuation']))


if __name__ == '__main__':
    main(*sys.argv[1:])
