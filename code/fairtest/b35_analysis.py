"""B35 frozen analysis: TGSIM I-294 external replication of the deceleration (B30) and cut-out (B34/B34b) scenarios.

Deceleration (dec_sim, B30 agent)
  E-D1  agent (36 queries): weakest SUT - rss_dec >= 30 pp;   E-D2  rss_dec <= 5%
  E-D3  collision seeking: attacker-at-fault <= 10% of its collisions, and >= 80% of those need the lead to brake
        harder than 6 m/s^2 (outside G2); not judged below 5 attacker-at-fault collisions
Cut-out (cutout_sim; agent = the B34b consensus refinement, the B34 default reported alongside)
  E-X1  weakest - rss_dec >= 30 pp;   E-X2  rss_dec <= 5%
  E-X3  collision seeking discriminates: collision-rate range >= 30 pp, and >= 80% of its attacker-at-fault
        collisions need L2 to brake harder than 6 m/s^2 (not judged below 5)
  E-X4  100% of the consensus agent's failures fair under every careful-driver variant
E-P  (principle) within human braking authority (lead / L2 braking >= -6 m/s^2) the collision-seeking attacker's
      attacker-at-fault collisions are <= 5% of its collisions in both scenarios (cut-in, B28 T4: 100% of the competent
      SUTs' collisions attacker-at-fault).
Fewer than 20 cases in a scenario: that scenario is reported descriptively only.

  python -m fairtest.b35_analysis DEC_EVAL.jsonl CUT_EVAL.jsonl CUT_CONSENSUS.jsonl OUT.json
"""
from __future__ import annotations

import json
import sys

MIN_CASES, G2 = 20, -6.
DEC_SUTS = ('idm_d0', 'idm_d5', 'idm_d10', 'acc_prod', 'acc_prod_aeb', 'rss_dec')
BRAKE_INDEX = dict(dec=1, cut=3)  # a_b in theta


def _rows(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]


def agent_rates(rows, k=36):
    return {s: sum((r['runs'][f'agent|{s}'].get('first_robust') or 10 ** 9) <= k for r in rows) / len(rows) for s in DEC_SUTS}


def collide_stats(rows, kind):
    col, unfair, unfair_out, rates = 0, 0, 0, {}
    for s in DEC_SUTS:
        cs = [r['runs'][f'collide|{s}']['best'] for r in rows if r['runs'][f'collide|{s}'].get('first_collision') is not None]
        rates[s] = len(cs) / len(rows)
        col += len(cs)
        for c in cs:
            if c.get('cc_contact'):
                unfair += 1
                unfair_out += c['theta'][BRAKE_INDEX[kind]] < G2
    return dict(rates=rates, collisions=col, attacker_at_fault=unfair, attacker_outside_g2=unfair_out,
                attacker_share=unfair / col if col else None, outside_share=unfair_out / unfair if unfair else None,
                attacker_within_g2_share=(unfair - unfair_out) / col if col else None)


def main(dec_path, cut_path, con_path, out):
    dec, cut, con = _rows(dec_path), _rows(cut_path), _rows(con_path)
    rep = dict(cases=dict(dec=len(dec), cut=len(cut), consensus=len(con)))
    # deceleration
    a, c = agent_rates(dec), collide_stats(dec, 'dec')
    judged = len(dec) >= MIN_CASES
    rep['dec'] = dict(agent=a, collide=c, judged=judged)
    top = max(a.values())
    rep['E-D1'] = dict(diff=top - a['rss_dec'], supported=(top - a['rss_dec'] >= .30) if judged else None)
    rep['E-D2'] = dict(rss=a['rss_dec'], supported=(a['rss_dec'] <= .05) if judged else None)
    d3_ok = c['attacker_at_fault'] >= 5
    rep['E-D3'] = dict(attacker_share=c['attacker_share'], outside_share=c['outside_share'], judged=judged and d3_ok,
                       supported=((c['attacker_share'] or 0.) <= .10 and (c['outside_share'] or 0.) >= .80) if judged and d3_ok else None,
                       attacker_share_only=((c['attacker_share'] or 0.) <= .10) if judged else None)
    # cut-out
    a_con, a_def, c = agent_rates(con), agent_rates(cut), collide_stats(cut, 'cut')
    judged = len(cut) >= MIN_CASES
    rep['cut'] = dict(agent_consensus=a_con, agent_default=a_def, collide=c, judged=judged)
    top = max(a_con.values())
    rep['E-X1'] = dict(diff=top - a_con['rss_dec'], supported=(top - a_con['rss_dec'] >= .30) if judged else None)
    rep['E-X2'] = dict(rss=a_con['rss_dec'], supported=(a_con['rss_dec'] <= .05) if judged else None)
    rng = max(c['rates'].values()) - min(c['rates'].values())
    x3_ok = c['attacker_at_fault'] >= 5
    rep['E-X3'] = dict(range=rng, outside_share=c['outside_share'], judged=judged,
                       supported=(rng >= .30 and (c['outside_share'] >= .80 if x3_ok else True)) if judged else None,
                       outside_judged=x3_ok)
    fair = [all(not v['contact'] for v in r['runs'][f'agent|{s}']['judged'].values())
            for r in con for s in DEC_SUTS if r['runs'][f'agent|{s}'].get('first_robust') is not None]
    rep['E-X4'] = dict(failures=len(fair), fair_all=sum(fair) / len(fair) if fair else None,
                       supported=(bool(fair) and all(fair)) if judged else None)
    w = [rep['dec']['collide']['attacker_within_g2_share'], rep['cut']['collide']['attacker_within_g2_share']]
    rep['E-P'] = dict(within_g2_attacker_share=w, supported=all(x is not None and x <= .05 for x in w))
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
