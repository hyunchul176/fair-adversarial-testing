"""B44 analysis (B44 design, available on request).

  U1   deceleration, the regulation's full reference (cc_dec_aeb): cases with a reference-collision plan within the
       human braking authority <= 5%
  U2   cut-out, the primary reference (cc_co): <= 10%;  U2b  cut-out, cc_co_aeb: <= 5%
  U4   (descriptive, the disclosed probe) the literal deceleration CC (cc_dec): case share, the share of its breaking
       grid plans with braking below the 5 m/s^2 perception threshold, the share the competent followers avoid

  python -m fairtest.b44_analysis OUT.json DEC.jsonl CUTOUT.jsonl
"""
from __future__ import annotations

import json
import sys


def summarise(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    out = {}
    for ref in rows[0]['refs']:
        s = {}
        for auth in ('human', 'full'):
            g = [r['refs'][ref]['grid'][auth] for r in rows]
            b = [r['refs'][ref]['bo'][auth] for r in rows]
            brk = [gg['n_break'] > 0 or bb['found'] for gg, bb in zip(g, b)]
            nb = sum(gg['n_break'] for gg in g)
            s[auth] = dict(cases=len(rows), cases_broken=sum(brk), share=round(sum(brk) / len(rows), 4),
                           by_grid=sum(gg['n_break'] > 0 for gg in g), by_bo=sum(bb['found'] for bb in b),
                           grid_breaking_plans=nb, grid_valid_plans=sum(gg['n_valid'] for gg in g),
                           breaking_below_5=round(sum(gg['sub5'] for gg in g) / nb, 4) if nb else None,
                           breaking_competent_avoid=round(sum(gg['competent_avoid'] for gg in g) / nb, 4) if nb else None,
                           cases_with_sub5_break=sum(gg['sub5'] > 0 for gg in g),
                           bo_competent_avoid=sum(bool(bb.get('competent_avoid')) for bb in b if bb['found']),
                           weakest_a_b=max((gg['weakest'][1 if rows[0]['scenario'] == 'dec' else 3] for gg in g if gg['weakest']), default=None))
        out[ref] = s
    return out


def main(out, dec_path, cut_path):
    rep = dict(dec=summarise(dec_path), cutout=summarise(cut_path))
    u1 = rep['dec']['cc_dec_aeb']['human']['share']
    u2 = rep['cutout']['cc_co']['human']['share']
    u2b = rep['cutout']['cc_co_aeb']['human']['share']
    rep['U1'] = dict(share=u1, supported=u1 <= .05)
    rep['U2'] = dict(share=u2, supported=u2 <= .10)
    rep['U2b'] = dict(share=u2b, supported=u2b <= .05)
    rep['U4'] = rep['dec']['cc_dec']
    rep['cut_in_existing'] = dict(source='landscape-v3-cc (exposed grid, dev 62 x 200 plans)',
                                  cc_r157_react050=53 / 62, cc_r157_react100=53 / 62)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    for sc in ('dec', 'cutout'):
        for ref, s in rep[sc].items():
            h, f = s['human'], s['full']
            print(f"{sc:6s} {ref:13s} human {h['cases_broken']:2d}/{h['cases']} (grid {h['by_grid']}, bo {h['by_bo']}) "
                  f"below5 {h['breaking_below_5']} competent-avoid {h['breaking_competent_avoid']} weakest a_b {h['weakest_a_b']} | "
                  f"full {f['cases_broken']}/{f['cases']}")
    print(json.dumps({k: rep[k] for k in ('U1', 'U2', 'U2b')}))


def posthoc(out, cut_path, cases_path):
    """Post hoc (after the design, labelled so): what breaks the cut-out references. For every breaking plan within the
    human authority (grid + BO) of cc_co and cc_co_aeb: which ordinary controllers also collide, and the plan's shape
    (L1-L2 minimum bumper gap while L1's body is in the lane, the follower's gap / closing speed / THW to L2 when L1's
    body leaves the lane)."""
    import itertools
    import statistics as st
    from fairtest import b44_reference_break as B
    from fairtest import cutout_sim as X
    cases = {c['case_id']: c for c in json.load(open(cases_path, encoding='utf-8'))['cases']}
    rows = [json.loads(l) for l in open(cut_path, encoding='utf-8') if l.strip()]
    grid = [p for p in itertools.product(*B.GRID['cutout']) if p[3] >= B.HUMAN - 1e-9]
    others = ('idm_d0', 'idm_d5', 'idm_d10', 'acc_prod', 'acc_prod_aeb', 'rss_dec', 'cc_co_aeb')

    def shape(case, th):
        r = X.run(case, th, 'cc_co', trace=True)
        g12, at = [], None
        for (t, y1, g1, g2, v, v2, a) in r['trace']:
            if y1 < X.BODY_OUT:
                g12.append(g2 - g1 - float(case['length_lead']))
            elif at is None:
                at = (g2, v - v2, g2 / v if v > 0 else None)
        return (min(g12) if g12 else None), at

    rep = {}
    for ref in ('cc_co', 'cc_co_aeb'):
        plans = []
        for r in rows:
            c = cases[r['case_id']]
            if r['refs'][ref]['grid']['human']['n_break']:
                plans += [(r['case_id'], th) for th in grid if (lambda x: x['valid'] and x['contact'])(X.run(c, th, ref))]
            if r['refs'][ref]['bo']['human']['found']:
                plans.append((r['case_id'], tuple(r['refs'][ref]['bo']['human']['theta'])))
        hit = {f: [bool(X.run(cases[c], th, f)['contact']) for c, th in plans] for f in others if f != ref}
        shapes = [shape(cases[c], th) for c, th in plans]
        g12 = [s[0] for s in shapes if s[0] is not None]
        at = [s[1] for s in shapes if s[1]]
        rep[ref] = dict(plans=len(plans), cases=len({c for c, _ in plans}),
                        avoid_share={f: round(1 - sum(v) / len(v), 4) for f, v in hit.items()},
                        cases_with_a_plan_breaking_idm_d0=len({c for (c, _), h in zip(plans, hit['idm_d0']) if h}),
                        a_b_median=st.median(th[3] for _, th in plans),
                        l1_l2_min_gap_median=st.median(g12), l1_l2_min_gap_below_2m=round(sum(x < 2 for x in g12) / len(g12), 4),
                        at_body_out=dict(gap_to_l2=st.median(x[0] for x in at), closing=st.median(x[1] for x in at),
                                         thw=st.median(x[2] for x in at if x[2])))
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'posthoc':
        posthoc(*sys.argv[2:5])
    else:
        main(*sys.argv[1:4])
