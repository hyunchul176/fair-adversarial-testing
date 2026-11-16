"""exiD frozen analysis (exiD protocol draft §5-6, available on request): the final agent v2s = the v2j policy stage + the screened
refinement bo_warm_g4s; the earlier version v2j (unscreened bo_warm_g4j) is the ablation. The B41 definitions are imported
unchanged from b41_analysis (load, intervals, rates); only the refinement method of the composition is chosen here.

Inputs in DIR (B41 layout): final_s{1,2,3}.jsonl, collide_commit_s*, cars_s*, dynasto_s* (policy stage), margin.jsonl,
policy_replay.jsonl, case_pool.json; REFINE files (b23_search --log-queries --split-suts) with bo_warm_g4s and
bo_warm_g4j; CONSENSUS (exid_consensus.py) for the reported failures on the competent SUTs.

  primary   P1  following H0 - longitudinal RSS (v2s, 37 queries) >= 50 points (high authority) or >= 10 (low)
            P2  for each of SPECIFIC: share of cases with a reported failure that is not consensus-fair <= 5%
                (an unjudged reported failure counts as not consensus-fair; their number is reported)
            P3  collision-seeking: >= 80% of its collisions with AEB H0/H0.5 and RSS attacker-caused (>= 10 collisions)
  secondary S1 policy-stage attacker-caused (implementation check); S2 Kendall tau on the ladder; S3 lane assignment -
            delay (learned); S4 screening effect v2s - v2j (k6, k37; DETECTABLE mean with case bootstrap); S4' weak
            rule-based 4: v2s - v2j lower bound > -5 points; S5 collision-seeking range; S6 CARS/Dynasto on competent
            SUTs; S12 executed inadmissible queries (v2s must be 0), C replays per query, stops; S13 per-case consensus on
            the competent SUTs; S14 the B41 definition of P2 (rate <= 5%) for v2s and v2j

  python -m fairtest.exid_analysis DIR OUT.json --refine R.jsonl [...] --consensus CONS.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest.b23_analysis import RANK
from fairtest.b41_analysis import (COMPETENT, K_FULL, LADDER, LANE_ASSIGN, LATENCY, SPECIFIC, boot_diff, boot_rate,
                                    load, rate, seed_avg)

V2S, V2J = 'bo_warm_g4s', 'bo_warm_g4j'
DETECTABLE = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'bc_human_H0', 'ads@2.0_H0',
              'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')
WEAK4 = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
COMPETENT8 = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
              'drl_sut@3_H0', 'drl_sut@6_H0')
P2_MAX, S4P_MIN = .05, -.05


def load_refine(paths):
    """{method: {(case, sut): first success or None}}, {method: {(case, sut): record}}, errors, conflicts (the same
    (case, SUT, method) in two files with different first successes)."""
    first, recs = collections.defaultdict(dict), collections.defaultdict(dict)
    errors, conflicts = collections.Counter(), []
    for path in paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                k = (r['case_id'], s)
                v = None if 'error' in rec else rec.get('first_fair')
                errors[m] += 'error' in rec
                if k in first[m] and first[m][k] != v:
                    conflicts.append([m, r['case_id'], s, first[m][k], v])
                first[m][k] = v
                if 'error' not in rec:
                    recs[m][k] = rec
    return first, recs, dict(errors), conflicts


def composed(evals, realistic, first, method, suts, cases, k):
    """{sut: {case: seed-averaged success within k queries}}: the policy stage's robust failure inside the joint support
    (with G2), or (k >= 2 and the refinement's first success q has 1 + q <= k). As b41_analysis.composed(joint=True)."""
    seeds = sorted(evals['final'])
    out = {}
    for s in suts:
        per = {}
        for c in cases:
            q = first[method].get((c, s))
            per[c] = float(np.mean([float((evals['final'][x].get((c, s), {}).get('robust', False) and realistic.get((x, c, s), False))
                                          or (k >= 2 and q is not None and 1 + q <= k)) for x in seeds]))
        out[s] = per
    return out


def load_consensus(path):
    pol, ref, rows = {}, {}, []
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        rows.append(r)
        cf = None if 'error' in r else bool(r.get('consensus_fair'))
        if r['stage'] == 'policy':
            pol[(str(r['seed']), r['case_id'], r['sut'])] = cf
        else:
            ref[(r['case_id'], r['sut'])] = cf
    return pol, ref, rows


def not_consensus(evals, realistic, first, pol, ref, s, cases, method=V2S, k=K_FULL):
    """{case: seed-averaged indicator that the reported failure is not consensus-fair}, unjudged count."""
    seeds = sorted(evals['final'])
    per, unjudged = {}, 0
    for c in cases:
        q = first[method].get((c, s))
        vals = []
        for x in seeds:
            if evals['final'][x].get((c, s), {}).get('robust', False) and realistic.get((x, c, s), False):
                cf = pol.get((x, c, s))
            elif k >= 2 and q is not None and 1 + q <= k:
                cf = ref.get((c, s))
            else:
                vals.append(0.)
                continue
            unjudged += cf is None
            vals.append(0. if cf else 1.)
        per[c] = float(np.mean(vals))
    return per, unjudged


def query_stats(recs, method, suts):
    e = dict(searches=0, executed=0, inadmissible=0, sut_contacts_inadmissible=0, cc_contact_among_them=0, cc_replays=0, stops={})
    for (c, s), rec in recs[method].items():
        if s not in suts:
            continue
        e['searches'] += 1
        qs = rec.get('queries', [])
        e['executed'] += len(qs)
        bad = [q for q in qs if not q['admissible']]
        e['inadmissible'] += len(bad)
        e['sut_contacts_inadmissible'] += sum(bool(q['sut_contact']) for q in bad)
        e['cc_contact_among_them'] += sum(bool(q['sut_contact'] and q['cc_contact']) for q in bad)
        e['cc_replays'] += rec.get('cc_replays', 0) or 0
        if rec.get('stop'):
            e['stops'][rec['stop']] = e['stops'].get(rec['stop'], 0) + 1
    e['logged'] = e['executed'] > 0
    e['cc_replays_per_query'] = round(e['cc_replays'] / e['executed'], 2) if e['executed'] and e['cc_replays'] else None  # g4j: no screening
    return e


def main(a):
    from scipy.stats import kendalltau
    d = Path(a.dir)
    evals, realistic, _, pool, tau = load(d)
    first, recs, ref_errors, conflicts = load_refine(a.refine)
    cases = sorted(c['case_id'] for c in pool['cases'])
    rec_of = {c['case_id']: c['event']['recording'] for c in pool['cases']}
    headroom = np.array([c['physical_initial_state'][1]['vx'] - 15. for c in pool['cases']])
    rep = dict(n_cases=len(cases), recordings=len(set(rec_of.values())), joint_tau=tau, refine_errors=ref_errors,
               refine_conflicts=conflicts[:50], n_refine_conflicts=len(conflicts),
               authority=dict(median=float(np.median(headroom)), p10=float(np.percentile(headroom, 10)),
                              p90=float(np.percentile(headroom, 90))) if len(cases) else None)
    high = bool(len(cases)) and rep['authority']['median'] >= 5.
    if rep['authority']:
        rep['authority']['grade'] = 'high' if high else 'low'
    suts = sorted({s for e in evals['final'].values() for (_, s) in e})
    comp = {m: {k: composed(evals, realistic, first, m, suts, cases, k) for k in (1, 6, K_FULL)} for m in (V2S, V2J)}
    rep['v2s'] = {s: {f'k{k}': rate(comp[V2S][k][s]) for k in (1, 6, K_FULL)} for s in suts}
    rep['v2j'] = {s: {f'k{k}': rate(comp[V2J][k][s]) for k in (1, 6, K_FULL)} for s in suts}
    for agent in evals:
        rep.setdefault('policy', {})[agent] = {s: {m: rate(seed_avg(evals, agent, [s], cases, m)[s])
                                                   for m in ('collision', 'fair', 'robust', 'unfair')} for s in suts}
    if len(cases) < 20:
        rep['verdicts'] = 'not tested: fewer than 20 cases (PROTOCOL section 2)'
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
        return rep
    full, fullj = comp[V2S][K_FULL], comp[V2J][K_FULL]
    v = {}
    thr = .5 if high else .1
    diff, ci = boot_diff(full['following_H0'], full['rss_sut_H0'])
    _, cci = boot_diff(full['following_H0'], full['rss_sut_H0'], clusters=rec_of)
    v['P1'] = dict(threshold=thr, diff=diff, ci=ci, cluster_ci=cci, supported=diff >= thr)
    pol, ref, rows = load_consensus(a.consensus) if a.consensus else ({}, {}, [])
    p2 = {}
    for s in SPECIFIC:
        if s in full:
            per, unj = not_consensus(evals, realistic, first, pol, ref, s, cases)
            r_, (lo, hi) = boot_rate(per)
            p2[s] = dict(not_consensus_rate=r_, ci=[lo, hi], unjudged=unj, supported=r_ <= P2_MAX, upper_le_5=hi <= P2_MAX)
    v['P2'] = dict(per_sut=p2, consensus_rows=len(rows), consensus_errors=sum('error' in r for r in rows),
                   supported=bool(p2) and all(x['supported'] for x in p2.values()))
    comp_cols = [m for x in evals['collide_commit'].values() for (c, s), m in x.items() if s in COMPETENT and m['collision']]
    share = (sum(m['unfair'] for m in comp_cols) / len(comp_cols)) if comp_cols else None
    v['P3'] = dict(competent_collisions=len(comp_cols), attacker_share=share, judged=len(comp_cols) >= 10,
                   supported=(share >= .8) if len(comp_cols) >= 10 else None)
    unfair = {s: rate(seed_avg(evals, 'final', [s], cases, 'unfair')[s]) for s in suts}
    v['S1_policy_attacker_at_fault'] = dict(per_sut=unfair, all_zero=all(x == 0. for x in unfair.values()))
    lad = [s for s in LADDER if s in full]
    v['S2_tau'] = float(kendalltau([RANK[s] for s in lad], [rate(full[s]) for s in lad]).statistic)
    la = {c: float(np.mean([full[s][c] for s in LANE_ASSIGN if s in full])) for c in cases}
    lt = {c: float(np.mean([full[s][c] for s in LATENCY if s in full])) for c in cases}
    d3, ci3 = boot_diff(la, lt)
    v['S3_lane_assignment_minus_latency'] = dict(diff=d3, ci=ci3, cluster_ci=boot_diff(la, lt, clusters=rec_of)[1])
    s4 = {}
    for k in (6, K_FULL):
        s4[f'k{k}'] = {s: rate(comp[V2S][k][s]) - rate(comp[V2J][k][s]) for s in DETECTABLE if s in full}
    det = [s for s in DETECTABLE if s in full]
    mean_s = {c: float(np.mean([full[s][c] for s in det])) for c in cases}
    mean_j = {c: float(np.mean([fullj[s][c] for s in det])) for c in cases}
    s4['detectable_mean_k37'] = dict(zip(('diff', 'ci'), boot_diff(mean_s, mean_j)))
    v['S4_screening_effect'] = s4
    w = [s for s in WEAK4 if s in full]
    ws = {c: float(np.mean([full[s][c] for s in w])) for c in cases}
    wj = {c: float(np.mean([fullj[s][c] for s in w])) for c in cases}
    d4, ci4 = boot_diff(ws, wj)
    v["S4'_noninferiority"] = dict(diff=d4, ci=ci4, margin=S4P_MIN, supported=ci4[0] > S4P_MIN)
    col = {s: rate(seed_avg(evals, 'collide_commit', [s], cases, 'collision')[s]) for s in lad}
    v['S5_collide_range'] = dict(collision=col, range=max(col.values()) - min(col.values()))
    for base in ('cars', 'dynasto'):
        if base in evals:
            cc = [m for x in evals[base].values() for (c, s), m in x.items() if s in COMPETENT and m['collision']]
            v[f'S6_{base}_competent'] = dict(collisions=len(cc), attacker_share=(sum(m['unfair'] for m in cc) / len(cc)) if cc else None,
                                            fair_rate={s: rate(seed_avg(evals, base, [s], cases, 'fair')[s]) for s in COMPETENT})
    v['S12_execution'] = {m: query_stats(recs, m, suts) for m in (V2S, V2J)}
    v['S12_execution']['v2s_inadmissible_zero'] = v['S12_execution'][V2S]['logged'] and v['S12_execution'][V2S]['inadmissible'] == 0
    s13 = {}
    for s in COMPETENT8:
        rs = [r for r in rows if r['sut'] == s]
        s13[s] = dict(v2s_k37=rate(full[s]) if s in full else None, judged=len(rs),
                      consensus_fair=sum(bool(r.get('consensus_fair')) for r in rs if 'error' not in r),
                      grades=dict(collections.Counter(r.get('grade') for r in rs if 'error' not in r)),
                      errors=sum('error' in r for r in rs))
    v['S13_competent'] = s13
    v['S14_B41_P2'] = {m: {s: dict(rate=rate(c[K_FULL][s]), le_5=rate(c[K_FULL][s]) <= P2_MAX) for s in SPECIFIC if s in c[K_FULL]}
                       for m, c in (('v2s', comp[V2S]), ('v2j', comp[V2J]))}
    rep['verdicts'] = v
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(dict(n_cases=rep['n_cases'], authority=rep['authority'], conflicts=len(conflicts),
                          primary={k: v[k].get('supported') for k in ('P1', 'P2', 'P3')}), indent=1))
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('out')
    ap.add_argument('--refine', nargs='+', required=True, help='b23_search outputs with bo_warm_g4s and bo_warm_g4j')
    ap.add_argument('--consensus', help='exid_consensus.py output (P2, S13)')
    main(ap.parse_args())
