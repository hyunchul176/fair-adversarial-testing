"""B41 frozen analysis: the MiTra (Milan A50) external evaluation of the final agent v2j (B41 protocol sections 4-6, available on request).

Inputs in DIR (written by run_b41.sh):
  final_s{1,2,3}.jsonl        v2j policy stage (fair_guarded2 + window shield + CC margin 0.5 m)
  collide_commit_s{1,2,3}.jsonl, cars_s{1,2,3}.jsonl, dynasto_s{1,2,3}.jsonl   baselines (policy stage, one query)
  margin.jsonl                G4 re-judgement of every fair failure of those files (b23_margin)
  policy_replay.jsonl         CC replays of the final_s* episodes (b45_joint_realism replay --dataset mitra)
  refine.jsonl                bo_warm_g4j (v2j) and bo_warm_g4g (v2, comparison) for all SUTs (b23_search)
  case_pool.json              the sealed cohort
v2j within k queries (one success contract, B46): the policy stage's robust fair failure (front contact, CC clearance
>= 0.5 m) that also passes G2 (CC-replay realism <= 1 IQR) and the natural joint support (5-NN <= tau), or
(k >= 2 and bo_warm_g4j's first success q has 1 + q <= k). v2 (comparison): policy robust, or bo_warm_g4g.

  python -m fairtest.b41_analysis DIR OUT.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

from fairtest.b23_analysis import RANK

DELTA, FRONT, K_FULL = .5, 3.5, 37
LADDER = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
SPECIFIC = ('rss_sut_H0', 'rss_lat_sut_H0', 'drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0')
LATENCY = ('drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0')
LANE_ASSIGN = ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')
N_BOOT, SEED = 10_000, 20260928


def _rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load(d):
    margin = {}
    for r in _rows(d / 'margin.jsonl'):
        margin[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= FRONT
                                                             and r['cc_clearance'] >= DELTA)
    evals = collections.defaultdict(dict)  # agent -> seed -> {(case, sut): metrics}
    for path in sorted(d.glob('*_s[0-9].jsonl')):
        agent, seed = path.stem.rsplit('_s', 1)
        for r in _rows(path):
            if 'valid' not in r:
                continue
            col = bool(r['valid'] and r['contact'])
            judged = not col or 'contact' in r['refs']['cc_r157']
            fair = bool(col and judged and not r['refs']['cc_r157']['contact'])
            evals[agent].setdefault(seed, {})[(r['case_id'], r['sut'])] = dict(
                collision=col, fair=fair, unfair=bool(col and judged and not fair), unjudged=not judged,
                robust=bool(fair and margin.get((path.name, r['case_id'], r['sut']), False)))
    from fairtest.b45_joint_realism import Support
    sup = Support()
    realistic = {}
    for r in _rows(d / 'policy_replay.jsonl'):
        if r['source'] == 'v2_policy':
            sc = sup.score(r['features']) if r['cut_in'] else None
            realistic[(str(r['seed']), r['case_id'], r['sut'])] = bool(sc is not None and sc <= sup.tau and r.get('exceedance') is not None
                                                                      and r['exceedance'] <= 1.)
    refine = collections.defaultdict(dict)  # method -> {(case, sut): first success query or None}
    for r in _rows(d / 'refine.jsonl'):
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            refine[m][(r['case_id'], s)] = None if 'error' in rec else rec.get('first_fair')
    pool = json.loads((d / 'case_pool.json').read_text(encoding='utf-8'))
    return evals, realistic, refine, pool, sup.tau


def rate(per_case):
    return float(np.mean(list(per_case.values()))) if per_case else float('nan')


def boot_diff(a, b, clusters=None, n=N_BOOT, seed=SEED):
    """95% CI of mean(a) - mean(b) over cases; clusters = case -> recording for the cluster bootstrap."""
    cases = sorted(set(a) & set(b))
    diff = np.array([a[c] - b[c] for c in cases], float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = diff[rng.integers(0, len(cases), (n, len(cases)))].mean(1)
    else:
        groups = collections.defaultdict(list)
        for i, c in enumerate(cases):
            groups[clusters[c]].append(i)
        keys = sorted(groups)
        means = np.array([diff[np.concatenate([groups[keys[p]] for p in rng.integers(0, len(keys), len(keys))])].mean()
                          for _ in range(n)])
    return float(diff.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_rate(a, clusters=None):
    zero = {c: 0. for c in a}
    return boot_diff(a, zero, clusters)


def composed(evals, realistic, refine, suts, cases, k, joint=True):
    """{sut: {case: seed-averaged success within k queries}} for v2j (joint=True) or v2."""
    out = {}
    seeds = sorted(evals['final'])
    method = 'bo_warm_g4j' if joint else 'bo_warm_g4g'
    for s in suts:
        per = {}
        for c in cases:
            q = refine[method].get((c, s))
            hits = []
            for x in seeds:
                p1 = evals['final'][x].get((c, s), {}).get('robust', False)
                if joint:
                    p1 = p1 and realistic.get((x, c, s), False)
                hits.append(float(p1 or (k >= 2 and q is not None and 1 + q <= k)))
            per[c] = float(np.mean(hits))
        out[s] = per
    return out


def seed_avg(evals, agent, suts, cases, key):
    seeds = sorted(evals[agent])
    return {s: {c: float(np.mean([evals[agent][x].get((c, s), {}).get(key, False) for x in seeds])) for c in cases} for s in suts}


def main(d, out):
    from scipy.stats import kendalltau
    d = Path(d)
    evals, realistic, refine, pool, tau = load(d)
    cases = sorted(c['case_id'] for c in pool['cases'])
    rec = {c['case_id']: c['event']['recording'] for c in pool['cases']}
    headroom = np.array([c['physical_initial_state'][1]['vx'] - 15. for c in pool['cases']])
    rep = dict(n_cases=len(cases), recordings=len(set(rec.values())), per_recording=dict(collections.Counter(rec.values())),
               joint_tau=tau,
               authority=dict(median=float(np.median(headroom)), p10=float(np.percentile(headroom, 10)),
                              p90=float(np.percentile(headroom, 90))) if len(cases) else None)
    high = bool(len(cases)) and rep['authority']['median'] >= 5.
    if rep['authority']:
        rep['authority']['grade'] = 'high' if high else 'low'
    suts = sorted({s for a in evals.values() for e in a.values() for (_, s) in e})
    comp = {k: composed(evals, realistic, refine, suts, cases, k) for k in (1, 6, K_FULL)}
    comp_v2 = {k: composed(evals, realistic, refine, suts, cases, k, joint=False) for k in (1, 6, K_FULL)}
    rep['v2j'] = {s: {f'k{k}': rate(comp[k][s]) for k in comp} for s in suts}
    rep['v2'] = {s: {f'k{k}': rate(comp_v2[k][s]) for k in comp_v2} for s in suts}
    for agent in evals:
        rep.setdefault('policy', {})[agent] = {s: {m: rate(v) for m, v in
                                                   ((m, seed_avg(evals, agent, [s], cases, m)[s]) for m in ('collision', 'fair', 'robust', 'unfair'))}
                                               for s in suts}
    if len(cases) < 20:
        rep['verdicts'] = 'not tested: fewer than 20 cases (PROTOCOL section 2)'
        Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
        print(json.dumps(rep, indent=1))
        return rep
    full = comp[K_FULL]
    v = {}
    # P1 discrimination (primary)
    thr = .5 if high else .1
    diff, ci = boot_diff(full['following_H0'], full['rss_sut_H0'])
    _, cci = boot_diff(full['following_H0'], full['rss_sut_H0'], clusters=rec)
    v['P1'] = dict(threshold=thr, diff=diff, ci=ci, cluster_ci=cci, supported=diff >= thr)
    # P2 specificity (primary)
    v['P2'] = {s: dict(rate=rate(full[s]), ci=boot_rate(full[s])[1], supported=rate(full[s]) <= .05) for s in SPECIFIC if s in full}
    v['P2']['supported'] = all(x['supported'] for x in v['P2'].values())
    # P3 collision-seeking contrast (primary)
    comp_cols = [m for x in evals['collide_commit'].values() for (c, s), m in x.items() if s in COMPETENT and m['collision']]
    share = (sum(m['unfair'] for m in comp_cols) / len(comp_cols)) if comp_cols else None
    v['P3'] = dict(competent_collisions=len(comp_cols), attacker_share=share, judged=len(comp_cols) >= 10,
                   supported=(share >= .8) if len(comp_cols) >= 10 else None)
    # secondary
    unfair = {s: rate(seed_avg(evals, 'final', [s], cases, 'unfair')[s]) for s in suts}
    v['S1_policy_attacker_at_fault'] = dict(per_sut=unfair, all_zero=all(x == 0. for x in unfair.values()))
    lad = [s for s in LADDER if s in full]
    v['S2_tau'] = float(kendalltau([RANK[s] for s in lad], [rate(full[s]) for s in lad]).statistic)
    la = {c: float(np.mean([full[s][c] for s in LANE_ASSIGN if s in full])) for c in cases}
    lt = {c: float(np.mean([full[s][c] for s in LATENCY if s in full])) for c in cases}
    d3, ci3 = boot_diff(la, lt)
    v['S3_lane_assignment_minus_latency'] = dict(diff=d3, ci=ci3, cluster_ci=boot_diff(la, lt, clusters=rec)[1])
    v['S4_v2j_minus_v2'] = {s: rate(full[s]) - rate(comp_v2[K_FULL][s]) for s in suts}
    col = {s: rate(seed_avg(evals, 'collide_commit', [s], cases, 'collision')[s]) for s in lad}
    v['S5_collide_range'] = dict(collision=col, range=max(col.values()) - min(col.values()))
    for base in ('cars', 'dynasto'):
        if base in evals:
            cc = [m for x in evals[base].values() for (c, s), m in x.items() if s in COMPETENT and m['collision']]
            v[f'S6_{base}_competent'] = dict(collisions=len(cc), attacker_share=(sum(m['unfair'] for m in cc) / len(cc)) if cc else None,
                                            fair_rate={s: rate(seed_avg(evals, base, [s], cases, 'fair')[s]) for s in COMPETENT})
    rep['verdicts'] = v
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(dict(n_cases=rep['n_cases'], authority=rep['authority'],
                          primary={k: v[k].get('supported') for k in ('P1', 'P2', 'P3')}), indent=1))
    return rep


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
