"""B28 frozen analysis: the TGSIM I-294 external evaluation of the final agent (B28 protocol sections 4-6, available on request).

Inputs in DIR (written by run_b28.sh):
  final_s{1,2,3}.jsonl        final agent policy stage (fair_guarded2 + window shield + CC margin 0.5 m)
  noshield_s{1,2,3}.jsonl     the same policy without the shield
  collide_commit_s{1,2,3}.jsonl, cars_s{1,2,3}.jsonl, dynasto_s{1,2,3}.jsonl, naive_fair_s1.jsonl   baselines
  margin.jsonl                G4 re-judgement of every fair failure of those files (b23_margin)
  e3*.jsonl                   refinement runs (b23_search: bo_warm_g4 for all SUTs; bo_cold_g4, random_g4 on the ladder)
  case_pool.json              the sealed cohort (authority: SUT initial speed - 15 m/s)
Robust (G4) = SUT contact with contact_dx >= 3.5 m while the CC avoids with clearance >= 0.5 m.
Final agent within k queries: policy robust, or (k >= 2 and the refinement's first robust query q has 1 + q <= k).

  python -m fairtest.b28_analysis DIR OUT.json
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

from fairtest.b23_analysis import RANK, feints

DELTA, FRONT, K_FULL = .5, 3.5, 37
LADDER = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
SPECIFIC = ('rss_sut_H0', 'rss_lat_sut_H0', 'drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0')
LATENCY = ('drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0')
LANE_ASSIGN = ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')
N_BOOT, SEED = 10_000, 20260927


def _rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load(d):
    margin = {}
    if (d / 'margin.jsonl').exists():
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
            judged = not col or 'contact' in r['refs']['cc_r157']  # deviation D1: a crashed CC replay is unjudged
            fair = bool(col and judged and not r['refs']['cc_r157']['contact'])
            evals[agent].setdefault(seed, {})[(r['case_id'], r['sut'])] = dict(
                collision=col, fair=fair, unfair=bool(col and judged and not fair), unjudged=not judged,
                feint=feints(r['actions']) > 0, robust=bool(fair and margin.get((path.name, r['case_id'], r['sut']), False)))
    refine = collections.defaultdict(dict)  # method -> {(case, sut): first robust query or None}
    for path in sorted(d.glob('e3*.jsonl')):
        for r in _rows(path):
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                refine[m][(r['case_id'], s)] = None if 'error' in rec else rec.get('first_fair')
    pool = json.loads((d / 'case_pool.json').read_text(encoding='utf-8'))
    return evals, refine, pool


def rate(per_case):
    return float(np.mean(list(per_case.values()))) if per_case else float('nan')


def boot_diff(a, b, clusters=None, n=N_BOOT, seed=SEED):
    """95% CI of mean(a) - mean(b) over cases (dicts case -> value); clusters = case -> recording for the cluster bootstrap."""
    cases = sorted(set(a) & set(b))
    diff = np.array([a[c] - b[c] for c in cases], float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(cases), (n, len(cases)))
        means = diff[idx].mean(1)
    else:
        groups = collections.defaultdict(list)
        for i, c in enumerate(cases):
            groups[clusters[c]].append(i)
        keys = sorted(groups)
        means = np.empty(n)
        for j in range(n):
            pick = rng.integers(0, len(keys), len(keys))
            sel = np.concatenate([groups[keys[p]] for p in pick])
            means[j] = diff[sel].mean()
    return float(diff.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def composed(evals, refine, suts, cases, k):
    """{sut: {case: seed-averaged success of the final agent within k queries}}."""
    out = {}
    seeds = sorted(evals['final'])
    for s in suts:
        per = {}
        for c in cases:
            q = refine['bo_warm_g4'].get((c, s))
            hits = []
            for x in seeds:
                p1 = evals['final'][x].get((c, s), {}).get('robust', False)
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
    evals, refine, pool = load(d)
    cases = sorted(c['case_id'] for c in pool['cases'])
    rec = {c['case_id']: c['event']['recording'] for c in pool['cases']}
    headroom = np.array([c['physical_initial_state'][1]['vx'] - 15. for c in pool['cases']])
    rep = dict(n_cases=len(cases), recordings=len(set(rec.values())),
               authority=dict(median=float(np.median(headroom)), p10=float(np.percentile(headroom, 10)),
                              p90=float(np.percentile(headroom, 90))))
    high = rep['authority']['median'] >= 5.
    rep['authority']['grade'] = 'high' if high else 'low'
    suts = sorted({s for a in evals.values() for e in a.values() for (_, s) in e})
    comp = {k: composed(evals, refine, suts, cases, k) for k in (1, 2, 6, K_FULL)}
    rep['final_agent'] = {s: {f'k{k}': rate(comp[k][s]) for k in comp} for s in suts}
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
    # T1a discrimination
    thr = .5 if high else .1
    diff, ci = boot_diff(full['following_H0'], full['rss_sut_H0'])
    _, cci = boot_diff(full['following_H0'], full['rss_sut_H0'], clusters=rec)
    v['T1a'] = dict(threshold=thr, diff=diff, ci=ci, cluster_ci=cci, supported=diff >= thr)
    # T1b ranking
    lad = [s for s in LADDER if s in full]
    tau = float(kendalltau([RANK[s] for s in lad], [rate(full[s]) for s in lad]).statistic)
    v['T1b'] = dict(tau=tau, supported=tau <= -.6)
    # T2 specificity
    v['T2'] = {s: dict(rate=rate(full[s]), supported=rate(full[s]) <= .05) for s in SPECIFIC if s in full}
    # T3 shield
    unfair = {s: rate(seed_avg(evals, 'final', [s], cases, 'unfair')[s]) for s in suts}
    f_final = seed_avg(evals, 'final', ['following_H0'], cases, 'fair')['following_H0']
    f_open = seed_avg(evals, 'noshield', ['following_H0'], cases, 'fair')['following_H0']
    d3, ci3 = boot_diff(f_final, f_open)
    v['T3'] = dict(attacker_at_fault=unfair, all_zero=all(x == 0. for x in unfair.values()), following_H0_diff=d3, ci=ci3,
                   supported=all(x == 0. for x in unfair.values()) and d3 >= -.05)
    # T4 collision-seeking
    col = {s: rate(seed_avg(evals, 'collide_commit', [s], cases, 'collision')[s]) for s in lad}
    comp_cols = [m for x in evals['collide_commit'].values() for (c, s), m in x.items() if s in COMPETENT and m['collision']]
    share = (sum(m['unfair'] for m in comp_cols) / len(comp_cols)) if comp_cols else None
    rng_ = max(col.values()) - min(col.values())
    v['T4'] = dict(collision=col, range=rng_, competent_collisions=len(comp_cols), attacker_share=share,
                   supported=rng_ <= .2 and (share is None or len(comp_cols) < 10 or share >= .8),
                   share_judged=len(comp_cols) >= 10)
    # T5 refinement
    a, b = {}, {}
    for s in ('following_H0', 'following_H0.5', 'following_H1'):
        for c in cases:
            qa, qb = refine['bo_warm_g4'].get((c, s)), refine['bo_cold_g4'].get((c, s))
            a[(s, c)] = float(qa is not None and qa <= 5)
            b[(s, c)] = float(qb is not None and qb <= 5)
    d5, ci5 = boot_diff(a, b)
    v['T5'] = dict(diff=d5, ci=ci5, supported=not ci5[1] < 0.)
    # T6 Goodhart
    nf = evals.get('naive_fair', {})
    rss = [m for x in nf.values() for (c, s), m in x.items() if s == 'rss_sut_H0']
    blame = (sum(m['fair'] for m in rss) / len(rss)) if rss else 0.
    feint = [m['feint'] for m in rss if m['fair']]
    v['T6'] = dict(rss_blame=blame, feint_share=(sum(feint) / len(feint)) if feint else None,
                   judged=blame >= .2, supported=(sum(feint) / len(feint) >= .8) if blame >= .2 and feint else None)
    # T7 learned SUTs
    thr7 = .2 if high else .05
    la = {c: float(np.mean([full[s][c] for s in LANE_ASSIGN if s in full])) for c in cases}
    lt = {c: float(np.mean([full[s][c] for s in LATENCY if s in full])) for c in cases}
    d7, ci7 = boot_diff(la, lt)
    _, cci7 = boot_diff(la, lt, clusters=rec)
    v['T7'] = dict(threshold=thr7, diff=d7, ci=ci7, cluster_ci=cci7, supported=d7 >= thr7)
    rep['verdicts'] = v
    # exploratory: ACC-follower subgroup
    acc = [c['case_id'] for c in pool['cases'] if c.get('acc_follower')]
    rep['acc_follower_subgroup'] = dict(n=len(acc), final_agent_k37={s: (float(np.mean([full[s][c] for c in acc])) if acc else None) for s in suts})
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(dict(n_cases=rep['n_cases'], authority=rep['authority'],
                          verdicts={k: (x.get('supported') if isinstance(x, dict) and 'supported' in x else x) for k, x in v.items()}), indent=1))
    return rep


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
