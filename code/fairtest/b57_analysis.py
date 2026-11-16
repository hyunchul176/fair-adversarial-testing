"""B57 analysis (B57 design, fixed before the run, available on request).

P(avoid) of an attack under a population = share of that population's drivers that avoid it (replay errors left out).
Groups (by the CC replay of this run, which must reproduce the stored verdict):
  v2j          every attack v2j reports on the highway-env SUTs (policy episodes + refinement plans; CC avoids all)
  autoware     every v2j failure reported against Autoware in B54 (CC avoids all)
  unfair       every baseline collision the CC does not avoid (attacker-caused by the CC verdict)
  fair_base    every baseline collision the CC avoids
Hypotheses on the main population (MAIN):
  H1  median P(avoid) over v2j >= H1_MIN            (most human-like drivers avoid what the agent reports)
  H2  median P(avoid) over unfair <= H2_MAX         (CC-unavoidable collisions are unavoidable for most humans too)
  H3  AUC of P(avoid), v2j against unfair, >= H3_MIN
  check  every CC replay reproduces its stored verdict
The same statistics are reported, without verdicts, for every other population (less attentive; paired brackets on the
flagged subset). Intervals: case-cluster bootstrap (4,000, rng 57). Descriptive: quantiles and shares per group and SUT
class, per-driver avoidance, Spearman correlation of P(avoid) with the CC clearance.

  python -m fairtest.b57_analysis OUT.json REJUDGE.jsonl
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

MAIN = 'attentive'
H1_MIN, H2_MAX, H3_MIN = .5, .1, .9
CLASSES = {
    'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'),
    'BC human': ('bc_human_H0',), 'ADS 2.0 m': ('ads@2.0_H0',),
    'learned lane-assignment': ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
    'Autoware': ('aw_default', 'aw_r157'),
}
GROUPS = ('v2j', 'autoware', 'unfair', 'fair_base')


def p_avoid(marks):
    ok, hit = marks.count('0'), marks.count('1')
    return ok / (ok + hit) if ok + hit else None


def group(r):
    if r['attacker'] == 'v2j':
        return 'autoware' if r['stage'].startswith('autoware') else 'v2j'
    return 'unfair' if r['cc_contact'] else 'fair_base'


def boot_cases(rows, stat, n=4000, seed=57):
    by = collections.defaultdict(list)
    for r in rows:
        by[r['case_id']].append(r)
    keys = sorted(by)
    if not keys:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = [by[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        v = stat([r for rs in pick for r in rs])
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind='stable')
    ranks[order] = np.arange(len(allv), dtype=float)
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def describe(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4), mean=round(float(v.mean()), 4),
                quantiles={q: round(float(np.quantile(v, float(q))), 4) for q in ('0.1', '0.25', '0.75', '0.9')},
                share_ge_0_5=round(float((v >= .5).mean()), 4), share_ge_0_8=round(float((v >= .8).mean()), 4),
                share_ge_0_95=round(float((v >= .95).mean()), 4), share_le_0_1=round(float((v <= .1).mean()), 4))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


def analyse(rows, label, verdicts):
    rows = [dict(r, p=p_avoid(r['collide'][label])) for r in rows if label in r['collide']]
    G = {g: [r for r in rows if r['g'] == g and r['p'] is not None] for g in GROUPS}
    rep = dict(attacks=len(rows), drivers=len(rows[0]['collide'][label]) if rows else 0,
               driver_errors=sum(r['collide'][label].count('e') for r in rows),
               groups={g: describe([r['p'] for r in rs]) for g, rs in G.items()})
    rep['by_attacker'] = {}
    for att in sorted({r['attacker'] for r in rows if r['attacker'] != 'v2j'}):
        for g in ('unfair', 'fair_base'):
            vals = [r['p'] for r in G[g] if r['attacker'] == att]
            if vals:
                rep['by_attacker'][f'{att}|{g}'] = describe(vals)
    rep['v2j_by_class'] = {c: describe([r['p'] for r in G['v2j'] + G['autoware'] if r['sut'] in s]) for c, s in CLASSES.items()}
    rep['v2j_by_stage'] = {st: describe([r['p'] for r in G['v2j'] + G['autoware'] if r['stage'] == st])
                           for st in ('policy', 'refine', 'autoware_policy', 'autoware_refine')}
    med = lambda rs: float(np.median([r['p'] for r in rs])) if rs else None  # noqa: E731
    area = lambda rs: auc([r['p'] for r in rs if r['g'] == 'v2j'], [r['p'] for r in rs if r['g'] == 'unfair'])  # noqa: E731
    rep['H1'] = dict(median=rep['groups']['v2j'].get('median'), ci=boot_cases(G['v2j'], med))
    rep['H2'] = dict(median=rep['groups']['unfair'].get('median'), ci=boot_cases(G['unfair'], med))
    a = area(G['v2j'] + G['unfair'])
    rep['H3'] = dict(auc=None if a is None else round(a, 4), ci=boot_cases(G['v2j'] + G['unfair'], area))
    if verdicts:
        rep['H1'].update(threshold=H1_MIN, supported=bool(rep['H1']['median'] is not None and rep['H1']['median'] >= H1_MIN))
        rep['H2'].update(threshold=H2_MAX, supported=bool(rep['H2']['median'] is not None and rep['H2']['median'] <= H2_MAX))
        rep['H3'].update(threshold=H3_MIN, supported=bool(a is not None and a >= H3_MIN))
    aw = auc([r['p'] for r in G['autoware']], [r['p'] for r in G['unfair']])
    rep['autoware_vs_unfair_auc'] = None if aw is None else round(aw, 4)
    n = rep['drivers']
    dv = [np.mean([r['collide'][label][k] == '0' for r in G['v2j']]) for k in range(n)] if G['v2j'] else []
    du = [np.mean([r['collide'][label][k] == '0' for r in G['unfair']]) for k in range(n)] if G['unfair'] else []
    rep['per_driver'] = dict(v2j_avoided=describe(dv), unfair_avoided=describe(du))
    rep['spearman_p_vs_cc_clearance'] = {g: spearman([r['p'] for r in G[g] if r.get('cc_clearance') is not None],
                                                     [r['cc_clearance'] for r in G[g] if r.get('cc_clearance') is not None])
                                         for g in ('v2j', 'autoware', 'fair_base')}
    return rep


def main(out, path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    rep = dict(jobs=len(rows), cc_errors=sum('cc_error' in r for r in rows),
               cc_matches_stored=sum(bool(r.get('cc_matches_stored')) for r in rows))
    rep['check_supported'] = rep['cc_errors'] == 0 and rep['cc_matches_stored'] == len(rows)
    rows = [dict(r, g=group(r)) for r in rows if 'cc_error' not in r]
    labels = []
    for r in rows:
        for label in r['collide']:
            if label not in labels:
                labels.append(label)
    rep['populations'] = {label: analyse(rows, label, label == MAIN) for label in labels}
    # the main population on the bracket subset, the reference of the paired brackets
    sub = [r for r in rows if r.get('bracket')]
    rep['main_on_bracket_subset'] = analyse(sub, MAIN, False)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    m = rep['populations'].get(MAIN, {})
    print(json.dumps({k: rep[k] for k in ('jobs', 'cc_errors', 'cc_matches_stored', 'check_supported')}, indent=1))
    print(json.dumps({k: m.get(k) for k in ('H1', 'H2', 'H3')}, indent=1))
    for label, e in rep['populations'].items():
        print(label, {g: e['groups'][g].get('median') for g in GROUPS}, 'AUC', e['H3']['auc'])


if __name__ == '__main__':
    main(*sys.argv[1:3])
