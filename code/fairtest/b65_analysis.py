"""B65 analysis (B65 design, fixed before the run, available on request).

P(avoid) of an attack under a population = share of its drivers that avoid it (replay errors left out), as B57.
Groups per dataset (development; MiTra post hoc), by the CC replay of this run (it must reproduce the stored verdict):
  v2s      the final agent's reported failures: policy-stage failures + first bo_warm_g4s successes
  v2s_S    the severity-floor arm: policy-stage failures with closing >= 3 m/s + first bo_warm_g4ss successes
  unfair   baseline collisions the CC does not avoid;  fair_base   baseline collisions the CC avoids
Hypotheses on the main population (attentive, 100 drivers):
  HV1  development: median P(avoid) over v2s >= .5
  HV2  MiTra: (a) median over v2s >= .5, (b) median over unfair <= .1, (c) AUC(v2s against unfair) >= .9
  HV3  median over v2s_S >= .5, development and MiTra each
  C1   every CC replay reproduces its stored verdict;  C2  every job carried over from B57 (b57_job) has, for every
       population both runs share, exactly its B57 marks
Bracket att_perceive_1.5 (drivers 1-50 of the main population with perception 1.5 m, later than the CC's 1.095 m):
descriptive, read with the rule of the design, beside the same 50 drivers unchanged (attentive_first50) and at 1.095 m.
Also by stage, SUT class and severity (closing < 3 or >= 3 m/s). Intervals: case-cluster bootstrap (4,000, rng 65).

  python -m fairtest.b65_analysis OUT.json B65_DIR --b57-rejudge B57/rejudge_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest.b57_analysis import auc, boot_cases, describe, p_avoid

MAIN, LATE, CCB = 'attentive', 'att_perceive_1.5', 'att_perceive_1.095'
H_MIN, U_MAX, AUC_MIN, SEV = .5, .1, .9, 3.
CLASSES = {
    'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'),
    'BC human': ('bc_human_H0',), 'ADS 2.0 m': ('ads@2.0_H0',),
    'learned lane-assignment': ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
    'competent': ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
                  'drl_sut@3_H0', 'drl_sut@6_H0'),
}
STAGES = ('policy', 'refine_g4s', 'refine_g4ss')


def rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def groups(r):
    out = []
    if r['attacker'] == 'v2s':
        if r['stage'] in ('policy', 'refine_g4s'):
            out.append('v2s')
        if r['stage'] == 'refine_g4ss' or (r['stage'] == 'policy' and r.get('closing') is not None and r['closing'] >= SEV):
            out.append('v2s_S')
    else:
        out.append('unfair' if r['cc_contact'] else 'fair_base')
    return out


def marks(r, label):
    if label == 'attentive_first50':
        m = r['collide'].get(MAIN)
        return None if m is None else m[:50]
    return r['collide'].get(label)


def analyse(rs, label, boot):
    vals = {}
    for r in rs:
        m = marks(r, label)
        p = None if m is None else p_avoid(m)
        if p is not None:
            for g in r['g']:
                vals.setdefault(g, []).append(dict(r, p=p))
    G = {g: vals.get(g, []) for g in ('v2s', 'v2s_S', 'unfair', 'fair_base')}
    rep = dict(groups={g: describe([x['p'] for x in v]) for g, v in G.items()})
    for g in ('v2s', 'v2s_S'):
        rep[f'{g}_auc_vs_unfair'] = None if not G[g] or not G['unfair'] else round(auc([x['p'] for x in G[g]], [x['p'] for x in G['unfair']]), 4)
    rep['v2s_by_stage'] = {st: describe([x['p'] for x in G['v2s'] + G['v2s_S'] if x['stage'] == st]) for st in STAGES}
    # a policy failure can sit in both groups; count each attack once in the class and severity splits
    once = {(x['stage'], x['seed'], x['case_id'], x['sut']): x for x in G['v2s'] + G['v2s_S']}
    rep['v2s_by_class'] = {c: describe([x['p'] for x in once.values() if x['sut'] in s]) for c, s in CLASSES.items()}
    rep['v2s_by_severity'] = {
        f'{st}|{"severe" if sev else "light"}': describe([x['p'] for x in once.values() if x['stage'] == st and x.get('closing') is not None
                                                         and (x['closing'] >= SEV) == sev])
        for st in STAGES for sev in (False, True)}
    rep['by_attacker'] = {f'{a}|{g}': describe([x['p'] for x in G[g] if x['attacker'] == a])
                          for g in ('unfair', 'fair_base') for a in sorted({x['attacker'] for x in G[g]})}
    if boot:
        med = lambda xs: float(np.median([x['p'] for x in xs])) if xs else None  # noqa: E731
        rep['ci'] = {g: boot_cases(G[g], med, seed=65) for g in ('v2s', 'v2s_S', 'unfair')}
        for g in ('v2s', 'v2s_S'):
            area = lambda xs, g=g: auc([x['p'] for x in xs if g in x['g']], [x['p'] for x in xs if 'unfair' in x['g']])  # noqa: E731
            both = {id(x): x for x in G[g] + G['unfair']}
            rep['ci'][f'{g}_auc_vs_unfair'] = boot_cases(list(both.values()), area, seed=65)
    return rep


def check_b57(rs, b57_path):
    old = {r['job']: r for r in rows(b57_path)}
    n = same = 0
    diff = []
    for r in rs:
        if r.get('b57_job') is None:
            continue
        o = old.get(r['b57_job'])
        if o is None or (o['case_id'], o['sut']) != (r['case_id'], r['sut']):
            diff.append([r['job'], 'missing_or_mismatched_job'])
            continue
        for label in set(r['collide']) & set(o['collide']):
            n += 1
            if r['collide'][label] == o['collide'][label]:
                same += 1
            else:
                diff.append([r['job'], label])
    return dict(compared=n, identical=same, differences=diff[:20], supported=bool(n and same == n and not diff))


def main(a):
    d = Path(a.dir)
    rep = {}
    for ds in ('dev', 'mitra'):
        raw = rows(d / f'rejudge_{ds}.jsonl')
        jobs = {j['job']: j for j in rows(d / f'jobs_{ds}.jsonl')}
        rs = []
        for r in raw:
            j = jobs[r['job']]
            rs.append(dict(r, closing=j.get('closing'), b57_job=j.get('b57_job')))
        e = dict(jobs=len(jobs), rejudged=len(rs), cc_errors=sum('cc_error' in r for r in rs),
                 cc_matches_stored=sum(bool(r.get('cc_matches_stored')) for r in rs),
                 driver_errors={lab: sum(r['collide'].get(lab, '').count('e') for r in rs) for lab in (MAIN, 'less_attentive', CCB, LATE)})
        e['C1'] = dict(supported=bool(e['rejudged'] == e['jobs'] and e['cc_errors'] == 0 and e['cc_matches_stored'] == e['jobs']))
        if ds == 'dev':
            e['C2'] = check_b57(rs, a.b57_rejudge)
        rs = [dict(r, g=groups(r)) for r in rs if 'cc_error' not in r]
        e['populations'] = {lab: analyse(rs, lab, lab == MAIN) for lab in (MAIN, 'less_attentive', 'attentive_first50', CCB, LATE)}
        rep[ds] = e
    m = {ds: rep[ds]['populations'][MAIN] for ds in rep}

    def med(ds, g):
        return m[ds]['groups'][g].get('median')
    rep['HV1'] = dict(median=med('dev', 'v2s'), ci=m['dev']['ci']['v2s'], supported=bool(med('dev', 'v2s') is not None and med('dev', 'v2s') >= H_MIN))
    a_ = m['mitra']['v2s_auc_vs_unfair']
    rep['HV2'] = dict(v2s_median=med('mitra', 'v2s'), v2s_ci=m['mitra']['ci']['v2s'], unfair_median=med('mitra', 'unfair'),
                      unfair_ci=m['mitra']['ci']['unfair'], auc=a_, auc_ci=m['mitra']['ci']['v2s_auc_vs_unfair'],
                      a=bool(med('mitra', 'v2s') is not None and med('mitra', 'v2s') >= H_MIN),
                      b=bool(med('mitra', 'unfair') is not None and med('mitra', 'unfair') <= U_MAX),
                      c=bool(a_ is not None and a_ >= AUC_MIN))
    rep['HV2']['supported'] = rep['HV2']['a'] and rep['HV2']['b'] and rep['HV2']['c']
    rep['HV3'] = {ds: dict(median=med(ds, 'v2s_S'), ci=m[ds]['ci']['v2s_S'], supported=bool(med(ds, 'v2s_S') is not None and med(ds, 'v2s_S') >= H_MIN))
                  for ds in rep if ds in ('dev', 'mitra')}
    rep['late_perception'] = {ds: {lab: dict(v2s=rep[ds]['populations'][lab]['groups']['v2s'].get('median'),
                                             v2s_S=rep[ds]['populations'][lab]['groups']['v2s_S'].get('median'),
                                             unfair=rep[ds]['populations'][lab]['groups']['unfair'].get('median'),
                                             auc=rep[ds]['populations'][lab]['v2s_auc_vs_unfair'])
                                   for lab in ('attentive_first50', CCB, LATE)} for ds in ('dev', 'mitra')}
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for ds in ('dev', 'mitra'):
        e = rep[ds]
        print(ds, {k: e[k] for k in ('jobs', 'rejudged', 'cc_errors', 'cc_matches_stored', 'driver_errors', 'C1')}, e.get('C2', ''))
        for lab, p in e['populations'].items():
            print(f'  {lab:18s}', {g: p['groups'][g].get('median') for g in ('v2s', 'v2s_S', 'unfair', 'fair_base')},
                  'AUC', p['v2s_auc_vs_unfair'], p['v2s_S_auc_vs_unfair'])
    print('HV1', rep['HV1'])
    print('HV2', rep['HV2'])
    print('HV3', rep['HV3'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('dir')
    ap.add_argument('--b57-rejudge', required=True)
    main(ap.parse_args())
