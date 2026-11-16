"""B68 analysis (B68 design, fixed before the run, available on request).

Inputs: the B65 job sets (development; MiTra post hoc) re-judged by four reference drivers (b68_cars_refs): cc_r157 (our C,
must reproduce the stored verdicts), cc_jrc (CARS/JRC reading of the R157 CC), ref_fsm (our FSM), fsm_cars (CARS FSM).
Groups as B65: v2s (policy-stage reported failures + first bo_warm_g4s successes), v2s_S (policy failures closing >= 3 m/s +
first bo_warm_g4ss successes), unfair / fair_base (baseline collisions C does not / does avoid).
  checks  C1 the cc_r157 pass reproduces every stored C verdict; M1 every replay of a port ran the port (trace model)
  JR      share of v2s reported failures cc_jrc also avoids, development and MiTra; read in tiers fixed before the run:
          >= .8 robust to the trigger reading, .5-.8 partly, < .5 reading-dependent
  FR      the same for fsm_cars (and, for comparison, ref_fsm)
  P1      lower bounds of the final agent's P1 at 37 queries when a reported failure also has to be avoided by cc_jrc, by
          fsm_cars, and by both (a failure the other reading rejects is dropped; the search is not re-run)
Shares per SUT class and per stage are descriptive. Intervals: case bootstrap as b41_analysis.

  python -m fairtest.b68_analysis OUT.json B68_DIR --b65 B65_DIR --dev-dir DEV_STANDIN --mitra-dir B41_DIR \
         --dev-refine B56_DEV.jsonl --mitra-refine B56_MITRA.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b41_analysis import K_FULL, boot_diff, load
from fairtest.exid_analysis import load_refine

VARIANTS = ('cc_r157', 'cc_jrc', 'ref_fsm', 'fsm_cars')
PORT_MODEL = dict(cc_jrc='cc_jrc', fsm_cars='fsm_cars', ref_fsm='ref_fsm')
CLASSES = {
    'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'),
    'BC human': ('bc_human_H0',), 'ADS 2.0 m': ('ads@2.0_H0',),
    'learned lane-assignment': ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
    'competent': ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
                  'drl_sut@3_H0', 'drl_sut@6_H0'),
}


def rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def tier(x):
    return None if x is None else ('robust' if x >= .8 else ('partly' if x >= .5 else 'reading-dependent'))


def groups(j):
    out = []
    if j['attacker'] == 'v2s':
        if j['stage'] in ('policy', 'refine_g4s'):
            out.append('v2s')
        if j['stage'] == 'refine_g4ss' or (j['stage'] == 'policy' and j.get('closing') is not None and j['closing'] >= 3.):
            out.append('v2s_S')
    else:
        out.append('unfair' if j['stored_cc'] else 'fair_base')
    return out


def share(vals):
    v = [x for x in vals if x is not None]
    return dict(n=len(v), avoided=round(float(np.mean(v)), 4) if v else None)


def key(j):
    return (j['stage'], None if j['seed'] is None else str(j['seed']), j['case_id'], j['sut'])


def analyse(ds, b68, b65, comp_dir, refine_path):
    jobs = {j['job']: j for j in rows(Path(b65) / f'jobs_{ds}.jsonl')}
    res = {v: {r['job']: r for r in rows(Path(b68) / f'rejudge_{v}_{ds}.jsonl')} for v in VARIANTS}
    rep = dict(jobs=len(jobs), rejudged={v: len(res[v]) for v in VARIANTS},
               errors={v: sum('error' in r for r in res[v].values()) for v in VARIANTS})
    c1 = [i for i, j in jobs.items() if i not in res['cc_r157'] or 'error' in res['cc_r157'][i]
          or res['cc_r157'][i]['contact'] != bool(j['stored_cc'])]
    rep['C1'] = dict(mismatches=len(c1), rows=c1[:10], supported=not c1)
    m1 = {v: sum(1 for r in res[v].values() if 'error' not in r and r.get('models') != [PORT_MODEL[v]]) for v in PORT_MODEL}
    rep['M1'] = dict(rows_without_the_port=m1, supported=not any(m1.values()))
    avoid = {v: {i: (None if 'error' in r else not r['contact']) for i, r in res[v].items()} for v in VARIANTS}
    G = collections.defaultdict(list)
    for i, j in jobs.items():
        for g in groups(j):
            G[g].append(i)
    rep['groups'] = {g: {v: share([avoid[v].get(i) for i in ids]) for v in VARIANTS} for g, ids in G.items()}
    for g, ids in G.items():
        both = [None if avoid['cc_jrc'].get(i) is None or avoid['fsm_cars'].get(i) is None else avoid['cc_jrc'][i] and avoid['fsm_cars'][i]
                for i in ids]
        rep['groups'][g]['cc_jrc_and_fsm_cars'] = share(both)
    once = {key(jobs[i]): i for g in ('v2s', 'v2s_S') for i in G[g]}
    rep['v2s_by_class'] = {c: {v: share([avoid[v].get(i) for k, i in once.items() if k[3] in s]) for v in ('cc_jrc', 'fsm_cars')}
                           for c, s in CLASSES.items()}
    rep['v2s_by_stage'] = {st: {v: share([avoid[v].get(i) for k, i in once.items() if k[0] == st]) for v in ('cc_jrc', 'fsm_cars', 'ref_fsm')}
                           for st in ('policy', 'refine_g4s', 'refine_g4ss')}
    jr = rep['groups']['v2s']['cc_jrc']['avoided']
    fr = rep['groups']['v2s']['fsm_cars']['avoided']
    rep['JR'] = dict(avoided=jr, tier=tier(jr))
    rep['FR'] = dict(avoided=fr, tier=tier(fr), ref_fsm=rep['groups']['v2s']['ref_fsm']['avoided'])
    # P1 lower bounds: a reported failure counts only if the other reading(s) also avoid it
    evals, realistic, _, pool, _ = load(Path(comp_dir))
    first = load_refine([refine_path])[0]['bo_warm_g4s']
    cases = sorted(c['case_id'] for c in pool['cases'])
    seeds = sorted(evals['final'])
    by_key = {key(j): i for i, j in jobs.items() if j['attacker'] == 'v2s' and j['stage'] in ('policy', 'refine_g4s')}

    def ok(k, need):
        i = by_key.get(k)
        return i is not None and all(avoid[v].get(i) is True for v in need)

    def comp(need):
        out = {}
        for s in ('following_H0', 'rss_sut_H0'):
            per = {}
            for c in cases:
                q = first.get((c, s))
                hits = []
                for x in seeds:
                    pol = bool(evals['final'][x].get((c, s), {}).get('robust') and realistic.get((x, c, s), False))
                    hits.append(float((pol and ok(('policy', x, c, s), need))
                                      or (q is not None and 1 + q <= K_FULL and ok(('refine_g4s', None, c, s), need))))
                per[c] = float(np.mean(hits))
            out[s] = per
        p1, ci = boot_diff(out['following_H0'], out['rss_sut_H0'])
        return dict(P1=round(p1, 4), ci=[round(v, 4) for v in ci])
    rep['P1'] = {name: comp(need) for name, need in (('C (stored)', ('cc_r157',)), ('C and cc_jrc', ('cc_r157', 'cc_jrc')),
                                                    ('C and fsm_cars', ('cc_r157', 'fsm_cars')),
                                                    ('C, cc_jrc and fsm_cars', ('cc_r157', 'cc_jrc', 'fsm_cars')))}
    return rep


def main(a):
    rep = {}
    rep['dev'] = analyse('dev', a.dir, a.b65, a.dev_dir, a.dev_refine)
    rep['mitra'] = analyse('mitra', a.dir, a.b65, a.mitra_dir, a.mitra_refine)
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for ds, e in rep.items():
        print(ds, 'jobs', e['jobs'], 'errors', e['errors'], 'C1', e['C1']['supported'], e['C1']['mismatches'], 'M1', e['M1'])
        for g, v in e['groups'].items():
            print(f'  {g:10s}', {k: (x['avoided'], x['n']) for k, x in v.items()})
        print('  JR', e['JR'], '| FR', e['FR'])
        print('  P1', e['P1'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('dir')
    ap.add_argument('--b65', required=True)
    ap.add_argument('--dev-dir', required=True)
    ap.add_argument('--mitra-dir', required=True)
    ap.add_argument('--dev-refine', required=True)
    ap.add_argument('--mitra-refine', required=True)
    main(ap.parse_args())
