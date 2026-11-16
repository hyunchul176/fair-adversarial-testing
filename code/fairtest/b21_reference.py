"""B21 (G2/G3): reference-model validity — does the fair/attacker-at-fault judgement survive the
regulation's time-based CC perception, and how do the fault oracles agree on the same collisions?

  python -m fairtest.b21_reference rl-reeval EVAL.jsonl OUT.jsonl [--commit] --workers W
      replay every RL collision episode's action sequence against cc_r157_time
  python -m fairtest.b21_reference analyze OUT.json
      exposed-highD testers: fair-failure rate under CC-distance vs CC-time, Cohen's kappa on collisions,
      disagreement by the attack's mean lateral speed (3.2 m / duration_10_90 of the CC run)
  python -m fairtest.b21_reference heldout OUT.json
      the same grid and stored-plan comparison on the B9 highD (post hoc) and NGSIM cohorts
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

from fairtest.strict_analysis import EV, E13, J

B14D = EV / 'b14_realism'
REF_T = 'cc_r157_time'


def _replay(args):
    row, commit = args
    from fairtest.b17_rl import B17Env
    from fairtest.runner import _STATE, _init
    _init()
    sut = row['sut'].rsplit('_H', 1)
    env = B17Env([row['case_id']], 'fair_guarded' if commit else 'fair', suts=[(sut[0], float(sut[1]))], label=False)
    env.case = _STATE['cases'][row['case_id']]
    env.actions = [tuple(a) for a in row['actions']]
    res = env.replay(REF_T)
    return dict(case_id=row['case_id'], sut=row['sut'], contact=res['contact'], valid=res['valid'])


def rl_reeval(a):
    rows = [json.loads(l) for l in open(a.eval, encoding='utf-8')]
    todo = [(r, a.commit) for r in rows if r['valid'] and r['contact']]
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(_replay, todo, chunksize=4):
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(finished=len(todo))))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not len(a):
        return float('nan')
    po = float(np.mean(a == b))
    pe = float(np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.


def analyze(a):
    from fairtest.analyze_landscape import load
    from fairtest.b20_validity import RANK
    from fairtest.report_b7 import rows_all
    import fairtest.boundary_planner as BP
    from fairtest import b9_analysis as A
    from fairtest.runner import _STATE, _init
    _init()
    cases = dict(_STATE['cases'])
    rows, _ = rows_all()
    rows += load(J / 'b10v2-a' / 'landscape.jsonl')[0] + load(J / 'b10v2-b' / 'landscape.jsonl')[0] + load(J / 'b21' / 'cctime_exposed.jsonl')[0]
    suts = tuple((k.rsplit('_H', 1)[0], float(k.rsplit('_H', 1)[1])) for k in RANK)
    A.SUTS, BP.SUTS = suts, suts
    rep = dict(grid={}, plans={})
    for ref in ('cc_r157', REF_T):
        per = A.per_case_outcomes(rows, cases, {}, ref, math.inf)
        rep['grid'][ref] = {k: dict(oracle=float(np.mean([m['oracle'][0] for m in v.values() if m])),
                                    boundary=float(np.mean([m['boundary'][0] for m in v.values() if m])),
                                    b5_fair=float(np.mean([m['b5'][0] for m in v.values() if m])),
                                    b5_unfair=float(np.mean([m['b5'][1] for m in v.values() if m]))) for k, v in per.items()}
    # grid plan-level agreement on SUT collisions, stratified by lateral speed of the CC run
    by = BP.index(rows)
    agree = collections.defaultdict(lambda: collections.defaultdict(list))
    for (c, p), v in by.items():
        if not p.startswith('plan_') or ('cc_r157', 0.) not in v or (REF_T, 0.) not in v:
            continue
        cd, ct = v[('cc_r157', 0.)], v[(REF_T, 0.)]
        dur = cd['features'].get('duration_10_90')
        vlat = 3.2 / dur if dur else None
        band = 'unknown' if vlat is None else ('<1.2' if vlat < 1.2 else '1.2-1.8' if vlat < 1.8 else '1.8-2.4' if vlat < 2.4 else '>=2.4')
        for k in RANK:
            s = suts[list(RANK).index(k)]
            r = v.get(s)
            if r is None or not (r['valid'] and r['contact']):
                continue
            agree[k][band].append((not cd['contact'], not ct['contact']))
    rep['grid_collision_agreement'] = {k: {band: dict(n=len(x), fair_distance=float(np.mean([a_ for a_, _ in x])), fair_time=float(np.mean([b_ for _, b_ in x])),
                                                       kappa=kappa([a_ for a_, _ in x], [b_ for _, b_ in x])) for band, x in sorted(bands.items())}
                                       for k, bands in agree.items()}
    # stored search plans (b16_reeval output with cc_r157_time) and RL replays
    ree = {(r['source'], r['case_id'], r['key']): r['refs'][REF_T]['contact'] for r in (json.loads(l) for l in open(J / 'b21' / 'reeval_exposed_cctime.jsonl', encoding='utf-8'))}
    n = len(cases)
    plans = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))  # tester -> sut -> [fair_dist, fair_time, collisions]
    for line in open(E13 / 'posthoc_highd' / 'exposed_cont_search_all7.jsonl', encoding='utf-8'):
        r = json.loads(line)
        for k, ss in r['sut_search'].items():
            b = ss.get('best')
            if b and b['fair']:
                x = plans['cont_search'][k]
                x[0] += 1
                x[1] += int(not ree[('cont_search', r['case_id'], k)])
    for line in open(E13 / 'falsify' / 'b13_falsify_highd_dev.jsonl', encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            start, k = key.split('|')
            if start != 'falsify_centre' or not (rec['best'] and rec['best']['contact']):
                continue
            x = plans['falsify_centre'][k]
            x[2] += 1
            x[0] += int(not rec['refs']['cc_r157']['contact'])
            x[1] += int(not ree[('falsify', r['case_id'], key)])
    rep['plans'] = {t: {k: dict(fair_distance=v[0] / n, fair_time=v[1] / n, collisions=v[2] / n if v[2] else None) for k, v in per.items()} for t, per in plans.items()}
    rl = {}
    for name, ev, rr in (('RL_fair_naive', 'eval_fair_s1_dev.jsonl', 'rl_fair_naive_cctime.jsonl'),
                         ('RL_fair_guarded', 'eval_fair_guarded2_s1_dev.jsonl', 'rl_fair_guarded_cctime.jsonl')):
        path = J / 'b21' / rr
        if not path.exists():
            continue
        tt = {(r['case_id'], r['sut']): r['contact'] for r in (json.loads(l) for l in open(path, encoding='utf-8'))}
        rows_ev = [json.loads(l) for l in open(J / 'b17' / ev, encoding='utf-8')]
        per = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows_ev:
            x = per[r['sut']]
            x[2] += 1
            if r['valid'] and r['contact']:
                x[0] += int(not r['refs']['cc_r157']['contact'])
                x[1] += int(not tt[(r['case_id'], r['sut'])])
        rl[name] = {k: dict(fair_distance=v[0] / v[2], fair_time=v[1] / v[2]) for k, v in per.items()}
    rep['rl'] = rl
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for ref, d in rep['grid'].items():
        print('grid', ref, {k: (round(100 * v['oracle'], 1), round(100 * v['b5_fair'], 1), round(100 * v['b5_unfair'], 1)) for k, v in d.items()})
    for k, bands in rep['grid_collision_agreement'].items():
        print('agree', k, {b: (x['n'], round(100 * x['fair_distance']), round(100 * x['fair_time']), round(x['kappa'], 2)) for b, x in bands.items()})
    for t, d in rep['plans'].items():
        print(t, {k: (round(100 * v['fair_distance'], 1), round(100 * v['fair_time'], 1)) for k, v in d.items()})
    for t, d in rl.items():
        print(t, {k: (round(100 * v['fair_distance'], 1), round(100 * v['fair_time'], 1)) for k, v in d.items()})


def heldout(a):
    """B9 (post hoc highD) and NGSIM: grid methods and stored search plans under CC-distance vs CC-time."""
    from fairtest import b16_analysis as B
    from fairtest.strict_analysis import SUT6, SUT7, rows_of
    from fairtest.runner import _init
    _init()
    b9_pool = EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json'
    sets = dict(
        highd_b9_posthoc=({c['case_id']: c for c in json.loads(b9_pool.read_text(encoding='utf-8'))['cases']},
                          [J / s / 'landscape.jsonl' for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')] + [J / 'b21' / 'b21_cctime_b9.jsonl'], SUT6,
                          E13 / 'posthoc_highd' / 'b9_cont_search_all7_POSTHOC.jsonl', E13 / 'posthoc_highd' / 'b9_falsify_POSTHOC.jsonl',
                          B14D / 'b14_b9_POSTHOC.jsonl', J / 'b21' / 'b21_reeval_b9.jsonl'),
        ngsim_posthoc=({c['case_id']: c for c in json.loads((E13 / 'cohort' / 'case_pool.json').read_text(encoding='utf-8'))['cases']},
                       [J / 'b13-ngsim-a' / 'landscape.jsonl', J / 'b13-ngsim-b' / 'landscape.jsonl', J / 'b21' / 'b21_cctime_ngsim.jsonl'], SUT7,
                       E13 / 'continuous' / 'b13_cont_search.jsonl', E13 / 'falsify' / 'b13_falsify.jsonl',
                       B14D / 'b14_ngsim_POSTHOC.jsonl', J / 'b21' / 'b21_reeval_ngsim.jsonl'))
    rep = {}
    for name, (cases, paths, suts, cont, fal, b14, ree_path) in sets.items():
        rows = rows_of(paths)
        grid = B.grid(rows, cases, {}, suts, ('cc_r157', REF_T))
        ree = {(r['source'], r['case_id'], r['key']): r['refs'][REF_T]['contact'] for r in (json.loads(l) for l in open(ree_path, encoding='utf-8'))}
        n = len(cases)
        plans = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))  # tester -> sut -> [fair_dist, fair_time, collisions]
        for line in open(cont, encoding='utf-8'):
            r = json.loads(line)
            for k, ss in r['sut_search'].items():
                b = ss.get('best')
                if b and b['fair']:
                    x = plans['cont_search'][k]
                    x[0] += 1
                    x[1] += int(not ree[('cont_search', r['case_id'], k)])
        for line in open(fal, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                start, k = key.split('|')
                if start != 'falsify_centre' or not (rec['best'] and rec['best']['contact']):
                    continue
                x = plans['falsify_centre'][k]
                x[2] += 1
                x[0] += int(not rec['refs']['cc_r157']['contact'])
                x[1] += int(not ree[('falsify', r['case_id'], key)])
        for line in open(b14, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                mode, k = key.split('|')
                if not (rec['best'] and rec['best']['hit']):
                    continue
                x = plans[mode][k]
                x[2] += 1
                x[0] += int(not rec['refs']['cc_r157'])
                x[1] += int(not ree[('b14', r['case_id'], key)])
        rep[name] = dict(n_cases=n, grid=grid,
                         plans={t: {k: dict(fair_distance=v[0] / n, fair_time=v[1] / n, collisions=v[2] / n if v[2] else None) for k, v in per.items()}
                                for t, per in plans.items()})
        print('==', name, n)
        for ref, d in grid.items():
            print('  grid', ref, {k: (round(100 * v['oracle'], 1), round(100 * v['b5_fair'], 1), round(100 * v['b5_unfair'], 1)) for k, v in d.items()})
        for t, d in rep[name]['plans'].items():
            print(' ', t, {k: (round(100 * v['fair_distance'], 1), round(100 * v['fair_time'], 1)) for k, v in d.items()})
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    h = sub.add_parser('heldout')
    h.add_argument('out')
    r = sub.add_parser('rl-reeval')
    r.add_argument('eval')
    r.add_argument('out')
    r.add_argument('--commit', action='store_true')
    r.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    q = sub.add_parser('analyze')
    q.add_argument('out')
    a = ap.parse_args()
    dict(heldout=heldout, analyze=analyze, **{'rl-reeval': rl_reeval})[a.cmd](a)
