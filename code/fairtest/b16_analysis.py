"""B16 analysis: which avoidability criterion restores the construct validity of adversarial testing?

For each reference criterion (human benchmarks: CC R157, FSM, RSS; solvability-style: ideal ADS
oracle_react0, omniscient oracle_t0) report, per SUT of the capability panel:
  grid methods (frozen B9 logic): B5 fair / attacker-at-fault, R157-boundary, grid oracle;
  generic falsification (falsify_centre) and plausibility-constrained falsification (falsify_real):
    share of their SUT collisions the criterion calls "fair" (i.e. attributes to the SUT);
  discrimination: Spearman rho between SUT capability rank and the falsifier's fair-failure rate.

  python -m fairtest.b16_analysis OUT.json
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from fairtest import b9_analysis as A
import fairtest.boundary_planner as BP
from fairtest.strict_analysis import EV, E13, J, SUT6, SUT7, rows_of

B14 = EV / 'b14_realism'
CRITERIA = ('cc_r157', 'ref_fsm', 'ref_rss', 'oracle_react0', 'oracle_t0')
CAPABILITY = ('following_H0', 'reference_idm_H0', 'following_H0.5', 'following_H1', 'following_aeb_H0',
              'following_aeb_H0.5', 'rss_sut_H0')  # increasing capability (ties allowed in rank)
RANK = {'following_H0': 1, 'reference_idm_H0': 1, 'following_H0.5': 2, 'following_H1': 3, 'following_aeb_H0': 4,
        'following_aeb_H0.5': 5, 'rss_sut_H0': 6}


def grid(rows, cases, models, suts, criteria):
    out = {}
    have = {r['target'] for r in rows}
    for ref in criteria:
        if ref not in have:
            continue
        A.SUTS, BP.SUTS = suts, suts
        per = A.per_case_outcomes(rows, cases, models, ref, math.inf)
        out[ref] = {s: dict(n=len([m for m in v.values() if m is not None]),
                            b5_fair=float(np.mean([m['b5'][0] for m in v.values() if m])),
                            b5_unfair=float(np.mean([m['b5'][1] for m in v.values() if m])),
                            boundary=float(np.mean([m['boundary'][0] for m in v.values() if m])),
                            oracle=float(np.mean([m['oracle'][0] for m in v.values() if m])))
                    for s, v in per.items()}
    return out


def plans(falsify, b14, reeval, n_cases):
    """Per SUT: collisions of falsify_centre / falsify_real and the share each criterion calls fair."""
    extra = {(r['source'], r['case_id'], r['key']): r['refs'] for r in (json.loads(l) for l in open(reeval, encoding='utf-8'))}
    res = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for line in open(falsify, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            start, sut = key.split('|')
            if start != 'falsify_centre' or not (rec['best'] and rec['best']['contact']):
                continue
            c = res['falsify_centre'][sut]
            c['collisions'] += 1
            for ref in ('cc_r157', 'ref_fsm', 'ref_rss'):
                c[f'fair_{ref}'] += int(not rec['refs'][ref]['contact'])
            for ref in ('oracle_react0', 'oracle_t0'):
                c[f'fair_{ref}'] += int(not extra[('falsify', r['case_id'], key)][ref]['contact'])
    for line in open(b14, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            mode, sut = key.split('|')
            if mode != 'falsify_real' or not (rec['best'] and rec['best']['hit']):
                continue
            c = res['falsify_real'][sut]
            c['collisions'] += 1
            for ref in ('cc_r157', 'ref_fsm', 'ref_rss'):
                c[f'fair_{ref}'] += int(not rec['refs'][ref])
            for ref in ('oracle_react0', 'oracle_t0'):
                c[f'fair_{ref}'] += int(not extra[('b14', r['case_id'], key)][ref]['contact'])
    out = {}
    for method, per in res.items():
        out[method] = {}
        for sut, c in per.items():
            out[method][sut] = dict(collision_rate=c['collisions'] / n_cases,
                                    **{f'fair_rate_{ref}': c[f'fair_{ref}'] / n_cases for ref in CRITERIA},
                                    **{f'attributed_share_{ref}': (c[f'fair_{ref}'] / c['collisions']) if c['collisions'] else None for ref in CRITERIA})
        disc = {}
        for ref in CRITERIA:
            suts = [s for s in CAPABILITY if s in out[method]]
            rho = spearmanr([RANK[s] for s in suts], [out[method][s][f'fair_rate_{ref}'] for s in suts])
            disc[ref] = dict(rho=float(rho.statistic), spread=max(out[method][s][f'fair_rate_{ref}'] for s in suts) -
                             min(out[method][s][f'fair_rate_{ref}'] for s in suts))
        out[method]['_discrimination'] = disc
    return out


def main(out):
    from fairtest.report_b7 import rows_all
    from fairtest.runner import _STATE, _init
    _init()
    b16 = J / 'b16'
    exposed, _ = rows_all()
    exposed += rows_of([J / 'b10v2-a' / 'landscape.jsonl', J / 'b10v2-b' / 'landscape.jsonl', b16 / 'b16_exposed.jsonl'])
    b9_pool = EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json'
    b9_cases = {c['case_id']: c for c in json.loads(b9_pool.read_text(encoding='utf-8'))['cases']}
    b9 = rows_of([J / s / 'landscape.jsonl' for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')] + [b16 / 'b16_b9.jsonl'])
    ng_cases = {c['case_id']: c for c in json.loads((E13 / 'cohort' / 'case_pool.json').read_text(encoding='utf-8'))['cases']}
    ng = rows_of([J / 'b13-ngsim-a' / 'landscape.jsonl', J / 'b13-ngsim-b' / 'landscape.jsonl', b16 / 'b16_ngsim.jsonl'])
    sets = dict(
        highd_exposed_development=(exposed, dict(_STATE['cases']), SUT7, E13 / 'falsify' / 'b13_falsify_highd_dev.jsonl',
                                   B14 / 'b14_exposed_development.jsonl', b16 / 'b16_reeval_exposed.jsonl'),
        highd_b9_posthoc=(b9, b9_cases, SUT6, E13 / 'posthoc_highd' / 'b9_falsify_POSTHOC.jsonl', B14 / 'b14_b9_POSTHOC.jsonl',
                          b16 / 'b16_reeval_b9.jsonl'),
        ngsim_posthoc=(ng, ng_cases, SUT7, E13 / 'falsify' / 'b13_falsify.jsonl', B14 / 'b14_ngsim_POSTHOC.jsonl',
                       b16 / 'b16_reeval_ngsim.jsonl'))
    rep = {}
    for name, (rows, cases, suts, fal, b14, ree) in sets.items():
        rep[name] = dict(n_cases=len(cases), grid=grid(rows, cases, {}, suts, CRITERIA), plans=plans(fal, b14, ree, len(cases)))
        print('==', name, len(cases))
        g = rep[name]['grid']
        for s in [x for x in CAPABILITY if x in g.get('cc_r157', {})]:
            print(f"  grid {s:20}", ' '.join(f"{ref}: B5 {100*g[ref][s]['b5_fair']:4.1f}/{100*g[ref][s]['b5_unfair']:4.1f} orc {100*g[ref][s]['oracle']:4.1f}" for ref in g))
        for method in ('falsify_centre', 'falsify_real'):
            p = rep[name]['plans'][method]
            print(f'  {method} discrimination', {ref: (round(v["rho"], 2), round(100 * v["spread"], 1)) for ref, v in p['_discrimination'].items()})
            for s in CAPABILITY:
                if s in p:
                    print(f"    {s:20} coll {100*p[s]['collision_rate']:5.1f}  attributed share:", {ref: round(100 * p[s][f'attributed_share_{ref}']) if p[s][f'attributed_share_{ref}'] is not None else None for ref in CRITERIA})
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
