"""B52 analysis (B52 design, written before the results, available on request).

Per policy-stage episode (one query, no shield) of the feasibility adversaries FEAS-T0 (oracle_t0) and FEAS-R0
(oracle_react0), seeds 1-3, 17 SUTs: collision = valid and contact; the CC and the arm's own reference re-judge every
collision (b17_rl REFS); robust (G4) from b23_margin.
  F1  among collisions its own reference avoids, the share the human CC does not avoid >= 30% (each arm)
  F2  own-reference "failure" rate (collision, own reference avoids) on AEB H0 / AEB H0.5 / longitudinal RSS pooled >= 10%
  F3  CC robust fair discrimination (following H0 - longitudinal RSS): the v2j policy stage (B26 dev) exceeds each arm by
      >= 20 pp
  EVAL_DIR holds <arm>_s<k>.jsonl and margin.jsonl; V2J_DIR is the B26 layout (final/final_s<k>_dev.jsonl + margin_dev).

  python -m fairtest.b52_analysis OUT.json EVAL_DIR --jobs fairtest_jobs [--dataset dev]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage

ARMS = {'feast0': 'oracle_t0', 'feasr0': 'oracle_react0'}
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
DELTA, FRONT = .5, 3.5


def _rows(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def main(a):
    d = Path(a.eval_dir)
    margin = {}
    for r in _rows(d / 'margin.jsonl'):
        margin[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= FRONT
                                                             and r['cc_clearance'] >= DELTA)
    rep = dict(arms={})
    stats = {}
    for arm, own in ARMS.items():
        eps = []
        for path in sorted(d.glob(f'{arm}_s[0-9].jsonl')):
            for r in _rows(path):
                if 'valid' not in r:
                    continue
                col = bool(r['valid'] and r['contact'])
                refs = r.get('refs') or {}
                cc = refs.get('cc_r157', {}).get('contact') if col else None
                mine = refs.get(own, {}).get('contact') if col else None
                eps.append(dict(sut=r['sut'], case=r['case_id'], collision=col, cc_contact=cc, own_contact=mine,
                                cc_fair=bool(col and cc is False), own_fair=bool(col and mine is False),
                                robust=bool(col and cc is False and margin.get((path.name, r['case_id'], r['sut']), False)),
                                realistic=bool((r.get('cc_replay') or {}).get('exc') is not None and r['cc_replay']['exc'] <= 1.)))
        stats[arm] = eps
        feas = [e for e in eps if e['own_fair']]
        f1 = sum(e['cc_contact'] is True for e in feas) / len(feas) if feas else None
        comp = [e for e in eps if e['sut'] in COMPETENT]
        f2 = float(np.mean([e['own_fair'] for e in comp])) if comp else None
        by = collections.defaultdict(list)
        for e in eps:
            by[e['sut']].append(e)
        per_sut = {s: {k: round(float(np.mean([e[k] for e in v])), 4) for k in ('collision', 'cc_fair', 'robust', 'own_fair', 'realistic')}
                   | dict(attacker_at_fault=round(sum(e['cc_contact'] is True for e in v) / max(1, sum(e['collision'] for e in v)), 4), n=len(v))
                   for s, v in sorted(by.items())}
        disc = per_sut['following_H0']['robust'] - per_sut['rss_sut_H0']['robust']
        rep['arms'][arm] = dict(reference=own, episodes=len(eps), F1_feasible_but_cc_unavoidable=None if f1 is None else round(f1, 4),
                                F2_own_failure_on_competent=None if f2 is None else round(f2, 4), cc_robust_discrimination=round(disc, 4),
                                per_sut=per_sut)
    # v2j policy stage on the same cases (dev: the B26 files; the comparison fixed before the run)
    robust, unfair = policy_stage(Path(a.jobs) / 'b26', a.dataset)
    seeds = sorted({k[0] for k in robust})
    r0 = np.mean([robust[(x, c, s)] for (x, c, s) in robust if s == 'following_H0'])
    r1 = np.mean([robust[(x, c, s)] for (x, c, s) in robust if s == 'rss_sut_H0'])
    rep['v2j_policy'] = dict(seeds=seeds, following_H0=round(float(r0), 4), rss=round(float(r1), 4), cc_robust_discrimination=round(float(r0 - r1), 4))
    rep['F1'] = {arm: dict(value=v['F1_feasible_but_cc_unavoidable'], supported=v['F1_feasible_but_cc_unavoidable'] is not None
                           and v['F1_feasible_but_cc_unavoidable'] >= .30) for arm, v in rep['arms'].items()}
    rep['F2'] = {arm: dict(value=v['F2_own_failure_on_competent'], supported=v['F2_own_failure_on_competent'] is not None
                           and v['F2_own_failure_on_competent'] >= .10) for arm, v in rep['arms'].items()}
    rep['F3'] = {arm: dict(gap=round(rep['v2j_policy']['cc_robust_discrimination'] - v['cc_robust_discrimination'], 4),
                           supported=rep['v2j_policy']['cc_robust_discrimination'] - v['cc_robust_discrimination'] >= .20)
                 for arm, v in rep['arms'].items()}
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('F1', 'F2', 'F3', 'v2j_policy')}, indent=1))
    for arm, v in rep['arms'].items():
        print('==', arm, v['reference'], 'episodes', v['episodes'])
        for s in ('following_H0', 'following_H1', 'following_aeb_H0', 'rss_sut_H0', 'bc_human_H0', 'drl_sut@p2.5_H0'):
            x = v['per_sut'].get(s)
            if x:
                print(f"   {s:18s} collision {100*x['collision']:5.1f} cc_fair {100*x['cc_fair']:5.1f} robust {100*x['robust']:5.1f} "
                      f"own_fair {100*x['own_fair']:5.1f} attacker(cc) {100*x['attacker_at_fault']:5.1f} realistic {100*x['realistic']:5.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('eval_dir')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--dataset', default='dev')
    main(ap.parse_args())
