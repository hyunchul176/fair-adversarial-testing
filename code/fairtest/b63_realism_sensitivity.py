"""B63 post hoc (2026-10-02, audit item A26): how much the reported detection depends on the realism thresholds of the
success contract. Grid: the G2 envelope excursion g in {0.5, 1, 1.5, 2} IQR (contract 1) and the joint-support threshold
tau at the {95, 99, 99.9}th percentile of the natural leave-one-out 5-NN distances (contract 99).

- Policy stage, one SUT query: exact for every threshold. The episodes do not depend on the thresholds (the shield does
  not use them); only the success filter changes.
- Final agent within 37 queries (policy + screened refinement bo_warm_g4s): exact at the contract. For a stricter
  threshold it is a lower bound: a stored refinement stops at its first success, so if that success fails the stricter
  test the later queries are unknown and count as no success, and pairs whose refinement never ran (all seeds succeeded
  under the contract) count the policy stage only. Not computed for looser thresholds (the screening would have admitted
  other plans).

  python -m fairtest.b63_realism_sensitivity OUT.json NAME DIR REFINE.jsonl [NAME DIR REFINE.jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b41_analysis import K_FULL, _rows, load, rate
from fairtest.b45_joint_realism import K, Support
from fairtest.exid_analysis import COMPETENT8, load_refine

METHOD = 'bo_warm_g4s'
G_GRID = (0.5, 1.0, 1.5, 2.0)
Q_GRID = (95.0, 99.0, 99.9)
GROUPS = {'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'),
          'BC human': ('bc_human_H0',), 'ADS 2.0 m': ('ads@2.0_H0',),
          'learned lane assignment': ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
          'competent (8)': COMPETENT8}


def taus():
    sup = Support()
    d = np.sqrt(((sup.z[:, None, :] - sup.z[None, :, :]) ** 2).sum(-1))
    loo = np.sort(d, 1)[:, K]
    t = {q: float(np.percentile(loo, q)) for q in Q_GRID}
    assert abs(t[99.0] - sup.tau) < 1e-9, (t[99.0], sup.tau)
    return sup, t


def policy_rows(d, sup):
    out = {}
    for r in _rows(Path(d) / 'policy_replay.jsonl'):
        if r['source'] == 'v2_policy':
            out[(str(r['seed']), r['case_id'], r['sut'])] = (sup.score(r['features']) if r['cut_in'] else None, r.get('exceedance'))
    return out


def refine_first(recs, g, tau):
    """{(case, sut): first success query under (g, tau)}; stored runs stop at the first success under the contract."""
    out = {}
    for key, rec in recs[METHOD].items():
        q = rec.get('first_fair')
        if q is None:
            out[key] = None
            continue
        hit = [h for h in rec['queries'] if h.get('n') == q]
        assert len(hit) == 1 and hit[0].get('success'), (key, q)
        h = hit[0]
        ok = h.get('exc') is not None and h['exc'] <= g + 1e-12 and h.get('js') is not None and h['js'] <= tau + 1e-12
        out[key] = q if ok else None
    return out


def composed(evals, real, first, suts, cases, k):
    seeds = sorted(evals['final'])
    return {s: {c: float(np.mean([float((evals['final'][x].get((c, s), {}).get('robust', False) and real.get((x, c, s), False))
                                        or (k >= 2 and first.get((c, s)) is not None and 1 + first[(c, s)] <= k))
                                  for x in seeds])) for c in cases} for s in suts}


def main(out, args):
    sup, tq = taus()
    rep = dict(tau=tq, g_grid=G_GRID, method=METHOD, datasets={})
    for name, d, refine in zip(args[::3], args[1::3], args[2::3]):
        evals, realistic0, _, pool, tau0 = load(Path(d))
        cases = sorted(c['case_id'] for c in pool['cases'])
        suts = sorted({s for e in evals['final'].values() for (_, s) in e})
        prow = policy_rows(d, sup)
        _, recs, errors, conflicts = load_refine([refine])
        first0 = {k: v.get('first_fair') for k, v in recs[METHOD].items()}
        # the contract reproduces the stored realism flags exactly
        real0 = {k: bool(v[0] is not None and v[0] <= tq[99.0] and v[1] is not None and v[1] <= 1.0) for k, v in prow.items()}
        assert real0 == {k: v for k, v in realistic0.items()}, 'contract flags differ from b41_analysis.load'
        ds = dict(n_cases=len(cases), refine_errors=errors, refine_conflicts=len(conflicts), grid={})
        for g in G_GRID:
            for q in Q_GRID:
                real = {k: bool(v[0] is not None and v[0] <= tq[q] and v[1] is not None and v[1] <= g) for k, v in prow.items()}
                k1 = composed(evals, real, {}, suts, cases, 1)
                cell = dict(g=g, q=q, tau=tq[q], policy_k1={s: round(rate(k1[s]), 4) for s in suts},
                            policy_reported=sum(1 for x in evals['final'] for (c, s), m in evals['final'][x].items()
                                                if m['robust'] and real.get((x, c, s), False)))
                strict = g <= 1.0 and q <= 99.0
                if strict:
                    first = first0 if (g == 1.0 and q == 99.0) else refine_first(recs, g, tq[q])
                    k37 = composed(evals, real, first, suts, cases, K_FULL)
                    cell['final_k37'] = {s: round(rate(k37[s]), 4) for s in suts}
                    cell['final_k37_kind'] = 'exact' if (g == 1.0 and q == 99.0) else 'lower bound'
                    cell['P1_k37'] = round(rate(k37['following_H0']) - rate(k37['rss_sut_H0']), 4)
                cell['groups_k1'] = {gname: round(float(np.mean([cell['policy_k1'][s] for s in mem if s in cell['policy_k1']])), 4)
                                     for gname, mem in GROUPS.items()}
                if 'final_k37' in cell:
                    cell['groups_k37'] = {gname: round(float(np.mean([cell['final_k37'][s] for s in mem if s in cell['final_k37']])), 4)
                                          for gname, mem in GROUPS.items()}
                ds['grid'][f'g{g}_q{q}'] = cell
        rep['datasets'][name] = ds
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    for name, ds in rep['datasets'].items():
        print('==', name, ds['n_cases'], 'cases; refine errors', ds['refine_errors'], 'conflicts', ds['refine_conflicts'])
        for key, cell in ds['grid'].items():
            k37 = cell.get('groups_k37')
            print(f"  {key:12s} tau {cell['tau']:.2f} reported(policy) {cell['policy_reported']:5d} | k1 "
                  + ' '.join(f"{v * 100:5.1f}" for v in cell['groups_k1'].values())
                  + (f" | k37 {cell['final_k37_kind'][:5]} " + ' '.join(f"{v * 100:5.1f}" for v in k37.values())
                     + f" P1 {cell['P1_k37'] * 100:5.1f}" if k37 else ''))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
