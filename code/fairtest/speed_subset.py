"""Post hoc sensitivity: results restricted to cases inside the R157 Rev.1 speed range (ALKS / follower initial speed
<= 130 km/h = 36.11 m/s).
  cut-in: the final agent v2, composed exactly as b26_analysis final (policy stage s1-3 + guarded refinement bo_warm_g4g)
  oned:   the 1-D deceleration (B30) and cut-out (B34, B34b) evaluations -- agent detection, baseline attacker-at-fault

  python -m fairtest.speed_subset OUT.json --dir fairtest_jobs/b26 --refine fairtest_jobs/b29/guarded_{ds}.jsonl
  python -m fairtest.speed_subset OUT.json --oned fairtest_jobs
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage, refine_stage

V_MAX = 130 / 3.6
POOLS = dict(dev='case_pool.json', b9='case_pool_b9.json', ngsim='case_pool_ngsim.json')
KS = (1, 6, 37)


def speeds(ds):
    d = json.loads((Path(__file__).parent / 'data' / POOLS[ds]).read_text(encoding='utf-8'))
    return {c['case_id']: float(c['physical_initial_state'][1]['vx']) for c in d['cases']}


def rates(robust, ref, s, cases, seeds):
    out = {}
    for k in KS:
        per_seed = [np.mean([robust.get((x, c, s), False) or (k >= 2 and ref.get((c, s)) is not None and 1 + ref[(c, s)] <= k)
                             for c in cases]) for x in seeds]
        out[f'k{k}'] = round(float(np.mean(per_seed)), 4)
    return out


def main(a):
    rep = {}
    for ds in POOLS:
        robust, unfair = policy_stage(Path(a.dir), ds)
        ref = refine_stage(Path(a.dir), ds, 'bo_warm_g4g', a.refine)
        if not robust:
            continue
        v = speeds(ds)
        seeds = sorted({k[0] for k in robust})
        rep[ds] = {}
        for s in sorted({k[2] for k in robust}):
            cases = sorted({k[1] for k in robust if k[2] == s})
            if not all((c, s) in ref for c in cases):
                continue
            inside = [c for c in cases if v[c] <= V_MAX]
            rep[ds][s] = dict(n_all=len(cases), n_inside=len(inside), all=rates(robust, ref, s, cases, seeds),
                              inside=rates(robust, ref, s, inside, seeds) if inside else None,
                              attacker_at_fault_inside=float(np.mean([unfair.get((x, c, s), False) for x in seeds for c in inside])) if inside else None)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for ds, t in rep.items():
        n = next(iter(t.values()))
        print(f"== {ds}: {n['n_inside']}/{n['n_all']} cases with SUT speed <= 130 km/h  (robust fair within 1 / 6 / 37 queries, %)")
        for s, e in t.items():
            al, ins = e['all'], e['inside']
            print(f"   {s:22s} all {100*al['k1']:5.1f} {100*al['k6']:5.1f} {100*al['k37']:5.1f}   inside "
                  + (f"{100*ins['k1']:5.1f} {100*ins['k6']:5.1f} {100*ins['k37']:5.1f}  unfair {100*e['attacker_at_fault_inside']:.1f}" if ins else '-'))


def oned(out, jobs):
    rep = {}
    for name, cf, ev in (('dec_b30', 'b30/dec_cases.json', 'b30/dec_eval_dev.jsonl'),
                         ('cutout_b34', 'b34/cutout_cases.json', 'b34/cutout_eval_dev.jsonl'),
                         ('cutout_b34b', 'b34/cutout_cases.json', 'b34/cutout_eval_dev_consensus.jsonl')):
        cases = {c['case_id']: c for c in json.loads((Path(jobs) / cf).read_text(encoding='utf-8'))['cases']}
        rows = [json.loads(l) for l in (Path(jobs) / ev).read_text(encoding='utf-8').splitlines() if l.strip()]
        inside = [r for r in rows if cases[r['case_id']]['v_follower'] <= V_MAX]
        t = dict(n_all=len(rows), n_inside=len(inside), suts={})
        for s in sorted({k.split('|')[1] for r in rows for k in r['runs']}):
            e = {}
            for label, rs in (('all', rows), ('inside', inside)):
                ok = [bool((r['runs'].get(f'agent|{s}') or {}).get('best', {}) and r['runs'][f'agent|{s}']['best'].get('success')) for r in rs]
                e[f'agent_{label}'] = round(sum(ok) / len(rs), 4)
                col = [r['runs'][f'collide|{s}']['best'] for r in rs if f'collide|{s}' in r['runs']
                       and (r['runs'][f'collide|{s}'].get('best') or {}).get('success')]
                if any(f'collide|{s}' in r['runs'] for r in rs):
                    e[f'collide_{label}'] = dict(collisions=len(col), attacker_at_fault=sum(bool(b.get('cc_contact')) for b in col))
            t['suts'][s] = e
        rep[name] = t
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for name, t in rep.items():
        print(f"== {name}: {t['n_inside']}/{t['n_all']} cases <= 130 km/h")
        for s, e in t['suts'].items():
            print(f"   {s:13s} agent {100*e['agent_all']:5.1f} -> {100*e['agent_inside']:5.1f}  collide inside {e.get('collide_inside')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--dir')
    ap.add_argument('--refine', nargs='*')
    ap.add_argument('--oned', help='fairtest_jobs directory (1-D scenarios)')
    a = ap.parse_args()
    oned(a.out, a.oned) if a.oned else main(a)
