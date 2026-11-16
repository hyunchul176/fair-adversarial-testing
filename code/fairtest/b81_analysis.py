"""B81 analysis (B81 design, fixed before the run, available on request): B61's SUTs
of known capability, built from the reference driver C with one factor changed, on the 120 exiD cases of B71 (reuse,
labelled post hoc), against the final agent (policy stage, seeds 1-3, window shield, C margin 0.5 m; then the screened
refinement bo_warm_g4s, budget 36). Same composition and hypotheses as B61, for the final agent only:
  K1  SUT = C: detection 0 at every budget (implementation check)
  K2  SUTs stronger than C in one factor (reaction 0.5 s, braking 8.5 m/s^2): k37 <= 3% each
  K3  along each degraded factor (reaction +0.25 / +0.5 / +1.0 s; braking 6.5 / 5.5 m/s^2; perception 1.5 / 2.0 m)
      k37 is non-decreasing (point estimates) and the most degraded level's 95% lower bound is > 0; all three factors
Inputs in JOBS/b81 (B41 layout, b41_analysis.load): final_s<k>.jsonl, margin.jsonl, policy_replay.jsonl, refine.jsonl,
case_pool.json.

  python -m fairtest.b81_analysis OUT.json --jobs fairtest_jobs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairtest.b56_analysis import refine
from fairtest.b61_analysis import FACTORS, KNOWN, STRONGER, boot, compose


def main(a):
    from fairtest.b41_analysis import load
    d = Path(a.jobs) / 'b81'
    evals, realistic, _, pool, _ = load(d)
    g4s, _, err = refine([d / 'refine.jsonl'], 'bo_warm_g4s')
    fin = evals['final']
    seeds = sorted(fin)
    robust = {(x, c, s): m['robust'] for x in seeds for (c, s), m in fin[x].items()}
    rep = dict(design='B81 design (available on request)', errors=dict(g4s=err),
               policy_attacker_caused=int(sum(m['unfair'] for x in seeds for m in fin[x].values())),
               policy_unjudged=int(sum(m['unjudged'] for x in seeds for m in fin[x].values())), known={})
    for label, s in KNOWN.items():
        cases = sorted({c for (x, c, ss) in robust if ss == s})
        if not cases:
            continue
        e = dict(n_cases=len(cases))
        for k in (1, 6, 37):
            v = compose(robust, realistic, g4s, seeds, cases, s, k)
            e[f'k{k}'] = dict(mean=round(float(v.mean()), 4), ci=boot(v))
        rep['known'][label] = e
    kn = rep['known']
    rep['K1'] = dict(supported=all(kn['C'][f'k{k}']['mean'] == 0 for k in (1, 6, 37))) if 'C' in kn else None
    rep['K2'] = dict(values={l: kn[l]['k37']['mean'] for l in STRONGER if l in kn},
                     supported=all(kn[l]['k37']['mean'] <= .03 for l in STRONGER if l in kn))
    k3 = {}
    for f, levels in FACTORS.items():
        v = [kn[l]['k37']['mean'] for l in levels if l in kn]
        lo = kn[levels[-1]]['k37']['ci'][0] if levels[-1] in kn else None
        k3[f] = dict(k37=v, monotone=all(b >= x for x, b in zip(v, v[1:])), last_lower_bound=lo,
                     supported=bool(len(v) == len(levels) and all(b >= x for x, b in zip(v, v[1:])) and lo is not None and lo > 0))
    rep['K3'] = dict(factors=k3, supported=all(x['supported'] for x in k3.values()))
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('errors', 'policy_attacker_caused', 'K1', 'K2', 'K3')}, indent=1))
    for label, e in kn.items():
        print(f"{label:26s} k1 {100 * e['k1']['mean']:5.1f}  k6 {100 * e['k6']['mean']:5.1f}  k37 {100 * e['k37']['mean']:5.1f} {e['k37']['ci']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    main(ap.parse_args())
