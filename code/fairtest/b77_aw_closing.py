"""B77 addendum (after the plan, descriptive, stored rows only): closing speed at contact of the agent's Autoware failures
(B59, development cases), which the Autoware harness logged per tick but no analysis computed.

closing = Autoware ego speed - adversary longitudinal speed at the first tick with clearance <= 0, the quantity of
b51_severity (target vx - generator vx at contact); the map is straight, so the ego speed is its longitudinal speed.
Reported failures and confirmed ones (>= 2 of 4 open-loop reruns collide) as in aw_export;
pairs: the same case and source (policy seed or refinement) confirmed under both variants.

  python -m fairtest.b77_aw_closing B59_DIR OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401

CONFIRM, SEVERE, HARD = 2, 3., 5.


def _load(d, phase):
    rows = []
    for p in sorted(Path(d).glob(f'{phase}_*.jsonl')):
        rows += [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    return rows


def failures(d):
    reruns = {(r['variant'], r['case_id'], r['source']): sum(bool(x['contact']) for x in (r['result'].get('runs') or []))
              for r in _load(d, 'repro')}
    items = []
    for r in _load(d, 'policy'):
        x = r['result']
        if r['attacker'] == 'v2j' and x.get('reportable'):
            items.append((r['variant'], r['case_id'], f"policy_s{r['seed']}", x['trace'], x['t_contact']))
    for r in _load(d, 'refine'):
        x = r['result']
        if x.get('found'):
            b = x['best']
            items.append((r['variant'], r['case_id'], 'refine', b['aw_trace'], b['aw']['t_contact']))
    out = []
    for v, c, src, trace, t_hit in items:
        k = next(i for i, p in enumerate(trace) if p['clear'] <= 0.)
        p = trace[k]
        out.append(dict(variant=v, case_id=c, source=src, t=p['t'], t_contact=t_hit, closing=round(p['ego'][2] - p['adv'][2], 3),
                        v_ego=p['ego'][2], v_adv=p['adv'][2], rear=p['gap'] is not None, reruns=reruns.get((v, c, src))))
    return out


def describe(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2), p10=round(float(np.percentile(x, 10)), 2),
                p90=round(float(np.percentile(x, 90)), 2), max=round(float(x.max()), 2),
                below_3=round(float(np.mean(x < SEVERE)), 4), at_least_5=round(float(np.mean(x >= HARD)), 4))


def paired(conf, keys):
    """AW-default minus AW-r157 closing speed over failures confirmed under both variants for the same key, with a case
    bootstrap of the median (a resampled case brings all of its pairs)."""
    d = np.array([conf['aw_default'][k] - conf['aw_r157'][k] for k in keys])
    cases = sorted({c for c, _ in keys})
    by_case = {c: d[[i for i, (cc, _) in enumerate(keys) if cc == c]] for c in cases}
    rng = np.random.default_rng(77)
    boot = [float(np.median(np.concatenate([by_case[cases[i]] for i in rng.integers(0, len(cases), len(cases))])))
            for _ in range(10_000)]
    return dict(n=len(keys), cases=len(cases), median_diff_default_minus_r157=round(float(np.median(d)), 2),
                ci95=[round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
                share_default_faster=round(float(np.mean(d > 0)), 4))


def main(a):
    rows = failures(a.b59)
    rep = dict(definition=__doc__.split('\n\n')[1], tick_mismatch=[(r['variant'], r['case_id'], r['source'], r['t'], r['t_contact'])
                                                                    for r in rows if abs(r['t'] - r['t_contact']) > .051])
    conf = {}
    for v in sorted({r['variant'] for r in rows}):
        rep_v = [r for r in rows if r['variant'] == v]
        con = [r for r in rep_v if (r['reruns'] or 0) >= CONFIRM]
        conf[v] = {(r['case_id'], r['source']): r['closing'] for r in con}
        rep[v] = dict(reported=describe([r['closing'] for r in rep_v]), confirmed=describe([r['closing'] for r in con]),
                      confirmed_policy=describe([r['closing'] for r in con if r['source'] != 'refine']),
                      confirmed_refine=describe([r['closing'] for r in con if r['source'] == 'refine']),
                      rear_end=sum(r['rear'] for r in con), side=sum(not r['rear'] for r in con))
    both = sorted(set(conf['aw_default']) & set(conf['aw_r157']))
    rep['paired'] = paired(conf, both)
    rep['paired_policy'] = paired(conf, [k for k in both if k[1] != 'refine'])  # same case and policy seed
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'definition'}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('b59')
    ap.add_argument('out')
    main(ap.parse_args())
