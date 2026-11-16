"""B54 figure data (descriptive, after aw_analysis): for every reported v2j failure against Autoware, the Autoware
ego's acceleration and speed per tick and the R157 CC's in the highway-env counterfactual of the same attack, both on
the same clock (t = 0 at the case's initial state), with the time the cut-in becomes visible (|adversary y| >= 1.095 m)
and the reruns' contact count.

  python -m fairtest.aw_export OUT_DIR FIGDATA.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairtest import devboot  # noqa: F401

PERCEIVE_M = 1.095


def _load(out, phase):
    rows = []
    for p in sorted(Path(out).glob(f'{phase}_*.jsonl')):
        rows += [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    return rows


def main(a):
    from fairtest.b23_margin import clearance
    from fairtest.b29_consensus import _run
    from fairtest.runner import _STATE, _init
    _init()
    reruns = {(r['variant'], r['case_id'], r['source']): sum(bool(x['contact']) for x in (r['result'].get('runs') or []))
              for r in _load(a.out, 'repro')}
    items = []
    for r in _load(a.out, 'policy'):
        x = r['result']
        if r['attacker'] == 'v2j' and x.get('reportable'):
            items.append((r['variant'], r['case_id'], f"policy_s{r['seed']}", 'actions', x['actions'], x['trace'], x['t_contact']))
    for r in _load(a.out, 'refine'):
        x = r['result']
        if x.get('found'):
            b = x['best']
            items.append((r['variant'], r['case_id'], 'refine', 'theta', b['theta'], b['aw_trace'], b['aw']['t_contact']))
    out = []
    for v, c, src, how, payload, trace, t_hit in items:
        case = _STATE['cases'][c]
        cc = _run(case, how, payload, 'cc_r157')
        aw = [dict(t=r['t'], a=r['ego'][3], v=r['ego'][2], y_adv=r['adv'][1], gap=r['gap'], clear=r['clear'], mrm=r.get('mrm'))
              for r in trace]
        ccs = [dict(t=round(.1 * (k + 1), 1), a=r['target_control']['applied_mps2'], v=r['target']['vx'],
                    y_adv=r['generator']['y'], gap=r['generator']['x'] - r['target']['x'] - 5.) for k, r in enumerate(cc['trace'])]
        k_aw = next((p['t'] for p in aw if abs(p['y_adv']) >= PERCEIVE_M), None)
        k_cc = next((p['t'] for p in ccs if abs(p['y_adv']) >= PERCEIVE_M), None)
        out.append(dict(variant=v, case_id=c, source=src, t_contact=t_hit, t_cutin_aw=k_aw, t_cutin_cc=k_cc,
                        cc_contact=bool(cc['pair_contact']), cc_clearance=round(float(clearance(cc['trace'])), 3),
                        rerun_contacts=reruns.get((v, c, src)), aw=aw, cc=ccs))
    Path(a.figdata).write_text(json.dumps(out), encoding='utf-8')
    print(json.dumps(dict(failures=len(out), by_variant={v: sum(1 for o in out if o['variant'] == v) for v in sorted({o['variant'] for o in out})})))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('figdata')
    main(ap.parse_args())
