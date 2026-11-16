"""E7/B24: exhaustive plan-grid oracle inside CARLA for one SUT kind (the 199 B7 grid plans per case).

The grid oracle bounds what any attack search could find with these plans: a case counts if some valid cut-in plan
makes the SUT collide while the CARLA-CC avoids. For each development case and plan the adversary trajectory comes
from the highway-env generator (plans never read the target, so it is SUT-independent), the SUT is run in CARLA and,
if it collides, the CC as well. Rows carry the decision-margin fields (G4: contact_dx, CC clearance). A case whose
nominal plan (never entering) already makes the SUT collide is excluded, as in the highway-env grid oracle.

  python -m fairtest.carla_grid OUT.jsonl --kind ba:normal [--cases 0:21] [--port 2010] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from fairtest import devboot  # noqa: F401


def main(a):
    from fairtest.carla_replay import BLUEPRINT, Road, connect, run
    from fairtest.carla_search import trajectory_row
    from fairtest.plans import grid
    from fairtest.runner import _STATE, _init
    _init()
    base = _STATE['base']
    plans = grid()
    ids = sorted(c for c, v in _STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    lo, hi = (int(x) for x in a.cases.split(':')) if a.cases else (0, len(ids))
    ids = ids[lo:hi]
    done = set()
    if a.resume and os.path.exists(a.out):  # restart after a CARLA server crash: skip finished (case, plan) pairs
        done = {(r['case_id'], r['plan']) for r in (json.loads(l) for l in open(a.out, encoding='utf-8') if l.strip())}
    world = connect(a.port)
    road = Road(world)
    bp = world.get_blueprint_library().find(BLUEPRINT)
    try:
        with open(a.out, 'a', encoding='utf-8') as f:
            for i, c in enumerate(ids):
                case, t0 = _STATE['cases'][c], time.time()
                for plan in plans:
                    if (c, plan.name) in done:
                        continue
                    valid, row = trajectory_row(case, plan)
                    out = dict(case_id=c, plan=plan.name, kind=a.kind, valid=valid,
                               cut_in=any(p['y'] >= 2. for p in row['trajectory']))
                    if valid and out['cut_in'] or plan.name.startswith('plan_enever'):
                        try:
                            out['sut'] = run(world, road, bp, row, a.kind, base)
                            out['cc'] = run(world, road, bp, row, 'cc', base) if out['sut']['contact'] else None
                        except Exception as e:  # recorded, never dropped
                            out['error'] = f'{type(e).__name__}: {e}'
                    f.write(json.dumps(out) + '\n')
                    f.flush()
                print(json.dumps(dict(done=i + 1, of=len(ids), case=c, seconds=round(time.time() - t0, 1))), flush=True)
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


def oracle(paths, delta=.5):
    """Per case: excluded (nominal collides), fair (some valid cut-in plan: SUT collides, CC avoids), robust (G4)."""
    import collections
    by = collections.defaultdict(list)
    for path in paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            by[r['case_id']].append(r)
    out = {}
    for c, rs in by.items():
        nom = next((r for r in rs if r['plan'].startswith('plan_enever')), None)
        if nom is None or 'sut' not in nom or nom['sut']['contact']:
            out[c] = None
            continue
        hits = [r for r in rs if not r['plan'].startswith('plan_enever') and r.get('sut') and r['sut']['contact']]
        fair = [r for r in hits if r['cc'] is not None and not r['cc']['contact']]
        robust = [r for r in fair if r['sut'].get('contact_dx', -1e9) >= 3.5 and r['cc']['min_clearance'] >= delta]
        out[c] = dict(plans=sum(1 for r in rs if 'sut' in r), collisions=len(hits), fair=bool(fair), robust=bool(robust),
                      attacker_at_fault=sum(1 for r in hits if r['cc'] is not None and r['cc']['contact']))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--kind', required=True)
    ap.add_argument('--partition', default='development')
    ap.add_argument('--cases', help='slice start:end of the sorted case ids')
    ap.add_argument('--port', type=int, default=2010)
    ap.add_argument('--resume', action='store_true')
    main(ap.parse_args())
