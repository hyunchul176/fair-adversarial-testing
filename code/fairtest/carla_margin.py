"""E7: decision margin of the fair failures found in CARLA (G4).

Re-runs every fair failure found by the in-CARLA search (stored plan) against the SUT and the CARLA-CC with the
clearance metric: how far the careful driver stayed from the adversary (smallest box distance), and whether the
SUT's contact was a front contact (adversary centre >= 3.5 m ahead) or a side/rear contact.

  python -m fairtest.carla_margin OUT.jsonl SEARCH.jsonl [...] [--port 2000]
"""
from __future__ import annotations

import argparse
import json


def main(a):
    from fairtest.carla_replay import BLUEPRINT, Road, connect, run
    from fairtest.carla_search import adversary_row
    from fairtest.runner import _STATE, _init
    _init()
    base = _STATE['base']
    world = connect(a.port)
    road = Road(world)
    bp = world.get_blueprint_library().find(BLUEPRINT)
    try:
        with open(a.out, 'w', encoding='utf-8') as f:
            for path in a.searches:
                for line in open(path, encoding='utf-8'):
                    r = json.loads(line)
                    for key, v in r['runs'].items():
                        objective, kind = key.split('|', 1)
                        found = v.get('found')
                        if not found:
                            continue
                        valid, row = adversary_row(_STATE['cases'][r['case_id']], found['theta'])
                        sut = run(world, road, bp, row, kind, base)
                        cc = run(world, road, bp, row, 'cc', base)
                        f.write(json.dumps(dict(case_id=r['case_id'], objective=objective, kind=kind, valid=valid, found=found,
                                                sut=sut, cc=cc)) + '\n')
                        f.flush()
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('searches', nargs='+')
    ap.add_argument('--port', type=int, default=2000)
    main(ap.parse_args())
