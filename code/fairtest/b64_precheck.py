"""B64 pre-check: bo_warm_g4ss with the severity floor off (v_min = -1000) must reproduce the stored B56 bo_warm_g4s runs
(first success, SUT queries, best plan) for every (case, SUT) it ran. Exit code 1 on any difference.

  python -m fairtest.b64_precheck PRECHECK.jsonl STORED_B56.jsonl
"""
from __future__ import annotations

import json
import sys


def runs(path, method):
    out = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == method:
                out[(r['case_id'], s)] = rec
    return out


def main(new_path, stored_path):
    new, old = runs(new_path, 'bo_warm_g4ss'), runs(stored_path, 'bo_warm_g4s')
    diff, errors = [], [k for k, v in new.items() if 'error' in v]
    for k, a in new.items():
        b = old.get(k)
        if b is None or 'error' in a or 'error' in b:
            diff.append([*k, 'missing or error'])
            continue
        same = (a.get('first_fair') == b.get('first_fair') and a.get('sut_queries') == b.get('sut_queries')
                and all(abs(x - y) < 1e-9 for x, y in zip(a['best']['theta'], b['best']['theta'])))
        if not same:
            diff.append([*k, a.get('first_fair'), b.get('first_fair'), a.get('sut_queries'), b.get('sut_queries')])
    rep = dict(pairs=len(new), errors=len(errors), differing=len(diff), rows=diff[:20], identical=len(new) - len(diff))
    print(json.dumps(rep))
    return 0 if (new and not diff and not errors) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
