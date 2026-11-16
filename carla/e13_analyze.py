"""E13 analysis (addendum 8): the collision-only search (inside CARLA) and its post-hoc verdicts, next to E4 S1.

  python e13_analyze.py WORK_DIR      -> runs/E13_RESULT.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_analyze import COMPETENT, KINDS, boot  # noqa: E402


def main(work):
    rows, errors = [], 0
    for line in open(work / ('runs/E13_collide_merged.jsonl' if (work / 'runs/E13_collide_merged.jsonl').exists() else 'runs/E13_collide.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        if 'error' in r:
            errors += 1
        else:
            rows.append(r)
    e4 = json.load(open(work / 'runs/E4_RESULT.json'))['by'] if (work / 'runs/E4_RESULT.json').exists() else {}
    res = dict(n=len(rows), errors=errors, by={})
    pool = dict(found=0, attacker=0, final=0)
    for kind in KINDS:
        sel = [r for r in rows if r['kind'] == kind]
        if not sel:
            continue
        found = [r['found'] for r in sel if r.get('found')]
        twin = [r['twin']['S1'] for r in sel if r.get('found')]
        final = [bool(f['robust'] and t['admissible']) for f, t in zip(found, twin)]
        q = [f['query'] for f in found]
        d = dict(n=len(sel), found=len(found), found_rate=round(100 * len(found) / len(sel), 1), ci=boot([bool(r.get('found')) for r in sel]),
                 attacker=sum(f['cc_contact'] for f in found), carla_fair=sum(f['fair'] for f in found),
                 carla_robust=sum(f['robust'] for f in found), twin_s1_admissible=sum(t['admissible'] for t in twin),
                 final=sum(final), final_rate_of_cases=round(100 * sum(final) / len(sel), 1),
                 queries_median=float(np.median(q)) if q else None, e4_s1_rate=e4.get(f'S1|{kind}', {}).get('rate'))
        d['attacker_share'] = round(100 * d['attacker'] / len(found), 1) if found else None
        res['by'][kind] = d
        if kind in COMPETENT:
            pool['found'] += len(found)
            pool['attacker'] += d['attacker']
            pool['final'] += d['final']
    res['E13_H1'] = dict(rates={k: v['found_rate'] for k, v in res['by'].items()},
                         supported=bool(res['by']) and all(v['found_rate'] >= 70. for v in res['by'].values()))
    res['E13_H2'] = dict(pool, share=round(100 * pool['attacker'] / pool['found'], 1) if pool['found'] else None,
                         by_kind={k: res['by'][k]['attacker_share'] for k in COMPETENT if k in res['by']},
                         supported=bool(pool['found']) and pool['attacker'] / pool['found'] >= .8)
    res['E13_H3'] = dict(final=pool['final'], found=pool['found'],
                         share=round(100 * pool['final'] / pool['found'], 1) if pool['found'] else None,
                         supported=bool(pool['found']) and pool['final'] / pool['found'] <= .05)
    json.dump(res, open(work / 'runs/E13_RESULT.json', 'w'), indent=1)
    print(f"E13 rows {res['n']} errors {errors}")
    print(f"{'kind':24} {'n':>3} {'found%':>6} {'95% CI':>14} {'at fault':>6} {'C fair':>5} {'C robust':>5} {'twin S1':>8} {'final':>6} {'med q':>6} {'E4 S1%':>7}")
    for k, d in res['by'].items():
        print(f"{k:24} {d['n']:3d} {d['found_rate']:6.1f} {str(d['ci']):>14} {d['attacker']:6d} {d['carla_fair']:5d} {d['carla_robust']:5d} "
              f"{d['twin_s1_admissible']:8d} {d['final']:6d} {str(d['queries_median']):>6} {str(d['e4_s1_rate']):>7}")
    for h in ('E13_H1', 'E13_H2', 'E13_H3'):
        print(h, res[h])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
