"""E4 analysis (addendum 3) and comparison with E14 (addendum 8, partial repeat of E4).

  python e4_analyze.py WORK_DIR      -> runs/E4_RESULT.json (+ runs/E14_RESULT.json if E14 exists)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

KINDS = ('ladder:following:0', 'ladder:following_aeb:0', 'ladder:rss_sut:0', 'ladder:rss_lat_sut:0', 'ba:cautious', 'ba:normal',
         'ba:aggressive', 'tm:default')
COMPETENT = ('ladder:following_aeb:0', 'ladder:rss_sut:0', 'ladder:rss_lat_sut:0', 'tm:default')


def boot(x, n=2000, seed=0):
    """Case bootstrap 95% interval (one search per case, so the 0/1 values are resampled)."""
    x = np.asarray(x, float)
    if not len(x):
        return None
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [round(100 * float(np.percentile(m, 2.5)), 1), round(100 * float(np.percentile(m, 97.5)), 1)]


def load(path):
    rows, errors = {}, 0
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if 'error' in r:
            errors += 1
            continue
        rows[(r['case_id'], r['kind'], r['arm'])] = r
    return rows, errors


def verdict(r):
    """Success and CARLA C check: (success, CARLA C collision, CARLA fair, CARLA robust)."""
    ok = r.get('first_success') is not None
    c = r.get('carla_cc') or {}
    if not ok:
        return False, None, None, None
    if 'contact' not in c:
        return True, None, None, None
    cc_hit = bool(c['contact'])
    robust = bool(not cc_hit and (c.get('min_clearance') or 0.) >= .5 and ((r.get('best') or {}).get('contact_dx') or -1e9) >= 3.5)
    return True, cc_hit, not cc_hit, robust


def main(work):
    rows, errors = load(work / 'runs/E4_search.jsonl')
    res = dict(n=len(rows), errors=errors, by={})
    for arm in ('S0', 'S1'):
        for kind in KINDS:
            sel = [r for (c, k, a), r in rows.items() if k == kind and a == arm]
            if not sel:
                continue
            v = [verdict(r) for r in sel]
            succ = [x[0] for x in v]
            hits = [x for x in v if x[0]]
            q = [r['first_success'] for r in sel if r.get('first_success') is not None]
            stops = collections.Counter(r.get('stop') for r in sel)
            res['by'][f'{arm}|{kind}'] = dict(
                n=len(sel), success=sum(succ), rate=round(100 * float(np.mean(succ)), 1), ci=boot(succ),
                carla_cc_hit=sum(1 for x in hits if x[1]), carla_fair=sum(1 for x in hits if x[2]),
                carla_robust=sum(1 for x in hits if x[3]), carla_unchecked=sum(1 for x in hits if x[1] is None),
                queries_median=float(np.median(q)) if q else None, queries_mean=round(float(np.mean(q)), 1) if q else None,
                sut_queries=int(sum(r['sut_queries'] for r in sel)), cc_replays=int(sum(r['cc_replays'] for r in sel)),
                stops=dict(stops))
    s1_hits = [verdict(r) for (c, k, a), r in rows.items() if a == 'S1' and verdict(r)[0]]
    s1_cc = sum(1 for x in s1_hits if x[1])
    res['E4_H1'] = dict(successes=len(s1_hits), carla_cc_hit=s1_cc, share=round(100 * s1_cc / len(s1_hits), 1) if s1_hits else None,
                        supported=bool(s1_hits) and s1_cc / len(s1_hits) <= .05)
    res['E4_H2'] = {k: dict(rate=res['by'].get(f'S1|{k}', {}).get('rate'),
                            supported=bool((res['by'].get(f'S1|{k}', {}).get('rate') or 0.) <= 3.)) for k in COMPETENT}
    json.dump(res, open(work / 'runs/E4_RESULT.json', 'w'), indent=1)
    print(f"E4 rows {res['n']} errors {errors}")
    print(f"{'arm|kind':32} {'n':>3} {'succ%':>6} {'95% CI':>14} {'C hit':>5} {'C fair':>5} {'C robust':>5} {'med q':>6} {'C rep/q':>9}  stops")
    for key, d in res['by'].items():
        per = d['cc_replays'] / max(d['sut_queries'], 1)
        print(f"{key:32} {d['n']:3d} {d['rate']:6.1f} {str(d['ci']):>14} {d['carla_cc_hit']:5d} {d['carla_fair']:5d} {d['carla_robust']:5d} "
              f"{str(d['queries_median']):>6} {per:9.1f}  {d['stops']}")
    print('E4-H1', res['E4_H1'])
    print('E4-H2', res['E4_H2'])
    p14 = work / 'runs/E14_search.jsonl'
    if p14.exists():
        rep, err14 = load(p14)
        out = dict(n=len(rep), errors=err14, by={})
        agree_all = []
        for kind in ('ladder:following:0', 'ba:normal'):
            keys = [k for k in rep if k[1] == kind and k[2] == 'S1' and k in rows]
            a = [verdict(rows[k])[0] for k in keys]
            b = [verdict(rep[k])[0] for k in keys]
            agree = [x == y for x, y in zip(a, b)]
            agree_all += agree
            qd = [rep[k]['first_success'] - rows[k]['first_success'] for k in keys
                  if rep[k].get('first_success') is not None and rows[k].get('first_success') is not None]
            out['by'][kind] = dict(n=len(keys), first=round(100 * float(np.mean(a)), 1) if a else None, second=round(100 * float(np.mean(b)), 1) if b else None,
                                   diff_pp=round(100 * float(np.mean(b) - np.mean(a)), 1) if a else None,
                                   agreement=round(100 * float(np.mean(agree)), 1) if agree else None,
                                   both_success=sum(x and y for x, y in zip(a, b)),
                                   query_diff_median=float(np.median(qd)) if qd else None,
                                   query_same=sum(1 for d in qd if d == 0), query_n=len(qd))
        out['agreement'] = round(100 * float(np.mean(agree_all)), 1) if agree_all else None
        out['E14_H1'] = dict(supported=bool(agree_all and np.mean(agree_all) >= .85
                             and all(abs(d['diff_pp'] or 0) <= 10 for d in out['by'].values())))
        json.dump(out, open(work / 'runs/E14_RESULT.json', 'w'), indent=1)
        print('E14', json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
