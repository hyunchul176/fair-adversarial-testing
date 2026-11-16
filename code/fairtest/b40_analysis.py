"""B40 analysis (B40 design, available on request).

  M0  stage 1 reproduces the stored v2 refinement (first-success query and plan) for >= 99% of (case, SUT)
  M1  following_H0 and reference_idm_H0: median r_up of the reported failure >= 1.25 s
  M2  pooled over SUTs with >= 10 failures, the share with r_up >= 1.25 s rises by >= 20 pp over stage 1

  python -m fairtest.b40_analysis OUT.json B40.jsonl STORED_V2.jsonl
"""
from __future__ import annotations

import json
import statistics
import sys


def main(out, path, stored_path):
    stored = {}
    for line in open(stored_path, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == 'bo_warm_g4g':
                b = rec.get('best') or {}
                stored[(r['case_id'], s)] = (rec.get('first_fair'), [round(x, 9) for x in b.get('theta', [])])
    same = total = errors = 0
    per = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        for s, rec in r['runs'].items():
            if 'error' in rec:
                errors += 1
                continue
            total += 1
            st = rec['stage1']
            q = st.get('first_fair')
            th = [round(x, 9) for x in (st.get('best') or {}).get('theta', [])]
            same += stored.get((r['case_id'], s)) == (q, th)
            d = per.setdefault(s, dict(n=0, found=0, up1=[], upf=[], level=[]))
            d['n'] += 1
            if 'final' in rec:
                d['found'] += 1
                d['up1'].append(rec['r_up_stage1'])
                d['upf'].append(rec['r_up_final'])
                d['level'].append(rec['level_reached'])
    rep = dict(total=total, errors=errors, M0=dict(identical=same, share=same / total if total else None,
                                                   supported=bool(total) and same / total >= .99), sut={})
    for s, d in sorted(per.items()):
        f = d['found']
        rep['sut'][s] = dict(cases=d['n'], detection=f / d['n'],
                             r_up_median_stage1=statistics.median(d['up1']) if f else None,
                             r_up_median_final=statistics.median(d['upf']) if f else None,
                             ge_1_25_stage1=sum(x >= 1.25 for x in d['up1']) / f if f else None,
                             ge_1_25_final=sum(x >= 1.25 for x in d['upf']) / f if f else None,
                             ge_1_5_final=sum(x >= 1.5 for x in d['upf']) / f if f else None,
                             case_share_fair_at_1_25=sum(x >= 1.25 for x in d['upf']) / d['n'],
                             case_share_fair_at_1_5=sum(x >= 1.5 for x in d['upf']) / d['n'],
                             levels={f'{lv:g}': d['level'].count(lv) for lv in sorted(set(d['level']))})
    m1 = {s: rep['sut'].get(s, {}).get('r_up_median_final') for s in ('following_H0', 'reference_idm_H0')}
    rep['M1'] = dict(median=m1, supported=all(v is not None and v >= 1.25 for v in m1.values()))
    big = [s for s, d in per.items() if d['found'] >= 10]
    a = sum(sum(x >= 1.25 for x in per[s]['up1']) for s in big) / max(1, sum(per[s]['found'] for s in big))
    b = sum(sum(x >= 1.25 for x in per[s]['upf']) for s in big) / max(1, sum(per[s]['found'] for s in big))
    rep['M2'] = dict(suts=big, stage1=a, final=b, rise=b - a, supported=b - a >= .20)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('total', 'errors', 'M0', 'M1', 'M2')}, indent=1))
    for s, x in rep['sut'].items():
        print(f"{s:22s} det {x['detection']:.2f}  r_up median {x['r_up_median_stage1']} -> {x['r_up_median_final']}  "
              f">=1.25: {x['ge_1_25_stage1']} -> {x['ge_1_25_final']}  cases fair@1.25 {x['case_share_fair_at_1_25']:.2f} "
              f"@1.5 {x['case_share_fair_at_1_5']:.2f}  levels {x['levels']}")


if __name__ == '__main__':
    main(*sys.argv[1:4])
