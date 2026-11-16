"""B38 analysis (B38 design, available on request).

The v2 refinement's robust fair-failure rate per SUT (dev, bo_warm_g4g, 5 and 36 queries) with the careful driver's
reaction at 0.75 s (B29, guarded_dev.jsonl) and at 1.0 / 1.25 / 1.5 s (B38).
  S1  following_H0 at 1.25 s >= 70%      S2  following_aeb_H0/H0.5, rss_sut_H0, drl_sut_H0 <= 5% at every level
  S3  following_H0 - rss_sut_H0 >= 50 pp at 1.25 s

  python -m fairtest.b38_analysis OUT.json R075.jsonl R1.jsonl R1.25.jsonl R1.5.jsonl
"""
from __future__ import annotations

import json
import sys

LEVELS = ('0.75', '1', '1.25', '1.5')
SPECIFIC = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'drl_sut_H0')


def rates(path, k):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    out, errors = {}, 0
    for r in rows:
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m != 'bo_warm_g4g':
                continue
            if 'error' in rec:
                errors += 1
            q = rec.get('first_fair')
            out.setdefault(s, []).append(q is not None and q <= k)
    return {s: sum(v) / len(v) for s, v in out.items()}, len(rows), errors


def main(out, *paths):
    rep = dict(levels=LEVELS, k36={}, k5={}, cases={}, errors={})
    for lvl, p in zip(LEVELS, paths):
        rep['k36'][lvl], rep['cases'][lvl], rep['errors'][lvl] = rates(p, 36)
        rep['k5'][lvl] = rates(p, 5)[0]
    k = rep['k36']
    rep['S1'] = dict(following_H0=k['1.25'].get('following_H0'), supported=k['1.25'].get('following_H0', 0.) >= .70)
    rep['S2'] = dict(max_specific={lvl: max(k[lvl].get(s, 0.) for s in SPECIFIC) for lvl in LEVELS},
                     supported=all(k[lvl].get(s, 0.) <= .05 for lvl in LEVELS for s in SPECIFIC))
    d = k['1.25'].get('following_H0', 0.) - k['1.25'].get('rss_sut_H0', 0.)
    rep['S3'] = dict(diff=d, supported=d >= .50)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({x: rep[x] for x in ('cases', 'errors', 'S1', 'S2', 'S3')}, indent=1))
    suts = sorted(k['0.75'])
    print('SUT                     ' + '  '.join(f'r={lvl:>4s}' for lvl in LEVELS) + '   (k = 36; k = 5 in brackets)')
    for s in suts:
        print(f'{s:22s}  ' + '  '.join(f"{k[lvl].get(s, float('nan')):.2f}({rep['k5'][lvl].get(s, float('nan')):.2f})" for lvl in LEVELS))


if __name__ == '__main__':
    main(*sys.argv[1:])
