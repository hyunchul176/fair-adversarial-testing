"""B14 analysis: realism-constrained falsification vs realism-constrained fair adversary.

  python -m fairtest.b14_analysis OUT.json NAME=B14.jsonl[,B14b.jsonl] [NAME=...]
Rates are % of cases; attacker-at-fault (AAF) shares are % of hits.
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import numpy as np

REFS = ('cc_r157', 'cc_r157_strict', 'ref_fsm', 'ref_rss')
SUTS = ('following_H0', 'reference_idm_H0', 'following_H0.5', 'following_H1', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')


def ci(v, seed=20260926, boot=4000):
    v = np.asarray(v, float)
    rng = random.Random(seed)
    idx = np.array([[rng.randrange(len(v)) for _ in range(len(v))] for _ in range(boot)])
    d = v[idx].mean(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def summarize(paths):
    per = collections.defaultdict(list)
    for p in paths:
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if r.get('error'):
                raise RuntimeError(f"case {r['case_id']}: {r['error']}")
            for key, rec in r['runs'].items():
                b = rec['best']
                hit = bool(b and b['hit'])
                refs = rec.get('refs') or {}
                per[key].append(dict(hit=float(hit), queries=rec['sut_queries'],
                                     **{f'crash_{n}': float(hit and refs.get(n, False)) for n in REFS}))
    out = {}
    for key, rows in sorted(per.items()):
        hits = sum(r['hit'] for r in rows)
        item = dict(n=len(rows), hit_rate=100 * float(np.mean([r['hit'] for r in rows])), hit_ci=[100 * x for x in ci([r['hit'] for r in rows])],
                    mean_queries=float(np.mean([r['queries'] for r in rows])))
        for n in REFS:
            item[f'aaf_share_{n}'] = (100 * sum(r[f'crash_{n}'] for r in rows) / hits) if hits else None
        item['consensus_fair_rate'] = 100 * float(np.mean([r['hit'] * (1 - max(r['crash_cc_r157'], r['crash_ref_fsm'], r['crash_ref_rss'])) for r in rows]))
        item['strict_fair_rate'] = 100 * float(np.mean([r['hit'] * (1 - r['crash_cc_r157_strict']) for r in rows]))
        out[key] = item
    return out


if __name__ == '__main__':
    rep = {}
    for spec in sys.argv[2:]:
        name, paths = spec.split('=')
        rep[name] = summarize(paths.split(','))
    Path(sys.argv[1]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for name, d in rep.items():
        print('==', name)
        for s in SUTS:
            f, a = d.get(f'falsify_real|{s}'), d.get(f'fair_real|{s}')
            if not f:
                continue
            fa = f['aaf_share_cc_r157']
            print(f"  {s:20} falsify_real hits {f['hit_rate']:5.1f}% AAF(CC) {'NA' if fa is None else round(fa)}% AAF(strict) "
                  f"{'NA' if f['aaf_share_cc_r157_strict'] is None else round(f['aaf_share_cc_r157_strict'])}% cons-fair {f['consensus_fair_rate']:4.1f}%"
                  f" | fair_real hits {a['hit_rate']:5.1f}% strict-fair {a['strict_fair_rate']:5.1f}% cons-fair {a['consensus_fair_rate']:5.1f}% q {a['mean_queries']:.1f}")
