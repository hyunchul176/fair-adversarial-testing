"""B32 analysis: how the R157 AEB layer (cc_r157_aeb, strongest reading) changes the cut-in verdicts.

  agent     share of the final agent's reported failures still avoided by the AEB reference (fair) and with >= 0.5 m
            (robust); the clearance change the layer brings
  baseline  attacker-at-fault share (the reference also collides) of the RL baselines' collisions under the CC and
            under the AEB reference; overall and for the competent SUTs (the B28 T4 group)
Every row must reproduce the stored CC verdict.

  python -m fairtest.b32_analysis OUT.json B32_DIR
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

DELTA = .5
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')


def family(source):
    for f in ('collide_commit', 'cars', 'dynasto', 'naive_fair'):
        if source.startswith(f):
            return f
    return source


def summarise(rows):
    out = dict(rows=len(rows), errors=sum('error' in r for r in rows))
    ok = [r for r in rows if 'error' not in r]
    out['reproduction_mismatches'] = sum(r['cc_contact'] != r['stored_cc_contact'] for r in ok)
    agent = [r for r in ok if r['stage'] in ('policy', 'refine')]
    if agent:
        a = {}
        for stage in ('policy', 'refine', 'all'):
            rs = [r for r in agent if stage == 'all' or r['stage'] == stage]
            if not rs:
                continue
            d = [r['aeb_clearance'] - r['cc_clearance'] for r in rs]
            a[stage] = dict(n=len(rs), fair_aeb=sum(not r['aeb_contact'] for r in rs) / len(rs),
                            robust_aeb=sum((not r['aeb_contact']) and r['aeb_clearance'] >= DELTA for r in rs) / len(rs),
                            engaged=sum(r['aeb_engaged'] for r in rs) / len(rs),
                            clearance_gain_median=sorted(d)[len(d) // 2], clearance_loss_cases=sum(x < -1e-9 for x in d))
        per_sut = collections.defaultdict(list)
        for r in agent:
            per_sut[r['sut']].append(r)
        a['not_fair_by_sut'] = {s: sum(x['aeb_contact'] for x in rs) for s, rs in per_sut.items() if any(x['aeb_contact'] for x in rs)}
        out['agent'] = a
    base = [r for r in ok if r['stage'] == 'baseline']
    if base:
        b = {}
        groups = collections.defaultdict(list)
        for r in base:
            groups[family(r['source'])].append(r)
            groups['all'].append(r)
        for g, rs in sorted(groups.items()):
            comp = [r for r in rs if r['sut'] in COMPETENT]
            b[g] = dict(n=len(rs), attacker_cc=sum(r['cc_contact'] for r in rs) / len(rs),
                        attacker_aeb=sum(r['aeb_contact'] for r in rs) / len(rs),
                        flipped_to_sut_fault=sum(r['cc_contact'] and not r['aeb_contact'] for r in rs),
                        competent_n=len(comp),
                        competent_attacker_cc=sum(r['cc_contact'] for r in comp) / len(comp) if comp else None,
                        competent_attacker_aeb=sum(r['aeb_contact'] for r in comp) / len(comp) if comp else None)
        out['baseline'] = b
    return out


def main(out, d):
    rep = {}
    for path in sorted(Path(d).glob('*.jsonl')):
        rows = [json.loads(line) for line in open(path, encoding='utf-8')]
        rep[path.stem] = summarise(rows)
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
