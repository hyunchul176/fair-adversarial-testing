"""B17 analysis: RL adversaries across the SUT capability panel.

Per evaluation file and SUT: collision rate; share judged fair by each reference (CC, strict CC, FSM,
RSS, CC-FSM-RSS consensus, ideal ADS, omniscient); lateral feints (lane-goal reversals after the first
entry); realistic-cut-in share of the CC replay (cut-in completed, exceedance <= 1 IQR); guarded fair
failures (fair under CC with a realistic CC-replay cut-in); discrimination spread across SUTs.

  python -m fairtest.b17_analysis OUT.json EVAL1.jsonl [EVAL2.jsonl ...]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ('following_H0', 'reference_idm_H0', 'following_H0.5', 'following_H1', 'following_aeb_H0',
         'following_aeb_H0.5', 'rss_sut_H0')
REFS = ('cc_r157', 'cc_r157_strict', 'ref_fsm', 'ref_rss', 'oracle_react0', 'oracle_t0')


def reversals(actions):
    lanes = [a[0] for a in actions]
    first = next((i for i, l in enumerate(lanes) if l == 4.), None)
    if first is None:
        return 0
    return sum(1 for a, b in zip(lanes[first:], lanes[first + 1:]) if a == 4. and b == 0.)


def realistic(cc):
    return bool(cc and cc.get('cut_in') and cc.get('exc') is not None and cc['exc'] <= 1.)


def summarize(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    per = collections.defaultdict(list)
    for r in rows:
        per[r['sut']].append(r)
    out = {}
    for s in ORDER:
        rs = per.get(s, [])
        n = len(rs)
        if not n:
            continue
        coll = [r for r in rs if r['valid'] and r['contact']]
        item = dict(n=n, collision_rate=len(coll) / n,
                    feint_share=float(np.mean([reversals(r['actions']) > 0 for r in rs])),
                    mean_reversals=float(np.mean([reversals(r['actions']) for r in rs])))
        for ref in REFS:
            item[f'fair_rate_{ref}'] = sum(1 for r in coll if not r['refs'][ref]['contact']) / n
        item['fair_rate_consensus'] = sum(1 for r in coll if not any(r['refs'][x]['contact'] for x in ('cc_r157', 'ref_fsm', 'ref_rss'))) / n
        if all('cc_replay' in r for r in rs):
            item['guarded_fair_rate'] = sum(1 for r in coll if not r['refs']['cc_r157']['contact'] and realistic(r.get('cc_replay'))) / n
            item['realistic_share_of_fair'] = (sum(1 for r in coll if not r['refs']['cc_r157']['contact'] and realistic(r.get('cc_replay')))
                                              / max(1, sum(1 for r in coll if not r['refs']['cc_r157']['contact'])))
        out[s] = item
    for key in ('collision_rate', 'fair_rate_cc_r157', 'fair_rate_consensus', 'guarded_fair_rate'):
        vals = [out[s][key] for s in out if key in out[s]]
        if vals:
            out.setdefault('_spread', {})[key] = max(vals) - min(vals)
    return out


if __name__ == '__main__':
    rep = {Path(p).name: summarize(p) for p in sys.argv[2:]}
    Path(sys.argv[1]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for name, d in rep.items():
        print('==', name, 'spread', {k: round(100 * v, 1) for k, v in d.get('_spread', {}).items()})
        for s in ORDER:
            if s not in d:
                continue
            x = d[s]
            print(f"  {s:20} coll {100*x['collision_rate']:5.1f} | fair CC {100*x['fair_rate_cc_r157']:5.1f} strict {100*x['fair_rate_cc_r157_strict']:5.1f} "
                  f"cons {100*x['fair_rate_consensus']:5.1f} t0 {100*x['fair_rate_oracle_t0']:5.1f} | guarded {100*x.get('guarded_fair_rate', float('nan')):5.1f} "
                  f"| feints {100*x['feint_share']:5.1f}% (mean rev {x['mean_reversals']:.1f})")
