"""Analysis of E5 and E8 (v2): closed-loop verdicts per condition (same rules as e2_analyze). E5-A no shield, E5-B shield with the default twin, E2 shield with the matched twin,
E8 the three learned baselines.

  python e5_e8_analyze.py WORK_DIR      -> runs/E5_RESULT.json, runs/E8_RESULT.json
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_analyze import COMPETENT, ORDER, load  # noqa: E402

BA = ('ba:cautious', 'ba:normal', 'ba:aggressive')


def condition(work, prefix):
    runs = {s: load(work / f'runs/{prefix}_{s}.jsonl') for s in ('s1', 's2', 's3') if (work / f'runs/{prefix}_{s}.jsonl').exists()}
    if not runs:
        return None
    allv = [v for d in runs.values() for v in d.values()]
    by = {}
    for kind in ORDER:
        c = collections.Counter(v for d in runs.values() for (case, k), v in d.items() if k == kind)
        n = sum(c.values()) - c['error']
        by[kind] = dict(n=n, **{x: c[x] for x in ('robust', 'fair', 'attacker', 'other', 'none', 'error')},
                        robust_rate=round(100 * c['robust'] / n, 1) if n else None)
    contacts = sum(v in ('robust', 'fair', 'attacker', 'other') for v in allv)
    att = sum(v == 'attacker' for v in allv)
    return dict(seeds=sorted(runs), runs=len(allv), attacker=att,
                attacker_by_seed={s: sum(v == 'attacker' for v in d.values()) for s, d in runs.items()},
                contact_rate=round(100 * contacts / len(allv), 1), attacker_share=round(100 * att / contacts, 1) if contacts else None,
                ba_attacker=sum(by[k]['attacker'] for k in BA), by_kind=by)


def main(work):
    names = dict(E5A='E5A_noshield', E5B='E5B_hwshield', E2='E2_rl')
    e5 = {k: condition(work, p) for k, p in names.items()}
    e5 = {k: v for k, v in e5.items() if v}
    if {'E5A', 'E5B', 'E2'} <= set(e5):
        a, b, m = (e5[k]['attacker'] for k in ('E5A', 'E5B', 'E2'))
        e5['E5_H1'] = dict(A=a, supported=a >= 1)
        e5['E5_H2'] = dict(A=a, B=b, E2=m, supported=a >= b >= m)
    json.dump(e5, open(work / 'runs/E5_RESULT.json', 'w'), indent=1)
    e8 = {k: condition(work, f'E8_{k}') for k in ('collide_commit', 'cars', 'dynasto')}
    e8 = {k: v for k, v in e8.items() if v}
    json.dump(e8, open(work / 'runs/E8_RESULT.json', 'w'), indent=1)
    for title, res in (('E5', e5), ('E8', e8)):
        print(f'== {title}')
        for k, v in res.items():
            if not isinstance(v, dict) or 'by_kind' not in v:
                print('  ', k, v)
                continue
            bk = v['by_kind']
            print(f"   {k:15} runs {v['runs']:4d} contact {v['contact_rate']:5.1f}% at fault {v['attacker']:4d} {v['attacker_by_seed']} ({v['attacker_share']}% of contacts)"
                  f" | H0 robust {bk['ladder:following:0']['robust_rate']}% | BA robust {[bk[x]['robust_rate'] for x in BA]} BA at fault {v['ba_attacker']}"
                  f" | competent robust {[bk[x]['robust_rate'] for x in COMPETENT]}")


if __name__ == '__main__':
    main(Path(sys.argv[1]))
