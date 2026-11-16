"""Analysis of E2 and E3: per-SUT results and repeatability of the closed loop (policies s1–s3 + shield with the C matched to CARLA).

  python e2_analyze.py WORK_DIR          -> WORK_DIR/runs/E2_RESULT.json, prints a table
Verdicts (one case and SUT): robust = fair failure with contact_dx >= 3.5 m and C minimum gap >= 0.5 m. fair = other fair failures.
attacker = collision and the CARLA C also collides (attacker at fault). other = other collisions (invalid etc.). none = no contact.
"""
import collections
import json
import sys
from pathlib import Path

ORDER = ('ladder:following:0', 'ladder:following_aeb:0', 'ladder:rss_sut:0', 'ladder:rss_lat_sut:0', 'ba:cautious', 'ba:normal',
         'ba:aggressive', 'tm:default')
COMPETENT = ('ladder:following_aeb:0', 'ladder:rss_sut:0', 'ladder:rss_lat_sut:0', 'tm:default')


def verdict(v):
    if 'error' in v:
        return 'error'
    if not v['contact']:
        return 'none'
    if v['fair']:
        return 'robust' if (v.get('contact_dx') or -1e9) >= 3.5 and (v.get('cc_min_clearance') or 0.) >= .5 else 'fair'
    if v.get('cc_contact'):
        return 'attacker'
    return 'other'


def load(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        for kind, v in r['runs'].items():
            out[(r['case_id'], kind)] = verdict(v)
    return out


def main(work):
    runs = {s: load(work / f'runs/E2_rl_{s}.jsonl') for s in ('s1', 's2', 's3') if (work / f'runs/E2_rl_{s}.jsonl').exists()}
    res = dict(by_seed={}, pooled={}, E2_H1=None, E2_H2=None, E3=None)
    for s, d in runs.items():
        res['by_seed'][s] = {}
        for kind in ORDER:
            c = collections.Counter(v for (case, k), v in d.items() if k == kind)
            n = sum(c.values()) - c['error']
            res['by_seed'][s][kind] = dict(n=n, **{k: c[k] for k in ('robust', 'fair', 'attacker', 'other', 'none', 'error')},
                                           robust_rate=round(c['robust'] / n, 4) if n else None)
    for kind in ORDER:
        c = collections.Counter(v for d in runs.values() for (case, k), v in d.items() if k == kind)
        n = sum(c.values()) - c['error']
        res['pooled'][kind] = dict(n=n, **{k: c[k] for k in ('robust', 'fair', 'attacker', 'other', 'none', 'error')},
                                   robust_rate=round(c['robust'] / n, 4) if n else None)
    att = sum(x['attacker'] for x in res['pooled'].values())
    res['E2_H1'] = dict(statement='attacker-caused collisions = 0 (all SUTs, all seeds)', attacker_collisions=att, supported=att == 0)
    comp = {k: res['pooled'][k]['robust_rate'] for k in COMPETENT}
    res['E2_H2'] = dict(statement='competent SUTs: robust fair failure <= 3% each (seeds pooled)', rates=comp,
                        supported=all(v is not None and v <= .03 for v in comp.values()))
    rep = work / 'runs/E3_rl_s1_rep.jsonl'
    if rep.exists() and 's1' in runs:
        a, b = runs['s1'], load(rep)
        keys = [k for k in a if k in b and 'error' not in (a[k], b[k])]
        same = sum(a[k] == b[k] for k in keys)
        same_bin = sum((a[k] == 'robust') == (b[k] == 'robust') for k in keys)
        res['E3'] = dict(statement='verdict agreement of repeated s1 runs >= 95%', pairs=len(keys), agreement=round(same / len(keys), 4) if keys else None,
                         agreement_robust_binary=round(same_bin / len(keys), 4) if keys else None,
                         differing=[[c, k, a[(c, k)], b[(c, k)]] for (c, k) in keys if a[(c, k)] != b[(c, k)]],
                         supported=bool(keys) and same / len(keys) >= .95)
    json.dump(res, open(work / 'runs/E2_RESULT.json', 'w'), indent=1)
    pct = lambda v: '   -' if v is None else f'{100 * v:5.1f}%'
    print(f"{'SUT':24} " + ' '.join(f'{s:>7}' for s in runs) + f" {'pooled':>7}  robust/fair/attacker/other/none (pooled)")
    for kind in ORDER:
        p = res['pooled'][kind]
        print(f"{kind:24} " + ' '.join(f"{pct(res['by_seed'][s][kind]['robust_rate']):>7}" for s in runs) +
              f" {pct(p['robust_rate']):>7}  {p['robust']}/{p['fair']}/{p['attacker']}/{p['other']}/{p['none']}" + (f" errors {p['error']}" if p['error'] else ''))
    print('E2-H1:', res['E2_H1']['attacker_collisions'], res['E2_H1']['supported'], '| E2-H2:', res['E2_H2']['supported'], res['E2_H2']['rates'])
    if res['E3']:
        print('E3:', res['E3']['pairs'], res['E3']['agreement'], res['E3']['agreement_robust_binary'], res['E3']['supported'])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
