"""B33 analysis (B33 design, available on request).

  R1  share of the final agent's reported failures that R157 5.2.5.2 explicitly mandates (descriptive)
  R2  among the baselines' CC-attacker-at-fault collisions, the mandated share <= 10% (primary)
  R3  among the CC grid's mandated cut-ins, the CC avoids >= 90%
  R4  regulation-explicit grid oracle per ladder / ACC SUT (dev): cases with a mandated SUT collision, next to the
      CC grid oracle (cases with a SUT collision the CC avoids); capability-rank tau

  python -m fairtest.b33_analysis OUT.json B33_DIR
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from scipy.stats import kendalltau

COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
RANK = {'following_H0': 1, 'reference_idm_H0': 1, 'following_H0.5': 2, 'following_H1': 3, 'following_aeb_H0': 4,
        'following_aeb_H0.5': 5, 'rss_sut_H0': 6}
LABELS = ('mandated', 'mandated_strict', 'mandated_loose')


def _rows(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]


def family(source):
    return next((f for f in ('collide_commit', 'cars', 'dynasto', 'naive_fair') if source.startswith(f)), source)


def share(rows, key):
    return round(sum(bool(r.get(key)) for r in rows) / len(rows), 4) if rows else None


def why(rows):
    """Which 5.2.5.2 condition fails (not exclusive), among non-mandated rows with an intrusion."""
    rs = [r for r in rows if r.get('intrusion') and not r['mandated']]
    if not rs:
        return {}
    return dict(n=len(rs), no_a=share([dict(x=not (r['adv_max_abs_acc'] <= .5 + 1e-9 and r['faster'])) for r in rs], 'x'),
                no_b=share([dict(x=not r['b_lenient']) for r in rs], 'x'), no_c=share([dict(x=not r['c_ok']) for r in rs], 'x'))


def jobs_summary(rows):
    ok = [r for r in rows if 'error' not in r]
    out = dict(rows=len(rows), errors=len(rows) - len(ok), sut_no_contact=sum(not r['sut_contact'] for r in ok))
    agent = [r for r in ok if r['stage'] in ('policy', 'refine')]
    if agent:
        out['agent'] = {st: dict(n=len(rs), **{k: share(rs, k) for k in LABELS}, intrusion=share(rs, 'intrusion'),
                                 why_not=why(rs))
                        for st in ('policy', 'refine', 'all')
                        for rs in [[r for r in agent if st == 'all' or r['stage'] == st]] if rs}
    base = [r for r in ok if r['stage'] == 'baseline']
    if base:
        groups = collections.defaultdict(list)
        for r in base:
            groups[family(r['source'])].append(r)
            groups['all'].append(r)
        out['baseline'] = {}
        for g, rs in sorted(groups.items()):
            att = [r for r in rs if r['cc_contact']]
            fair = [r for r in rs if not r['cc_contact']]
            comp = [r for r in att if r['sut'] in COMPETENT]
            out['baseline'][g] = dict(n=len(rs), attacker=len(att), **{f'attacker_{k}': share(att, k) for k in LABELS},
                                      fair=len(fair), fair_mandated=share(fair, 'mandated'),
                                      competent_attacker=len(comp), competent_attacker_mandated=share(comp, 'mandated'),
                                      why_not_attacker=why(att))
    return out


def grid_cc(rows):
    ok = [r for r in rows if 'error' not in r and r.get('valid', True)]
    intr = [r for r in ok if r['intrusion']]
    out = dict(rows=len(rows), errors=sum('error' in r for r in rows), valid=len(ok), intrusion=len(intr))
    for k in LABELS:
        m = [r for r in intr if r[k]]
        out[k] = dict(n=len(m), cc_avoids=share([dict(x=not r['contact']) for r in m], 'x'))
    cc_col = [r for r in intr if r['contact']]
    out['cc_collisions'] = dict(n=len(cc_col), mandated=share(cc_col, 'mandated'))
    bins = collections.defaultdict(list)
    for r in intr:
        if r['mandated']:
            bins[min(int(r['v_rel_int'] // 5) * 5, 20)].append(r)
    out['mandated_by_vrel'] = {f'{b}-{b + 5}' if b < 20 else '20+': dict(n=len(rs), cc_avoids=share([dict(x=not r['contact']) for r in rs], 'x'))
                               for b, rs in sorted(bins.items())}
    return out


def grid_sut(rows, cc_rows):
    cc = {(r['case_id'], json.dumps(r['spec'])): r for r in cc_rows if 'error' not in r}
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(mandated=False, cc_fair=False, conflict=False)))
    cases = set()
    errors = 0
    for r in rows:
        if 'error' in r:
            errors += 1
            continue
        if not r.get('valid', True):
            continue
        sut = f"{r['target']}_H{r['H']:g}"
        cases.add(r['case_id'])
        c = cc.get((r['case_id'], json.dumps(r['spec'])))
        rec = per[sut][r['case_id']]
        if r['contact'] and r['mandated']:
            rec['mandated'] = True
            if c is not None and c['contact']:
                rec['conflict'] = True
        if r['contact'] and c is not None and not c['contact']:
            rec['cc_fair'] = True
    n = len(cases)
    out = dict(errors=errors, cases=n, sut={})
    for sut, recs in sorted(per.items()):
        m = sum(v['mandated'] for v in recs.values()) / n
        f = sum(v['cc_fair'] for v in recs.values()) / n
        u = sum(v['mandated'] or v['cc_fair'] for v in recs.values()) / n
        k = sum(v['conflict'] for v in recs.values()) / n
        out['sut'][sut] = dict(mandated_oracle=round(m, 4), cc_oracle=round(f, 4), union_oracle=round(u, 4),
                               mandated_cc_collides=round(k, 4))
    lad = [s for s in RANK if s in out['sut']]
    for key in ('mandated_oracle', 'cc_oracle', 'union_oracle'):
        out[f'tau_{key}'] = float(kendalltau([RANK[s] for s in lad], [out['sut'][s][key] for s in lad]).statistic)
    return out


def main(out, d):
    d = Path(d)
    rep = dict(jobs={}, grid_cc={}, grid_sut=None)
    for p in sorted(d.glob('jobs_*.jsonl')):
        rep['jobs'][p.stem[5:]] = jobs_summary(_rows(p))
    pooled = [r for p in sorted(d.glob('jobs_*.jsonl')) for r in _rows(p) if 'error' not in r]
    att = [r for r in pooled if r['stage'] == 'baseline' and r['cc_contact']]
    agent = [r for r in pooled if r['stage'] in ('policy', 'refine')]
    rep['R1'] = {k: share(agent, k) for k in LABELS} | dict(n=len(agent))
    rep['R2'] = {k: share(att, k) for k in LABELS} | dict(n=len(att), supported=share(att, 'mandated') <= .10)
    cc_rows = {}
    for p in sorted(d.glob('grid_cc_*.jsonl')):
        cc_rows[p.stem[8:]] = _rows(p)
        rep['grid_cc'][p.stem[8:]] = grid_cc(cc_rows[p.stem[8:]])
    allm = [r for rs in cc_rows.values() for r in rs if 'error' not in r and r.get('valid', True) and r.get('mandated')]
    rep['R3'] = dict(n=len(allm), cc_avoids=share([dict(x=not r['contact']) for r in allm], 'x'))
    rep['R3']['supported'] = rep['R3']['cc_avoids'] is not None and rep['R3']['cc_avoids'] >= .9
    if (d / 'grid_sut_dev.jsonl').exists() and 'dev' in cc_rows:
        rep['grid_sut'] = rep['R4'] = grid_sut(_rows(d / 'grid_sut_dev.jsonl'), cc_rows['dev'])
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('R1', 'R2', 'R3')}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:3])
