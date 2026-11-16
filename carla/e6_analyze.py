"""E6 analysis (design addendum 2): per dataset, the CARLA transfer rate of the final version's attacks, why some did not transfer, the transfer rate per
highway-env C gap bin, and the transfer rate after screening with the C matched to CARLA. The highway-env reference gap is that of the re-judged default C (cc_r157).

  python e6_analyze.py WORK_DIR [tag ...]      -> WORK_DIR/runs/E6_RESULT.json, prints a table
"""
import json
import sys
from pathlib import Path

KIND = {'following_H0': 'ladder:following:0', 'following_H0.5': 'ladder:following:0.5', 'following_H1': 'ladder:following:1',
        'reference_idm_H0': 'ladder:reference_idm:0', 'following_aeb_H0': 'ladder:following_aeb:0',
        'following_aeb_H0.5': 'ladder:following_aeb:0.5', 'rss_sut_H0': 'ladder:rss_sut:0', 'rss_lat_sut_H0': 'ladder:rss_lat_sut:0'}
WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
BINS = ((0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 99.))


def rows_of(work, tag):
    comp = json.load(open(work / f'runs/E6_{tag}_composition.json'))
    rep = {}
    for p in work.glob(f'runs/E6_{tag}_replay_*.jsonl'):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            rep[f"{r['source']}|{r['case_id']}|{r['sut']}"] = r
    rj = {r['key']: r for r in map(json.loads, open(work / f'runs/E6_{tag}_rejudge.jsonl', encoding='utf-8'))}
    rows = []
    for key, v in comp['composition'].items():
        if not v:
            continue
        seed, case, sut = key.split('|')
        r, j = rep.get(v['key']), rj.get(v['key'])
        row = dict(ds=comp['ds'], seed=seed, case=int(case), sut=sut, stage=v['stage'], traj=v['key'])
        if r is None or j is None:
            rows.append(dict(row, status='missing'))
            continue
        s, c = r['carla'][KIND[sut]], r['carla']['cc']
        if 'error' in s or 'error' in c:
            rows.append(dict(row, status='error'))
            continue
        hit, cch = bool(s['contact']), bool(c['contact'])
        cal = j['cc_r157_cal']
        rows.append(dict(row, status='ok', sut_hit=hit, cc_hit=cch, transfer=hit and not cch,
                         robust=hit and not cch and (s.get('contact_dx') or -1e9) >= 3.5 and c['min_clearance'] >= .5,
                         hw_clear=j['cc_r157']['clearance'], hw_cc_contact=j['cc_r157']['contact'],
                         cal_ok=(not cal['contact']) and cal['clearance'] >= .5, carla_clear=c['min_clearance']))
    return rows


def summ(sel):
    ok = [x for x in sel if x['status'] == 'ok']
    n = len(ok)
    d = dict(n=len(sel), n_ok=n, errors=sum(x['status'] != 'ok' for x in sel))
    if n:
        keep = [x for x in ok if x['cal_ok']]
        d.update(transfer=round(sum(x['transfer'] for x in ok) / n, 4), robust=round(sum(x['robust'] for x in ok) / n, 4),
                 cc_hit=sum(x['cc_hit'] for x in ok), sut_avoided=sum(not x['sut_hit'] for x in ok),
                 cal_kept=len(keep), cal_transfer=round(sum(x['transfer'] for x in keep) / len(keep), 4) if keep else None)
    return d


def main(work, tags):
    res = dict(by_ds={}, pooled={}, bins={}, E6_H1={}, E6_H2=None)
    allrows = []
    for tag in tags:
        rows = rows_of(work, tag)
        allrows += rows
        d = res['by_ds'][tag] = {sut: summ([x for x in rows if x['sut'] == sut]) for sut in KIND if any(x['sut'] == sut for x in rows)}
        d['weak_pooled'] = summ([x for x in rows if x['sut'] in WEAK])
        d['by_stage'] = {st: summ([x for x in rows if x['stage'] == st]) for st in ('policy', 'refine')}
        if 'following_H0' in d:
            res['E6_H1'][tag] = dict(value=d['following_H0'].get('transfer'), supported=(d['following_H0'].get('transfer') or 0) >= .90)
    ok = [x for x in allrows if x['status'] == 'ok']
    res['pooled'] = dict(all=summ(allrows), weak=summ([x for x in allrows if x['sut'] in WEAK]))
    for lo, hi in BINS:
        sel = [x for x in ok if x['hw_clear'] is not None and lo <= x['hw_clear'] < hi]
        res['bins'][f'{lo}-{hi}'] = dict(n=len(sel), transfer=round(sum(x['transfer'] for x in sel) / len(sel), 4) if sel else None,
                                          cc_hit=sum(x['cc_hit'] for x in sel), sut_avoided=sum(not x['sut_hit'] for x in sel))
    kept = [x for x in ok if x['cal_ok']]
    res['E6_H2'] = dict(statement='calibrated-C 0.5 m filter: transfer of kept attacks >= 90% (all datasets pooled)', kept=len(kept),
                        of=len(ok), value=round(sum(x['transfer'] for x in kept) / len(kept), 4) if kept else None,
                        supported=bool(kept) and sum(x['transfer'] for x in kept) / len(kept) >= .90)
    json.dump(res, open(work / 'runs/E6_RESULT.json', 'w'), indent=1)
    pct = lambda v: '    -' if v is None else f'{100 * v:5.1f}%'
    for tag in tags:
        d = res['by_ds'][tag]
        print(f'== {tag}')
        for sut, x in d.items():
            if sut == 'by_stage':
                continue
            print(f"   {sut:18} n {x['n']:4d} transfer {pct(x.get('transfer'))} robust {pct(x.get('robust'))} C hit {x.get('cc_hit', '-'):>4} SUT avoided {x.get('sut_avoided', '-'):>4} "
                  f"| screened with matched C: kept {x.get('cal_kept', '-'):>4}, transfer {pct(x.get('cal_transfer'))} errors {x['errors']}")
        print('   by stage:', {k: (v['n'], pct(v.get('transfer'))) for k, v in d['by_stage'].items()})
    print('== pooled', {k: (v['n'], pct(v.get('transfer')), 'cal', v.get('cal_kept'), pct(v.get('cal_transfer'))) for k, v in res['pooled'].items()})
    print('== gap bins', {k: (v['n'], pct(v['transfer'])) for k, v in res['bins'].items()})
    print('E6-H1', res['E6_H1'])
    print('E6-H2', res['E6_H2'])


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2:] or ['tgsim', 'mitra', 'highsim', 'exid'])
