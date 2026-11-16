"""E1 analysis: CARLA transfer rate of the final version's attacks (seed, case and SUT combinations) and why some did not transfer.

  python e1_analyze.py WORK_DIR FAIRTEST_DIR      -> WORK_DIR/runs/E1_RESULT.json, prints a table
transfer = in CARLA the SUT collides and C avoids. robust = transfer with CARLA contact offset contact_dx >= 3.5 m and CARLA C minimum gap >= 0.5 m.
"""
import collections
import json
import statistics
import sys
from pathlib import Path

KIND = {'following_H0': 'ladder:following:0', 'following_H0.5': 'ladder:following:0.5', 'following_H1': 'ladder:following:1',
        'reference_idm_H0': 'ladder:reference_idm:0', 'following_aeb_H0': 'ladder:following_aeb:0', 'rss_sut_H0': 'ladder:rss_sut:0'}
WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')


def main(work, b7):
    comp = json.load(open(work / 'runs/E1_composition.json'))
    rep = {}
    for p in work.glob('runs/E1_replay_*.jsonl'):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            rep[f"{r['source']}|{r['case_id']}|{r['sut']}"] = r
    hw = {}                                    # highway-env reference values: C minimum gap, contact offset
    for line in open(b7 / 'b26/final/margin_dev.jsonl', encoding='utf-8'):
        r = json.loads(line)
        hw[f"policy_{r['source'].split('_')[1]}|{r['case_id']}|{r['sut']}"] = (r['cc_clearance'], r['contact_dx'])
    for line in open(work / 'runs/E1_trajectories.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['source'] == 'refine_g4s':
            hw[f"refine_g4s|{r['case_id']}|{r['sut']}"] = (r['hwenv']['cc_clearance'], r['hwenv']['contact_dx'])
    rows = []
    for key, v in comp['composition'].items():
        if not v:
            continue
        seed, case, sut = key.split('|')
        r = rep.get(v['key'])
        row = dict(seed=seed, case=int(case), sut=sut, stage=v['stage'], traj=v['key'])
        if r is None:
            rows.append(dict(row, status='missing'))
            continue
        s, c = r['carla'][KIND[sut]], r['carla']['cc']
        if 'error' in s or 'error' in c:
            rows.append(dict(row, status='error', error=s.get('error') or c.get('error')))
            continue
        sut_hit, cc_hit = bool(s['contact']), bool(c['contact'])
        hcl, hdx = hw.get(v['key'], (None, None))
        rows.append(dict(row, status='ok', sut_hit=sut_hit, cc_hit=cc_hit, transfer=sut_hit and not cc_hit,
                         robust=sut_hit and not cc_hit and (s.get('contact_dx') or -1e9) >= 3.5 and c['min_clearance'] >= .5,
                         carla_cc_clear=c['min_clearance'], hw_cc_clear=hcl, carla_dx=s.get('contact_dx'), hw_dx=hdx,
                         carla_t=s.get('t_contact'), sut_min_gap=s.get('min_gap')))

    def summ(sel):
        ok = [x for x in sel if x['status'] == 'ok']
        n = len(ok)
        d = dict(n=len(sel), n_ok=n, errors=sum(x['status'] == 'error' for x in sel), missing=sum(x['status'] == 'missing' for x in sel))
        if n:
            d.update(transfer=round(sum(x['transfer'] for x in ok) / n, 4), robust=round(sum(x['robust'] for x in ok) / n, 4),
                     lost_fair_cc_hit=sum(x['cc_hit'] for x in ok), lost_fail_sut_avoided=sum(not x['sut_hit'] for x in ok),
                     both=sum(x['cc_hit'] and not x['sut_hit'] for x in ok))
            pairs = [(x['carla_cc_clear'], x['hw_cc_clear']) for x in ok if not x['cc_hit'] and x['hw_cc_clear'] is not None]
            if pairs:
                d['cc_clear_diff_median'] = round(statistics.median(a - b for a, b in pairs), 3)
                d['cc_clear_carla_median'] = round(statistics.median(a for a, _ in pairs), 3)
                d['cc_clear_hw_median'] = round(statistics.median(b for _, b in pairs), 3)
        return d

    res = dict(by_sut={}, following_H0_by_seed={}, weak_pooled=None, unique_trajectories={})
    for sut in KIND:
        res['by_sut'][sut] = summ([x for x in rows if x['sut'] == sut])
    for s in comp['seeds']:
        res['following_H0_by_seed'][s] = summ([x for x in rows if x['sut'] == 'following_H0' and x['seed'] == s])
    res['weak_pooled'] = summ([x for x in rows if x['sut'] in WEAK])
    uniq = {x['traj']: x for x in rows}                       # count a trajectory once if several seeds use it
    for sut in KIND:
        res['unique_trajectories'][sut] = summ([x for x in uniq.values() if x['sut'] == sut])
    res['by_stage'] = {st: summ([x for x in rows if x['stage'] == st]) for st in ('policy', 'refine')}
    fh0 = res['by_sut']['following_H0']
    res['E1_H1'] = dict(statement='following H0 transfer (3 seeds pooled) >= 90%', value=fh0.get('transfer'),
                        supported=(fh0.get('transfer') or 0) >= .90)
    res['not_transferred'] = [x for x in rows if x['status'] == 'ok' and not x['transfer']]
    res['rows'] = rows
    json.dump(res, open(work / 'runs/E1_RESULT.json', 'w'), indent=1)
    pct = lambda v: '-' if v is None else f'{100 * v:5.1f}%'
    print(f"{'SUT':18} {'n':>4} {'transfer':>7} {'robust':>7} {'C hit':>6} {'SUT avoid':>7} {'err':>4}  C gap (CARLA/hw)")
    for sut, d in list(res['by_sut'].items()) + [('four weak, pooled', res['weak_pooled'])]:
        print(f"{sut:18} {d['n']:4d} {pct(d.get('transfer')):>7} {pct(d.get('robust')):>7} {d.get('lost_fair_cc_hit', '-'):>6} "
              f"{d.get('lost_fail_sut_avoided', '-'):>7} {d['errors']:>4}  {d.get('cc_clear_carla_median', '-')}/{d.get('cc_clear_hw_median', '-')}")
    print('following H0 by seed:', {s: pct(d.get('transfer')) for s, d in res['following_H0_by_seed'].items()})
    print('by stage:', {s: (d['n'], pct(d.get('transfer'))) for s, d in res['by_stage'].items()})
    print('E1-H1:', res['E1_H1'])


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
