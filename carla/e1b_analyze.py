"""E1b analysis (post hoc, outside the design): how well the highway-env verdicts of the C matched to CARLA (cc_r157_cal) predict the CARLA C outcomes,
and the transfer rate of the attacks screened with those verdicts.

  python e1b_analyze.py WORK_DIR      -> WORK_DIR/runs/E1b_RESULT.json
"""
import json
import sys
from pathlib import Path

SUTS = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'rss_sut_H0')


def main(work):
    cal = {r['key']: r for r in map(json.loads, open(work / 'runs/E1b_calibrated.jsonl', encoding='utf-8'))}
    rows = [x for x in json.load(open(work / 'runs/E1_RESULT.json'))['rows'] if x['status'] == 'ok']
    check = sum(abs((cal[x['traj']]['cc_r157']['clearance'] or 0.) - (x['hw_cc_clear'] or 0.)) < 1e-3 for x in rows
                if x['hw_cc_clear'] is not None and not cal[x['traj']]['cc_r157']['contact'])

    def pred(x, m):
        c = cal[x['traj']]['cc_r157_cal']
        return (not c['contact']) and c['clearance'] >= m

    res = dict(n=len(rows), stored_clearance_reproduced=check, by_margin={}, by_sut_margin05={})
    for m in (0.0, 0.5, 1.0):
        keep = [x for x in rows if pred(x, m)]
        agree = sum(pred(x, m) == (not x['cc_hit']) for x in rows)
        res['by_margin'][str(m)] = dict(kept=len(keep), agreement_with_carla_cc=round(agree / len(rows), 4),
                                        predicted_fair_but_carla_cc_hit=sum(pred(x, m) and x['cc_hit'] for x in rows),
                                        predicted_unfair_but_carla_cc_avoided=sum((not pred(x, m)) and not x['cc_hit'] for x in rows),
                                        transfer_of_kept=round(sum(x['transfer'] for x in keep) / len(keep), 4) if keep else None)
    for sut in SUTS:
        sel = [x for x in rows if x['sut'] == sut]
        keep = [x for x in sel if pred(x, .5)]
        res['by_sut_margin05'][sut] = dict(n=len(sel), transfer_all=round(sum(x['transfer'] for x in sel) / len(sel), 4) if sel else None,
                                           kept=len(keep), transfer_kept=round(sum(x['transfer'] for x in keep) / len(keep), 4) if keep else None,
                                           kept_cc_hit=sum(x['cc_hit'] for x in keep), kept_sut_avoided=sum(not x['sut_hit'] for x in keep))
    json.dump(res, open(work / 'runs/E1b_RESULT.json', 'w'), indent=1)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
