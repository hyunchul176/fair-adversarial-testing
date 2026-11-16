"""E16v2-SUT analysis (v2 addendum v2-2): do the SUT collision verdicts of twins U and M predict the CARLA (E1, E6) SUT verdicts, and the CARLA transfer
rate of the attacks re-confirmed as fair failures in M (the SUT collides, C avoids with a gap >= 0.5 m).

  python e16v2_sut_analyze.py WORK_DIR      -> runs/E16v2_SUT_RESULT.json
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e11_e12_analyze import KIND  # noqa: E402

TAGS = ('dev', 'tgsim', 'mitra', 'highsim', 'exid')
WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')


def main(work):
    sut_twin, cc_twin, carla = {}, {}, {}
    for l in open(work / 'runs/E16v2_sut.jsonl', encoding='utf-8'):
        r = json.loads(l)
        sut_twin[(r['tag'], r['key'])] = r
    for l in open(work / 'runs/E16v2_matched.jsonl', encoding='utf-8'):
        r = json.loads(l)
        cc_twin[(r['tag'], r['key'])] = r
    for tag in TAGS:
        pat = 'runs/E1_replay_*.jsonl' if tag == 'dev' else f'runs/E6_{tag}_replay_*.jsonl'
        for p in glob.glob(str(work / pat)):
            for line in open(p, encoding='utf-8'):
                r = json.loads(line)
                s, c = r['carla'].get(KIND[r['sut']], {}), r['carla'].get('cc', {})
                if 'contact' in s and 'contact' in c:
                    carla[(tag, f"{r['source']}|{r['case_id']}|{r['sut']}")] = (bool(s['contact']), bool(c['contact']))
    res = dict(by_sut={})
    for sut in KIND:
        keys = [k for k in carla if k[1].split('|')[2] == sut and k in sut_twin]
        if not keys:
            continue
        d = dict(n=len(keys), carla_sut_hit=round(100 * sum(carla[k][0] for k in keys) / len(keys), 1))
        for c in ('U', 'M'):
            d[f'agree_{c}'] = round(100 * sum(sut_twin[k][c]['contact'] == carla[k][0] for k in keys) / len(keys), 1)
            d[f'twin_{c}_hit'] = round(100 * sum(sut_twin[k][c]['contact'] for k in keys) / len(keys), 1)
        res['by_sut'][sut] = d
    weak = [k for k in carla if k[1].split('|')[2] in WEAK and k in sut_twin]
    wa = {c: sum(sut_twin[k][c]['contact'] == carla[k][0] for k in weak) / len(weak) for c in ('U', 'M')}
    res['H1'] = dict(weak_agree_U=round(100 * wa['U'], 1), weak_agree_M=round(100 * wa['M'], 1), supported=bool(wa['M'] >= .9 and wa['M'] > wa['U']))
    h0 = [k for k in carla if k[1].split('|')[2] == 'following_H0' and k in sut_twin and k in cc_twin]
    sel = [k for k in h0 if sut_twin[k]['M']['contact'] and not cc_twin[k]['M']['contact'] and (cc_twin[k]['M']['clearance'] or 0) >= .5]
    tr = sum(carla[k][0] and not carla[k][1] for k in sel) / len(sel) if sel else None
    raw = sum(carla[k][0] and not carla[k][1] for k in h0) / len(h0) if h0 else None
    res['H2'] = dict(following_H0=len(h0), recertified=len(sel), transfer_recertified=None if tr is None else round(100 * tr, 1),
                     transfer_all=None if raw is None else round(100 * raw, 1), supported=bool(tr is not None and tr >= .9))
    json.dump(res, open(work / 'runs/E16v2_SUT_RESULT.json', 'w'), indent=1)
    print('SUT | n | CARLA SUT collision | U predicted | M predicted | U agreement | M agreement')
    for s, d in res['by_sut'].items():
        print(f"{s:18} {d['n']:5d} {d['carla_sut_hit']:5.1f} {d['twin_U_hit']:5.1f} {d['twin_M_hit']:5.1f} {d['agree_U']:5.1f} {d['agree_M']:5.1f}")
    print('H1', res['H1'])
    print('H2', res['H2'])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
