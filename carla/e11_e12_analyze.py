"""Analysis of E11 and E12 (addendum 6).
E11: per-trajectory agreement of transfer between the second replay of the E1 trajectories and the first (E1), and the difference in transfer rate per SUT.
E12: transfer rates per dataset and SUT of the E1 and E6 trajectories replayed with the distance-preserving mapping (y' = y + w - 4), next to the proportional mapping (E1, E6).

  python e11_e12_analyze.py WORK_DIR      -> runs/E11_RESULT.json, runs/E12_RESULT.json
"""
import json
import sys
from pathlib import Path

KIND = {'following_H0': 'ladder:following:0', 'following_H0.5': 'ladder:following:0.5', 'following_H1': 'ladder:following:1',
        'reference_idm_H0': 'ladder:reference_idm:0', 'following_aeb_H0': 'ladder:following_aeb:0',
        'following_aeb_H0.5': 'ladder:following_aeb:0.5', 'rss_sut_H0': 'ladder:rss_sut:0', 'rss_lat_sut_H0': 'ladder:rss_lat_sut:0'}


def outcomes(paths):
    out = {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            s, c = r['carla'].get(KIND[r['sut']], {}), r['carla'].get('cc', {})
            if 'error' in s or 'error' in c or 'contact' not in s or 'contact' not in c:
                continue
            out[f"{r['source']}|{r['case_id']}|{r['sut']}"] = dict(transfer=bool(s['contact'] and not c['contact']),
                                                                  sut_hit=bool(s['contact']), cc_hit=bool(c['contact']))
    return out


def composed(work, tag):
    path = work / ('runs/E1_composition.json' if tag == 'dev' else f'runs/E6_{tag}_composition.json')
    comp = json.load(open(path))['composition']
    return [(k.split('|')[2], v['key']) for k, v in comp.items() if v]


def rate(pairs, oc, sut):
    sel = [oc[key] for s, key in pairs if s == sut and key in oc]
    return (round(sum(x['transfer'] for x in sel) / len(sel), 4), len(sel)) if sel else (None, 0)


def main(work):
    res11 = {}
    a = outcomes(work.glob('runs/E1_replay_*.jsonl'))
    b = outcomes(work.glob('runs/E11_replay_*.jsonl'))
    keys = [k for k in a if k in b]
    if keys:
        res11 = dict(trajectories=len(keys), agreement_transfer=round(sum(a[k]['transfer'] == b[k]['transfer'] for k in keys) / len(keys), 4),
                     agreement_sut=round(sum(a[k]['sut_hit'] == b[k]['sut_hit'] for k in keys) / len(keys), 4),
                     agreement_cc=round(sum(a[k]['cc_hit'] == b[k]['cc_hit'] for k in keys) / len(keys), 4), by_sut={})
        pairs = composed(work, 'dev')
        for sut in KIND:
            r1, n = rate(pairs, a, sut)
            r2, _ = rate(pairs, b, sut)
            if n:
                res11['by_sut'][sut] = dict(n=n, first=r1, second=r2, diff=None if r1 is None or r2 is None else round(r2 - r1, 4))
        json.dump(res11, open(work / 'runs/E11_RESULT.json', 'w'), indent=1)
        print('E11', json.dumps({k: v for k, v in res11.items() if k != 'by_sut'}))
        for s, v in res11['by_sut'].items():
            print(f'   {s:18} n {v["n"]:4d}  first {v["first"]}  second {v["second"]}  diff {v["diff"]}')
    res12 = {}
    for tag in ('dev', 'tgsim', 'mitra', 'highsim', 'exid'):
        e12 = outcomes(work.glob(f'runs/E12_{tag}_replay_*.jsonl'))
        if not e12:
            continue
        base = outcomes(work.glob('runs/E1_replay_*.jsonl') if tag == 'dev' else work.glob(f'runs/E6_{tag}_replay_*.jsonl'))
        pairs = composed(work, tag)
        res12[tag] = {}
        for sut in KIND:
            r0, n = rate(pairs, base, sut)
            r1, m = rate(pairs, e12, sut)
            if n:
                res12[tag][sut] = dict(n=n, proportional=r0, metric=r1, n_metric=m)
    if res12:
        res12['E12_H1'] = {t: dict(value=d.get('following_H0', {}).get('metric'),
                                   supported=(d.get('following_H0', {}).get('metric') or 0) >= .90) for t, d in res12.items()}
        json.dump(res12, open(work / 'runs/E12_RESULT.json', 'w'), indent=1)
        for tag, d in res12.items():
            if tag == 'E12_H1':
                continue
            print(f'== E12 {tag}')
            for s, v in d.items():
                print(f'   {s:18} n {v["n"]:4d}  proportional {v["proportional"]}  distance-preserving {v["metric"]} (n {v["n_metric"]})')
        print('E12-H1', res12['E12_H1'])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
