"""E9 analysis (addendum 5, post hoc): split the reasons for non-transfer into geometry and physics. The :hw kinds let the controller see highway-env geometry under CARLA physics.

  python e9_analyze.py WORK_DIR      -> runs/E9_RESULT.json, prints a table
"""
import collections
import json
import sys
from pathlib import Path

KIND = {'following_H0': 'ladder:following:0', 'following_H0.5': 'ladder:following:0.5', 'following_H1': 'ladder:following:1',
        'reference_idm_H0': 'ladder:reference_idm:0', 'following_aeb_H0': 'ladder:following_aeb:0',
        'following_aeb_H0.5': 'ladder:following_aeb:0.5', 'rss_sut_H0': 'ladder:rss_sut:0', 'rss_lat_sut_H0': 'ladder:rss_lat_sut:0'}
WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')


def main(work):
    sel = json.load(open(work / 'runs/E9_selection.json'))
    rep = {}
    for p in work.glob('runs/E9_*_replay_*.jsonl'):
        tag = p.stem.split('_')[1]
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            rep[(tag, f"{r['source']}|{r['case_id']}|{r['sut']}")] = r['carla']
    cnt = collections.defaultdict(collections.Counter)
    missing = 0
    for x in sel:
        c = rep.get((x['tag'], x['traj']))
        if c is None:
            missing += 1
            continue
        s, cc = c.get(KIND[x['sut']] + ':hw', {}), c.get('cc:hw', {})
        if 'error' in s or 'error' in cc or not s or not cc:
            cnt['errors']['n'] += 1
            continue
        grp = 'weak' if x['sut'] in WEAK else 'capable'
        if x['group'] == 'control':                       # transferred trajectories: does :hw change the verdict (baseline variation)
            cnt['control']['n'] += 1
            cnt['control']['sut_flip'] += not s['contact']
            cnt['control']['cc_flip'] += bool(cc['contact'])
            continue
        if not x['sut_hit']:                              # the CARLA SUT avoided
            k = f'sut_avoided|{grp}'
            cnt[k]['n'] += 1
            cnt[k]['geometry'] += bool(s['contact'])      # collides when it sees highway-env geometry -> geometry is the reason
            cnt[k]['physics'] += not s['contact']
        if x['cc_hit']:                                   # the CARLA C also collided
            k = f'cc_hit|{grp}'
            cnt[k]['n'] += 1
            cnt[k]['geometry'] += not cc['contact']       # avoids when it sees highway-env geometry -> geometry is the reason
            cnt[k]['physics'] += bool(cc['contact'])
    res = {k: dict(v) for k, v in cnt.items()}
    res['missing'] = missing
    json.dump(res, open(work / 'runs/E9_RESULT.json', 'w'), indent=1)
    for k, v in sorted(res.items()):
        if isinstance(v, dict) and 'n' in v and v['n']:
            extra = {a: f"{b} ({100 * b / v['n']:.0f}%)" for a, b in v.items() if a != 'n'}
            print(f'{k:22} n={v["n"]:4d}', extra)
    print('missing', missing)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
