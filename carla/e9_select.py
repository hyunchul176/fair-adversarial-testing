"""E9 (addendum 5, post hoc): select all distinct non-transferred trajectories of E1 and E6 and 200 random transferred ones, and build the replay inputs per dataset and SUT.

  python e9_select.py WORK_DIR      -> runs/E9_<tag>_traj_<sut>.jsonl, runs/E9_selection.json
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_analyze import rows_of  # noqa: E402


def main(work):
    picked, ctrl_pool = {}, []                         # (tag, traj key) -> info
    e1 = json.load(open(work / 'runs/E1_RESULT.json'))['rows']
    for tag, rows in [('dev', e1)] + [(t, rows_of(work, t)) for t in ('tgsim', 'mitra', 'highsim', 'exid')]:
        for x in rows:
            if x['status'] != 'ok':
                continue
            key = (tag, x['traj'])
            info = dict(tag=tag, traj=x['traj'], sut=x['sut'], cc_hit=x['cc_hit'], sut_hit=x['sut_hit'], transfer=x['transfer'])
            if not x['transfer']:
                picked[key] = dict(info, group='not_transferred')
            else:
                ctrl_pool.append((key, info))
    ctrl = {}
    for key, info in ctrl_pool:
        ctrl.setdefault(key, info)
    random.seed(0)
    for key in random.sample(sorted(ctrl), 200):
        picked.setdefault(key, dict(ctrl[key], group='control'))
    by_file = {}
    for (tag, traj), info in picked.items():
        by_file.setdefault((tag, info['sut']), set()).add(traj)
    n = 0
    for (tag, sut), keys in by_file.items():
        src = work / ('runs/E1_trajectories.jsonl' if tag == 'dev' else f'runs/E6_{tag}_trajectories.jsonl')
        with open(work / f'runs/E9_{tag}_traj_{sut}.jsonl', 'w', encoding='utf-8') as f:
            for line in open(src, encoding='utf-8'):
                r = json.loads(line)
                if f"{r['source']}|{r['case_id']}|{r['sut']}" in keys:
                    f.write(line)
                    n += 1
    json.dump([dict(v, key=f'{k[0]}|{k[1]}') for k, v in picked.items()], open(work / 'runs/E9_selection.json', 'w'), indent=0)
    groups = {}
    for v in picked.values():
        groups[v['group']] = groups.get(v['group'], 0) + 1
    print(json.dumps(dict(selected=len(picked), written=n, groups=groups, files=len(by_file))))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
