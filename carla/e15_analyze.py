"""E15 analysis (addendum 8): vehicle-model sensitivity. The E1 trajectories replayed with the Tesla Model 3 (a: feed-forward braking, b: PI braking), shown next to
the Mustang (E1).

  python e15_analyze.py WORK_DIR      -> runs/E15_RESULT.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e11_e12_analyze import KIND, composed, outcomes  # noqa: E402


def main(work):
    e1 = outcomes(work.glob('runs/E1_replay_*.jsonl'))
    pairs = composed(work, 'dev')
    res = {}
    for cond in ('a', 'b'):
        e15 = outcomes(work.glob(f'runs/E15{cond}_replay_*.jsonl'))
        if not e15:
            continue
        keys = [k for k in e1 if k in e15]
        res[cond] = dict(trajectories=len(keys),
                         agreement_transfer=round(100 * sum(e1[k]['transfer'] == e15[k]['transfer'] for k in keys) / len(keys), 1),
                         agreement_sut=round(100 * sum(e1[k]['sut_hit'] == e15[k]['sut_hit'] for k in keys) / len(keys), 1),
                         agreement_cc=round(100 * sum(e1[k]['cc_hit'] == e15[k]['cc_hit'] for k in keys) / len(keys), 1), by_sut={})
        for sut in KIND:
            sel = [key for s, key in pairs if s == sut and key in e1 and key in e15]
            if not sel:
                continue
            res[cond]['by_sut'][sut] = dict(
                n=len(sel), mustang=round(100 * sum(e1[k]['transfer'] for k in sel) / len(sel), 1),
                tesla=round(100 * sum(e15[k]['transfer'] for k in sel) / len(sel), 1),
                tesla_cc_hit=round(100 * sum(e15[k]['cc_hit'] for k in sel) / len(sel), 1),
                mustang_cc_hit=round(100 * sum(e1[k]['cc_hit'] for k in sel) / len(sel), 1),
                tesla_sut_avoid=round(100 * sum(not e15[k]['sut_hit'] for k in sel) / len(sel), 1),
                mustang_sut_avoid=round(100 * sum(not e1[k]['sut_hit'] for k in sel) / len(sel), 1))
    if 'a' in res and 'following_H0' in res['a']['by_sut']:
        d = res['a']['by_sut']['following_H0']
        res['E15_H1'] = dict(mustang=d['mustang'], tesla=d['tesla'], supported=abs(d['tesla'] - d['mustang']) <= 5.)
    json.dump(res, open(work / 'runs/E15_RESULT.json', 'w'), indent=1)
    for cond in ('a', 'b'):
        if cond not in res:
            continue
        r = res[cond]
        print(f"== E15{cond} trajectories {r['trajectories']}  transfer agreement {r['agreement_transfer']}%  SUT {r['agreement_sut']}%  C {r['agreement_cc']}%")
        for s, v in r['by_sut'].items():
            print(f"   {s:18} n {v['n']:4d}  transfer Mustang {v['mustang']}  Tesla {v['tesla']}  |  C collision {v['mustang_cc_hit']} -> {v['tesla_cc_hit']}"
                  f"  SUT avoided {v['mustang_sut_avoid']} -> {v['tesla_sut_avoid']}")
    print('E15-H1', res.get('E15_H1'))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
