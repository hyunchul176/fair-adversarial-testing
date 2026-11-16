"""E16 (design addendum 9): re-judge the stored attacks with six C settings in a twin (highway-env) matched to the run environment.

The twin's two event positions (the perception threshold of C and the lateral distance at which the bodies overlap) are matched to the run environment. The width
is set with env.VEH_W (both vehicles) and the perception threshold by swapping in perceive_m of env.CCDriver (the code is unchanged). The adversary vehicle follows the stored actions or plan (as in E1b).

  python e16_matched.py [--workers 10] [--limit N]                                 -> runs/E16_matched.jsonl
  python e16_matched.py --conds PM2,PT2 --out runs/E16b_matched.jsonl                (addendum 10, E16b)
  python e16_matched.py --conds PMx,MMx,PTx --out runs/E16c_matched.jsonl            (addendum 11, E16c)
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / 'code'))
S = 4. / 3.5
CONDS = {  # name: (C target, twin width (None = 2 m), perception threshold m, reaction-phase deceleration m/s^2)
    'U': ('cc_r157', None, 1.095, -.4),
    'Cal': ('cc_r157_cal', None, 1.095, -.4),
    'Prev': ('cc_r157_cal', 2.39, 1.095, -.4),
    'PM': ('cc_r157_cal', round(S * 1.895, 3), round(1.095 * S, 3), -.4),
    'MM': ('cc_r157_cal', 1.895, 1.595, -.4),
    'PT': ('cc_r157_cal', round(S * 2.16, 3), round(1.095 * S, 3), -.4),
    # addendum 10 (E16b): reaction-phase deceleration set to the value measured in CARLA (Mustang about -0.3, Tesla about 0)
    'PM2': ('cc_r157_cal', round(S * 1.895, 3), round(1.095 * S, 3), -.3),
    'PT2': ('cc_r157_cal', round(S * 2.16, 3), round(1.095 * S, 3), 0.),
    # addendum 11 (E16c): R157 reaction 0.75 s, emergency level set to the measured 7.35 m/s^2 (fifth value)
    'PMx': ('cc_r157', round(S * 1.895, 3), round(1.095 * S, 3), -.4, -7.35),
    'MMx': ('cc_r157', 1.895, 1.595, -.4, -7.35),
    'PTx': ('cc_r157', round(S * 2.16, 3), round(1.095 * S, 3), 0., -7.35),
}
DEV_ONLY = ('PT', 'PT2', 'PTx')
TAGS = ('dev', 'tgsim', 'mitra', 'highsim', 'exid')
POLICY = {'dev': 'jobs/b26/final/final_s{n}_dev.jsonl', 'tgsim': 'jobs/b28/final_s{n}.jsonl', 'mitra': 'jobs/b41/final_s{n}.jsonl',
          'highsim': 'jobs/b70/final_s{n}.jsonl', 'exid': 'jobs/b71/final_s{n}.jsonl'}
EMERGENCY0 = None
DATA = {'dev': 'data', 'tgsim': 'data_tgsim', 'mitra': 'data_mitra', 'highsim': 'data_highsim', 'exid': 'data_exid'}


def _task(job):
    import fairtest.cc_target as CT
    import fairtest.env as E
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_crossjudge import replay
    from fairtest.b23_margin import clearance
    from fairtest.cc_target import CCDriver
    from fairtest.runner import _STATE, _init, run_episode
    _init()
    global EMERGENCY0
    if EMERGENCY0 is None:
        EMERGENCY0 = dict(E.CC_EMERGENCY)
    tag, key, case_id, actions, theta, conds = job
    case = _STATE['cases'][case_id]
    out = dict(tag=tag, key=key)
    for name in conds:
        target, width, perceive, react = CONDS[name][:4]
        CT.REACTION_MPS2 = react
        E.CC_EMERGENCY = dict(EMERGENCY0, **({target: CONDS[name][4]} if len(CONDS[name]) > 4 else {}))

        class GeoCC(CCDriver):
            def __init__(self, *a, perceive_m=perceive, **k):
                super().__init__(*a, perceive_m=perceive_m, **k)
        E.VEH_W, E.CCDriver = width, GeoCC
        if actions is not None:
            r = replay(case, actions, target, 0.)
        else:
            r = run_episode(case, ContinuousPlan.from_theta(theta), target, 0., keep_trace=True)
        hit = bool(r['pair_contact'])
        out[name] = dict(contact=hit, clearance=None if hit else round(float(clearance(r['trace'])), 3))
    return out


def jobs_of(tag, names):
    acts = {}
    for n in (1, 2, 3):
        for line in open(W / POLICY[tag].format(n=n), encoding='utf-8'):
            r = json.loads(line)
            acts[(f'policy_s{n}', r['case_id'], r['sut'])] = r['actions']
    src = W / ('runs/E1_trajectories.jsonl' if tag == 'dev' else f'runs/E6_{tag}_trajectories.jsonl')
    conds = [c for c in names if tag == 'dev' or c not in DEV_ONLY]
    out = []
    for line in open(src, encoding='utf-8'):
        r = json.loads(line)
        key = f"{r['source']}|{r['case_id']}|{r['sut']}"
        if r['source'].startswith('policy_'):
            out.append((tag, key, r['case_id'], acts[(r['source'], r['case_id'], r['sut'])], None, conds))
        else:
            out.append((tag, key, r['case_id'], None, r['hwenv']['theta'], conds))
    return out


def main(a):
    ctx = mp.get_context('fork')
    n = 0
    names = a.conds.split(',')
    with open(W / a.out, 'w', encoding='utf-8') as f:
        for tag in TAGS:
            jobs = jobs_of(tag, names)[:a.limit or None]
            os.environ['FAIRTEST_DATA'] = str(W / DATA[tag])     # each dataset has its own case pool, so a new worker pool is started
            with ctx.Pool(a.workers) as pool:
                for res in pool.imap(_task, jobs, chunksize=4):
                    f.write(json.dumps(res) + '\n')
                    n += 1
            f.flush()
            print(json.dumps(dict(tag=tag, done=len(jobs))), flush=True)
    print(json.dumps(dict(total=n)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--conds', default='U,Cal,Prev,PM,MM,PT')
    ap.add_argument('--out', default='runs/E16_matched.jsonl')
    main(ap.parse_args())
