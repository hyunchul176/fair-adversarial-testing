"""E16v2 (v2 design §3): re-judge the stored attacks in the four v2 twins (U, M, MM, T). Under harness H2 the C in CARLA follows the R157 actuation
as is, so the twin C is the R157 C as is, and only the geometry (perception threshold, width, length) is changed with env.VEH_W, VEH_L and CC_PERCEIVE.

  python e16v2_matched.py [--workers 10]      -> runs/E16v2_matched.jsonl
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
CONDS = {  # v2 (harness H2): name -> (C target, width, length, perception threshold m). C is the R157 C as is, only the geometry is matched (design v2 §2)
    'U': ('cc_r157', None, None, 1.095),
    'M': ('cc_r157', round(S * 1.895, 3), 4.718, round(1.095 * S, 3)),
    'MM': ('cc_r157', 1.895, 4.718, 1.095 + 4. - 3.5),
    'T': ('cc_r157', round(S * 2.16, 3), 4.79, round(1.095 * S, 3)),
}
DEV_ONLY = ('T',)
TAGS = ('dev', 'tgsim', 'mitra', 'highsim', 'exid')
POLICY = {'dev': 'jobs/b26/final/final_s{n}_dev.jsonl', 'tgsim': 'jobs/b28/final_s{n}.jsonl', 'mitra': 'jobs/b41/final_s{n}.jsonl',
          'highsim': 'jobs/b70/final_s{n}.jsonl', 'exid': 'jobs/b71/final_s{n}.jsonl'}
DATA = {'dev': 'data', 'tgsim': 'data_tgsim', 'mitra': 'data_mitra', 'highsim': 'data_highsim', 'exid': 'data_exid'}


def _task(job):
    import fairtest.env as E
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_crossjudge import replay
    from fairtest.b23_margin import clearance
    from fairtest.runner import _STATE, _init, run_episode
    _init()
    tag, key, case_id, actions, theta, conds = job
    case = _STATE['cases'][case_id]
    out = dict(tag=tag, key=key)
    for name in conds:
        target, width, length, perceive = CONDS[name]
        E.VEH_W, E.VEH_L, E.CC_PERCEIVE = width, length, perceive
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
    ap.add_argument('--conds', default='U,M,MM,T')
    ap.add_argument('--out', default='runs/E16v2_matched.jsonl')
    main(ap.parse_args())
