"""B45 (design fixed before the run): joint (multivariate) realism of the attacks' CC replays vs highD natural cut-ins.

G2 is a per-feature bound (each of the six cut-in features within natural p1..p99 + 1 IQR). Here the six features are
standardised by the natural median / IQR and scored by the distance to the 5th nearest natural cut-in; the joint
support is the natural leave-one-out 99th percentile of the same score.
  replay   every stored RL episode (development) is replayed against the CC (`cc_r157`) to recover its features;
           the stored CC-replay exceedance, where recorded, checks the replay
  analyse  R1 (v2's reported failures >= 90% inside), R2 (G2-less ablations >= 20 pp below the v2 policy), R3 (per-
           feature G2 and the joint support agree on >= 90% of the v2 policy episodes)

  python -m fairtest.b45_joint_realism replay OUT.jsonl --jobs fairtest_jobs [--workers 15]
  python -m fairtest.b45_joint_realism analyse OUT.json REPLAY.jsonl --jobs fairtest_jobs
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401

K = 5
SOURCES = {  # name: (file pattern under fairtest_jobs, seeds)
    'v2_policy': ('b26/final/final_s{s}_dev.jsonl', (1, 2, 3)),
    'collide_commit': ('b23/eval/collide_commit_s{s}_dev.jsonl', (1, 2, 3)),
    'cars': ('b23/eval/cars_s{s}_dev.jsonl', (1, 2, 3)),
    'abl_none': ('b23/eval/abl_none_s{s}_dev.jsonl', (1,)),
    'abl_commit': ('b23/eval/abl_commit_s{s}_dev.jsonl', (1,)),
}


_REALISM = []


def _task(job):
    source, seed, row = job
    from fairtest.b23_crossjudge import replay
    from fairtest.features import FEATURES, Realism, simulated_features
    from fairtest.runner import _STATE, _init
    _init()
    if not _REALISM:
        _REALISM.append(Realism())
    r = replay(_STATE['cases'][row['case_id']], row['actions'], 'cc_r157', 0.)
    f = simulated_features(r)
    exc = _REALISM[0].score(f)['max_exceedance'] if f.get('cut_in') else None
    return dict(source=source, seed=seed, case_id=row['case_id'], sut=row['sut'], contact=bool(row['valid'] and row['contact']),
                cc_contact=bool(r['pair_contact']), cut_in=bool(f.get('cut_in')), features={k: f.get(k) for k in FEATURES},
                exceedance=exc, stored_exc=(row.get('cc_replay') or {}).get('exc'))


POLICY = {  # B47 (post hoc): the v2 policy stage on the other datasets (FAIRTEST_DATA must hold that dataset's cases)
    'b9': ('b26/final/final_s{s}_b9.jsonl', (1, 2, 3)),
    'ngsim': ('b26/final/final_s{s}_ngsim.jsonl', (1, 2, 3)),
    'tgsim': ('b28/final_s{s}.jsonl', (1, 2, 3)),
    'mitra': ('b41/final_s{s}.jsonl', (1, 2, 3)),  # B41 (pre-specified): the v2j policy stage on MiTra
    'b61': ('b61/final_s{s}.jsonl', (1, 2, 3)),  # B61 controls: v2j policy stage against SUTs of known capability
    'b61c': ('b61/collide_shield_s{s}.jsonl', (1, 2, 3)),  # B61 controls: the collision-seeking policy with the shield
    'b62': ('b62/final_s{s}.jsonl', (1, 2, 3)),  # B62: the policy trained against the SUT population
    'b81': ('b81/final_s{s}.jsonl', (1, 2, 3)),  # B81: the policy stage against SUTs of known capability on exiD (b81_analysis)
}


def replay_main(a):
    jobs = []
    sources = SOURCES if a.dataset == 'dev' else {'v2_policy': POLICY[a.dataset]}
    for name, (pat, seeds) in sources.items():
        for s in seeds:
            for line in open(Path(a.jobs) / pat.format(s=s), encoding='utf-8'):
                row = json.loads(line)
                if 'valid' in row and row.get('actions'):
                    jobs.append((name, s, row))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(_task, jobs, chunksize=4):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(replayed=len(jobs))))


def _refine_task(job):
    """B45 correction (review W5 P1-A): the refinement records' `features` are the SUT run's; recover the CC replay's."""
    case_id, sut, theta = job
    from fairtest.b11_search import ContinuousPlan
    from fairtest.features import FEATURES, Realism, simulated_features
    from fairtest.runner import _STATE, _init, run_episode
    _init()
    if not _REALISM:
        _REALISM.append(Realism())
    raw = run_episode(_STATE['cases'][case_id], ContinuousPlan.from_theta(theta), 'cc_r157', 0., keep_trace=True)
    f = simulated_features(raw)
    exc = _REALISM[0].score(f)['max_exceedance'] if f.get('cut_in') else None
    return dict(case_id=case_id, sut=sut, theta=theta, cc_contact=bool(raw['pair_contact']), cut_in=bool(f.get('cut_in')),
                cc_features={k: f.get(k) for k in FEATURES}, cc_exceedance=exc)


def refine_cc_main(a):
    """Replay every reported refinement success (first robust success of METHOD) against the CC."""
    jobs = []
    for line in open(a.refine, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == a.method and rec.get('first_fair') is not None and 'error' not in rec:
                jobs.append((r['case_id'], s, rec['best']['theta']))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(_refine_task, jobs, chunksize=2):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(replayed=len(jobs))))


class Support:
    def __init__(self):
        from fairtest.features import FEATURES, natural_reference
        self.features = FEATURES
        nat = [r for r in natural_reference() if all(r.get(k) is not None for k in FEATURES)]
        n = np.array([[r[k] for k in FEATURES] for r in nat], float)
        self.med = np.median(n, 0)
        self.iqr = np.subtract(*np.percentile(n, [75, 25], 0))
        self.z = (n - self.med) / self.iqr
        d = np.sqrt(((self.z[:, None, :] - self.z[None, :, :]) ** 2).sum(-1))
        loo = np.sort(d, 1)[:, K]  # column 0 is the point itself
        self.n, self.tau = len(nat), float(np.percentile(loo, 99))

    def score(self, feats):
        x = [feats.get(k) for k in self.features]
        if any(v is None for v in x):
            return None
        z = (np.array(x, float) - self.med) / self.iqr
        return float(np.sort(np.sqrt(((self.z - z) ** 2).sum(1)))[K - 1])


def analyse(a):
    from fairtest.b26_analysis import policy_stage
    sup = Support()
    rows = [json.loads(l) for l in open(a.replay, encoding='utf-8') if l.strip()]
    for r in rows:
        r['score'] = sup.score(r['features']) if r['cut_in'] else None
        r['inside'] = r['score'] is not None and r['score'] <= sup.tau
        r['g2'] = r['exceedance'] is not None and r['exceedance'] <= 1.
    chk = [r for r in rows if r['stored_exc'] is not None and r['exceedance'] is not None]
    rep = dict(natural=sup.n, k=K, tau=sup.tau,
               replay_check=dict(n=len(chk), same=sum(abs(r['exceedance'] - r['stored_exc']) < 1e-6 for r in chk)))
    rep['sources'] = {}
    for name in SOURCES:
        v = [r for r in rows if r['source'] == name]
        c = [r for r in v if r['cut_in']]
        hit = [r for r in c if r['contact']]
        scored = [r['score'] for r in c if r['score'] is not None]
        rep['sources'][name] = dict(episodes=len(v), cut_in=len(c), inside=round(sum(r['inside'] for r in c) / len(c), 4) if c else None,
                                    g2=round(sum(r['g2'] for r in c) / len(c), 4) if c else None,
                                    sut_contact_inside=round(sum(r['inside'] for r in hit) / len(hit), 4) if hit else None,
                                    incomplete_features=len(c) - len(scored),  # counted as outside the joint support
                                    score_median=float(np.median(scored)) if scored else None)
    # v2's reported failures: the policy stage's robust fair failure, else the refinement's first robust success
    robust, _ = policy_stage(Path(a.jobs) / 'b26', 'dev')
    pol = {(f"s{r['seed']}", r['case_id'], r['sut']): r for r in rows if r['source'] == 'v2_policy'}
    ref = {}
    if a.refine_cc:  # corrected (W5 P1-A): the CC replay's features, as for the policy stage
        for line in open(a.refine_cc, encoding='utf-8'):
            r = json.loads(line)
            ref[(r['case_id'], r['sut'])] = r['cc_features'] if r['cut_in'] else {}
    else:  # original B45: the stored `features` of the refinement record, which are the SUT run's
        for line in open(Path(a.jobs) / 'b29' / 'guarded_dev.jsonl', encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                b = rec.get('best') or {}
                if m == 'bo_warm_g4g' and rec.get('first_fair') is not None and b.get('features'):
                    ref[(r['case_id'], s)] = b['features']
    reported = []
    for key, ok in robust.items():
        if ok and key in pol:
            reported.append(('policy', pol[key]['inside']))
        elif not ok and (key[1], key[2]) in ref:
            sc = sup.score(ref[(key[1], key[2])])
            reported.append(('refinement', sc is not None and sc <= sup.tau))
    r1 = sum(x for _, x in reported) / len(reported) if reported else None
    rep['R1'] = dict(n=len(reported), by_stage={s: sum(1 for t, _ in reported if t == s) for s in ('policy', 'refinement')},
                     inside=r1, supported=r1 is not None and r1 >= .90)
    abl = [r for r in rows if r['source'] in ('abl_none', 'abl_commit') and r['cut_in']]
    v2 = [r for r in rows if r['source'] == 'v2_policy' and r['cut_in']]
    a2, b2 = sum(r['inside'] for r in abl) / len(abl), sum(r['inside'] for r in v2) / len(v2)
    rep['R2'] = dict(ablations=a2, v2_policy=b2, difference=b2 - a2, supported=b2 - a2 >= .20)
    agree = sum(r['inside'] == r['g2'] for r in v2) / len(v2)
    rep['R3'] = dict(agree=agree, g2_only=sum(r['g2'] and not r['inside'] for r in v2), joint_only=sum(r['inside'] and not r['g2'] for r in v2),
                     supported=agree >= .90)
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('replay')
    p.add_argument('out')
    p.add_argument('--jobs', required=True)
    p.add_argument('--workers', type=int, default=15)
    p.add_argument('--dataset', default='dev', choices=['dev', 'b9', 'ngsim', 'tgsim', 'mitra', 'b61', 'b61c', 'b62', 'b81'])
    q = sub.add_parser('analyse')
    q.add_argument('out')
    q.add_argument('replay')
    q.add_argument('--jobs', required=True)
    q.add_argument('--refine-cc', help='CC replays of the refinement successes (refine-cc); default: the stored SUT features (original B45)')
    c = sub.add_parser('refine-cc')
    c.add_argument('out')
    c.add_argument('refine', help='refinement output (e.g. b29/guarded_dev.jsonl)')
    c.add_argument('--method', default='bo_warm_g4g')
    c.add_argument('--workers', type=int, default=15)
    a = ap.parse_args()
    {'replay': replay_main, 'analyse': analyse, 'refine-cc': refine_cc_main}[a.cmd](a)
