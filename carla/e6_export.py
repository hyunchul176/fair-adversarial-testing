"""E6 (addendum 2): export the final version's fair failures on one dataset as trajectories for CARLA replay, and re-judge them in highway-env with the default C
and the C matched to CARLA. The composition is the same as in b77_analysis.dataset (the policy-stage attack if it succeeded, otherwise the first success of the g4s refinement within 37 queries).

  FAIRTEST_DATA=data_<ds> python e6_export.py DS --jobs JOBS [--suts a,b] [--workers 8]
    -> runs/E6_<ds>_trajectories.jsonl, runs/E6_<ds>_composition.json, runs/E6_<ds>_rejudge.jsonl
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

W = Path(__file__).resolve().parent
SUTS = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'following_aeb_H0.5',
        'rss_sut_H0', 'rss_lat_sut_H0')
K = 37
POLICY = {'dev': 'b26/final/final_s{n}_dev.jsonl', 'tgsim': 'b28/final_s{n}.jsonl', 'mitra': 'b41/final_s{n}.jsonl',
          'highsim': 'b70/final_s{n}.jsonl', 'exid': 'b71/final_s{n}.jsonl'}


def _policy_task(job):
    from fairtest.carla_export import _task
    seed, row = job
    out = _task((f'policy_s{seed}', row))
    out['hwenv'].update(stage='policy', seed=f's{seed}')
    return out


def _refine_task(job):
    from fairtest.carla_search import adversary_row
    from fairtest.runner import _STATE, _init
    _init()
    case_id, sut, best = job
    valid, row = adversary_row(_STATE['cases'][case_id], best['theta'])
    row.update(source='refine_g4s', case_id=case_id, sut=sut,
               hwenv=dict(stage='refine', valid=valid, sut_contact=bool(best.get('sut_contact')), cc_contact=bool(best.get('cc_contact')),
                          robust=bool(best.get('robust')), cc_clearance=best.get('cc_clearance'), contact_dx=best.get('contact_dx'),
                          theta=best['theta']))
    return row


def _rejudge_task(job):
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_crossjudge import replay
    from fairtest.b23_margin import clearance
    from fairtest.runner import _STATE, _init, run_episode
    _init()
    key, case_id, actions, theta = job
    case = _STATE['cases'][case_id]
    out = dict(key=key)
    for target in ('cc_r157', 'cc_r157_cal'):
        r = replay(case, actions, target, 0.) if actions is not None else \
            run_episode(case, ContinuousPlan.from_theta(theta), target, 0., keep_trace=True)
        hit = bool(r['pair_contact'])
        out[target] = dict(contact=hit, clearance=None if hit else round(float(clearance(r['trace'])), 3))
    return out


def main(a):
    from fairtest.b77_analysis import dataset
    jobs = Path(a.jobs)
    ok, g4s, seeds, cases = dataset(jobs, a.ds)
    suts = tuple(a.suts.split(',')) if a.suts else SUTS
    seed_n = {x: x.lstrip('s') for x in seeds}                 # 's1' or '1' -> '1'
    rows = {}
    for x in seeds:
        for line in open(jobs / POLICY[a.ds].format(n=seed_n[x]), encoding='utf-8'):
            r = json.loads(line)
            rows[(seed_n[x], r['case_id'], r['sut'])] = r
    comp, pol_jobs, ref_jobs = {}, {}, {}
    for x in seeds:
        n = seed_n[x]
        for c in cases:
            for sut in suts:
                q = g4s.get((c, sut), (None, None))[0]
                if ok.get((x, c, sut), False):
                    key = f'policy_s{n}|{c}|{sut}'
                    comp[f's{n}|{c}|{sut}'] = dict(stage='policy', key=key)
                    pol_jobs[key] = (n, rows[(n, c, sut)])
                elif q is not None and 1 + q <= K:
                    key = f'refine_g4s|{c}|{sut}'
                    comp[f's{n}|{c}|{sut}'] = dict(stage='refine', key=key, first_fair=q)
                    ref_jobs[key] = (c, sut, g4s[(c, sut)][1]['best'])
                else:
                    comp[f's{n}|{c}|{sut}'] = None
    tag = a.tag or a.ds
    with mp.Pool(a.workers) as pool:
        traj = pool.map(_policy_task, list(pol_jobs.values())) + pool.map(_refine_task, list(ref_jobs.values()))
        rj = [(f"{t['source']}|{t['case_id']}|{t['sut']}", t['case_id'],
               pol_jobs[f"{t['source']}|{t['case_id']}|{t['sut']}"][1]['actions'] if t['source'].startswith('policy_') else None,
               t['hwenv'].get('theta')) for t in traj]
        rejudge = pool.map(_rejudge_task, rj)
    with open(W / f'runs/E6_{tag}_trajectories.jsonl', 'w', encoding='utf-8') as f:
        for t in traj:
            f.write(json.dumps(t) + '\n')
    with open(W / f'runs/E6_{tag}_rejudge.jsonl', 'w', encoding='utf-8') as f:
        for t in rejudge:
            f.write(json.dumps(t) + '\n')
    succ = {s: sum(1 for k, v in comp.items() if v and k.endswith('|' + s)) for s in suts}
    json.dump(dict(ds=a.ds, cases=cases, suts=suts, seeds=[f's{seed_n[x]}' for x in seeds], k=K, composition=comp, successes=succ,
                   unique=dict(policy=len(pol_jobs), refine=len(ref_jobs))), open(W / f'runs/E6_{tag}_composition.json', 'w'), indent=1)
    print(json.dumps(dict(ds=a.ds, successes=succ, unique_policy=len(pol_jobs), unique_refine=len(ref_jobs), cases=len(cases))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ds', choices=sorted(POLICY))
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--suts')
    ap.add_argument('--tag')
    ap.add_argument('--workers', type=int, default=8)
    main(ap.parse_args())
