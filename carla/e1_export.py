"""E1: select the fair failures that the final agent (v2s) found on the 62 development cases and export them as trajectories for CARLA replay.

The composition is the same as in b56_analysis: per seed, case and SUT, the policy stage's robust fair failure if it is within joint realism, otherwise the first
success of the refinement (bo_warm_g4s) within 37 queries. Policy attacks are turned into trajectories as in carla_export (action sequence -> oracle_t0 replay),
refinement attacks with carla_search.adversary_row (plan -> trajectory). Each trajectory is exported once.

  python e1_export.py OUT_TRAJ.jsonl OUT_COMPOSITION.json --jobs JOBS --b56 B56_DEV.jsonl --b26 B26_FINAL_DIR [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

SUTS = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'following_aeb_H0', 'rss_sut_H0')
SEEDS = ('s1', 's2', 's3')
K = 37


def _policy_task(job):
    from fairtest.carla_export import _task
    seed, row = job
    out = _task((f'policy_{seed}', row))
    out['hwenv'].update(stage='policy', seed=seed)
    return out


def _refine_task(job):
    from fairtest.carla_search import adversary_row
    from fairtest.runner import _STATE, _init
    _init()
    case_id, sut, best = job
    valid, row = adversary_row(_STATE['cases'][case_id], best['theta'])     # (is the plan valid up to contact, trajectory row)
    row.update(source='refine_g4s', case_id=case_id, sut=sut,
               hwenv=dict(stage='refine', valid=valid, sut_contact=bool(best.get('sut_contact')), cc_contact=bool(best.get('cc_contact')),
                          robust=bool(best.get('robust')), cc_clearance=best.get('cc_clearance'), contact_dx=best.get('contact_dx'),
                          theta=best['theta']))
    return row


def main(a):
    from fairtest.b56_analysis import policy, refine
    robust, inside = policy(Path(a.jobs), 'dev')
    first, recs, err = refine([a.b56], 'bo_warm_g4s')
    assert err == 0, err
    cases = sorted({c for (_, c, _) in robust})
    comp, pol_jobs, ref_jobs = {}, [], {}
    rows = {s: {(r['case_id'], r['sut']): r for r in map(json.loads, open(Path(a.b26) / f'final_{s}_dev.jsonl'))} for s in SEEDS}
    for s in SEEDS:
        for c in cases:
            for sut in SUTS:
                if robust.get((s, c, sut), False) and inside.get((s, c, sut), False):
                    comp[f'{s}|{c}|{sut}'] = dict(stage='policy', key=f'policy_{s}|{c}|{sut}')
                    pol_jobs.append((s, rows[s][(c, sut)]))
                elif first.get((c, sut)) is not None and 1 + first[(c, sut)] <= K:
                    comp[f'{s}|{c}|{sut}'] = dict(stage='refine', key=f'refine_g4s|{c}|{sut}', first_fair=first[(c, sut)])
                    ref_jobs[(c, sut)] = (c, sut, recs[(c, sut)]['best'])
                else:
                    comp[f'{s}|{c}|{sut}'] = None
    with mp.Pool(a.workers) as pool:
        out = pool.map(_policy_task, pol_jobs) + pool.map(_refine_task, list(ref_jobs.values()))
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    n_succ = {sut: sum(1 for k, v in comp.items() if v and k.endswith('|' + sut)) for sut in SUTS}
    json.dump(dict(cases=cases, suts=SUTS, seeds=SEEDS, k=K, composition=comp, successes=n_succ,
                   unique=dict(policy=len(pol_jobs), refine=len(ref_jobs))), open(a.comp, 'w'), indent=1)
    print(json.dumps(dict(successes=n_succ, unique_policy=len(pol_jobs), unique_refine=len(ref_jobs), cases=len(cases))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('comp')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--b56', required=True)
    ap.add_argument('--b26', required=True)
    ap.add_argument('--workers', type=int, default=8)
    main(ap.parse_args())
