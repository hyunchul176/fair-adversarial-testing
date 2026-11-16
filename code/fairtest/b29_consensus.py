"""B29 (W1, W2, W4): what the final agent's failures are, judged by reference drivers it never used.

For every failure the final agent reports (policy stage: fair failures of final_s*_<ds>.jsonl, replayed from their
action sequences; refinement stage: the first robust plan of bo_warm_g4, replayed from theta) this replays
  * the held-out reference drivers: strict CC, the time-based (original R157 text) CC, FSM, the RSS reference driver
    and the two solvability oracles -- none of them was used by the shield, the objective or the verdict;
  * the CC itself with its trace: the braking it needs (max DRAC while overlapping) -> R157 Annex 5 grade
    (avoidable < 5, difficult 5-7.6, unavoidable >= 7.6 m/s^2);
  * the SUT with its trace: the failure-mode features (time of the cut-in, gap and headway then, the adversary's
    braking after entry, the SUT's braking onset delay and peak deceleration).
Consensus-fair = avoided by the CC, strict CC, FSM and the RSS reference driver.

  python -m fairtest.b29_consensus OUT.jsonl --final FINAL_DIR --dataset dev|b9|ngsim --refine E3.jsonl [E3.jsonl ...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from fairtest import devboot  # noqa: F401

HELD_OUT = ('cc_r157_strict', 'cc_r157_time', 'ref_fsm', 'ref_rss', 'oracle_t0', 'oracle_react0')
HUMAN = ('cc_r157_strict', 'ref_fsm', 'ref_rss')
PERCEIVE_M = 1.095


def grade(drac):
    return 'avoidable' if drac < 5. else ('difficult' if drac < 7.6 else 'unavoidable')


def _run(case, how, payload, target, h=0.):
    from fairtest import runner
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_crossjudge import replay
    if how == 'actions':
        return replay(case, payload, target, h)
    return runner.run_episode(case, ContinuousPlan.from_theta(payload), target, h, keep_trace=True)


def guards(raw_cc):
    """G1/G2 of an attack from its CC counterfactual (the full cut-in): the lateral retreat after the CC perception
    threshold was first crossed (m; > 0.2 = a feint-like approach-retreat-entry) and the realism exceedance of the
    cut-in (IQR units, as the RL guard G2)."""
    from fairtest.features import Realism, simulated_features
    ys = [abs(r['generator']['y']) for r in raw_cc['trace']]
    k = next((i for i, y in enumerate(ys) if y >= PERCEIVE_M), None)
    retreat = 0.
    if k is not None:
        peak = ys[k]
        for y in ys[k:]:
            peak = max(peak, y)
            retreat = max(retreat, peak - y)
    feats = simulated_features(raw_cc)
    exc = Realism().score(feats)['max_exceedance'] if feats.get('cut_in') else None
    return dict(retreat_m=round(retreat, 3), feint=retreat > .2, cut_in=bool(feats.get('cut_in')),
                realism_exceedance=None if exc is None else round(float(exc), 3),
                realistic=bool(feats.get('cut_in') and exc is not None and exc <= 1.))


def features(case, raw):
    """Failure-mode features from the SUT's own run (adversary trajectory identical across targets until contact)."""
    tr = raw['trace']
    g0, t0 = case['physical_initial_state']
    k_in = next((k for k, r in enumerate(tr) if abs(r['generator']['y']) >= PERCEIVE_M), None)
    out = dict(steps=len(tr), contact=bool(raw['pair_contact']))
    if k_in is None:
        return out
    g, t = tr[k_in]['generator'], tr[k_in]['target']
    gap = g['x'] - t['x'] - (g.get('length', 5.) + t.get('length', 5.)) / 2
    vg = [g0['vx']] + [r['generator']['vx'] for r in tr]
    acc_g = [(b - a) / .1 for a, b in zip(vg, vg[1:])]
    # the SUT's first braking at or after the moment the cut-in is visible (0 if it is already braking then)
    brake = next((k for k in range(k_in, len(tr)) if tr[k]['target_control']['applied_mps2'] <= -1.), None)
    after = [r['target_control']['applied_mps2'] for r in tr[k_in:]]
    out.update(t_cutin_s=round(.1 * (k_in + 1), 1), gap_cutin_m=round(gap, 2), thw_cutin_s=round(gap / max(t['vx'], .1), 2),
               closing_cutin=round(t['vx'] - g['vx'], 2), adv_min_acc_after=round(min(acc_g[k_in:]) if acc_g[k_in:] else 0., 2),
               sut_brake_delay_s=None if brake is None else round(.1 * (brake - k_in), 1),
               sut_min_acc=round(min(r['target_control']['applied_mps2'] for r in tr), 2),
               sut_min_acc_after=round(min(after), 2) if after else None,
               sut_max_acc_after=round(max(after), 2) if after else None,
               sut_speed_change_after=round(tr[-1]['target']['vx'] - t['vx'], 2))
    return out


FEATURES_ONLY = os.environ.get('FAIRTEST_B29_FEATURES_ONLY') == '1'


def task(job):
    source, stage, case_id, sut, how, payload = job
    if FEATURES_ONLY:  # only the SUT replay (failure-mode features)
        from fairtest import runner
        runner._init()
        case = runner._STATE['cases'][case_id]
        name, h = sut.rsplit('_H', 1)
        try:
            return dict(source=source, stage=stage, case_id=case_id, sut=sut,
                        features=features(case, _run(case, how, payload, name, float(h))),
                        guards=guards(_run(case, how, payload, 'cc_r157')))
        except Exception as e:
            return dict(source=source, stage=stage, case_id=case_id, sut=sut, error=f'{type(e).__name__}: {e}')
    from fairtest import runner
    from fairtest.b23_margin import clearance
    runner._init()
    case = runner._STATE['cases'][case_id]
    name, h = sut.rsplit('_H', 1)
    try:
        cc = _run(case, how, payload, 'cc_r157')
        drac = runner.interaction_metrics(cc)['max_drac']
        s = _run(case, how, payload, name, float(h))
        refs = {r: bool(_run(case, how, payload, r)['pair_contact']) for r in HELD_OUT}
        return dict(source=source, stage=stage, case_id=case_id, sut=sut, cc_contact=bool(cc['pair_contact']),
                    cc_demand=round(drac, 3), grade=grade(drac), cc_clearance=round(clearance(cc['trace']), 3),
                    refs_contact=refs, consensus_fair=not cc['pair_contact'] and not any(refs[r] for r in HUMAN),
                    features=features(case, s), guards=guards(cc))
    except Exception as e:  # recorded, never dropped
        return dict(source=source, stage=stage, case_id=case_id, sut=sut, error=f'{type(e).__name__}: {e}')


def jobs_from(final_dir, ds, refine_paths, method='bo_warm_g4', policy=True):
    jobs = []
    for path in (sorted(Path(final_dir).glob(f'final_s*_{ds}.jsonl')) if policy else []):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r.get('valid') and r.get('contact') and r.get('refs') and not r['refs']['cc_r157']['contact']:
                jobs.append((path.name, 'policy', r['case_id'], r['sut'], 'actions', r['actions']))
    for path in refine_paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == method and rec.get('first_fair') and rec.get('best'):
                    jobs.append((Path(path).name, 'refine', r['case_id'], s, 'theta', rec['best']['theta']))
    return jobs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--final', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--refine', nargs='*', default=[])
    ap.add_argument('--method', default='bo_warm_g4', help='refinement method whose robust plans are judged')
    ap.add_argument('--no-policy', action='store_true', help='judge only the refinement plans')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = jobs_from(a.final, a.dataset, a.refine, a.method, not a.no_policy)
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(task, jobs, chunksize=2):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(jobs=len(jobs))))
