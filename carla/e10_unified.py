"""E10 (addendum 6): apply the final version's success criteria (G2 and joint realism) to the robust successes of the CARLA closed loop.

The adversary vehicle was driven by actions in the highway-env twin (target oracle_t0). From the stored trajectory (adv_traj, [x, y, vx, vy] per 0.1 s,
three decimals) each decision re-runs 10 candidates (lane 0/4 × 5 requests) and keeps the action that matches up to the collision. With these actions, exactly as
in B45, it computes the features of the C replay (cc_r157) -> G2 exceedance and joint support score (same rule as 'inside' in b56_analysis).

  python e10_unified.py validate OUT.jsonl --b26 B26_FINAL_DIR [--n 120]     (validates the tool on highway-env records with known actions)
  python e10_unified.py run OUT.jsonl RUNS.jsonl [...] [--workers 8]           (applies it to the CARLA closed-loop results)
"""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

TOL = .01
LANES, REQUESTS = (0., 4.), (1., 0., -1., -2., -3.)


def _state(r, k):
    g = r['trace'][k]['generator']
    return (g['x'], g['y'], g['vx'], g['vy'])


def recover(case, traj, n_ticks, beam=1):
    """traj[k] = state of the twin adversary vehicle at t = k*0.1 s. The first n_ticks ticks were driven by actions. -> (actions, max error per decision).
    The actuator applies an acceleration request slowly, so a single decision interval may not separate the candidates within the rounding (0.0005). So each
    decision expands 10 candidates and keeps the `beam` best by cumulative max error (ties: error sum) through the next interval (beam search)."""
    from fairtest.b23_crossjudge import replay
    beams = [([], 0., 0., [])]                                  # (actions, cumulative max error, error sum, error per decision)
    for i in range(math.ceil(n_ticks / 5)):
        ks = range(5 * i, min(5 * i + 5, n_ticks))
        cand = []
        for acts, emax, esum, errs in beams:
            for lane in LANES:
                for req in REQUESTS:
                    r = replay(case, acts + [(lane, req)], 'oracle_t0', 0.)
                    if len(r['trace']) < ks[-1] + 1:
                        continue
                    err = max(abs(a - b) for k in ks for a, b in zip(_state(r, k), traj[k + 1]))
                    cand.append((acts + [(lane, req)], max(emax, err), esum + err, errs + [round(err, 5)]))
        cand.sort(key=lambda c: (c[1], c[2]))
        beams = cand[:beam]
    acts, _, _, errs = beams[0]
    return acts, errs


def judge(case, actions):
    """C replay as in B45 -> (cut-in, features, G2 exceedance, support score, tau, inside)."""
    from fairtest.b23_crossjudge import replay
    from fairtest.b45_joint_realism import Support
    from fairtest.features import FEATURES, Realism, simulated_features
    if not _CACHE:
        _CACHE.update(realism=Realism(), support=Support())
    r = replay(case, actions, 'cc_r157', 0.)
    f = simulated_features(r)
    cut = bool(f.get('cut_in'))
    exc = _CACHE['realism'].score(f)['max_exceedance'] if cut else None
    sc = _CACHE['support'].score(f) if cut else None
    tau = _CACHE['support'].tau
    inside = bool(sc is not None and sc <= tau and exc is not None and exc <= 1.)
    return dict(cut_in=cut, exceedance=None if exc is None else round(float(exc), 4), joint=None if sc is None else round(float(sc), 4),
                tau=round(float(tau), 4), inside=inside, cc_contact=bool(r['pair_contact']), features={k: f.get(k) for k in FEATURES})


_CACHE = {}


def _validate_task(job):
    from fairtest.b23_crossjudge import replay
    from fairtest.runner import _STATE, _init
    _init()
    seed, row = job
    case = _STATE['cases'][row['case_id']]
    acts = [tuple(a) for a in row['actions']]
    r = replay(case, acts, 'oracle_t0', 0.)                     # same twin trajectory as carla_export
    g0 = r['physical_initial_state'][0]
    traj = [(g0['x'], g0['y'], g0['vx'], g0['vy'])] + [tuple(round(v, 3) for v in _state(r, k)) for k in range(len(r['trace']))]
    n = min(len(r['trace']), 5 * len(acts))
    rec, errs = recover(case, traj, n)
    out = dict(seed=seed, case_id=row['case_id'], sut=row['sut'], n_actions=len(acts), match=rec == acts[:len(rec)],
               max_err=max(errs) if errs else 0.)
    out.update(judge(case, rec))
    return out


def _run_task(job):
    from fairtest.runner import _STATE, _init
    _init()
    src, case_id, kind, traj, t_contact = job
    case = _STATE['cases'][case_id]
    n = int(round(t_contact / .1))
    rec, errs = recover(case, traj, n)
    out = dict(source=src, case_id=case_id, kind=kind, t_contact=t_contact, actions=rec, max_err=max(errs) if errs else 0.,
               recovered=bool(not errs or max(errs) <= TOL))
    out.update(judge(case, rec))
    return out


def main(a):
    if a.mode == 'validate':
        jobs = []
        for s in (1, 2, 3):
            for line in open(Path(a.b26) / f'final_s{s}_dev.jsonl', encoding='utf-8'):
                r = json.loads(line)
                if r['sut'] == 'following_H0' and r['contact']:
                    jobs.append((f's{s}', r))
        jobs = jobs[:a.n]
        fn = _validate_task
    else:
        jobs = []
        for path in a.runs:
            for line in open(path, encoding='utf-8'):
                r = json.loads(line)
                for kind, v in r['runs'].items():
                    if (v.get('fair') and (v.get('contact_dx') or -1e9) >= 3.5 and (v.get('cc_min_clearance') or 0.) >= .5
                            and 'adv_traj' in v):
                        jobs.append((Path(path).stem, r['case_id'], kind, v['adv_traj'], v['t_contact']))
        fn = _run_task
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap(fn, jobs, chunksize=2):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(mode=a.mode, done=len(jobs))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('validate', 'run'))
    ap.add_argument('out')
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--b26')
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--workers', type=int, default=8)
    main(ap.parse_args())
