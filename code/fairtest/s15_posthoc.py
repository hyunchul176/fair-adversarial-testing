"""S15 post hoc (descriptive, after the frozen runs): the B68 reference-reading re-judgement for an external prospective run
in the B41 layout (B70 HIGH-SIM, B71 exiD). The frozen runs left S15 out because the B65 job tool covered only the
development and MiTra sets; this tool builds the same job set for a run directory, so b68_cars_refs and the B68 analysis run
on it unchanged.

Jobs (as b65_jobs.mitra_jobs): every v2s policy-stage reported failure (robust fair failure with the G4 margin whose C
replay is inside the joint support and passes G2: b41_analysis.load), the first bo_warm_g4s and bo_warm_g4ss success per
(case, SUT) of the run, and N_BASE baseline collisions with a stored C verdict drawn at random (rng 65) from the
collision-seeking, CARS-style and Dynasto-style evaluations. Closing speeds: policy contacts and g4s successes from the run's
closing.jsonl, g4ss successes from their own rows.

  python -m fairtest.s15_posthoc jobs RUN_DIR JOBS_DIR NAME          -> JOBS_DIR/jobs_NAME.jsonl (needs FAIRTEST_DATA)
  python -m fairtest.s15_posthoc analyse OUT.json RUN_DIR REJUDGE_DIR JOBS_DIR NAME
      (REJUDGE_DIR/rejudge_<variant>_NAME.jsonl from b68_cars_refs for cc_r157, cc_jrc, ref_fsm, fsm_cars)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b65_jobs import BASELINES, N_BASE, closings, refine_jobs, rows, write


def run_jobs(d, refine=None, g4ss=None, closing=None):
    """Run-directory names by default; the explicit paths reproduce the stored MiTra job set (check below)."""
    from fairtest.b41_analysis import load
    d = Path(d)
    evals, realistic, _, _, _ = load(d)
    close = closings(closing or [d / 'closing.jsonl'])
    out = []
    for path in sorted(d.glob('final_s[0-9].jsonl')):
        seed = path.stem.rsplit('_s', 1)[1]
        for r in rows(path):
            if 'valid' not in r:
                continue
            k = (r['case_id'], r['sut'])
            if evals['final'][seed].get(k, {}).get('robust') and realistic.get((seed, r['case_id'], r['sut'])):
                out.append(dict(stage='policy', attacker='v2s', seed=int(seed), case_id=r['case_id'], sut=r['sut'], how='actions',
                                payload=r['actions'], stored_cc=False, source=path.name,
                                closing=close.get(('policy', seed, r['case_id'], r['sut']))))
    out += refine_jobs(refine or d / 'refine.jsonl', 'bo_warm_g4s', 'refine_g4s', close)
    out += refine_jobs(g4ss or d / 'refine_g4ss.jsonl', 'bo_warm_g4ss', 'refine_g4ss', None)
    base = []
    for b in BASELINES:
        for path in sorted(d.glob(f'{b}_s[0-9].jsonl')):
            seed = int(path.stem.rsplit('_s', 1)[1])
            for r in rows(path):
                if r.get('valid') and r.get('contact') and r.get('refs') and 'contact' in r['refs'].get('cc_r157', {}):
                    base.append(dict(stage='baseline', attacker=b, seed=seed, case_id=r['case_id'], sut=r['sut'], how='actions',
                                     payload=r['actions'], stored_cc=bool(r['refs']['cc_r157']['contact']), source=path.name,
                                     closing=None))
    pick = sorted(int(i) for i in np.random.default_rng(65).choice(len(base), size=min(N_BASE, len(base)), replace=False))
    return out + [base[i] for i in pick], len(base)


def main(argv):
    if argv[0] == 'jobs':
        run_dir, jobs_dir, name = argv[1:4]
        jobs, n_base = run_jobs(run_dir)
        Path(jobs_dir).mkdir(parents=True, exist_ok=True)
        rep = write(Path(jobs_dir) / f'jobs_{name}.jsonl', jobs)
        rep['baseline_pool'] = n_base
        print(json.dumps(rep, indent=1))
    elif argv[0] == 'analyse':
        from fairtest.b68_analysis import analyse
        out, run_dir, rejudge_dir, jobs_dir, name = argv[1:6]
        rep = analyse(name, rejudge_dir, jobs_dir, run_dir, str(Path(run_dir) / 'refine.jsonl'))
        rep['posthoc'] = 'S15 run after the frozen evaluation (descriptive; not in the frozen run)'
        json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
        print(name, 'jobs', rep['jobs'], 'errors', rep['errors'], 'C1', rep['C1']['supported'], rep['C1']['mismatches'], 'M1', rep['M1'])
        for g, v in rep['groups'].items():
            print(f'  {g:10s}', {k: (x['avoided'], x['n']) for k, x in v.items()})
        print('  JR', rep['JR'], '| FR', rep['FR'])
        print('  P1', rep['P1'])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
