"""B36 (post hoc): the reaction-time margin of every fairness verdict (cut-in).

The verdict uses the R157 careful driver with a 0.75 s reaction. Every failure the final agent v2 reports and every
judged baseline collision (the B32 job set) is replayed with the same CC and other reaction times r (the perception
point is unchanged; target cc_r157@r<r>):
  the CC avoided it (fair)             r = 1.0 ... 3.0: r_up = the largest r such that the CC avoids at every grid
                                       value up to r -- how much slower than the norm a driver may be
  the CC collided (attacker-at-fault)  r = 0, 0.25, 0.5: r_need = the largest of them at which the CC avoids (None if
                                       even an instantaneous reaction at the perception point collides)

  python -m fairtest.b36_reaction_margin OUT.jsonl [--final DIR --dataset DS --refine E3.jsonl --method M]
         [--baselines EVAL.jsonl ...] --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

from fairtest import devboot  # noqa: F401
from fairtest.b29_consensus import _run, jobs_from
from fairtest.b32_aeb_reeval import baseline_jobs

UP = (1., 1.25, 1.5, 1.75, 2., 2.5, 3.)
DOWN = (0., .25, .5)


def task(job):
    source, stage, case_id, sut, how, payload, stored_cc = job
    from fairtest import runner
    runner._init()
    case = runner._STATE['cases'][case_id]
    grid = DOWN if stored_cc else UP
    try:
        contact = {r: bool(_run(case, how, payload, f'cc_r157@r{r:g}')['pair_contact']) for r in grid}
        out = dict(source=source, stage=stage, case_id=case_id, sut=sut, stored_cc_contact=stored_cc,
                   contacts={f'{r:g}': c for r, c in contact.items()})
        if stored_cc:
            ok = [r for r in grid if not contact[r]]
            out['r_need'] = max(ok) if ok else None
        else:
            r_up = .75
            for r in grid:
                if contact[r]:
                    break
                r_up = r
            out['r_up'] = r_up
        return out
    except Exception as e:  # recorded, never dropped
        return dict(source=source, stage=stage, case_id=case_id, sut=sut, error=f'{type(e).__name__}: {e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--final')
    ap.add_argument('--dataset')
    ap.add_argument('--refine', nargs='*', default=[])
    ap.add_argument('--method', default='bo_warm_g4g')
    ap.add_argument('--baselines', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    jobs = [j + (False,) for j in jobs_from(a.final, a.dataset, a.refine, a.method, True)] if a.final else []
    jobs += baseline_jobs(a.baselines)
    from fairtest import runner
    with mp.Pool(a.workers, initializer=runner._init) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(task, jobs, chunksize=2):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(jobs=len(jobs))))
