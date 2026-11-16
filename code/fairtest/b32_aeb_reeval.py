"""B32 (post hoc): the AEB layer of the R157 reference driver model, cut-in scenario.

R157 (original series) Annex 4 Appendix 3, Figure 1 and Table 1: 'Reference Driver + AEB'. After the careful driver's
risk perception and 0.75 s reaction the driver brakes 0.6 s to 0.774 g; an AEB layer derived from UN R152, triggered at
'xx% wrap & TTC < TTC_AEB', brakes 0.6 s to 0.85 g 'after full wrap'. xx%, TTC_AEB and the AEB delay are not specified,
and every CC of this study omits the layer. cc_r157_aeb adds its strongest reading (cc_target), so it bounds the layer's
effect on the two kinds of verdict:
  agent     every failure the final agent reports (policy stage and refinement plans, as b29_consensus.jobs_from):
            still avoided by the reference, and with the 0.5 m margin?
  baseline  every judged collision in an RL evaluation file: still attacker-at-fault (the reference also collides)?
Each job also replays the primary CC; it must reproduce the stored verdict (the edited CCDriver is unchanged without
the layer).

  python -m fairtest.b32_aeb_reeval OUT.jsonl [--final DIR --dataset DS --refine E3.jsonl ... --method M]
         [--baselines EVAL.jsonl ...] --workers W
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from fairtest import devboot  # noqa: F401
from fairtest.b29_consensus import _run, jobs_from

AEB = 'cc_r157_aeb'


def baseline_jobs(paths):
    jobs = []
    for path in paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r.get('valid') and r.get('contact') and r.get('refs') and 'contact' in r['refs'].get('cc_r157', {}):
                jobs.append((Path(path).name, 'baseline', r['case_id'], r['sut'], 'actions', r['actions'],
                             bool(r['refs']['cc_r157']['contact'])))
    return jobs


def task(job):
    source, stage, case_id, sut, how, payload, stored_cc = job
    from fairtest import runner
    from fairtest.b23_margin import clearance
    runner._init()
    case = runner._STATE['cases'][case_id]
    try:
        cc, aeb = _run(case, how, payload, 'cc_r157'), _run(case, how, payload, AEB)
        engaged = any(r['target_control'].get('recognition', {}).get('aeb') for r in aeb['trace'])
        return dict(source=source, stage=stage, case_id=case_id, sut=sut, stored_cc_contact=stored_cc,
                    cc_contact=bool(cc['pair_contact']), cc_clearance=round(clearance(cc['trace']), 3),
                    aeb_contact=bool(aeb['pair_contact']), aeb_clearance=round(clearance(aeb['trace']), 3),
                    aeb_engaged=bool(engaged))
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
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(task, jobs, chunksize=2):
            f.write(json.dumps(res) + '\n')
    print(json.dumps(dict(jobs=len(jobs))))
