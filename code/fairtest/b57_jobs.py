"""B57 job set (stored rows only, no simulation): every attack to re-judge with the human-like reference population.

Development partition, cut-in scenario.
  v2j policy   each policy-stage episode (fair_guarded2 s1-3, window shield) that v2j reports under the unified contract:
               robust fair failure (b26 margin: front contact dx >= 3.5 m, CC clearance >= 0.5 m) whose CC replay is inside
               the joint support and passes G2 (b45 replay_dev, as b46_analysis --policy-g2)
  v2j refine   each first bo_warm_g4j success of the stored B46 refinement (one plan per case and SUT)
  baseline     each judged collision of the stored RL baseline evaluations (collision-seeking collide_commit s1-3, CARS
               s1-3, Dynasto s1-3, naive attribution s1), with its stored CC verdict (as b32_aeb_reeval.baseline_jobs)
  autoware     each v2j failure reported against Autoware in B54 (development run, AW-default and AW-r157): the policy
               episode's realised actions or the refinement plan, judged in highway-env like every other SUT
`bracket` marks the jobs of the sensitivity populations: every v2j job and BRACKET_BASELINES baseline jobs drawn at
random (rng 57).

  python -m fairtest.b57_jobs OUT.jsonl --jobs fairtest_jobs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage
from fairtest.b45_joint_realism import Support

BRACKET_BASELINES = 1000
BASELINES = ('collide_commit_s1', 'collide_commit_s2', 'collide_commit_s3', 'cars_s1', 'cars_s2', 'cars_s3', 'dynasto_s1',
             'dynasto_s2', 'dynasto_s3', 'naive_fair_s1')


def main(a):
    jobs_dir = Path(a.jobs)
    sup = Support()
    robust = policy_stage(jobs_dir / 'b26', 'dev')[0]
    inside = {}
    for line in open(jobs_dir / 'b45' / 'replay_dev.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['source'] == 'v2_policy':
            sc = sup.score(r['features']) if r['cut_in'] else None
            inside[(f"s{r['seed']}", r['case_id'], r['sut'])] = bool(sc is not None and sc <= sup.tau and r.get('exceedance') is not None
                                                                     and r['exceedance'] <= 1.)
    out = []
    for path in sorted((jobs_dir / 'b26' / 'final').glob('final_s*_dev.jsonl')):
        seed = path.name.split('_')[1]
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'valid' not in r:
                continue
            k = (seed, r['case_id'], r['sut'])
            if robust.get(k) and inside.get(k):
                out.append(dict(stage='policy', attacker='v2j', seed=int(seed[1:]), case_id=r['case_id'], sut=r['sut'], how='actions',
                                payload=r['actions'], stored_cc=False, source=path.name))
    for name in ('joint_dev_rule.jsonl', 'joint_dev_learned.jsonl'):
        for line in open(jobs_dir / 'b46' / name, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == 'bo_warm_g4j' and 'error' not in rec and rec.get('first_fair') is not None:
                    out.append(dict(stage='refine', attacker='v2j', seed=None, case_id=r['case_id'], sut=s, how='theta',
                                    payload=rec['best']['theta'], stored_cc=bool(rec['best']['cc_contact']), source=name))
    for b in BASELINES:
        for line in open(jobs_dir / 'b23' / 'eval' / f'{b}_dev.jsonl', encoding='utf-8'):
            r = json.loads(line)
            if r.get('valid') and r.get('contact') and r.get('refs') and 'contact' in r['refs'].get('cc_r157', {}):
                out.append(dict(stage='baseline', attacker=b.rsplit('_s', 1)[0], seed=int(b.rsplit('_s', 1)[1]), case_id=r['case_id'],
                                sut=r['sut'], how='actions', payload=r['actions'], stored_cc=bool(r['refs']['cc_r157']['contact']),
                                source=f'{b}_dev.jsonl'))
    for path in sorted((jobs_dir / 'b54' / 'b54_dev').glob('policy_*.jsonl')) + sorted((jobs_dir / 'b54' / 'b54_dev').glob('refine_*.jsonl')):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            res = r['result']
            if path.name.startswith('policy') and r.get('attacker') == 'v2j' and res.get('reportable'):
                out.append(dict(stage='autoware_policy', attacker='v2j', seed=r['seed'], case_id=r['case_id'], sut=r['variant'],
                                how='actions', payload=res['actions'], stored_cc=False, source=path.name))
            elif path.name.startswith('refine') and res.get('found'):
                out.append(dict(stage='autoware_refine', attacker='v2j', seed=None, case_id=r['case_id'], sut=r['variant'],
                                how='theta', payload=res['best']['theta'], stored_cc=False, source=path.name))
    base = [i for i, j in enumerate(out) if j['stage'] == 'baseline']
    pick = {int(i) for i in np.random.default_rng(57).choice(base, size=min(BRACKET_BASELINES, len(base)), replace=False)}
    with open(a.out, 'w', encoding='utf-8') as f:
        for i, j in enumerate(out):
            f.write(json.dumps(dict(job=i, bracket=j['attacker'] == 'v2j' or i in pick, **j)) + '\n')
    counts = {}
    for j in out:
        key = f"{j['attacker']}|{j['stage']}|cc_{'contact' if j['stored_cc'] else 'avoid'}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(dict(jobs=len(out), counts=counts), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    main(ap.parse_args())
