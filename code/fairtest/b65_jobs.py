"""B65 job set (stored rows only, no simulation): the final agent v2s's reported failures and baseline collisions to
re-judge with the B57 human-like reference population plus a later-perceiving bracket (B65 design,
available on request).

Development partition (jobs_dev.jsonl, run with the development data):
  policy       the v2s policy-stage reported failures = the B57 'policy' jobs (fair_guarded2 s1-3, window shield; v2j and
               v2s share the policy stage), relabelled attacker v2s; the B57 job id is kept for the identity check
  refine_g4s   the first bo_warm_g4s success per (case, SUT) of the stored B56 run (the screened refinement stops at its
               first success, so the stored best plan is that success; checked below)
  refine_g4ss  the first bo_warm_g4ss success per (case, SUT) of the stored B64 run (severity floor 3 m/s, the v2s-S arm)
  baseline     the B57 bracket subset of judged baseline collisions (1,000, rng 57); the B57 job id is kept
MiTra, post hoc (jobs_mitra.jsonl, run with the MiTra data):
  policy       each policy-stage episode v2s reports: robust fair failure with the G4 margin whose CC replay is inside the
               joint support and passes G2 (b41_analysis.load, the composition of exid_analysis)
  refine_g4s, refine_g4ss   as above, from b56_mitra.jsonl and b64_mitra.jsonl
  baseline     N_BASE judged collisions drawn at random (rng 65) from the stored B41 evaluations of BASELINES
Every job is flagged `bracket` (every population runs). `closing` (SUT minus generator speed at contact, m/s) is attached
where stored: policy episodes from the B51 replays, g4s successes from the B64 replays, g4ss successes from the run.
The population file must list the B57 drivers first, unchanged (b65_spec.json = b57_spec.json + one appended bracket).

  python -m fairtest.b65_jobs OUT_DIR --b57 B57 --b56 B56 --b64 B64 --b51 B51 --b41 B41 --population POPULATION.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

N_BASE = 1000
BASELINES = ('collide_commit', 'cars', 'dynasto')


def rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def closings(paths):
    out = {}
    for p in paths:
        for r in rows(p):
            if r.get('sut_contact') and r.get('closing') is not None:
                out[(r['kind'], None if r['seed'] is None else str(r['seed']), r['case_id'], r['sut'])] = float(r['closing'])
    return out


def refine_jobs(path, method, stage, closing):
    """First success per (case, SUT); the stored best plan must be that success (screened runs stop at it)."""
    out, bad = [], []
    for r in rows(path):
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m != method or 'error' in rec or rec.get('first_fair') is None:
                continue
            b = rec['best']
            if not (b['robust'] and b['sut_contact'] and not b['cc_contact'] and rec['sut_queries'] == rec['first_fair']):
                bad.append([r['case_id'], s])
                continue
            c = b.get('closing') if closing is None else closing.get((method, None, r['case_id'], s))
            out.append(dict(stage=stage, attacker='v2s', seed=None, case_id=r['case_id'], sut=s, how='theta', payload=b['theta'],
                            stored_cc=False, source=Path(path).name, closing=c))
    if bad:
        raise SystemExit(f'{path} {method}: {len(bad)} successes whose best plan is not the first success: {bad[:5]}')
    return out


def dev_jobs(a, b51, b64c):
    b57 = list(rows(Path(a.b57) / 'jobs_dev.jsonl'))
    out = []
    for j in b57:
        if j['stage'] == 'policy' and j['attacker'] == 'v2j':
            out.append(dict({k: v for k, v in j.items() if k not in ('job', 'bracket')}, attacker='v2s', b57_job=j['job'],
                            closing=b51.get(('policy', str(j['seed']), j['case_id'], j['sut']))))
    out += refine_jobs(Path(a.b56) / 'b56_dev.jsonl', 'bo_warm_g4s', 'refine_g4s', b64c)
    out += refine_jobs(Path(a.b64) / 'b64_dev.jsonl', 'bo_warm_g4ss', 'refine_g4ss', None)
    for j in b57:
        if j['stage'] == 'baseline' and j['bracket']:
            out.append(dict({k: v for k, v in j.items() if k not in ('job', 'bracket')}, b57_job=j['job'], closing=None))
    return out


def mitra_jobs(a, b51, b64c):
    from fairtest.b41_analysis import load
    d = Path(a.b41)
    evals, realistic, _, _, _ = load(d)
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
                                closing=b51.get(('policy', seed, r['case_id'], r['sut']))))
    out += refine_jobs(Path(a.b56) / 'b56_mitra.jsonl', 'bo_warm_g4s', 'refine_g4s', b64c)
    out += refine_jobs(Path(a.b64) / 'b64_mitra.jsonl', 'bo_warm_g4ss', 'refine_g4ss', None)
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


def check_population(a):
    old = json.load(open(Path(a.b57) / 'population.json', encoding='utf-8'))['drivers']
    new = json.load(open(a.population, encoding='utf-8'))['drivers']
    if new[:len(old)] != old:
        raise SystemExit('population: the B57 drivers are not the unchanged prefix of the new population')
    return dict(b57_drivers=len(old), drivers=len(new), added=dict(collections.Counter(d['pop'] for d in new[len(old):])))


def write(path, jobs):
    with open(path, 'w', encoding='utf-8') as f:
        for i, j in enumerate(jobs):
            f.write(json.dumps(dict(job=i, bracket=True, **j)) + '\n')
    counts = collections.Counter(f"{j['attacker']}|{j['stage']}|cc_{'contact' if j['stored_cc'] else 'avoid'}" for j in jobs)
    return dict(jobs=len(jobs), counts=dict(sorted(counts.items())),
                with_closing=sum(j.get('closing') is not None for j in jobs if j['attacker'] == 'v2s'),
                v2s=sum(j['attacker'] == 'v2s' for j in jobs))


def main(a):
    rep = dict(population=check_population(a))
    b51_dev = closings([Path(a.b51) / 'closing_dev.jsonl'])
    b51_mitra = closings([Path(a.b51) / 'closing_mitra.jsonl'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep['dev'] = write(out / 'jobs_dev.jsonl', dev_jobs(a, b51_dev, closings([Path(a.b64) / 'closing_g4s_dev.jsonl'])))
    mj, n_base = mitra_jobs(a, b51_mitra, closings([Path(a.b64) / 'closing_g4s_mitra.jsonl']))
    rep['mitra'] = write(out / 'jobs_mitra.jsonl', mj)
    rep['mitra']['baseline_pool'] = n_base
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    for k in ('b57', 'b56', 'b64', 'b51', 'b41', 'population'):
        ap.add_argument(f'--{k}', required=True)
    main(ap.parse_args())
    sys.exit(0)
