"""Addendum of a v2j external evaluation (exiD protocol draft v2j, sections 5-6; on MiTra B41 it is post hoc).

Reads the output directory of an external evaluation in the B41 layout (b41_analysis.load) and adds
  S4'  non-inferiority, v2j - v2 on the weak rule-based SUTs: case-bootstrap 95% interval; lower bound > -5 pp?
  S7   impact severity: the SUT's closing speed at contact (target vx - generator vx at the last step) of every v2j
       reported failure, and P1 recomputed without the reported failures that close below 3 m/s
  S8   reference reaction margin: r_up of every v2j reported failure on the B36 grid (1.0 ... 3.0 s; b36 rule)
  an optional extra cluster key of the case events (exiD: the location) for the P1 interval
A reported failure is the policy stage's success under the one contract, else the first bo_warm_g4j success (seed-
independent, so one replay per case and SUT). Every replay error is recorded, never dropped.

  python -m fairtest.v2j_extra DIR OUT.json [--workers 24] [--cluster-key location]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b36_reaction_margin import UP
from fairtest.b41_analysis import K_FULL, _rows, boot_diff, composed, load, rate

WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
GROUPS = {
    'weak rule-based': WEAK,
    'BC human': ('bc_human_H0',),
    'ADS 2.0 m': ('ads@2.0_H0',),
    'learned': ('drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0', 'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
    'competent': ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0'),
}
LOW_CLOSING, NONINF = 3., -.05


def reported(d, evals, realistic, refine):
    """{(seed, case, sut): (stage, how, payload)} of the v2j reported failures within K_FULL queries."""
    actions = {}
    for path in sorted(d.glob('final_s[0-9].jsonl')):
        seed = path.stem.rsplit('_s', 1)[1]
        for r in _rows(path):
            if 'valid' in r:
                actions[(seed, r['case_id'], r['sut'])] = r['actions']
    theta = {}
    for r in _rows(d / 'refine.jsonl'):
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == 'bo_warm_g4j' and 'error' not in rec and rec.get('first_fair') is not None:
                theta[(r['case_id'], s)] = rec['best']['theta']
    out = {}
    for x, e in evals['final'].items():
        for (c, s), m in e.items():
            if m['robust'] and realistic.get((x, c, s), False):
                out[(x, c, s)] = ('policy', 'actions', actions[(x, c, s)])
            elif refine['bo_warm_g4j'].get((c, s)) is not None:
                out[(x, c, s)] = ('refine', 'theta', theta[(c, s)])
    return out


def task(job):
    ident, case_id, sut, how, payload = job
    from fairtest import runner
    from fairtest.b29_consensus import _run
    runner._init()
    case = runner._STATE['cases'][case_id]
    name, h = sut.rsplit('_H', 1)
    try:
        s = _run(case, how, payload, name, float(h))
        last = s['trace'][-1]
        contact = {r: bool(_run(case, how, payload, f'cc_r157@r{r:g}')['pair_contact']) for r in UP}
        r_up = .75
        for r in UP:
            if contact[r]:
                break
            r_up = r
        return dict(ident=ident, case_id=case_id, sut=sut, sut_contact=bool(s['pair_contact']),
                    closing=round(float(last['target']['vx'] - last['generator']['vx']), 3), r_up=r_up,
                    cc_contacts={f'{r:g}': c for r, c in contact.items()})
    except Exception as e:  # recorded, never dropped
        return dict(ident=ident, case_id=case_id, sut=sut, error=f'{type(e).__name__}: {e}')


def main(a):
    d = Path(a.dir)
    evals, realistic, refine, pool, _ = load(d)
    cases = sorted(c['case_id'] for c in pool['cases'])
    rec = {c['case_id']: c['event']['recording'] for c in pool['cases']}
    suts = sorted({s for e in evals.values() for x in e.values() for (_, s) in x})
    rep = dict(n_cases=len(cases))
    full = composed(evals, realistic, refine, suts, cases, K_FULL)
    full_v2 = composed(evals, realistic, refine, suts, cases, K_FULL, joint=False)
    # S4' non-inferiority on the weak rule-based SUTs (pooled per case over the four SUTs)
    w = [s for s in WEAK if s in full]
    j = {c: float(np.mean([full[s][c] for s in w])) for c in cases}
    v = {c: float(np.mean([full_v2[s][c] for s in w])) for c in cases}
    diff, ci = boot_diff(j, v)
    rep['S4prime'] = dict(suts=w, v2j=rate(j), v2=rate(v), diff=diff, ci=ci, margin=NONINF, non_inferior=ci[0] > NONINF,
                          per_sut={s: rate(full[s]) - rate(full_v2[s]) for s in w})
    # S7 / S8 replays of the reported failures (refinement plans once per case and SUT)
    rep_f = reported(d, evals, realistic, refine)
    jobs, seen = [], {}
    for (x, c, s), (stage, how, payload) in sorted(rep_f.items()):
        ident = f'{stage}|{x}|{c}|{s}' if stage == 'policy' else f'refine|{c}|{s}'
        if ident not in seen:
            seen[ident] = True
            jobs.append((ident, c, s, how, payload))
    with mp.Pool(a.workers) as p:
        res = {r['ident']: r for r in p.imap_unordered(task, jobs, chunksize=2)}
    errors = [r for r in res.values() if 'error' in r]
    rep['replays'] = dict(reported_failures=len(rep_f), unique=len(jobs), errors=len(errors), error_rows=errors[:20],
                          sut_contact_reproduced=sum(r.get('sut_contact', False) for r in res.values()))

    def ident_of(x, c, s):
        return f'policy|{x}|{c}|{s}' if rep_f[(x, c, s)][0] == 'policy' else f'refine|{c}|{s}'
    rep['S7'] = {}
    rep['S8'] = {}
    for g, members in GROUPS.items():
        rows = [res[ident_of(x, c, s)] for (x, c, s) in rep_f if s in members and 'error' not in res[ident_of(x, c, s)]]
        cl = [r['closing'] for r in rows]
        ru = [r['r_up'] for r in rows]
        rep['S7'][g] = dict(n=len(rows), closing_median=float(np.median(cl)) if cl else None,
                            closing_p10=float(np.percentile(cl, 10)) if cl else None,
                            below_3=round(float(np.mean([x < LOW_CLOSING for x in cl])), 4) if cl else None)
        rep['S8'][g] = dict(n=len(rows), r_up_median=float(np.median(ru)) if ru else None,
                            at_least_1_25=round(float(np.mean([x >= 1.25 for x in ru])), 4) if ru else None,
                            at_least_1_5=round(float(np.mean([x >= 1.5 for x in ru])), 4) if ru else None,
                            distribution=dict(collections.Counter(f'{x:g}' for x in ru)))
    # P1 without the reported failures closing below 3 m/s (the seed average of the remaining successes)
    seeds = sorted(evals['final'])

    def kept(s):
        per = {}
        for c in cases:
            hits = []
            for x in seeds:
                r = rep_f.get((x, c, s))
                ok = r is not None and 'error' not in res[ident_of(x, c, s)] and res[ident_of(x, c, s)]['closing'] >= LOW_CLOSING
                hits.append(float(ok))
            per[c] = float(np.mean(hits))
        return per
    if 'following_H0' in full and 'rss_sut_H0' in full:
        p1, ci1 = boot_diff(full['following_H0'], full['rss_sut_H0'])
        k0, kr = kept('following_H0'), kept('rss_sut_H0')
        p1s, ci1s = boot_diff(k0, kr)
        rep['S7']['P1_sensitivity'] = dict(p1=p1, ci=ci1, p1_without_low_closing=p1s, ci_without=ci1s,
                                           following_H0_kept=rate(k0), rss_kept=rate(kr))
        if a.cluster_key:
            key = {c['case_id']: c['event'].get(a.cluster_key) for c in pool['cases']}
            rep['P1_extra_cluster'] = dict(key=a.cluster_key, clusters=len(set(key.values())),
                                           ci=boot_diff(full['following_H0'], full['rss_sut_H0'], clusters=key)[1],
                                           recording_ci=boot_diff(full['following_H0'], full['rss_sut_H0'], clusters=rec)[1])
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('S4prime', 'replays')}, indent=1))
    for g in GROUPS:
        print(g, rep['S7'][g], rep['S8'][g])
    print(json.dumps(rep['S7'].get('P1_sensitivity'), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cluster-key', help='an extra case-event cluster key (exiD: location)')
    main(ap.parse_args())
