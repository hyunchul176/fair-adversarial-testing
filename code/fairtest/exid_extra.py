"""exiD appendix analyses for the final agent v2s (exiD protocol draft §5, available on request: S7, S8, S10, S11, extra cluster key).
The definitions are those of v2j_extra.py (S7 closing speed, S8 r_up on the B36 grid, P1 without low-closing failures)
and authority_strata.py (S10 bins); only the refinement method is a parameter (default bo_warm_g4s), and the refinement
results come from the REFINE files (exid_analysis.load_refine). v2j_extra.py and authority_strata.py stay unchanged for
the MiTra post hoc results they produced.

  S7   closing speed at contact of every reported failure; P1 recomputed without the failures closing below 3 m/s
  S8   r_up of every reported failure (the CC's reaction raised along the B36 grid until it collides)
  S10  detection within 37 queries by the relative deceleration headroom (SUT initial speed - 15 m/s)
  S11  AEB H0 / H0.5: natural cut-in conditions at p10 (headway, closing speed = SUT - cut-in speed, SUT speed) of the
       cases with a reported failure, against all cases
  extra cluster key (exiD: location) for the P1 interval

  python -m fairtest.exid_extra DIR OUT.json --refine R.jsonl [...] [--method bo_warm_g4s] [--workers 24]
                                 [--cluster-key location] [--no-replay]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.authority_strata import BINS, SUTS as STRATA_SUTS
from fairtest.b41_analysis import K_FULL, _rows, boot_diff, load, rate
from fairtest.exid_analysis import composed, load_refine
from fairtest.v2j_extra import GROUPS, LOW_CLOSING, task

AEB = ('following_aeb_H0', 'following_aeb_H0.5')


def reported(d, evals, realistic, first, recs, method):
    """{(seed, case, sut): (stage, how, payload)} of the reported failures within K_FULL queries."""
    actions = {}
    for path in sorted(d.glob('final_s[0-9].jsonl')):
        seed = path.stem.rsplit('_s', 1)[1]
        for r in _rows(path):
            if 'valid' in r:
                actions[(seed, r['case_id'], r['sut'])] = r['actions']
    out = {}
    for x, e in evals['final'].items():
        for (c, s), m in e.items():
            q = first[method].get((c, s))
            if m['robust'] and realistic.get((x, c, s), False):
                out[(x, c, s)] = ('policy', 'actions', actions[(x, c, s)])
            elif q is not None and 1 + q <= K_FULL:
                out[(x, c, s)] = ('refine', 'theta', recs[method][(c, s)]['best']['theta'])
    return out


def conditions(pool, cases):
    """{case: (p10 headway s, p10 closing m/s = SUT - cut-in speed, p10 SUT speed m/s)} from the sealed case events."""
    out = {}
    for c in pool['cases']:
        if c['case_id'] in cases:
            p = c['event']['metrics']['p10']
            out[c['case_id']] = (p['headway'], p['target_speed'] - p['generator_speed'], p['target_speed'])
    return out


def main(a):
    d = Path(a.dir)
    evals, realistic, _, pool, _ = load(d)
    first, recs, _, _ = load_refine(a.refine)
    cases = sorted(c['case_id'] for c in pool['cases'])
    rec_of = {c['case_id']: c['event']['recording'] for c in pool['cases']}
    suts = sorted({s for e in evals['final'].values() for (_, s) in e})
    full = composed(evals, realistic, first, a.method, suts, cases, K_FULL)
    rep = dict(n_cases=len(cases), method=a.method)
    rep_f = reported(d, evals, realistic, first, recs, a.method)
    # S10 authority strata
    head = {c['case_id']: c['physical_initial_state'][1]['vx'] - 15. for c in pool['cases']}
    rep['S10'] = dict(headroom_median=float(np.median(list(head.values()))) if head else None, bins={})
    for lo, hi in BINS:
        cs = [c for c in cases if lo <= head[c] < hi]
        rep['S10']['bins'][f'{lo},{hi}'] = dict(n=len(cs), **{s: (round(float(np.mean([full[s][c] for c in cs])), 4) if cs else None)
                                                               for s in STRATA_SUTS if s in full})
    # S11 AEB failure conditions
    cond = conditions(pool, set(cases))
    allv = np.array(list(cond.values())) if cond else np.zeros((0, 3))
    rep['S11'] = dict(all=dict(n=len(cond), headway_median=float(np.median(allv[:, 0])) if len(allv) else None,
                               closing_median=float(np.median(allv[:, 1])) if len(allv) else None,
                               sut_speed_median=float(np.median(allv[:, 2])) if len(allv) else None))
    for s in AEB:
        fc = sorted({c for (x, c, ss) in rep_f if ss == s})
        v = np.array([cond[c] for c in fc]) if fc else np.zeros((0, 3))
        rep['S11'][s] = dict(cases=len(fc), headway_median=float(np.median(v[:, 0])) if len(v) else None,
                             closing_median=float(np.median(v[:, 1])) if len(v) else None,
                             sut_speed_median=float(np.median(v[:, 2])) if len(v) else None)
    if a.cluster_key and 'following_H0' in full and 'rss_sut_H0' in full:
        key = {c['case_id']: c['event'].get(a.cluster_key) for c in pool['cases']}
        rep['P1_extra_cluster'] = dict(key=a.cluster_key, clusters=len(set(key.values())),
                                       ci=boot_diff(full['following_H0'], full['rss_sut_H0'], clusters=key)[1],
                                       recording_ci=boot_diff(full['following_H0'], full['rss_sut_H0'], clusters=rec_of)[1])
    if not a.no_replay:
        jobs, seen = [], set()
        for (x, c, s), (stage, how, payload) in sorted(rep_f.items()):
            ident = f'{stage}|{x}|{c}|{s}' if stage == 'policy' else f'refine|{c}|{s}'
            if ident not in seen:
                seen.add(ident)
                jobs.append((ident, c, s, how, payload))
        with mp.Pool(a.workers) as p:
            res = {r['ident']: r for r in p.imap_unordered(task, jobs, chunksize=2)}
        errors = [r for r in res.values() if 'error' in r]
        rep['replays'] = dict(reported_failures=len(rep_f), unique=len(jobs), errors=len(errors), error_rows=errors[:20],
                              sut_contact_reproduced=sum(r.get('sut_contact', False) for r in res.values()))

        def ident_of(x, c, s):
            return f'policy|{x}|{c}|{s}' if rep_f[(x, c, s)][0] == 'policy' else f'refine|{c}|{s}'
        rep['S7'], rep['S8'] = {}, {}
        for g, members in GROUPS.items():
            rows = [res[ident_of(x, c, s)] for (x, c, s) in rep_f if s in members and 'error' not in res[ident_of(x, c, s)]]
            cl, ru = [r['closing'] for r in rows], [r['r_up'] for r in rows]
            rep['S7'][g] = dict(n=len(rows), closing_median=float(np.median(cl)) if cl else None,
                                closing_p10=float(np.percentile(cl, 10)) if cl else None,
                                below_3=round(float(np.mean([x < LOW_CLOSING for x in cl])), 4) if cl else None)
            rep['S8'][g] = dict(n=len(rows), r_up_median=float(np.median(ru)) if ru else None,
                                at_least_1_25=round(float(np.mean([x >= 1.25 for x in ru])), 4) if ru else None,
                                at_least_1_5=round(float(np.mean([x >= 1.5 for x in ru])), 4) if ru else None,
                                distribution=dict(collections.Counter(f'{x:g}' for x in ru)))
        seeds = sorted(evals['final'])

        def kept(s):
            return {c: float(np.mean([float((x, c, s) in rep_f and 'error' not in res[ident_of(x, c, s)]
                                            and res[ident_of(x, c, s)]['closing'] >= LOW_CLOSING) for x in seeds])) for c in cases}
        if 'following_H0' in full and 'rss_sut_H0' in full:
            p1, ci1 = boot_diff(full['following_H0'], full['rss_sut_H0'])
            k0, kr = kept('following_H0'), kept('rss_sut_H0')
            p1s, ci1s = boot_diff(k0, kr)
            rep['S7']['P1_sensitivity'] = dict(p1=p1, ci=ci1, p1_without_low_closing=p1s, ci_without=ci1s,
                                               following_H0_kept=rate(k0), rss_kept=rate(kr))
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in rep if k in ('n_cases', 'S10', 'S11', 'replays')}, indent=1)[:3000])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('out')
    ap.add_argument('--refine', nargs='+', required=True)
    ap.add_argument('--method', default='bo_warm_g4s')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cluster-key', help='an extra case-event cluster key (exiD: location)')
    ap.add_argument('--no-replay', action='store_true', help='only S10, S11 and the cluster interval (no simulation)')
    main(ap.parse_args())
