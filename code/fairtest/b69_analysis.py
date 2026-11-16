"""B69 analysis (B69 design, available on request): the final agent v2s, the
collision-seeking baseline and attack-free cut-ins on four third-party SUTs, development cases. Each SUT has its own
B41-layout folder DIR/<sut>/ (natural.jsonl, final_s{1,2,3}.jsonl, collide_commit_s{1,2,3}.jsonl, margin.jsonl,
policy_replay.jsonl, refine.jsonl, consensus.jsonl, closing.jsonl, case_pool.json; commonroad_rp also
refine_g4ss.jsonl); every file labels the SUT acc_prod_aeb_H0 (the slot it runs in). The definitions are imported
unchanged from b41_analysis (load, rates, bootstrap) and exid_analysis (composition, refinement loading, consensus,
execution statistics).

  T1  implementation check, each SUT: attack-free cut-ins without SUT contact or run error; no attacker-caused collision
      in the v2s policy stage; no executed inadmissible query in the v2s refinement
  T2  collision-seeking (collide_commit s1-3), pooled over the four SUTs: >= 80% of its collisions attacker-caused
      (C collides too), judged when there are >= 10 collisions
  T3  each SUT with at least one reported failure: share of cases whose reported v2s failure (37 queries) is not
      consensus-fair <= 5% (an unjudged reported failure counts as not consensus-fair)
  D   v2s detection at 1, 6 and 37 queries (case bootstrap); the policy stage alone; collision-seeking rates per SUT;
      closing speed of the reported failures (share >= 3 m/s, median); refinement execution statistics; severe detection
      as in b64_analysis: `v2s sev` (the failure v2s reports, kept only if it closes at >= 3 m/s) and, where the
      severity-floor refinement ran (refine_g4ss.jsonl, b64_severity_screened), `v2s-S` (a policy failure closing at
      >= 3 m/s, or the bo_warm_g4ss first success within the budget)

  python -m fairtest.b69_analysis DIR OUT.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest.b41_analysis import K_FULL, boot_rate, load, rate, seed_avg
from fairtest.exid_analysis import V2S, composed, load_consensus, load_refine, not_consensus, query_stats

V2SS = 'bo_warm_g4ss'

SUTS = ('commonroad_rp', 'sumo_acc', 'sumo_krauss', 'sumo_idm')
LABEL = 'acc_prod_aeb_H0'
T2_MIN, T2_N, T3_MAX, V_MIN = .8, 10, .05, 3.


def natural(path, cases):
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    runs = {r['case_id']: r['runs'].get(LABEL, {}) for r in rows}
    return dict(cases=len(runs), missing=sorted(set(cases) - set(runs)), errors=sum('error' in x for x in runs.values()),
                sut_contact=sum(bool(x.get('sut_contact')) for x in runs.values()),
                invalid=sum(not x.get('valid', False) for x in runs.values() if 'error' not in x))


def closings(path):
    pol, ref, errors = {}, {}, 0
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if 'error' in r:
            errors += 1
        elif r['kind'] == 'policy':
            pol[(r['seed'], r['case_id'], r['sut'])] = r['closing']
        elif r['kind'] == V2S:
            ref[(r['case_id'], r['sut'])] = r['closing']
    return pol, ref, errors


def severity(evals, realistic, first, pol_cl, ref_cl, cases):
    """Closing speed of every reported failure (one per seed and case: the policy episode if it reports, else the
    refinement's first success within 37 queries, as exid_consensus pairs them)."""
    vals, missing = [], 0
    for x in sorted(evals['final']):
        for c in cases:
            q = first[V2S].get((c, LABEL))
            if evals['final'][x].get((c, LABEL), {}).get('robust', False) and realistic.get((x, c, LABEL), False):
                v = pol_cl.get((x, c, LABEL))
            elif q is not None and 1 + q <= K_FULL:
                v = ref_cl.get((c, LABEL))
            else:
                continue
            if v is None:
                missing += 1
            else:
                vals.append(v)
    return dict(reported=len(vals) + missing, missing=missing, severe_share=float(np.mean([v >= V_MIN for v in vals])) if vals else None,
                closing_median=float(np.median(vals)) if vals else None)


def severe(evals, realistic, first, pol_cl, ref_cl, first_ss, cases, k):
    """{variant: {case: seed-averaged severe success}} for `v2s sev` and (with first_ss) `v2s-S`, as b64_analysis."""
    seeds = sorted(evals['final'])
    missing = dict(policy=0, g4s=0)

    def pol(x, c, sev):
        if not (evals['final'][x].get((c, LABEL), {}).get('robust') and realistic.get((x, c, LABEL), False)):
            return False
        if not sev:
            return True
        v = pol_cl.get((x, c, LABEL))
        if v is None:
            missing['policy'] += 1
            return False
        return v >= V_MIN

    def ref(c, variant):
        if k < 2:
            return False
        if variant == 'v2s-S':
            q = first_ss.get((c, LABEL))
            return q is not None and 1 + q <= k
        q = first[V2S].get((c, LABEL))
        if q is None or 1 + q > k:
            return False
        v = ref_cl.get((c, LABEL))
        if v is None:
            missing['g4s'] += 1
            return False
        return v >= V_MIN

    def one_(x, c, variant):
        if variant == 'v2s sev':  # the failure v2s reports, dropped if it closes below V_MIN
            return pol(x, c, True) if pol(x, c, False) else ref(c, variant)
        return pol(x, c, True) or ref(c, variant)  # v2s-S: a light policy failure is no success, the refinement runs

    out = {}
    for variant in ('v2s sev',) + (('v2s-S',) if first_ss is not None else ()):
        out[variant] = {c: float(np.mean([float(one_(x, c, variant)) for x in seeds])) for c in cases}
    return out, missing


def one(d):
    evals, realistic, _, pool, tau = load(d)
    cases = sorted(c['case_id'] for c in pool['cases'] if c['event'].get('partition') == 'development')
    first, recs, ref_errors, conflicts = load_refine([d / 'refine.jsonl'])
    rep = dict(n_cases=len(cases), joint_tau=tau, refine_errors=ref_errors, refine_conflicts=len(conflicts),
               seeds={a: sorted(evals[a]) for a in evals})
    rep['natural'] = natural(d / 'natural.jsonl', cases)
    det = {}
    for k in (1, 6, K_FULL):
        per = composed(evals, realistic, first, V2S, [LABEL], cases, k)[LABEL]
        r_, ci = boot_rate(per)
        det[f'k{k}'] = dict(rate=r_, ci=ci)
    rep['v2s'] = det
    rep['v2s']['policy_share_of_k37'] = (det['k1']['rate'] / det[f'k{K_FULL}']['rate']) if det[f'k{K_FULL}']['rate'] else None
    rep['policy'] = {a: {m: rate(seed_avg(evals, a, [LABEL], cases, m)[LABEL]) for m in ('collision', 'fair', 'robust', 'unfair', 'unjudged')}
                     for a in evals}
    rep['collide_collisions'] = [m for x in evals['collide_commit'].values() for (c, s), m in x.items() if s == LABEL and c in cases
                                 and m['collision']]
    pol, ref, rows = load_consensus(d / 'consensus.jsonl')
    per, unjudged = not_consensus(evals, realistic, first, pol, ref, LABEL, cases)
    r_, ci = boot_rate(per)
    reported = det[f'k{K_FULL}']['rate'] > 0
    rep['T3'] = dict(not_consensus_rate=r_, ci=ci, unjudged=unjudged, reported_any=reported, consensus_rows=len(rows),
                     consensus_errors=sum('error' in r for r in rows),
                     grades=dict(collections.Counter(r.get('grade') for r in rows if 'error' not in r)),
                     supported=(r_ <= T3_MAX) if reported else None)
    ex = query_stats(recs, V2S, [LABEL])
    rep['execution'] = ex
    pol_cl, ref_cl, cl_errors = closings(d / 'closing.jsonl')
    rep['severity'] = dict(severity(evals, realistic, first, pol_cl, ref_cl, cases), closing_errors=cl_errors)
    first_ss, ex_ss = None, None
    if (d / 'refine_g4ss.jsonl').exists():
        f_ss, recs_ss, err_ss, conf_ss = load_refine([d / 'refine_g4ss.jsonl'])
        first_ss = f_ss[V2SS]
        ex_ss = dict(query_stats(recs_ss, V2SS, [LABEL]), errors=err_ss, conflicts=len(conf_ss))
        rep['execution_g4ss'] = ex_ss
    for k in (1, K_FULL):
        per, missing = severe(evals, realistic, first, pol_cl, ref_cl, first_ss, cases, k)
        for variant, pc in per.items():
            r_, ci = boot_rate(pc)
            rep['severity'][f'{variant} k{k}'] = dict(rate=r_, ci=ci, missing=missing)
    nat = rep['natural']
    rep['T1'] = dict(natural_ok=nat['sut_contact'] == 0 and nat['errors'] == 0 and not nat['missing'],
                     policy_attacker_caused=rep['policy']['final']['unfair'],
                     executed_inadmissible=ex['inadmissible'], logged=ex['logged'])
    if ex_ss is not None:
        rep['T1']['executed_inadmissible_g4ss'] = ex_ss['inadmissible']
    rep['T1']['supported'] = bool(rep['T1']['natural_ok'] and rep['T1']['policy_attacker_caused'] == 0.
                                  and ex['logged'] and ex['inadmissible'] == 0
                                  and (ex_ss is None or (ex_ss['logged'] and ex_ss['inadmissible'] == 0)))
    return rep


def main(a):
    d = Path(a.dir)
    rep = dict(suts={})
    for s in SUTS:
        rep['suts'][s] = one(d / s)
    cols = [m for s in SUTS for m in rep['suts'][s].pop('collide_collisions')]
    share = (sum(m['unfair'] for m in cols) / len(cols)) if cols else None
    rep['T2'] = dict(collisions=len(cols), attacker_share=share, judged=len(cols) >= T2_N,
                     supported=(share >= T2_MIN) if len(cols) >= T2_N else None,
                     per_sut={s: dict(collision=rep['suts'][s]['policy']['collide_commit']['collision'],
                                      attacker_caused=rep['suts'][s]['policy']['collide_commit']['unfair'])
                              for s in SUTS})
    rep['T1'] = {s: rep['suts'][s]['T1']['supported'] for s in SUTS}
    rep['T3'] = {s: rep['suts'][s]['T3']['supported'] for s in SUTS}
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for s in SUTS:
        r = rep['suts'][s]
        print(f"{s:14s} natural contact {r['natural']['sut_contact']}/{r['natural']['cases']} | v2s k1 {r['v2s']['k1']['rate'] * 100:5.1f} "
              f"k6 {r['v2s']['k6']['rate'] * 100:5.1f} k37 {r['v2s'][f'k{K_FULL}']['rate'] * 100:5.1f} | not consensus "
              f"{r['T3']['not_consensus_rate'] * 100:4.1f} (unjudged {r['T3']['unjudged']}) | collide col "
              f"{r['policy']['collide_commit']['collision'] * 100:5.1f} attacker {r['policy']['collide_commit']['unfair'] * 100:5.1f} "
              f"| severe share {r['severity']['severe_share']} | v2s sev k37 {r['severity'][f'v2s sev k{K_FULL}']['rate'] * 100:5.1f}"
              + (f" v2s-S k37 {r['severity'][f'v2s-S k{K_FULL}']['rate'] * 100:5.1f}" if f'v2s-S k{K_FULL}' in r['severity'] else '')
              + f" | T1 {r['T1']['supported']}")
    print('T1', rep['T1'], '| T2', {k: rep['T2'][k] for k in ('collisions', 'attacker_share', 'supported')}, '| T3', rep['T3'])
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('out')
    main(ap.parse_args())
