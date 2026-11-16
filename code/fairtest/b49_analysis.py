"""B49 analysis (B49 design, fixed before the run, available on request): cost accounting
and statistical units of the final agent v2j on the 62 development cases (review W5 P2-A).

  C1     the callable (fair_agent_v2 --joint, policy seeds 1-3 x 62 cases x 17 SUTs) equals the B46 unified-contract
         composition (found, stage, queries) on every attack
  cost   per attack, from the callable's counter: SUT search episodes (= queries), SUT verification replays, reference-
         driver episodes, shield CC rollouts, wall-clock; the policy stage alone (k = 1) from the same records; the
         training-free boundary-start BO (bo_warm_g4j alone, the stored B46 records + one certificate per success); the
         one-query RL baselines (stored B23 development evaluation); per-case precomputation and training
  units  case-cluster (and, with only three highD recordings, recording-cluster) bootstrap of v2j k37 and of the
         corrected B45 R1

  python -m fairtest.b49_analysis OUT.json --jobs fairtest_jobs --boundary exposed_cont_search_all7.jsonl \
         --train train_logs.json B49_s1.jsonl B49_s2.jsonl B49_s3.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage
from fairtest.b45_joint_realism import Support
from fairtest.fair_agent_v2 import CERT_REF, CERT_SUT, FRONT_REF, FRONT_SUT, MARGIN

GROUPS = {
    'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'),
    'BC human': ('bc_human_H0',),
    'ADS 2.0 m': ('ads@2.0_H0',),
    'learned': ('drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0', 'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
    'competent': ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0'),
}
GRID_PLANS = 200  # CC grid per case: the warm start of the CC-boundary search
BASELINES = ('collide_commit', 'cars', 'dynasto')
B, SEED = 10000, 20260928


def _rows(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def composition(jobs, sup):
    """B46 unified contract: policy robust fair failure with G2 and the joint support on its CC replay, else bo_warm_g4j."""
    robust, _ = policy_stage(jobs / 'b26', 'dev')
    ok = {}
    for r in _rows(jobs / 'b45' / 'replay_dev.jsonl'):
        if r['source'] == 'v2_policy':
            sc = sup.score(r['features']) if r['cut_in'] else None
            ok[(f"s{r['seed']}", r['case_id'], r['sut'])] = (sc is not None and sc <= sup.tau and r.get('exceedance') is not None
                                                            and r['exceedance'] <= 1.)
    bo = {}
    for name in ('joint_dev_rule.jsonl', 'joint_dev_learned.jsonl'):
        for r in _rows(jobs / 'b46' / name):
            per_run = r['seconds'] / max(1, len(r['runs']))
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == 'bo_warm_g4j' and 'error' not in rec:
                    bo[(r['case_id'], s)] = dict(first=rec.get('first_fair'), used=rec.get('sut_queries'), seconds=per_run)
    return robust, ok, bo


def policy_only_cost(r):
    """The stage-1 part of one callable record: the SUT query, its certificate / front-contact check / CC realism replay."""
    p = r.get('policy') or {}
    cert = p.get('certificate')
    verif = ref = 0
    if cert is not None:
        verif, ref = verif + CERT_SUT, ref + CERT_REF
        if 'error' not in cert and not cert['cc_contact'] and cert['cc_clearance'] >= MARGIN:
            verif, ref = verif + FRONT_SUT, ref + FRONT_REF
    if 'realism' in p:
        ref += 1
    return dict(search=1, verif=verif, ref=ref, shield=p.get('cc_rollouts', 0), found=r.get('stage') == 'policy' and r['found'])


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=round(float(x.mean()), 2), median=round(float(np.median(x)), 2), p90=round(float(np.percentile(x, 90)), 2),
                max=round(float(x.max()), 2)) if len(x) else None


def summarise(recs):
    """recs: dicts with found, search, verif, ref, shield, seconds (one per attack)."""
    found = sum(r['found'] for r in recs)
    out = dict(attacks=len(recs), detection=round(found / len(recs), 4) if recs else None)
    for k in ('search', 'verif', 'ref', 'shield', 'seconds'):
        v = [r[k] for r in recs if r.get(k) is not None]
        out[k] = stats(v)
        out[k + '_per_detection'] = round(sum(v) / found, 2) if found and v else None
    return out


def cluster_ci(per_case, recording, rng):
    """per_case: {case: value}; case bootstrap and recording bootstrap (recordings, then their cases) of the mean."""
    cases = sorted(per_case)
    v = np.array([per_case[c] for c in cases], float)
    idx = rng.integers(0, len(v), (B, len(v)))
    case_ci = np.percentile(v[idx].mean(1), [2.5, 97.5])
    recs = sorted({recording[c] for c in cases})
    by = {g: np.array([per_case[c] for c in cases if recording[c] == g], float) for g in recs}
    means = []
    for _ in range(B):
        pick = rng.choice(recs, len(recs))
        vals = np.concatenate([by[g][rng.integers(0, len(by[g]), len(by[g]))] for g in pick])
        means.append(vals.mean())
    rec_ci = np.percentile(means, [2.5, 97.5])
    return dict(mean=round(float(v.mean()), 4), n_cases=len(v), n_recordings=len(recs),
                case_ci=[round(float(x), 4) for x in case_ci], recording_ci=[round(float(x), 4) for x in rec_ci])


def b45_r1(jobs, sup, robust):
    """Corrected B45 R1 as (case, inside) per reported v2 failure (policy robust fair failure, else the bo_warm_g4g
    refinement's first robust success, its CC replay)."""
    pol = {(f"s{r['seed']}", r['case_id'], r['sut']): r for r in _rows(jobs / 'b45' / 'replay_dev.jsonl') if r['source'] == 'v2_policy'}
    for r in pol.values():
        sc = sup.score(r['features']) if r['cut_in'] else None
        r['inside'] = sc is not None and sc <= sup.tau
    ref = {(r['case_id'], r['sut']): (r['cc_features'] if r['cut_in'] else {}) for r in _rows(jobs / 'b45' / 'refine_cc_dev.jsonl')}
    out = []
    for key, ok in robust.items():
        if ok and key in pol:
            out.append((key[1], pol[key]['inside']))
        elif not ok and (key[1], key[2]) in ref:
            sc = sup.score(ref[(key[1], key[2])])
            out.append((key[1], sc is not None and sc <= sup.tau))
    return out


def main(a):
    jobs = Path(a.jobs)
    sup = Support()
    rng = np.random.default_rng(SEED)
    robust, ok, bo = composition(jobs, sup)
    pool = json.loads((Path(a.data) / 'case_pool.json').read_text(encoding='utf-8'))
    recording = {c['case_id']: c['event']['recording'] for c in pool['cases']}
    calls = {}
    for path in a.calls:
        seed = 's' + Path(path).stem.split('_s')[-1][0]
        for r in _rows(path):
            calls[(seed, r['case_id'], r['sut'])] = r
    # C1: callable = composition
    errors = [dict(seed=k[0], case_id=k[1], sut=k[2], error=r['error']) for k, r in calls.items() if 'error' in r]
    diffs, same = [], 0
    for (seed, c, s), r in calls.items():
        if 'error' in r:
            continue
        q = (bo.get((c, s)) or {}).get('first')
        if robust.get((seed, c, s), False) and ok.get((seed, c, s), False):
            want = (True, 'policy', 1)
        elif q is not None:
            want = (True, 'refine', 1 + q)
        else:
            want = (False, None, 37)
        got = (bool(r['found']), r['stage'] if r['found'] else None, r['queries'])
        same += got == want
        if got != want:
            diffs.append(dict(seed=seed, case_id=c, sut=s, callable=got, composition=want, status=r.get('status')))
    n = len(calls) - len(errors)
    rep = dict(C1=dict(attacks=len(calls), errors=len(errors), compared=n, identical=same, supported=same == n and not errors,
                       differences=diffs[:40], error_rows=errors[:20]))
    # costs per SUT group
    boundary = {r['case_id']: r.get('cc_sims') for r in _rows(a.boundary)} if a.boundary else {}
    dev_cases = sorted({k[1] for k in calls})
    base, unjudged = {}, 0
    margin = {}
    for r in _rows(jobs / 'b23' / 'eval' / 'margin_dev.jsonl'):
        margin[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= 3.5
                                                             and r['cc_clearance'] >= MARGIN)
    for b in BASELINES:
        for s in (1, 2, 3):
            name = f'{b}_s{s}_dev.jsonl'
            for r in _rows(jobs / 'b23' / 'eval' / name):
                if 'valid' not in r:
                    continue
                col = bool(r['valid'] and r['contact'])
                refs = r.get('refs') or {}
                if col and 'cc_r157' not in refs:  # a contact without its CC judgement (not expected)
                    unjudged += 1
                    continue
                cc = bool(refs['cc_r157']['contact']) if col else False
                base.setdefault(b, []).append(dict(case_id=r['case_id'], sut=r['sut'], contact=col, fair=col and not cc,
                                                   attacker=col and cc, robust=margin.get((name, r['case_id'], r['sut']), False)))
    rep['baseline_contacts_without_cc_judgement'] = unjudged
    rep['groups'] = {}
    for g, suts in list(GROUPS.items()) + [('all 17', tuple(sorted({k[2] for k in calls})))]:
        agent, pol1, bo_only = [], [], []
        for (seed, c, s), r in calls.items():
            if s not in suts or 'error' in r:
                continue
            cost = r['cost']
            agent.append(dict(found=r['found'], search=cost['sut_search_episodes'], verif=cost['sut_verification_replays'],
                              ref=cost['reference_episodes'], shield=cost['shield_cc_rollouts'], seconds=r['seconds']))
            pol1.append(policy_only_cost(r))
        for c in dev_cases:
            for s in suts:
                b_ = bo.get((c, s))
                if b_ is None:
                    continue
                f = b_['first'] is not None
                bo_only.append(dict(found=f, search=b_['used'], verif=CERT_SUT * f, ref=b_['used'] + CERT_REF * f, shield=0,
                                    seconds=b_['seconds']))
        e = dict(suts=list(suts), v2j_agent=summarise(agent), policy_only=summarise(pol1), bo_only=summarise(bo_only),
                 bo_only_k1=round(float(np.mean([r['search'] == 1 and r['found'] for r in bo_only])), 4) if bo_only else None)
        for b in BASELINES:
            v = [x for x in base.get(b, []) if x['sut'] in suts]
            e[b] = dict(attacks=len(v), contact=round(float(np.mean([x['contact'] for x in v])), 4) if v else None,
                        fair=round(float(np.mean([x['fair'] for x in v])), 4) if v else None,
                        robust_fair=round(float(np.mean([x['robust'] for x in v])), 4) if v else None,
                        attacker_share=round(sum(x['attacker'] for x in v) / max(1, sum(x['contact'] for x in v)), 4) if v else None)
        # units: v2j k37 per case (mean over seeds and the group's SUTs), case- and recording-cluster intervals
        per_case = {}
        for c in dev_cases:
            vals = [calls[(sd, c, s)]['found'] for sd in ('s1', 's2', 's3') for s in suts
                    if (sd, c, s) in calls and 'error' not in calls[(sd, c, s)]]
            if vals:
                per_case[c] = float(np.mean(vals))
        e['v2j_k37_units'] = cluster_ci(per_case, recording, rng) if per_case else None
        rep['groups'][g] = e
    rep['precompute_per_case'] = dict(cc_grid=GRID_PLANS, cc_boundary_search=stats([boundary[c] for c in dev_cases if boundary.get(c) is not None]),
                                      note='SUT-independent: computed once per case for every SUT and seed')
    if a.train:
        tr = json.load(open(a.train, encoding='utf-8'))
        rep['training'] = tr
    r1 = b45_r1(jobs, sup, policy_stage(jobs / 'b26', 'dev')[0])
    per_case = {}
    for c, x in r1:
        per_case.setdefault(c, []).append(float(x))
    rep['B45_R1_units'] = dict(reports=len(r1), inside=round(float(np.mean([x for _, x in r1])), 4),
                               unique_cases=len(per_case),
                               case_weighted=cluster_ci({c: float(np.mean(v)) for c, v in per_case.items()}, recording, rng))
    # the report-level share with a case-cluster bootstrap (resample cases, keep all their reports)
    cases = sorted(per_case)
    sums = np.array([sum(per_case[c]) for c in cases]); cnts = np.array([len(per_case[c]) for c in cases])
    idx = rng.integers(0, len(cases), (B, len(cases)))
    share = sums[idx].sum(1) / cnts[idx].sum(1)
    rep['B45_R1_units']['report_share_case_ci'] = [round(float(x), 4) for x in np.percentile(share, [2.5, 97.5])]
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep['C1'], indent=1)[:1500])
    for g, e in rep['groups'].items():
        ag, p1, bo_ = e['v2j_agent'], e['policy_only'], e['bo_only']
        print(f"{g:16s} v2j {100*ag['detection']:5.1f}% search {ag['search']['mean']:5.1f} (per det {ag['search_per_detection']}) "
              f"verif {ag['verif']['mean']:.2f} ref {ag['ref']['mean']:5.1f} shield {ag['shield']['mean']:5.1f} sec {ag['seconds']['mean']:5.1f} | "
              f"policy {100*p1['detection']:5.1f}% | BO {100*bo_['detection']:5.1f}% search {bo_['search']['mean']:5.1f} "
              f"(per det {bo_['search_per_detection']}) BO k1 {100*e['bo_only_k1']:5.1f}% | "
              + ' '.join(f"{b} fair {100*e[b]['fair']:.1f}%" for b in BASELINES if e[b]['fair'] is not None))
        u = e['v2j_k37_units']
        print(f"{'':16s} units: mean {u['mean']} case CI {u['case_ci']} recording CI {u['recording_ci']} ({u['n_recordings']} recordings)")
    print(json.dumps(rep['B45_R1_units'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--data', required=True, help='data dir with the development case_pool.json (recordings)')
    ap.add_argument('--boundary', help='stored CC-boundary search (cc_sims per case)')
    ap.add_argument('--train', help='training logs summary (JSON)')
    ap.add_argument('calls', nargs='+')
    main(ap.parse_args())
