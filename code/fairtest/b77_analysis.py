"""B77 (B77 analysis plan, fixed before computing, available on request): post hoc,
descriptive analyses of stored rows only (no simulation), review of manuscript v7 items R6 (C7), R1 (C1) and STR-01/03.

  budget    learning versus search on the development cases (in-sample), TGSIM and MiTra (post hoc) and the prospective
            HIGH-SIM and exiD runs (read post hoc): detection of the agent (policy query first, then the screened
            refinement) within k = 1..37 SUT queries and of the screened search alone within k = 1..36, per SUT group, with
            case-bootstrap 95% bands; paired differences (agent minus search alone) at k = 1, 6 and the full budget with
            intervals; the smallest search budget that matches the policy's single query. Reproduces Table S5 where it exists.
  readings  Table S4 rows for HIGH-SIM and exiD: detection within 37 queries when a reported failure counts only if the
            CARS-code reading of the CC (cc_jrc), of the FSM (fsm_cars), or both, also avoid it; per SUT group, and P1.
  headway   HIGH-SIM and exiD: share of the reported failures that the CARS-code CC also avoids and P1 under C and under C
            and the CARS CC, by the cut-in's initial headway bin [0, 1, 1.5, 2.5, 4, inf) s.
  cost      per test (case and SUT, development): SUT queries and CC replays of the agent and of the search alone, and the
            SUT-query cost at which the agent's total cost equals the search's, with the policy's training amortised.

  python -m fairtest.b77_analysis OUT.json --jobs fairtest_jobs
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401

GROUPS = {'weak rule-based': ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0'),
          'learned lane-assignment': ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0'),
          'BC human': ('bc_human_H0',), 'ADS 2.0 m': ('ads@2.0_H0',),
          'competent (8)': ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
                            'drl_sut@3_H0', 'drl_sut@6_H0')}
K_AGENT, K_SEARCH, N_BOOT, SEED = 37, 36, 10_000, 77
BINS = (0., 1., 1.5, 2.5, 4., float('inf'))
TABLE_S5 = {  # Table S5: (search 1/6/36, agent 1/6/37, queries to match)
    ('dev', 'weak rule-based'): ((17.3, 22.2, 70.6), (62.2, 66.1, 74.3), 21), ('tgsim', 'weak rule-based'): ((27.8, 32.5, 67.2), (45.7, 55.7, 69.6), 12),
    ('mitra', 'weak rule-based'): ((28.5, 37.1, 59.2), (30.3, 44.6, 59.2), 2),
    ('dev', 'learned lane-assignment'): ((4.8, 9.7, 59.7), (27.2, 32.4, 62.5), 11), ('tgsim', 'learned lane-assignment'): ((7.1, 16.7, 57.9), (23.5, 32.6, 60.1), 9),
    ('mitra', 'learned lane-assignment'): ((9.9, 27.5, 62.2), (17.6, 34.6, 63.0), 4)}


def _rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _g4s(paths):
    """{(case, sut): (first success or None, record)} of bo_warm_g4s."""
    out = {}
    for p in paths:
        for r in _rows(p):
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == 'bo_warm_g4s' and 'error' not in rec:
                    out[(r['case_id'], s)] = (rec.get('first_fair'), rec)
    return out


def dataset(J, ds):
    """(policy_ok {(seed, case, sut)}, g4s {(case, sut): (q, rec)}, seeds, cases)."""
    if ds in ('dev', 'tgsim', 'mitra'):
        from fairtest.b56_analysis import policy
        robust, inside = policy(J, ds)
        ok = {k: bool(v and inside.get(k, False)) for k, v in robust.items()}
        g4s = _g4s([J / 'b56' / f'b56_{ds}.jsonl'])
    else:
        from fairtest.b41_analysis import load
        d = J / ('b70' if ds == 'highsim' else 'b71')
        evals, realistic, _, _, _ = load(d)
        ok = {(x, c, s): bool(m.get('robust') and realistic.get((x, c, s), False)) for x in evals['final'] for (c, s), m in evals['final'][x].items()}
        g4s = _g4s([d / 'refine.jsonl'])
    seeds = sorted({k[0] for k in ok})
    cases = sorted({k[1] for k in ok})
    return ok, g4s, seeds, cases


def per_case(ok, g4s, seeds, cases, suts, k, agent):
    """[case] group value at budget k (mean over the group's SUTs)."""
    vals = []
    for c in cases:
        v = []
        for s in suts:
            q = g4s.get((c, s), (None, None))[0]
            if agent:
                v.append(np.mean([ok.get((x, c, s), False) or (k >= 2 and q is not None and 1 + q <= k) for x in seeds]))
            else:
                v.append(float(q is not None and q <= k))
        vals.append(float(np.mean(v)))
    return np.array(vals)


def band(M):
    """M: [k, case] -> mean and case-bootstrap 2.5/97.5 percentiles per k."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M.shape[1], (N_BOOT, M.shape[1]))
    boots = M[:, idx].mean(2)  # [k, boot]
    return M.mean(1), np.percentile(boots, 2.5, 1), np.percentile(boots, 97.5, 1)


def diff_ci(a, b):
    rng = np.random.default_rng(SEED)
    d = a - b
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(1)
    return round(float(d.mean()) * 100, 1), [round(float(np.percentile(m, 2.5)) * 100, 1), round(float(np.percentile(m, 97.5)) * 100, 1)]


def budget(J):
    out = {}
    for ds in ('dev', 'tgsim', 'mitra', 'highsim', 'exid'):
        ok, g4s, seeds, cases = dataset(J, ds)
        out[ds] = dict(cases=len(cases), seeds=seeds, groups={})
        for g, suts in GROUPS.items():
            A = np.array([per_case(ok, g4s, seeds, cases, suts, k, True) for k in range(1, K_AGENT + 1)])
            S = np.array([per_case(ok, g4s, seeds, cases, suts, k, False) for k in range(1, K_SEARCH + 1)])
            am, alo, ahi = band(A)
            sm, slo, shi = band(S)
            one = am[0]
            match = next((k for k in range(1, K_SEARCH + 1) if sm[k - 1] >= one - 1e-12), None)
            e = dict(agent=[round(float(x) * 100, 2) for x in am], agent_lo=[round(float(x) * 100, 2) for x in alo],
                     agent_hi=[round(float(x) * 100, 2) for x in ahi], search=[round(float(x) * 100, 2) for x in sm],
                     search_lo=[round(float(x) * 100, 2) for x in slo], search_hi=[round(float(x) * 100, 2) for x in shi],
                     queries_to_match=match, diff_k1=diff_ci(A[0], S[0]), diff_k6=diff_ci(A[5], S[5]), diff_full=diff_ci(A[-1], S[-1]))
            ref = TABLE_S5.get((ds, g))
            if ref:
                got = ((round(float(sm[0]) * 100, 1), round(float(sm[5]) * 100, 1), round(float(sm[-1]) * 100, 1)),
                       (round(float(am[0]) * 100, 1), round(float(am[5]) * 100, 1), round(float(am[-1]) * 100, 1)), match)
                e['table_s5_check'] = dict(got=got, printed=ref, ok=got == ref)
            out[ds]['groups'][g] = e
    return out


def _s15(J, ds):
    from fairtest.b68_analysis import VARIANTS, key
    d = J / f's15_{ds}'
    jobs = {j['job']: j for j in _rows(d / f'jobs_{ds}.jsonl')}
    res = {v: {r['job']: r for r in _rows(d / f'rejudge_{v}_{ds}.jsonl')} for v in VARIANTS}
    avoid = {v: {i: (None if 'error' in r else not r['contact']) for i, r in res[v].items()} for v in VARIANTS}
    by_key = {key(j): i for i, j in jobs.items() if j['attacker'] == 'v2s' and j['stage'] in ('policy', 'refine_g4s')}
    return jobs, avoid, by_key


def readings(J):
    from fairtest.b41_analysis import load
    out = {}
    for ds, run in (('highsim', 'b70'), ('exid', 'b71')):
        jobs, avoid, by_key = _s15(J, ds)
        evals, realistic, _, pool, _ = load(J / run)
        g4s = _g4s([J / run / 'refine.jsonl'])
        cases = sorted(c['case_id'] for c in pool['cases'])
        seeds = sorted(evals['final'])

        def det(s, need):
            per = []
            for c in cases:
                q = g4s.get((c, s), (None, None))[0]
                hits = []
                for x in seeds:
                    pol = bool(evals['final'][x].get((c, s), {}).get('robust') and realistic.get((x, c, s), False))
                    i = by_key.get(('policy', x, c, s))
                    j = by_key.get(('refine_g4s', None, c, s))
                    okp = pol and i is not None and all(avoid[v].get(i) is True for v in need)
                    # as b68_analysis: a policy failure that the other reading rejects is no success, so the stored refinement's
                    # first success counts if that reading also avoids it (later queries unknown: no success)
                    okr = q is not None and 1 + q <= K_AGENT and j is not None and all(avoid[v].get(j) is True for v in need)
                    hits.append(float(okp or okr))
                per.append(float(np.mean(hits)))
            return np.array(per)
        rep = {}
        for name, need in (('C', ('cc_r157',)), ('C and CARS CC', ('cc_r157', 'cc_jrc')), ('C and CARS FSM', ('cc_r157', 'fsm_cars'))):
            v = {s: det(s, need) for g in GROUPS.values() for s in g}
            row = dict(following_H0=round(float(v['following_H0'].mean()) * 100, 1), RSS=round(float(v['rss_sut_H0'].mean()) * 100, 1))
            for g, suts in GROUPS.items():
                m = np.mean([v[s] for s in suts], 0)
                row[g] = round(float(m.mean()) * 100, 1)
                if g == 'weak rule-based':
                    rng = np.random.default_rng(SEED)
                    b = m[rng.integers(0, len(m), (N_BOOT, len(m)))].mean(1)
                    row['weak_ci'] = [round(float(np.percentile(b, 2.5)) * 100, 1), round(float(np.percentile(b, 97.5)) * 100, 1)]
            from fairtest.b41_analysis import boot_diff  # the interval of the stored S15 and B70/B71 analyses
            p1, ci = boot_diff(dict(zip(cases, v['following_H0'])), dict(zip(cases, v['rss_sut_H0'])))
            row['P1'] = (round(p1 * 100, 1), [round(x * 100, 1) for x in ci])
            rep[name] = row
        reported = [i for i in by_key.values()]
        rep['also_avoided'] = {v: round(float(np.mean([avoid[v][i] for i in reported if avoid[v].get(i) is not None])) * 100, 1)
                               for v in ('cc_jrc', 'fsm_cars')}
        out[ds] = rep
    return out


def headway(J):
    out = {}
    for ds, run in (('highsim', 'b70'), ('exid', 'b71')):
        jobs, avoid, by_key = _s15(J, ds)
        pool = json.load(open(J / run / 'case_pool.json', encoding='utf-8'))
        head = {c['case_id']: c['event']['metrics']['p10']['headway'] for c in pool['cases']}
        rows = []
        for (stage, x, c, s), i in by_key.items():
            rows.append((head.get(c), avoid['cc_jrc'].get(i)))
        bins = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sel = [a for h, a in rows if h is not None and lo <= h < hi and a is not None]
            cases = sum(1 for c, h in head.items() if lo <= h < hi)
            bins.append(dict(bin=[lo, None if hi == float('inf') else hi], cases=cases, failures=len(sel),
                             cars_cc_also_avoids=round(float(np.mean(sel)) * 100, 1) if sel else None))
        out[ds] = bins
    return out


def headway_p1(J):
    """Addendum (the plan's headway item also named P1 by headway bin, which headway() left out): detection of following H0
    and longitudinal RSS within 37 queries and their difference, under C and under C and the CARS CC, per initial-headway bin,
    composed as readings()."""
    from fairtest.b41_analysis import load
    out = {}
    for ds, run in (('highsim', 'b70'), ('exid', 'b71')):
        jobs, avoid, by_key = _s15(J, ds)
        evals, realistic, _, pool, _ = load(J / run)
        g4s = _g4s([J / run / 'refine.jsonl'])
        head = {c['case_id']: c['event']['metrics']['p10']['headway'] for c in pool['cases']}
        cases = sorted(c['case_id'] for c in pool['cases'])
        seeds = sorted(evals['final'])

        def hit(c, s, need):
            q = g4s.get((c, s), (None, None))[0]
            hs = []
            for x in seeds:
                pol = bool(evals['final'][x].get((c, s), {}).get('robust') and realistic.get((x, c, s), False))
                i = by_key.get(('policy', x, c, s))
                j = by_key.get(('refine_g4s', None, c, s))
                okp = pol and i is not None and all(avoid[v].get(i) is True for v in need)
                okr = q is not None and 1 + q <= K_AGENT and j is not None and all(avoid[v].get(j) is True for v in need)
                hs.append(float(okp or okr))
            return float(np.mean(hs))
        bins = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sel = [c for c in cases if lo <= head[c] < hi]
            row = dict(bin=[lo, None if hi == float('inf') else hi], cases=len(sel))
            for name, need in (('C', ('cc_r157',)), ('C and CARS CC', ('cc_r157', 'cc_jrc'))):
                h0 = [hit(c, 'following_H0', need) for c in sel]
                rss = [hit(c, 'rss_sut_H0', need) for c in sel]
                row[name] = dict(following_H0_cases=round(float(np.sum(h0)), 2), RSS_cases=round(float(np.sum(rss)), 2),
                                 P1=round(100 * float(np.mean(h0) - np.mean(rss)), 1) if sel else None)
            row['P1_points_lost_overall'] = round(100 * (row['C']['following_H0_cases'] - row['C']['RSS_cases']
                                                         - row['C and CARS CC']['following_H0_cases']
                                                         + row['C and CARS CC']['RSS_cases']) / len(cases), 1)
            bins.append(row)
        out[ds] = bins
    return out


def cost(J):
    """Development, per test (case, SUT) over the 17 SUTs: SUT queries and CC replays used until the first success or the
    budget, for the agent (policy query with its shield replays, then the refinement only if the policy failed) and for
    the screened search alone; seed-averaged for the agent."""
    from fairtest.b56_analysis import policy
    robust, inside = policy(J, 'dev')
    ok = {k: bool(v and inside.get(k, False)) for k, v in robust.items()}
    g4s = _g4s([J / 'b56' / 'b56_dev.jsonl'])
    shield = {}
    for s in (1, 2, 3):
        for r in _rows(J / 'b26' / 'final' / f'final_s{s}_dev.jsonl'):
            if 'valid' in r:
                shield[(f's{s}', r['case_id'], r['sut'])] = (r.get('shield') or {}).get('cc_rollouts', 0)
    seeds = sorted({k[0] for k in ok})
    tests = sorted({(c, s) for (_, c, s) in ok})
    a_q, a_r, s_q, s_r, a_det, s_det = [], [], [], [], [], []
    for c, s in tests:
        q, rec = g4s.get((c, s), (None, {}))
        sq, sr = rec.get('sut_queries', 0), rec.get('cc_replays', 0)
        s_q.append(sq)
        s_r.append(sr)
        s_det.append(q is not None)
        aq, ar, ad = [], [], []
        for x in seeds:
            if ok.get((x, c, s), False):
                aq.append(1)
                ar.append(shield.get((x, c, s), 0))
                ad.append(True)
            else:
                aq.append(1 + sq)
                ar.append(shield.get((x, c, s), 0) + sr)
                ad.append(q is not None)
        a_q.append(np.mean(aq))
        a_r.append(np.mean(ar))
        a_det.append(np.mean(ad))
    A = dict(sut_queries=float(np.sum(a_q)), cc_replays=float(np.sum(a_r)), detections=float(np.sum(a_det)))
    S = dict(sut_queries=float(np.sum(s_q)), cc_replays=float(np.sum(s_r)), detections=float(np.sum(s_det)))
    rep = dict(tests=len(tests), agent=A, search_alone=S,
               per_detection=dict(agent=dict(sut_queries=round(A['sut_queries'] / A['detections'], 2), cc_replays=round(A['cc_replays'] / A['detections'], 1)),
                                  search_alone=dict(sut_queries=round(S['sut_queries'] / S['detections'], 2),
                                                    cc_replays=round(S['cc_replays'] / S['detections'], 1))))
    # break-even SUT-query cost: agent total = search total, with a CC replay at 0.4 s (Table S6: 15.8 replays ~ 6.3 s)
    # and training 3 seeds x 1.6 h amortised over the 1,054 development tests of one campaign
    c_cc, train_s = 6.3 / 15.8, 3 * 1.6 * 3600
    dq = S['sut_queries'] - A['sut_queries']
    dr = A['cc_replays'] - S['cc_replays']
    rep['break_even'] = dict(cc_replay_s=round(c_cc, 3), training_s=train_s,
                             sut_query_s_no_training=round(dr * c_cc / dq, 3) if dq > 0 else None,
                             sut_query_s_with_training=round((dr * c_cc + train_s) / dq, 3) if dq > 0 else None,
                             note='agent cheaper when an SUT query costs more than this; training amortised over one development campaign')
    return rep


def main(a):
    J = Path(a.jobs)
    if a.only == 'headway_p1':
        rep = dict(headway_p1=headway_p1(J))
        json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
        print(json.dumps(rep, indent=1))
        return
    rep = dict(budget=budget(J), readings=readings(J), headway=headway(J), cost=cost(J))
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for ds, e in rep['budget'].items():
        for g, v in e['groups'].items():
            print(ds, g, 'k1 diff', v['diff_k1'], 'k6', v['diff_k6'], 'full', v['diff_full'], 'match', v['queries_to_match'],
                  v.get('table_s5_check', {}).get('ok', ''))
    print(json.dumps(rep['readings'], indent=1))
    print(json.dumps(rep['headway'], indent=1))
    print(json.dumps(rep['cost'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--only', choices=['headway_p1'], help='addendum: P1 by headway bin only (separate output)')
    main(ap.parse_args())
