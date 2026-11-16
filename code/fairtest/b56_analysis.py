"""B56 analysis (B56 design, fixed before the run, available on request).

Composition as b46_analysis with the unified contract (--policy-g2): the v2j policy stage's robust fair failure that is
inside the joint support and passes G2, else the refinement. v2j = refinement bo_warm_g4j (B56 re-run, logged),
v2s = refinement bo_warm_g4s (screened). Development partition, seeds 1-3, k = 1 / 6 / 37 SUT queries.
  S1  every executed bo_warm_g4s query is admissible (implementation check); the g4j re-run's inadmissible executed
      queries and the SUT collisions among them are described
  S2  detectable SUTs (DETECTABLE): the mean over them of v2s k37 - v2j k37 >= -3 points and no SUT below -10 points
  S3  competent SUTs (COMPETENT): v2s k37 <= 3% each
  S4  cost (descriptive): CC replays per executed g4s query, stop reasons
  S5  the g4j re-run reproduces the stored B46 first successes (first_fair) exactly; the stored B46 files composed here
      give the B46 unified numbers

  python -m fairtest.b56_analysis OUT.json --jobs fairtest_jobs --stored B46_rule.jsonl B46_learned.jsonl --run B56.jsonl
  post hoc (reuse labelled): --dataset tgsim --stored b47/joint_tgsim.jsonl --run B56_tgsim.jsonl;
                            --dataset mitra --stored b41/refine.jsonl --run B56_mitra.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage
from fairtest.b45_joint_realism import Support

KS = (1, 6, 37)
DETECTABLE = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'bc_human_H0', 'ads@2.0_H0',
              'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')  # v2j k37 >= 10% in B46 (unified)
COMPETENT = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
             'drl_sut@3_H0', 'drl_sut@6_H0')  # fixed before the experiments (manuscript Section 5)
S2_MEAN, S2_EACH, S3_MAX = -.03, -.10, .03


def refine(paths, method):
    """{(case, sut): first_fair or None}, {(case, sut): record} of METHOD from b23_search outputs; errors counted."""
    first, recs, errors = {}, {}, 0
    for path in paths:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m != method:
                    continue
                if 'error' in rec:
                    errors += 1
                    continue
                first[(r['case_id'], s)] = rec.get('first_fair')
                recs[(r['case_id'], s)] = rec
    return first, recs, errors


def boot(vals, n, seed):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    m = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def compose(robust, inside, ref, seeds, cases, s, k):
    """Per-case success share over the seeds at SUT-query budget k (policy query 1, refinement queries 2..k)."""
    per_seed = [np.array([(robust.get((x, c, s), False) and inside.get((x, c, s), False))
                          or (k >= 2 and ref.get((c, s)) is not None and 1 + ref[(c, s)] <= k) for c in cases], float)
                for x in seeds]
    return np.mean(per_seed, axis=0)


def policy(jobs, dataset):
    """(robust, inside) of the v2j policy stage, keyed (seed label, case, sut), unified contract (as b46_analysis
    --policy-g2): dev = b26 final + b45 replay; tgsim = b28 final + margin (b46_analysis.policy_tgsim) + b47 replay;
    mitra = b41 final + margin + policy replay (b41_analysis.load, B41's frozen definitions)."""
    sup = Support()
    if dataset == 'mitra':
        from fairtest.b41_analysis import load
        evals, realistic, _, _, _ = load(jobs / 'b41')
        robust = {(seed, c, s): m['robust'] for seed, e in evals['final'].items() for (c, s), m in e.items()}
        return robust, dict(realistic)
    if dataset == 'tgsim':
        from fairtest.b46_analysis import policy_tgsim
        robust, replay = policy_tgsim(jobs), jobs / 'b47' / 'policy_replay_tgsim.jsonl'
    else:
        robust, replay = policy_stage(jobs / 'b26', 'dev')[0], jobs / 'b45' / 'replay_dev.jsonl'
    inside = {}
    for line in open(replay, encoding='utf-8'):
        r = json.loads(line)
        if r['source'] == 'v2_policy':
            sc = sup.score(r['features']) if r['cut_in'] else None
            ok = sc is not None and sc <= sup.tau and r.get('exceedance') is not None and r['exceedance'] <= 1.
            inside[(f"s{r['seed']}", r['case_id'], r['sut'])] = ok
    return robust, inside


def main(a):
    jobs = Path(a.jobs)
    robust, inside = policy(jobs, a.dataset)
    stored, _, err_stored = refine(a.stored, 'bo_warm_g4j')
    g4j, rec_j, err_j = refine(a.run, 'bo_warm_g4j')
    g4s, rec_s, err_s = refine(a.run, 'bo_warm_g4s')
    seeds = sorted({k[0] for k in robust})
    suts = sorted({s for (_, s) in g4s})
    rep = dict(errors=dict(stored=err_stored, g4j=err_j, g4s=err_s), suts={})
    # S5: reproduction of the stored B46 refinement
    common = sorted(set(stored) & set(g4j))
    same = sum(stored[k] == g4j[k] for k in common)
    rep['S5'] = dict(pairs=len(common), identical_first_fair=same, differing=[list(k) for k in common if stored[k] != g4j[k]][:20],
                     supported=len(common) == len(stored) and same == len(common))
    if a.b46_unified:
        u = json.load(open(a.b46_unified, encoding='utf-8'))['suts']
        mism = []
        for s in suts:
            cases = sorted({k[1] for k in robust if k[2] == s})
            for k in KS:
                v = round(float(compose(robust, inside, stored, seeds, cases, s, k).mean()), 4)
                if s in u and abs(v - u[s][f'v2j_k{k}']['mean']) > 1e-4:
                    mism.append([s, k, v, u[s][f'v2j_k{k}']['mean']])
        rep['S5']['stored_composition_matches_B46_unified'] = not mism
        rep['S5']['composition_mismatches'] = mism
    # per SUT
    for s in suts:
        cases = sorted({k[1] for k in robust if k[2] == s})
        if not all((c, s) in g4s and (c, s) in g4j for c in cases):
            rep['suts'][s] = dict(incomplete=True)
            continue
        e = dict(n_cases=len(cases))
        for label, ref in (('v2j', g4j), ('v2s', g4s)):
            for k in KS:
                vals = compose(robust, inside, ref, seeds, cases, s, k)
                e[f'{label}_k{k}'] = dict(mean=round(float(vals.mean()), 4), ci=boot(vals, 4000, 20260928))
            e[f'{label}_refine_only'] = round(float(np.mean([ref.get((c, s)) is not None for c in cases])), 4)
        for k in (6, 37):
            d = compose(robust, inside, g4s, seeds, cases, s, k) - compose(robust, inside, g4j, seeds, cases, s, k)
            e[f'diff_k{k}'] = dict(mean=round(float(d.mean()), 4), ci=boot(d, 10000, 0))
        dr = np.array([float(g4s.get((c, s)) is not None) - float(g4j.get((c, s)) is not None) for c in cases])
        e['diff_refine_only'] = dict(mean=round(float(dr.mean()), 4), ci=boot(dr, 10000, 0))
        # S1: executed queries
        qs = [q for c in cases for q in rec_s[(c, s)].get('queries', [])]
        qj = [q for c in cases for q in rec_j[(c, s)].get('queries', [])]
        bad_j = [q for q in qj if not q['admissible']]
        e['g4s_executed'] = len(qs)
        e['g4s_inadmissible_executed'] = sum(not q['admissible'] for q in qs)
        e['g4j_executed'] = len(qj)
        e['g4j_inadmissible_executed'] = len(bad_j)
        e['g4j_inadmissible_reason'] = dict(
            cc_contact=sum(q['cc_contact'] for q in bad_j),
            clearance=sum(not q['cc_contact'] and (q['cc_clearance'] or 0.) < .5 for q in bad_j),
            guard=sum(not q['cc_contact'] and (q['cc_clearance'] or 0.) >= .5 for q in bad_j))
        e['g4j_sut_contacts_inadmissible'] = sum(q['sut_contact'] for q in bad_j)
        e['g4j_sut_contacts_cc_contact'] = sum(q['sut_contact'] and q['cc_contact'] for q in bad_j)
        e['g4j_sut_contacts_admissible'] = sum(q['sut_contact'] for q in qj if q['admissible'])
        # S4: cost
        per = [rec_s[(c, s)]['cc_replays'] / max(1, rec_s[(c, s)]['sut_queries']) for c in cases]
        e['g4s_cc_replays_per_query'] = dict(mean=round(float(np.mean(per)), 2), median=round(float(np.median(per)), 2),
                                             p95=round(float(np.quantile(per, .95)), 2), max=round(float(np.max(per)), 2))
        e['g4s_cc_replays_total'] = int(sum(rec_s[(c, s)]['cc_replays'] for c in cases))
        e['g4s_sut_queries_total'] = int(sum(rec_s[(c, s)]['sut_queries'] for c in cases))
        e['g4j_sut_queries_total'] = int(sum(rec_j[(c, s)]['sut_queries'] for c in cases))
        stops = {}
        for c in cases:
            stops[rec_s[(c, s)]['stop']] = stops.get(rec_s[(c, s)]['stop'], 0) + 1
        e['g4s_stop'] = stops
        rep['suts'][s] = e
    ok = {s: e for s, e in rep['suts'].items() if not e.get('incomplete')}
    rep['S1'] = dict(g4s_inadmissible_executed=sum(e['g4s_inadmissible_executed'] for e in ok.values()),
                     g4s_executed=sum(e['g4s_executed'] for e in ok.values()),
                     g4j_inadmissible_executed=sum(e['g4j_inadmissible_executed'] for e in ok.values()),
                     g4j_executed=sum(e['g4j_executed'] for e in ok.values()),
                     g4j_sut_contacts_inadmissible=sum(e['g4j_sut_contacts_inadmissible'] for e in ok.values()),
                     g4j_sut_contacts_cc_contact=sum(e['g4j_sut_contacts_cc_contact'] for e in ok.values()),
                     g4j_sut_contacts_admissible=sum(e['g4j_sut_contacts_admissible'] for e in ok.values()))
    rep['S1']['supported'] = rep['S1']['g4s_inadmissible_executed'] == 0 and rep['S1']['g4s_executed'] > 0
    d = {s: ok[s]['diff_k37']['mean'] for s in DETECTABLE if s in ok}
    rep['S2'] = dict(diff_k37=d, mean=round(float(np.mean(list(d.values()))), 4) if d else None,
                     supported=len(d) == len(DETECTABLE) and np.mean(list(d.values())) >= S2_MEAN and min(d.values()) >= S2_EACH)
    c3 = {s: ok[s]['v2s_k37']['mean'] for s in COMPETENT if s in ok}
    rep['S3'] = dict(v2s_k37=c3, supported=len(c3) == len(COMPETENT) and all(v <= S3_MAX for v in c3.values()))
    rep['S3']['supported'] = bool(rep['S3']['supported'])
    rep['S2']['supported'] = bool(rep['S2']['supported'])
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('errors', 'S1', 'S2', 'S3', 'S5')}, indent=1))
    for s, e in ok.items():
        print(f"{s:20s} v2j {100*e['v2j_k1']['mean']:5.1f} {100*e['v2j_k6']['mean']:5.1f} {100*e['v2j_k37']['mean']:5.1f} | "
              f"v2s {100*e['v2s_k1']['mean']:5.1f} {100*e['v2s_k6']['mean']:5.1f} {100*e['v2s_k37']['mean']:5.1f} | "
              f"refine j {100*e['v2j_refine_only']:5.1f} s {100*e['v2s_refine_only']:5.1f} | g4j bad {e['g4j_inadmissible_executed']}/"
              f"{e['g4j_executed']} | cc/q {e['g4s_cc_replays_per_query']['median']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--stored', nargs='+', required=True, help='stored B46 bo_warm_g4j outputs')
    ap.add_argument('--run', nargs='+', required=True, help='B56 outputs (bo_warm_g4j re-run and bo_warm_g4s, logged)')
    ap.add_argument('--b46-unified', help='B46_ANALYSIS_unified.json: check that the stored files compose to it')
    ap.add_argument('--dataset', default='dev', choices=('dev', 'tgsim', 'mitra'), help='B56 post hoc: the external datasets')
    main(ap.parse_args())
