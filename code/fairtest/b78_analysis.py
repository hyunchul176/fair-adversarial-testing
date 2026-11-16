"""B78 (B78 design, fixed before the run, available on request): analyses.

  style    post hoc, before the run (the design records its numbers): the cut-in style of natural highD cut-ins and of the
           agent's stored reported failures -- speed relative to the follower at the 50% lateral crossing (closing_p50 =
           follower minus cut-in vehicle; > 0 slower cut-in) and the cut-in vehicle's mean acceleration 3 s before and 2 s after
           it -- and the stored failures that would pass the braking bound (a filter of the stored searches, not a re-search)
  b78      the conditions a1 and a5 (b78_variants --acc-pct 1 / 5): the agent composed as in B75 (a policy-stage failure counts
           only if its stored CC replay also passes the braking bound; otherwise the re-run refinement's first success within
           36 queries), P1 (following H0 - longitudinal RSS, 37 queries) with a case-bootstrap interval, the weak ladder, the
           refinement alone, and the style of the new successes. NB1: a1 P1 >= 0.5; NB2: a5 P1 >= 0.5 (judged).

  python -m fairtest.b78_analysis style OUT.json --jobs fairtest_jobs
  python -m fairtest.b78_analysis b78 OUT.json --jobs fairtest_jobs --run a1=FILE --run a5=FILE
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import b75_analysis as A

WEAK = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
COMP = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0', 'drl_sut@3_H0',
        'drl_sut@6_H0')


def natural():
    from fairtest.features import natural_reference
    return [r for r in natural_reference() if r.get('closing_p50') is not None]


def bounds(pct):
    nat = natural()
    pre = np.array([r['acc_pre'] for r in nat if r.get('acc_pre') is not None], float)
    post = np.array([r['acc_post'] for r in nat if r.get('acc_post') is not None], float)
    return float(np.percentile(pre, pct)), float(np.percentile(post, pct))


def passes(f, b):
    return (f is not None and f.get('acc_pre') is not None and f.get('acc_post') is not None
            and f['acc_pre'] >= b[0] and f['acc_post'] >= b[1])


def style(feats):
    c = np.array([f['closing_p50'] for f in feats if f.get('closing_p50') is not None], float)
    pre = np.array([f['acc_pre'] for f in feats if f.get('acc_pre') is not None], float)
    post = np.array([f['acc_post'] for f in feats if f.get('acc_post') is not None], float)
    both = [f for f in feats if f.get('closing_p50') is not None and f.get('acc_post') is not None]
    q = lambda x: [round(float(v), 2) for v in np.percentile(x, (10, 50, 90))] if len(x) else None  # noqa: E731
    return dict(n=len(feats), slower_share=round(float(np.mean(c > 0)), 4) if len(c) else None,
                faster_share=round(float(np.mean(c < 0)), 4) if len(c) else None, closing_p10_50_90=q(c),
                acc_pre_p10_50_90=q(pre), acc_post_p10_50_90=q(post),
                braking_post_below_05=round(float(np.mean(post < -.5)), 4) if len(post) else None,
                braking_post_below_10=round(float(np.mean(post < -1.)), 4) if len(post) else None,
                slower_and_braking=round(float(np.mean([f['closing_p50'] > 0 and f['acc_post'] < -.5 for f in both])), 4) if both else None)


class BrakingPolicy(A.Policy):
    """The B75 policy stage plus the braking bound on each stored policy failure's CC replay."""

    def __init__(self, jobs, bound):
        super().__init__(jobs)
        self.bound = bound
        self.features = {}
        for r in A._rows(Path(jobs) / 'b45' / 'replay_dev.jsonl'):
            if r['source'] == 'v2_policy':
                self.features[(str(r['seed']), r['case_id'], r['sut'])] = r['features'] if r['cut_in'] else None

    def ok(self, k, cfg):
        return super().ok(k, cfg) and (self.bound is None or passes(self.features.get(k), self.bound))


def _stored_failures(J):
    """CC-replay features of the stored reported failures on the development cases: policy stage by (seed, case, SUT) under
    the success contract (B75 Policy.ok with the contract values), and the B56 refinement's first successes by (case, SUT)."""
    pol = BrakingPolicy(J, None)
    cfg = dict(g2=1., q=99., delta=.5, v_min=-1000., tau=A._tau99() if hasattr(A, '_tau99') else None)
    if cfg['tau'] is None:
        from fairtest.b63_realism_sensitivity import taus
        cfg['tau'] = taus()[1][99.0]
    policy = {k: pol.features[k] for k in pol.fair if pol.ok(k, cfg)}
    rec, _ = A.runs(Path(J) / 'b56' / 'b56_dev.jsonl', 'bo_warm_g4s')
    refine = {k: r['best']['cc_features'] for k, r in rec.items() if 'error' not in r and r.get('first_fair') is not None}
    return pol, cfg, policy, refine


def style_main(a):
    J = Path(a.jobs)
    nat = natural()
    pol, cfg, policy, refine = _stored_failures(J)
    rep = dict(natural=style(nat), bounds={p: bounds(p) for p in (1, 5)})
    rep['policy_weak4'] = style([f for (x, c, s), f in policy.items() if s in WEAK])
    rep['policy_following_H0'] = style([f for (x, c, s), f in policy.items() if s == 'following_H0'])
    rep['refine_all'] = style(list(refine.values()))
    rep['refine_weak4'] = style([f for (c, s), f in refine.items() if s in WEAK])
    # natural cut-ins of the 62 development cases themselves (recorded state at the 10% crossing)
    from fairtest import runner
    runner._init()
    rel = [v['event']['metrics']['p10']['generator_speed'] - v['event']['metrics']['p10']['target_speed']
           for v in runner._STATE['cases'].values() if v['event'].get('partition') == 'development']
    rep['dev_cases_recorded'] = dict(n=len(rel), cut_in_vehicle_slower_at_p10=round(float(np.mean(np.array(rel) < 0)), 4),
                                     median_generator_minus_follower=round(float(np.median(rel)), 2))
    # the stored searches filtered by the bound (not a re-search): seed-averaged composition as in B75
    filt = {}
    for p in (None, 1, 5):
        b = None if p is None else bounds(p)
        bp = BrakingPolicy(J, b)
        rec, _ = A.runs(J / 'b56' / 'b56_dev.jsonl', 'bo_warm_g4s')
        if b is not None:
            rec = {k: (r if ('error' in r or r.get('first_fair') is None or passes(r['best']['cc_features'], b))
                       else dict(r, first_fair=None)) for k, r in rec.items()}
        s = A.summary(bp, rec, cfg)
        filt['contract' if p is None else f'p{p}'] = dict(P1_k37=s.get('P1_k37'), groups=s['groups'],
                                                          following_H0=s['suts']['following_H0'], rss=s['suts']['rss_sut_H0'])
    rep['stored_filtered'] = filt
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep, indent=1))


def b78_main(a):
    J = Path(a.jobs)
    from fairtest.b63_realism_sensitivity import taus
    rep = dict(conditions={})
    for spec in a.run:
        name, path = spec.split('=', 1)
        rec, cfg = A.runs(path, 'bo_warm_g4v')
        pct = cfg.get('acc_pct')
        b = (cfg['acc_min_pre'], cfg['acc_min_post']) if pct is not None else None
        cfg = dict(g2=cfg['g2'], q=cfg['q'], delta=cfg['delta'], v_min=cfg['v_min'], tau=cfg['tau'] or taus()[1][99.0])
        pol = BrakingPolicy(J, b)
        s = A.summary(pol, rec, cfg)
        s['bound'] = dict(pct=pct, acc_min_pre=b[0] if b else None, acc_min_post=b[1] if b else None)
        s['search_only'] = {k: round(float(np.mean(A.search_only(rec, k, pol.cases))), 4) for k in sorted({k for (_, k) in rec})}
        succ = [r['best']['cc_features'] for r in rec.values() if 'error' not in r and r.get('first_fair') is not None]
        s['style_refine_successes'] = style(succ)
        s['policy_failures_passing'] = sum(1 for k in pol.fair if pol.ok(k, cfg) and k[2] in WEAK)
        rep['conditions'][name] = s
    for h, name in (('NB1', 'a1'), ('NB2', 'a5')):
        c = rep['conditions'].get(name)
        if c and 'P1_k37' in c:
            rep[h] = dict(P1=c['P1_k37']['value'], ci=c['P1_k37']['ci'], threshold=.5, supported=c['P1_k37']['value'] >= .5)
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'conditions'}))
    for n, c in rep['conditions'].items():
        print(n, c['bound'], c.get('P1_k37'), c['groups'], c['search_only'], c['search'], c['style_refine_successes'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('style')
    p.add_argument('out')
    p.add_argument('--jobs', required=True)
    q = sub.add_parser('b78')
    q.add_argument('out')
    q.add_argument('--jobs', required=True)
    q.add_argument('--run', action='append', required=True, help='name=FILE (b78_variants output)')
    a = ap.parse_args()
    (style_main if a.cmd == 'style' else b78_main)(a)
