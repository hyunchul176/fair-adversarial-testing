"""B79 analysis (B79 design, written before the results, available on request).

  python -m fairtest.b79_analysis precheck NEW.jsonl STORED.jsonl          -> identical first success, queries and best plan?
  python -m fairtest.b79_analysis b79 OUT.json --run fb_bo=FILE --run fb_rs=FILE --run fb_om=FILE --run or_om=FILE
         [--ref b56=b56_dev.jsonl] [--ref b61=random_g4s.jsonl]

A fair failure of a (case, SUT) is a first success of the contract within 36 SUT queries (first_fair not None). Its attack
style is read from the stored best plan's CC replay (cc_features, as B78_STYLE): closing_p50 = follower minus cut-in vehicle
speed at the 50% lateral crossing (> 0: the cut-in vehicle is slower), acc_pre / acc_post = its mean acceleration in the 3 s
before / 2 s after the crossing; and from the plan theta = (entry_s, pre, post, post2, duration): pre is the acceleration
before the lane change, post2 the acceleration from 2 s after its start. Per SUT: the case rate with a case-bootstrap 95%
interval (2,000 resamples, seed 79), and the styles of the failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUTS = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0', 'bc_human_H0', 'rss_sut_H0')
RULE = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')


def rows(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def runs(path):
    """{(case, SUT): run} for the single method of a file (bo_warm_g4v / bo_warm_g4s / random_g4s)."""
    out = {}
    for r in rows(path):
        for k, v in r['runs'].items():
            out[(r['case_id'], k.split('|', 1)[1])] = v
    return out


def precheck(a):
    new, old = runs(a.new), runs(a.stored)
    diff = []
    for key, v in new.items():
        o = old.get(key)
        if o is None:
            diff.append((key, 'missing in stored'))
            continue
        same = (v.get('first_fair') == o.get('first_fair') and v.get('sut_queries') == o.get('sut_queries')
                and np.allclose((v.get('best') or {}).get('theta', [0] * 5), (o.get('best') or {}).get('theta', [0] * 5), atol=1e-9))
        if not same:
            diff.append((key, dict(new=[v.get('first_fair'), v.get('sut_queries')], stored=[o.get('first_fair'), o.get('sut_queries')])))
    res = dict(pairs=len(new), identical=len(new) - len(diff), differences=diff[:20])
    print(json.dumps(res, default=str))
    if diff:
        raise SystemExit(1)


def boot(x, n=2000, seed=79):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return [round(float(np.percentile(m, 2.5)) * 100, 1), round(float(np.percentile(m, 97.5)) * 100, 1)]


def style(fails):
    if not fails:
        return dict(n=0)
    f = [v['best'].get('cc_features') or {} for v in fails]
    th = np.array([v['best']['theta'] for v in fails], float)
    c = np.array([x['closing_p50'] for x in f if x.get('closing_p50') is not None], float)
    pre = np.array([x['acc_pre'] for x in f if x.get('acc_pre') is not None], float)
    post = np.array([x['acc_post'] for x in f if x.get('acc_post') is not None], float)
    q = lambda v: np.percentile(v, [10, 50, 90]).round(2).tolist() if len(v) else None
    return dict(n=len(fails), faster_at_crossing=round(float(np.mean(c < 0)), 3) if len(c) else None,
                closing_p10_50_90=q(c), acc_pre_p10_50_90=q(pre), acc_post_p10_50_90=q(post),
                brakes_after_crossing=round(float(np.mean(post < -.5)), 3) if len(post) else None,
                plan_entry_p10_50_90=q(th[:, 0]), plan_pre_p10_50_90=q(th[:, 1]), plan_post_p10_50_90=q(th[:, 2]),
                plan_post2_p10_50_90=q(th[:, 3]), plan_pre_above_1=round(float(np.mean(th[:, 1] > 1.)), 3))


def summary(rs):
    out = {}
    cases = sorted({c for c, _ in rs})
    for s in SUTS:
        vs = [rs[(c, s)] for c in cases if (c, s) in rs]
        if not vs:
            continue
        hit = [1. if (v.get('first_fair') is not None and 'error' not in v) else 0. for v in vs]
        fails = [v for v, h in zip(vs, hit) if h]
        fast = [1. if h and ((v['best'].get('cc_features') or {}).get('closing_p50') is not None
                             and v['best']['cc_features']['closing_p50'] < 0) else 0. for v, h in zip(vs, hit)]
        out[s] = dict(cases=len(vs), rate=round(float(np.mean(hit)) * 100, 1), ci=boot(hit),
                      overtaking_rate=round(float(np.mean(fast)) * 100, 1), overtaking_ci=boot(fast),
                      errors=sum('error' in v for v in vs), style=style(fails))
    return out


def b79(a):
    rep = dict(design='B79 design (available on request)', runs={}, refs={})
    for spec in a.run or []:
        name, path = spec.split('=', 1)
        rep['runs'][name] = summary(runs(path))
    for spec in a.ref or []:
        name, path = spec.split('=', 1)
        rep['refs'][name] = summary(runs(path))
    # interpretation rule fixed in DESIGN section 4: overtaking fair failures of the screened GP-BO on the rule-based SUTs
    fb = rep['runs'].get('fb_bo', {})
    rule = [s for s in RULE if s in fb]
    hi = [s for s in rule if fb[s]['overtaking_rate'] >= 25.]
    allc = [rep['runs'][k][s]['overtaking_rate'] for k in ('fb_bo', 'fb_rs', 'fb_om', 'or_om') if k in rep['runs']
            for s in RULE if s in rep['runs'][k]]
    if len(hi) >= 2:
        verdict = 'a: overtaking attacks are feasible under the contract and the screened search finds them'
    elif allc and max(allc) < 10.:
        verdict = 'b: the contract leaves little room for overtaking attacks on the rule-based SUTs'
    else:
        verdict = 'c: in between, reported as is'
    rep['interpretation'] = dict(rule_based_suts_with_overtaking_rate_ge_25=hi, max_overtaking_rate_rule_based=max(allc) if allc else None,
                                 verdict=verdict)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep['interpretation']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('precheck')
    p.add_argument('new')
    p.add_argument('stored')
    p.set_defaults(fn=precheck)
    p = sub.add_parser('b79')
    p.add_argument('out')
    p.add_argument('--run', action='append')
    p.add_argument('--ref', action='append')
    p.set_defaults(fn=b79)
    a = ap.parse_args()
    a.fn(a)
