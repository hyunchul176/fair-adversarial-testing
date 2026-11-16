"""B75 and B76 analyses (B75 and B76 designs, fixed before the runs, available on request). Development cases, local
stored rows for the policy stage (b26/final policy rows and margins, b45 replay, b51 policy closing speeds).

Composition at SUT-query budget k (policy query 1, refinement queries 2..k), per seed and case:
  policy success  fair, front contact (dx >= 3.5 m), CC clearance >= delta, G2 exceedance <= g2, joint score <= tau(q), and
                  closing >= v_min when a floor is set (a lighter policy failure is no success, so the refinement runs)
  else            the run's bo_warm_g4v first success q with 1 + q <= k
The policy stage is exact for the realism and floor variants (its episodes do not depend on them). For another CC margin
it is exact only for delta >= 0.5 (the shield used 0.5 m), so B76 judges the refinement alone (36 queries).

  precheck  the contract-parameter run reproduces the stored bo_warm_g4s (first success, queries, best plan)
  b75       RS1 (q 95) and RS2 (g2 0.5): P1 at 37 queries >= 50 points (judged); groups, post-filter comparison, screening cost
  b76       FS1: median closing of the refinement's reported failures, pooled over following H0.5, H1, BC human and ADS 2.0 m,
            lower at delta 1.0 than at delta 0.25 (judged); FS2: severe (>= 3 m/s) search-only detection, same SUTs, lower at
            delta 1.0 than at 0.25 (judged); FS3 floor ladder 3 (stored B64) / 5 / 7 m/s; FS4 AEB SUTs (descriptive)

  python -m fairtest.b75_analysis precheck NEW.jsonl STORED_B56.jsonl
  python -m fairtest.b75_analysis b75 OUT.json --jobs fairtest_jobs --run q95=FILE --run g05=FILE
  python -m fairtest.b75_analysis b76 OUT.json --jobs fairtest_jobs --run d025=FILE --run d10=FILE --run v5=FILE --run v7=FILE \n         [--ref d05=B56.jsonl@bo_warm_g4s@CLOSING_G4S.jsonl --ref v3=B64.jsonl@bo_warm_g4ss]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

WEAK4 = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
LEARNED3 = ('drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')
COMPETENT8 = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
              'drl_sut@3_H0', 'drl_sut@6_H0')
GROUPS = {'weak rule-based': WEAK4, 'BC human': ('bc_human_H0',), 'ADS 2.0 m': ('ads@2.0_H0',), 'learned lane assignment': LEARNED3,
          'competent (8)': COMPETENT8}
LATE_HARD = ('following_H0.5', 'following_H1', 'bc_human_H0', 'ads@2.0_H0')
AEB = ('following_aeb_H0', 'following_aeb_H0.5')
FRONT, K_FULL, N_BOOT, SEED = 3.5, 37, 10_000, 75
POST_FILTER = {'q95': .113, 'g05': .742}  # Table S7 (B63): the stored searches filtered at the stricter threshold


def _rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def runs(path, method):
    out, cfg = {}, None
    for r in _rows(path):
        cfg = r.get('cfg', cfg)
        for key, rec in r['runs'].items():
            m, s = key.split('|')
            if m == method:
                out[(r['case_id'], s)] = rec
    return out, cfg


def precheck(a):
    new, _ = runs(a.new, 'bo_warm_g4v')
    old, _ = runs(a.stored, 'bo_warm_g4s')
    diff = []
    for k, x in new.items():
        y = old.get(k)
        if y is None or 'error' in x or 'error' in y:
            diff.append([*k, 'missing or error'])
            continue
        same = (x.get('first_fair') == y.get('first_fair') and x.get('sut_queries') == y.get('sut_queries')
                and all(abs(u - v) < 1e-9 for u, v in zip(x['best']['theta'], y['best']['theta'])))
        if not same:
            diff.append([*k, x.get('first_fair'), y.get('first_fair')])
    rep = dict(pairs=len(new), differing=len(diff), rows=diff[:20])
    print(json.dumps(rep))
    return 0 if new and not diff else 1


class Policy:
    """The policy stage on the development cases from the stored rows."""

    def __init__(self, jobs):
        from fairtest.b45_joint_realism import Support
        sup = Support()
        self.sup = sup
        self.fair, self.margin, self.score, self.exc, self.closing = {}, {}, {}, {}, {}
        for s in (1, 2, 3):
            for r in _rows(jobs / 'b26' / 'final' / f'final_s{s}_dev.jsonl'):
                if 'valid' not in r:
                    continue
                col = bool(r['valid'] and r['contact'])
                self.fair[(str(s), r['case_id'], r['sut'])] = bool(col and not r['refs']['cc_r157']['contact'])
        for r in _rows(jobs / 'b26' / 'final' / 'margin_dev.jsonl'):
            x = r['source'].split('_')[1][1:]
            self.margin[(x, r['case_id'], r['sut'])] = (r['contact_dx'], r['cc_clearance'])
        for r in _rows(jobs / 'b45' / 'replay_dev.jsonl'):
            if r['source'] == 'v2_policy':
                k = (str(r['seed']), r['case_id'], r['sut'])
                self.score[k] = sup.score(r['features']) if r['cut_in'] else None
                self.exc[k] = r.get('exceedance')
        for r in _rows(jobs / 'b51' / 'closing_dev.jsonl'):
            if r.get('kind') == 'policy' and 'error' not in r:
                self.closing[(str(r['seed']), r['case_id'], r['sut'])] = r['closing']
        self.seeds = ('1', '2', '3')
        self.cases = sorted({k[1] for k in self.fair})

    def ok(self, k, cfg):
        if not self.fair.get(k, False):
            return False
        dx, clear = self.margin.get(k, (0., 0.))
        sc, exc = self.score.get(k), self.exc.get(k)
        if not (dx >= FRONT and clear >= cfg['delta'] and exc is not None and exc <= cfg['g2'] and sc is not None and sc <= cfg['tau']):
            return False
        if cfg['v_min'] > -1000.:
            v = self.closing.get(k)
            return v is not None and v >= cfg['v_min']
        return True


def compose(pol, rec, cfg, sut, k):
    """{case: seed-averaged success} at budget k."""
    out = {}
    for c in pol.cases:
        r = rec.get((c, sut))
        q = None if (r is None or 'error' in r) else r.get('first_fair')
        out[c] = float(np.mean([pol.ok((x, c, sut), cfg) or (k >= 2 and q is not None and 1 + q <= k) for x in pol.seeds]))
    return out


def boot(vals):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(SEED)
    m = vals[rng.integers(0, len(vals), (N_BOOT, len(vals)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def summary(pol, rec, cfg):
    suts = sorted({s for (_, s) in rec})
    det = {k: {s: compose(pol, rec, cfg, s, k) for s in suts} for k in (1, K_FULL)}
    out = dict(cfg=cfg, suts={s: {f'k{k}': round(float(np.mean(list(det[k][s].values()))), 4) for k in det} for s in suts})
    out['groups'] = {g: {f'k{k}': round(float(np.mean([np.mean(list(det[k][s].values())) for s in m if s in det[k]])), 4)
                         for k in det} for g, m in GROUPS.items() if all(s in det[K_FULL] for s in m)}
    if 'following_H0' in det[K_FULL] and 'rss_sut_H0' in det[K_FULL]:
        d = [det[K_FULL]['following_H0'][c] - det[K_FULL]['rss_sut_H0'][c] for c in pol.cases]
        out['P1_k37'] = dict(value=round(float(np.mean(d)), 4), ci=boot(d))
    succ = [r for r in rec.values() if 'error' not in r and r.get('first_fair') is not None]
    allr = [r for r in rec.values() if 'error' not in r]
    out['search'] = dict(pairs=len(rec), errors=sum('error' in r for r in rec.values()), successes=len(succ),
                         stops=dict(collections.Counter(r.get('stop') for r in allr)),
                         cc_replays_per_query=round(sum(r.get('cc_replays', 0) for r in allr) / max(1, sum(r.get('sut_queries', 0) for r in allr)), 1))
    return out


def search_only(rec, sut, cases, v_min=None):
    """Refinement alone within 36 queries: per-case success (optionally only failures closing >= v_min)."""
    out = []
    for c in cases:
        r = rec.get((c, sut))
        ok = r is not None and 'error' not in r and r.get('first_fair') is not None
        if ok and v_min is not None:
            ok = (r['best'].get('closing') or -1e9) >= v_min
        out.append(float(ok))
    return out


def closings(rec, suts):
    return [r['best']['closing'] for (c, s), r in rec.items() if s in suts and 'error' not in r and r.get('first_fair') is not None
            and r['best'].get('closing') is not None]


def _load(spec):
    name, path = spec.split('=', 1)
    rec, cfg = runs(path, 'bo_warm_g4v')
    return name, rec, cfg


def _load_ref(spec, tau99):
    """name=FILE@METHOD[@CLOSING] -- a stored run (B56 bo_warm_g4s: CC margin 0.5, no floor; B64 bo_warm_g4ss: floor 3 m/s),
    with the closing speeds of its successes from a b51 closing file when the records lack them."""
    name, rest = spec.split('=', 1)
    parts = rest.split('@')
    path, method = parts[0], parts[1]
    rec, _ = runs(path, method)
    if len(parts) > 2:
        cl = {(r['case_id'], r['sut']): r['closing'] for r in _rows(parts[2]) if r.get('kind') == method and 'error' not in r}
        for k, r in rec.items():
            if 'error' not in r and r.get('first_fair') is not None and r['best'].get('closing') is None and k in cl:
                r['best']['closing'] = cl[k]
    v_min = 3. if method == 'bo_warm_g4ss' else -1000.
    return name, rec, dict(g2=1., q=99., delta=.5, v_min=v_min, tau=tau99, stored=method)


def b75(a):
    pol = Policy(Path(a.jobs))
    rep = dict(conditions={})
    for spec in a.run:
        name, rec, cfg = _load(spec)
        rep['conditions'][name] = summary(pol, rec, cfg)
        rep['conditions'][name]['post_filter_P1'] = POST_FILTER.get(name)
    for h, name in (('RS1', 'q95'), ('RS2', 'g05')):
        c = rep['conditions'].get(name)
        if c and 'P1_k37' in c:
            rep[h] = dict(P1=c['P1_k37']['value'], ci=c['P1_k37']['ci'], threshold=.5, supported=c['P1_k37']['value'] >= .5)
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'conditions'}))
    for n, c in rep['conditions'].items():
        print(n, c.get('P1_k37'), c['groups'], c['search'])


def b76(a):
    pol = Policy(Path(a.jobs))
    rep = dict(conditions={})
    recs = {}
    from fairtest.b63_realism_sensitivity import taus
    tau99 = taus()[1][99.0]
    for spec in list(a.run) + [('ref', x) for x in (a.ref or [])]:
        name, rec, cfg = _load_ref(spec[1], tau99) if isinstance(spec, tuple) else _load(spec)
        recs[name] = rec
        cl = closings(rec, LATE_HARD)
        rep['conditions'][name] = dict(cfg=cfg, search_only={s: round(float(np.mean(search_only(rec, s, pol.cases))), 4)
                                                             for s in sorted({s for (_, s) in rec})},
                                       severe_search_only={s: round(float(np.mean(search_only(rec, s, pol.cases, 3.))), 4)
                                                           for s in sorted({s for (_, s) in rec})},
                                       late_hard_closing=dict(n=len(cl), median=round(float(np.median(cl)), 3) if cl else None),
                                       search=summary(pol, rec, cfg)['search'])
        if cfg['v_min'] > -1000. and cfg['delta'] >= .5:  # the composed agent is exact here (B64 v2s-S semantics)
            rep['conditions'][name]['composed'] = summary(pol, rec, cfg)
    lo, hi = rep['conditions'].get('d025'), rep['conditions'].get('d10')
    if lo and hi:
        rep['FS1'] = dict(median_d025=lo['late_hard_closing']['median'], median_d10=hi['late_hard_closing']['median'],
                          supported=bool(hi['late_hard_closing']['median'] is not None and lo['late_hard_closing']['median'] is not None
                                         and hi['late_hard_closing']['median'] < lo['late_hard_closing']['median']))
        sev = lambda c: float(np.mean([c['severe_search_only'][s] for s in LATE_HARD]))  # noqa: E731
        rep['FS2'] = dict(severe_d025=round(sev(lo), 4), severe_d10=round(sev(hi), 4), supported=sev(hi) < sev(lo))
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'conditions'}))
    for n, c in rep['conditions'].items():
        print(n, c['cfg'], c['late_hard_closing'], {s: c['search_only'][s] for s in LATE_HARD + AEB if s in c['search_only']})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('precheck')
    p.add_argument('new')
    p.add_argument('stored')
    for name in ('b75', 'b76'):
        q = sub.add_parser(name)
        q.add_argument('out')
        q.add_argument('--jobs', required=True)
        q.add_argument('--run', action='append', required=True, help='name=FILE')
        q.add_argument('--ref', action='append', help='b76: name=FILE@METHOD[@CLOSING] stored reference runs')
    a = ap.parse_args()
    if a.cmd == 'precheck':
        sys.exit(precheck(a))
    (b75 if a.cmd == 'b75' else b76)(a)
