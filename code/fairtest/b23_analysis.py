"""B23 analysis for the agent paper.

E3 (query efficiency, highD development cases, unseen by the RL agent): for each SUT the share of cases whose
first fair failure is found within k SUT queries, for
  ours        guarded RL policy (1 query), then CMA-ES from the R157-boundary plan (the stored B11 `cont_search`)
  rl_guarded  the policy alone (1 query)             cmaes_warm  the B11 CMA-ES alone (boundary start)
  boundary    the zero-query R157-boundary plan, evaluated once
  random, cmaes_cold, bo_cold, bo_warm   `b23_search` optimiser baselines (same objective and budget)

  python -m fairtest.b23_analysis e3 OUT.json OUT.png

e3x: the same per dataset (dev, or b9 post hoc) with the policy's query taken from the B23 evaluation of all three
training seeds (curves averaged over seeds), plus
  ours_bo     guarded RL policy (1 query), then GP-BO from the R157-boundary plan (refinement optimiser chosen on dev
              after E3; confirmed on B9 post hoc)

  python -m fairtest.b23_analysis e3x OUT.json OUT.png --dataset dev|b9

E1/E4/E5 (1-query RL adversaries; fairtest_jobs/b23/eval/<agent>_s<seed>_<dataset>.jsonl): per agent, dataset and
SUT, the per-scenario collision rate, fair-failure rate (CC distance), attacker-at-fault collision rate, CARS-style
conditional attribution (FSM, among collisions), feint share (lane reversals after entering) and realism; validity
across the ladder (range weakest - longitudinal RSS, RSS rate, Kendall tau vs the capability rank); mean and
spread over training seeds; a case-level bootstrap 95% CI of the seed-averaged fair-failure rate. When present:
  margin_<dataset>.jsonl (b23_margin)     robust fair failures: front contact and CC clearance >= 0.5 m (G4)
  cross_<dataset>.jsonl (b23_crossjudge)  share of collisions that the Dynasto validity rules call valid

  python -m fairtest.b23_analysis e1 OUT.json [--dir EVAL_DIR]

Two-stage agent per dataset (dev, B9 post hoc, NGSIM post hoc): the policy's single query (each training seed) and,
if it is not a fair failure, the stored B11/B13 CMA-ES refinement from the R157-boundary plan (budget 36); a case
counts at query 1 + q when the refinement finds its first fair failure after q queries.

  python -m fairtest.b23_analysis two_stage OUT.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest.strict_analysis import E13, J

B23 = J / 'b23'
SUT7 = ('following_H0', 'reference_idm_H0', 'following_H0.5', 'following_H1', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')
ADS = ('ads@0.5_H0', 'ads@2.0_H0')
BUDGET = 36
KS = (1, 2, 3, 5, 10, 20, 36, 37)
AGENTS = ('ours', 'naive_fair', 'collide_commit', 'collide', 'dynasto', 'cars', 'abl_none', 'abl_commit', 'abl_real')
DELTA = .5  # G4 robust fair failure: CC keeps >= DELTA m from the adversary and the SUT's contact is frontal
DATASETS = ('dev', 'b9', 'ngsim')
RANK = {'following_H0': 1, 'reference_idm_H0': 1, 'following_H0.5': 2, 'following_H1': 3, 'following_aeb_H0': 4,
        'following_aeb_H0.5': 5, 'rss_sut_H0': 6}


def _rows(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def stored_search(dev):
    """{(case, sut): {'boundary': first|None, 'cmaes_warm': first|None}} from the stored B11-format searches."""
    out = {}
    for path in (E13 / 'posthoc_highd' / 'exposed_cont_search_all7.jsonl', J / 'b19' / 'b19_cont.jsonl'):
        for r in _rows(path):
            if r['case_id'] not in dev:
                continue
            ev = ((r.get('cc_boundary') or {}).get('sut_eval')) or {}
            for k, ss in r['sut_search'].items():
                b = ss.get('best')
                out[(r['case_id'], k)] = dict(boundary=1 if ev.get(k, {}).get('contact') else None,
                                              cmaes_warm=ss['sut_queries'] if b and b['fair'] else None)
    return out


def stored_rl(dev):
    """{(case, sut): 1|None}: fair failure of the deterministic guarded policy (fair_guarded2 seed 1)."""
    out = {}
    for path in (J / 'b17' / 'eval_fair_guarded2_s1_dev.jsonl', J / 'b19' / 'b19_rl_fair.jsonl'):
        for r in _rows(path):
            if r['case_id'] in dev:
                fair = bool(r['valid'] and r['contact'] and not r['refs']['cc_r157']['contact'])
                out[(r['case_id'], r['sut'])] = 1 if fair else None
    return out


CONT = {'dev': E13 / 'posthoc_highd' / 'exposed_cont_search_all7.jsonl',
        'b9': E13 / 'posthoc_highd' / 'b9_cont_search_all7_POSTHOC.jsonl',
        'ngsim': E13 / 'continuous' / 'b13_cont_search.jsonl'}


def two_stage(a):
    ev = J / 'b23' / 'eval'
    rep = {}
    for ds, path in CONT.items():
        refine = {}
        for r in _rows(path):
            ev_b = ((r.get('cc_boundary') or {}).get('sut_eval')) or {}
            for k, ss in r['sut_search'].items():
                b = ss.get('best')
                refine[(r['case_id'], k)] = dict(q=ss['sut_queries'] if b and b['fair'] else None,
                                                 boundary=bool(ev_b.get(k, {}).get('contact')))
        seeds = sorted(ev.glob(f'ours_s*_{ds}.jsonl'))
        rep[ds] = {}
        for sut in SUT7:
            per = collections.defaultdict(list)
            for path_s in seeds:
                for r in _rows(path_s):
                    if r['sut'] != sut or (r['case_id'], sut) not in refine:
                        continue
                    fair = bool(r['valid'] and r['contact'] and not r['refs']['cc_r157']['contact'])
                    q = refine[(r['case_id'], sut)]['q']
                    per['policy'].append(fair)
                    per['two_stage'].append(fair or q is not None)
                    per['queries'].append(1 if fair else (None if q is None else 1 + q))
            evaluated = {r['case_id'] for r in _rows(seeds[0])} if seeds else set()
            cases = [(c, v) for (c, k), v in refine.items() if k == sut and c in evaluated]  # same cases as the policy
            rep[ds][sut] = dict(n=len(per['policy']) // max(1, len(seeds)), seeds=len(seeds),
                                policy=float(np.mean(per['policy'])) if per['policy'] else None,
                                refinement_only=float(np.mean([v['q'] is not None for _, v in cases])) if cases else None,
                                boundary_only=float(np.mean([v['boundary'] for _, v in cases])) if cases else None,
                                two_stage=float(np.mean(per['two_stage'])) if per['two_stage'] else None,
                                mean_queries=float(np.mean([q for q in per['queries'] if q is not None])) if any(q is not None for q in per['queries']) else None)
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    pct = lambda x: '  n/a' if x is None else f'{100 * x:5.1f}'
    for ds, d in rep.items():
        print(f'== {ds}')
        for sut, x in d.items():
            print(f"   {sut:20} n={x['n']:3d} x{x['seeds']} policy {pct(x['policy'])}  boundary {pct(x['boundary_only'])}  "
                  f"refinement {pct(x['refinement_only'])}  two-stage {pct(x['two_stage'])}  mean queries {x['mean_queries']}")


def found_at(firsts, k):
    return float(np.mean([f is not None and f <= k for f in firsts])) if firsts else None


def e3(a):
    from fairtest.runner import _STATE, _init
    _init()
    dev = {c for c, v in _STATE['cases'].items() if v['event'].get('partition') == 'development'}
    opt = collections.defaultdict(dict)
    for r in _rows(B23 / 'e3_dev_optimisers.jsonl'):
        for key, rec in r['runs'].items():
            method, sut = key.split('|')
            opt[(r['case_id'], sut)][method] = rec.get('first_fair')
    st, rl = stored_search(dev), stored_rl(dev)
    methods = ('ours', 'rl_guarded', 'cmaes_warm', 'boundary', 'bo_warm', 'cmaes_cold', 'bo_cold', 'random')
    rep = {}
    for sut in SUT7 + ADS:
        firsts = collections.defaultdict(list)
        for c in sorted(dev):
            s, o, q = st.get((c, sut)), opt.get((c, sut)), rl.get((c, sut), 'missing')
            if s is None or o is None or q == 'missing':
                continue
            firsts['rl_guarded'].append(q)
            firsts['ours'].append(1 if q else (None if s['cmaes_warm'] is None else 1 + s['cmaes_warm']))
            firsts['cmaes_warm'].append(s['cmaes_warm'])
            firsts['boundary'].append(s['boundary'])
            for m in ('bo_warm', 'cmaes_cold', 'bo_cold', 'random'):
                firsts[m].append(o.get(m))
        rep[sut] = {m: dict(n=len(firsts[m]), found={str(k): found_at(firsts[m], k) for k in KS},
                            mean_first=float(np.mean([f for f in firsts[m] if f is not None])) if any(f is not None for f in firsts[m]) else None)
                    for m in methods if firsts[m]}
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    for sut, d in rep.items():
        print(f'== {sut}')
        for m, x in d.items():
            f = x['found']
            print(f"   {m:11} n={x['n']:3d}  @1 {100*f['1']:5.1f}  @5 {100*f['5']:5.1f}  @10 {100*f['10']:5.1f}  @36 {100*f['36']:5.1f}  @37 {100*f['37']:5.1f}  mean first {x['mean_first']}")
    plot_e3(rep, a.png)


def e3x(a):
    ds = a.dataset
    ev = J / 'b23' / 'eval'
    cont = [CONT[ds]] + ([J / 'b19' / 'b19_cont.jsonl'] if ds == 'dev' else [])
    st = {}
    for path in cont:
        for r in _rows(path):
            ev_b = ((r.get('cc_boundary') or {}).get('sut_eval')) or {}
            for k, ss in r['sut_search'].items():
                b = ss.get('best')
                st[(r['case_id'], k)] = dict(boundary=1 if ev_b.get(k, {}).get('contact') else None,
                                             cmaes_warm=ss['sut_queries'] if b and b['fair'] else None)
    opt = collections.defaultdict(dict)
    for r in _rows(B23 / f'e3_{ds}_optimisers.jsonl'):
        for key, rec in r['runs'].items():
            method, sut = key.split('|')
            opt[(r['case_id'], sut)][method] = rec.get('first_fair')
    seeds = {}
    for path in sorted(ev.glob(f'ours_s*_{ds}.jsonl')):
        seeds[path.name] = {(r['case_id'], r['sut']): (1 if (r['valid'] and r['contact'] and not r['refs']['cc_r157']['contact']) else None)
                            for r in _rows(path)}
    methods = ('ours', 'ours_bo', 'rl_guarded', 'cmaes_warm', 'boundary', 'bo_warm', 'cmaes_cold', 'bo_cold', 'random')
    rep = {}
    for sut in SUT7 + (ADS if ds == 'dev' else ()):
        firsts = collections.defaultdict(list)
        cases = sorted({c for (c, k) in opt if k == sut})
        for c in cases:
            s_, o = st.get((c, sut)), opt.get((c, sut))
            if s_ is None or o is None or any((c, sut) not in rl for rl in seeds.values()):
                continue
            for rl in seeds.values():  # policy-based methods: one entry per training seed
                q = rl[(c, sut)]
                firsts['rl_guarded'].append(q)
                firsts['ours'].append(1 if q else (None if s_['cmaes_warm'] is None else 1 + s_['cmaes_warm']))
                firsts['ours_bo'].append(1 if q else (None if o.get('bo_warm') is None else 1 + o['bo_warm']))
            firsts['cmaes_warm'].append(s_['cmaes_warm'])
            firsts['boundary'].append(s_['boundary'])
            for m in ('bo_warm', 'cmaes_cold', 'bo_cold', 'random'):
                firsts[m].append(o.get(m))
        rep[sut] = {m: dict(n=len(firsts[m]), found={str(k): found_at(firsts[m], k) for k in KS},
                            mean_first=float(np.mean([f for f in firsts[m] if f is not None])) if any(f is not None for f in firsts[m]) else None)
                    for m in methods if firsts[m]}
    Path(a.out).write_text(json.dumps(dict(dataset=ds, seeds=sorted(seeds), per_sut=rep), indent=1), encoding='utf-8')
    for sut, d in rep.items():
        print(f'== {ds} {sut}')
        for m, x in d.items():
            f = x['found']
            print(f"   {m:11} n={x['n']:4d}  @1 {100*f['1']:5.1f}  @2 {100*f['2']:5.1f}  @5 {100*f['5']:5.1f}  @10 {100*f['10']:5.1f}  "
                  f"@20 {100*f['20']:5.1f}  @37 {100*f['37']:5.1f}  mean first {x['mean_first']}")
    plot_e3(rep, a.png, with_bo=True, ds=ds)


def plot_e3(rep, out, with_bo=False, ds='dev'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # validated categorical slots in fixed order (reference palette)
    COLOR = dict(ours='#2a78d6', cmaes_warm='#eb6834', bo_warm='#1baf7a', cmaes_cold='#eda100', bo_cold='#e87ba4', random='#008300')
    LABEL = dict(ours='ours: guarded RL (1 query) + CMA-ES refinement', cmaes_warm='CMA-ES, R157-boundary start',
                 bo_warm='GP-BO, R157-boundary start', cmaes_cold='CMA-ES, cold start', bo_cold='GP-BO, cold start', random='random search')
    if with_bo:  # the BO-refinement variant shares the agent's hue, dashed
        COLOR = dict(ours='#2a78d6', ours_bo='#2a78d6', **{k: v for k, v in COLOR.items() if k != 'ours'})
        LABEL['ours_bo'] = 'ours: guarded RL (1 query) + GP-BO refinement'
    INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
    if ds == 'dev':
        panels = (('following_H0', 'weakest SUT (following H0)'), ('following_H1', 'mid SUT (following H1)'),
                  ('ads@0.5_H0', 'unseen failure mode (over-sensitive ADS, d_p 0.5 m)'))
    else:
        panels = (('following_H0', 'weakest SUT (following H0)'), ('following_H0.5', 'following H0.5'),
                  ('following_H1', 'mid SUT (following H1)'))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE, sharey=False)
    for ax, (sut, title) in zip(axes, panels):
        d = rep.get(sut, {})
        for m in COLOR:
            if m not in d:
                continue
            ks = [k for k in KS if k <= 37]
            ax.step(ks, [100 * d[m]['found'][str(k)] for k in ks], where='post', color=COLOR[m],
                    linewidth=2.2 if m.startswith('ours') else 1.5, linestyle='--' if m == 'ours_bo' else '-', label=LABEL[m])
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 36], ['1', '2', '5', '10', '20', '36'])
        ax.set_xlabel('SUT queries (log)', fontsize=8, color=INK2)
        ax.set_ylabel('cases with a fair failure found, %', fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(color=GRID, linewidth=.7)
        ax.tick_params(colors=INK2, labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, labelcolor=INK, loc='lower center', ncol=4)
    fig.subplots_adjust(left=.05, right=.99, top=.9, bottom=.25, wspace=.18)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def feints(actions):
    """Lane reversals after the first entry (lane goal 4 -> 0)."""
    n, entered, last = 0, False, 0.
    for lane, _ in actions:
        if lane == 4.:
            entered = True
        elif entered and last == 4.:
            n += 1
        last = lane
    return n


def episode_metrics(r):
    col = bool(r['valid'] and r['contact'])
    refs = r.get('refs') or {}
    fair = col and not refs['cc_r157']['contact']
    return dict(collision=col, fair=fair, unfair=col and not fair,
                fsm_attr=col and not refs['ref_fsm']['contact'],
                strict_fair=col and not refs['cc_r157_strict']['contact'],
                feint=feints(r['actions']) > 0,
                realistic=bool(r.get('cut_in') and (r.get('realism_exceedance') or 0.) <= 1.))


def _side_tables(ev, ds):
    """{(source, case, sut): robust?} from the G4 margins, {(source, case, sut): Dynasto-valid?} from the cross-judgement."""
    robust, dyn = {}, {}
    if (ev / f'margin_{ds}.jsonl').exists():
        for r in _rows(ev / f'margin_{ds}.jsonl'):
            robust[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= 3.5
                                                                 and r['cc_clearance'] >= DELTA)
    if (ev / f'cross_{ds}.jsonl').exists():
        for r in _rows(ev / f'cross_{ds}.jsonl'):
            dyn[(r['source'], r['case_id'], r['sut'])] = bool(r['dynasto_valid'])
    return robust, dyn


def bootstrap_ci(per_case, n_boot=2000, seed=0):
    """95% CI of the mean over cases of the seed-averaged indicator."""
    vals = np.array(list(per_case.values()), float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def e1(a):
    from scipy.stats import kendalltau
    ev = Path(a.dir) if a.dir else J / 'b23' / 'eval'
    rep = {}
    for agent in AGENTS:
        for ds in DATASETS:
            robust, dyn = _side_tables(ev, ds)
            per_seed, per_case = {}, collections.defaultdict(lambda: collections.defaultdict(list))
            for path in sorted(ev.glob(f'{agent}_s*_{ds}.jsonl')):
                seed = path.name[len(agent) + 2:].split('_')[0]
                by = collections.defaultdict(list)
                for r in _rows(path):
                    m = episode_metrics(r)
                    key = (path.name, r['case_id'], r['sut'])
                    m['robust'] = robust.get(key) if m['fair'] else False
                    m['dynasto_valid'] = dyn.get(key) if m['collision'] else None
                    by[r['sut']].append(m)
                    per_case[r['sut']][r['case_id']].append(float(m['fair']))
                sut = {}
                for k, ms in by.items():
                    n = len(ms)
                    col = sum(m['collision'] for m in ms)
                    fair = [m for m in ms if m['fair']]
                    judged = [m for m in fair if m['robust'] is not None]
                    dv = [m['dynasto_valid'] for m in ms if m['collision'] and m['dynasto_valid'] is not None]
                    sut[k] = dict(n=n, collision=col / n, fair=len(fair) / n, unfair=sum(m['unfair'] for m in ms) / n,
                                  strict_fair=sum(m['strict_fair'] for m in ms) / n,
                                  robust_fair=(sum(m['robust'] for m in judged) / n) if robust and len(judged) == len(fair) else None,
                                  attribution_fsm=(sum(m['fsm_attr'] for m in ms) / col) if col else None,
                                  attribution_cc=(len(fair) / col) if col else None,
                                  dynasto_valid_share=(sum(dv) / len(dv)) if dv else None,
                                  feint_share_of_fair=(sum(m['feint'] for m in fair) / len(fair)) if fair else None,
                                  realistic_share_of_fair=(sum(m['realistic'] for m in fair) / len(fair)) if fair else None)
                ladder = [k for k in RANK if k in sut]
                val = dict(range=sut['following_H0']['fair'] - sut['rss_sut_H0']['fair'] if 'following_H0' in sut and 'rss_sut_H0' in sut else None,
                           rss_fair=sut.get('rss_sut_H0', {}).get('fair'),
                           tau=float(kendalltau([RANK[k] for k in ladder], [sut[k]['fair'] for k in ladder]).statistic) if len(ladder) > 2 else None,
                           collision_range=sut['following_H0']['collision'] - sut['rss_sut_H0']['collision'] if 'following_H0' in sut and 'rss_sut_H0' in sut else None)
                per_seed[seed] = dict(sut=sut, validity=val)
            if per_seed:
                rep[f'{agent}|{ds}'] = dict(seeds=per_seed, fair_ci={k: bootstrap_ci({c: float(np.mean(v)) for c, v in cs.items()})
                                                                      for k, cs in per_case.items()})
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    def ms(vals):
        vals = [v for v in vals if v is not None]
        return (100 * float(np.mean(vals)), 100 * float(np.std(vals))) if vals else (float('nan'), float('nan'))
    for key, entry in rep.items():
        per_seed, ci = entry['seeds'], entry['fair_ci']
        seeds = list(per_seed)
        print(f'== {key}  seeds {seeds}')
        v = {m: ms([per_seed[s]['validity'][m] for s in seeds]) for m in ('range', 'rss_fair', 'collision_range')}
        taus = [per_seed[s]['validity']['tau'] for s in seeds if per_seed[s]['validity']['tau'] is not None]
        print(f"   validity: fair range {v['range'][0]:.1f}±{v['range'][1]:.1f}  RSS fair {v['rss_fair'][0]:.1f}  collision range {v['collision_range'][0]:.1f}  tau {np.mean(taus) if taus else float('nan'):+.2f}")
        suts = sorted({k for s in seeds for k in per_seed[s]['sut']}, key=lambda k: (RANK.get(k, 9), k))
        for k in suts:
            row = {m: ms([per_seed[s]['sut'][k][m] for s in seeds if k in per_seed[s]['sut']])
                   for m in ('collision', 'fair', 'robust_fair', 'unfair', 'attribution_fsm', 'dynasto_valid_share',
                             'feint_share_of_fair', 'realistic_share_of_fair')}
            lo, hi = (100 * x for x in ci[k]) if ci.get(k) else (float('nan'), float('nan'))
            print(f"   {k:20} coll {row['collision'][0]:5.1f}  fair {row['fair'][0]:5.1f}±{row['fair'][1]:4.1f} [{lo:4.1f},{hi:5.1f}]  "
                  f"robust {row['robust_fair'][0]:5.1f}  unfair {row['unfair'][0]:5.1f}  FSM-attr {row['attribution_fsm'][0]:5.1f}  "
                  f"Dynasto-valid {row['dynasto_valid_share'][0]:5.1f}  feint {row['feint_share_of_fair'][0]:5.1f}  real {row['realistic_share_of_fair'][0]:5.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('e3')
    q.add_argument('out')
    q.add_argument('png')
    w = sub.add_parser('e1')
    w.add_argument('out')
    w.add_argument('--dir', help='evaluation directory (default fairtest_jobs/b23/eval)')
    t = sub.add_parser('two_stage')
    t.add_argument('out')
    x = sub.add_parser('e3x')
    x.add_argument('out')
    x.add_argument('png')
    x.add_argument('--dataset', default='dev', choices=('dev', 'b9', 'ngsim'))
    a = ap.parse_args()
    dict(e3=e3, e1=e1, two_stage=two_stage, e3x=e3x)[a.cmd](a)
