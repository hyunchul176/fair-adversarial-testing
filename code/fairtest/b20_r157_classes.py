"""B20 (G5): the severity dial in regulatory units and the UN R157 Annex 5 difficulty classes.

The CC braking demand d (max DRAC while overlapping / 7.59) converts to m/s^2 as 7.59 d. R157 Annex 5,
Appendix 1 classes (Rev.1 / 01 series; not in the original series): Avoidable < 5 m/s^2 (d < 0.659),
Difficult 5-7.6 m/s^2, Unavoidable >= 7.6 m/s^2.
Only CC-preventable cut-in plans are attacks here (CC-contact plans are attacker-at-fault). Per SUT:
  case level  share of cases with at least one fair failure among the case's plans in each class;
  plan level  P(SUT contact | CC-preventable plan, demand bin), with an isotonic (monotone) fit and a
              case-bootstrap 95% band; pairwise first-order dominance of adjacent SUT curves.

  python -m fairtest.b20_r157_classes OUT.json OUT.png
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

import fairtest.boundary_planner as BP
from fairtest.analyze_landscape import load
from fairtest.b20_validity import RANK, SUT_T
from fairtest.strict_analysis import J

CC = ('cc_r157', 0.)
CLASSES = (('Avoidable (<5 m/s²)', 0., 5. / 7.59), ('Difficult (5–7.6 m/s²)', 5. / 7.59, 7.6 / 7.59))
BINS = np.array([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
BOOT, SEED = 1000, 20260926


def collect(rows):
    by = BP.index(rows)
    have = {(r['target'], r['H']) for r in rows}
    suts = [k for k in RANK if SUT_T[k] in have]
    per = collections.defaultdict(list)  # case -> [(demand, {sut: fail})]
    for (c, p), v in by.items():
        if not p.startswith('plan_') or p == BP.NOMINAL or CC not in v:
            continue
        cc = v[CC]
        if not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
            continue
        d = BP.cc_demand(cc)
        fails = {k: float(v[SUT_T[k]]['valid'] and v[SUT_T[k]]['contact']) for k in suts if SUT_T[k] in v}
        per[c].append((d, fails))
    return suts, per


def case_level(suts, per):
    out = {}
    cases = sorted(per)
    for name, lo, hi in CLASSES:
        out[name] = {}
        for k in suts:
            hits, n = 0, 0
            for c in cases:
                plans = [f[k] for d, f in per[c] if lo <= d < hi and k in f]
                if not plans:
                    continue
                n += 1
                hits += any(plans)
            out[name][k] = dict(cases_with_class=n, fair_failure=hits / n if n else None)
    return out


def curves(suts, per):
    cases = sorted(per)
    centres = (BINS[:-1] + BINS[1:]) / 2

    def fit(idx_cases):
        res = {}
        for k in suts:
            xs, ys = [], []
            for c in idx_cases:
                for d, f in per[c]:
                    if k in f and d < 1.:
                        xs.append(d)
                        ys.append(f[k])
            if not xs:
                continue
            iso = IsotonicRegression(y_min=0., y_max=1., increasing=True, out_of_bounds='clip').fit(xs, ys)
            res[k] = iso.predict(centres)
        return res

    point = fit(cases)
    rng = random.Random(SEED)
    boots = collections.defaultdict(list)
    for _ in range(BOOT):
        sample = [cases[rng.randrange(len(cases))] for _ in cases]
        for k, v in fit(sample).items():
            boots[k].append(v)
    out = {}
    for k in point:
        b = np.array(boots[k])
        out[k] = dict(demand_bin_centres=centres.tolist(), m_per_s2=(centres * 7.59).tolist(), isotonic=point[k].tolist(),
                      lo=np.percentile(b, 2.5, axis=0).tolist(), hi=np.percentile(b, 97.5, axis=0).tolist())
    dominance = {}
    order = [k for k in suts if k in out and k != 'reference_idm_H0']
    for a, b_ in zip(order, order[1:]):
        wa, wb = np.array(boots[a]), np.array(boots[b_])
        dominance[f'{a} >= {b_}'] = float(np.mean(np.all(wa >= wb - 1e-9, axis=1)))
    return out, dominance


def main(out_json, out_png):
    from fairtest.report_b7 import rows_all
    exp, _ = rows_all()
    exp += load(J / 'b10v2-a' / 'landscape.jsonl')[0] + load(J / 'b10v2-b' / 'landscape.jsonl')[0]
    b9 = sum((load(J / s / 'landscape.jsonl')[0] for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')), [])
    rep = {}
    for name, rows in (('highD_exposed_266', exp), ('highD_B9_148_posthoc', b9)):
        suts, per = collect(rows)
        cl = case_level(suts, per)
        cv, dom = curves(suts, per)
        rep[name] = dict(case_level=cl, curves=cv, dominance_bootstrap_share=dom)
        print('==', name)
        for cname, d in cl.items():
            print('  ', cname, {k: (None if v['fair_failure'] is None else round(100 * v['fair_failure'], 1)) for k, v in d.items()})
        print('   dominance', {k: round(v, 3) for k, v in dom.items()})
    Path(out_json).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    plot(rep['highD_B9_148_posthoc'], rep['highD_exposed_266'], out_png)


def plot(b9, exp, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
    COLORS = {'following_H0': '#2a78d6', 'following_H0.5': '#eb6834', 'following_H1': '#1baf7a', 'following_aeb_H0': '#8a5cd6',
              'following_aeb_H0.5': '#c2185b', 'rss_sut_H0': '#8a8983'}
    LABELS = {'following_H0': 'following H0', 'following_H0.5': 'following H0.5', 'following_H1': 'following H1',
              'following_aeb_H0': 'AEB H0', 'following_aeb_H0.5': 'AEB H0.5', 'rss_sut_H0': 'longitudinal RSS'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE, sharey=True)
    for ax, (title, rep) in zip(axes, (('highD exposed (development, 266 cases)', exp), ('highD reserved B9 (post hoc, 148 cases)', b9))):
        for k, cv in rep['curves'].items():
            if k not in COLORS:
                continue
            x = np.array(cv['m_per_s2'])
            ax.fill_between(x, 100 * np.array(cv['lo']), 100 * np.array(cv['hi']), color=COLORS[k], alpha=.12, linewidth=0)
            ax.plot(x, 100 * np.array(cv['isotonic']), color=COLORS[k], linewidth=2, label=LABELS[k])
        ax.axvspan(0, 5., color='#e6e5e0', alpha=.35, linewidth=0)
        ax.axvline(5., color='#8a8983', linewidth=1, linestyle='--')
        ax.axvline(.12 * 7.59, color='#8a8983', linewidth=1, linestyle=':')
        ax.text(.12 * 7.59 - .12, 55, 'natural highD p99', fontsize=7, color=INK2, va='center', ha='right', rotation=90)
        ax.text(3.0, 108, 'R157 Avoidable (< 5 m/s²)', fontsize=8, color=INK2, ha='center')
        ax.text(6.3, 108, 'R157 Difficult (5–7.6 m/s²)', fontsize=8, color=INK2, ha='center')
        ax.set_xlim(0, 7.6)
        ax.set_ylim(-2, 114)
        ax.set_xlabel('braking the careful driver needs to avoid the attack (m/s²)', fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9, color=INK, loc='left')
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(color=GRID, linewidth=.7)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel('P(SUT collides | careful-driver-preventable attack) %', fontsize=8, color=INK2)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='center right')
    fig.suptitle('Fragility curves in UN R157 units: where each system stops matching the careful and competent driver', fontsize=10, color=INK, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
