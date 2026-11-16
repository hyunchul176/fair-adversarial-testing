"""Severity controllability of the R157-referenced adversary (post-hoc analysis of stored grid landscapes).

The severity dial s is a target CC braking demand (max DRAC while overlapping / 7.59 m/s^2). For each
case the planner picks the CC-preventable plan whose CC demand is closest to s (rule 'target',
optionally within a realism limit). Reported per dataset: achieved CC demand (median, IQR), Spearman
rho(s, achieved demand) over case x level pairs, and the fair-failure rate of each SUT at each level.

  python -m fairtest.severity_dial OUT.json OUT.png
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import fairtest.boundary_planner as BP
from fairtest.analyze_landscape import load
from fairtest.features import Realism

EV = Path(os.environ.get('FAIRTEST_EVIDENCE', 'evidence'))  # local evidence folder of the study (not distributed)
J = EV / 'fairtest_jobs'
LEVELS = (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)
SUTS = (('following', 0.), ('following', .5), ('following', 1.), ('following_aeb', 0.), ('rss_sut', 0.))
CC = ('cc_r157', 0.)


def dial(rows, realism, exceedance=math.inf):
    by = BP.index(rows)
    cases = sorted({c for c, _ in by})
    out = dict(levels=list(LEVELS), achieved={}, fair={f'{s[0]}_H{s[1]:g}': [] for s in SUTS}, n_cases=len(cases))
    pairs_s, pairs_d = [], []
    for s in LEVELS:
        demands, hits = [], {k: [] for k in out['fair']}
        for c in cases:
            p = BP.select(by, c, realism, severity=s, max_exceedance=exceedance, rule='target')
            if p is None:
                continue
            v = by[(c, p)]
            d = BP.cc_demand(v[CC])
            demands.append(d)
            pairs_s.append(s)
            pairs_d.append(d)
            for sut in SUTS:
                r = v.get(sut)
                if r is not None:
                    hits[f'{sut[0]}_H{sut[1]:g}'].append(float(r['valid'] and r['contact']))
        out['achieved'][f'{s:g}'] = dict(median=float(np.median(demands)), q25=float(np.percentile(demands, 25)),
                                          q75=float(np.percentile(demands, 75)), n=len(demands))
        for k, h in hits.items():
            out['fair'][k].append(float(np.mean(h)) if h else None)
    rho = spearmanr(pairs_s, pairs_d)
    out['spearman_level_vs_achieved'] = dict(rho=float(rho.statistic), p=float(rho.pvalue), n=len(pairs_s))
    return out


def main(out_json, out_png):
    realism = Realism()
    from fairtest.report_b7 import rows_all
    exposed, _ = rows_all()
    exposed += load(J / 'b10v2-a' / 'landscape.jsonl')[0] + load(J / 'b10v2-b' / 'landscape.jsonl')[0]
    b9 = sum((load(J / s / 'landscape.jsonl')[0] for s in ('b9-s0', 'b9-s1', 'b9-s2b', 'b9-s3b')), [])
    ng = sum((load(J / s / 'landscape.jsonl')[0] for s in ('b13-ngsim-a', 'b13-ngsim-b')), [])
    rep = {}
    for name, rows in (('highD exposed (development)', exposed), ('highD reserved B9 (post hoc)', b9), ('NGSIM US-101 (post hoc)', ng)):
        rep[name] = dict(no_limit=dial(rows, realism), realism_1iqr=dial(rows, realism, 1.))
        a = rep[name]['no_limit']
        print(name, 'rho(level, achieved)', round(a['spearman_level_vs_achieved']['rho'], 3),
              {k: [None if x is None else round(100 * x) for x in v] for k, v in a['fair'].items()})
    Path(out_json).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    plot(rep, out_png)


def plot(rep, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
    COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#8a5cd6', '#8a8983')
    LABELS = ('following H0', 'following H0.5', 'following H1', 'AEB H0', 'longitudinal RSS')
    names = list(rep)
    fig, axes = plt.subplots(2, len(names), figsize=(4.4 * len(names), 6.4), facecolor=SURFACE, sharex=True)
    for j, name in enumerate(names):
        a = rep[name]['no_limit']
        lv = np.array(a['levels'])
        ach = a['achieved']
        med = np.array([ach[f'{s:g}']['median'] for s in lv])
        lo = np.array([ach[f'{s:g}']['q25'] for s in lv])
        hi = np.array([ach[f'{s:g}']['q75'] for s in lv])
        ax = axes[0, j]
        ax.fill_between(lv, lo, hi, color='#2a78d6', alpha=.18, linewidth=0)
        ax.plot(lv, med, color='#2a78d6', linewidth=2, marker='o', markersize=4)
        ax.plot([0, 1], [0, 1], color='#8a8983', linewidth=1, linestyle='--')
        ax.set_title(f"{name}\nrho = {a['spearman_level_vs_achieved']['rho']:.2f}  (n = {a['n_cases']} cases)", fontsize=9, color=INK, loc='left')
        ax.set_ylim(0, 1.1)
        if j == 0:
            ax.set_ylabel('achieved CC braking demand\n(median, IQR)', fontsize=8, color=INK2)
        ax2 = axes[1, j]
        for (k, v), color, label in zip(a['fair'].items(), COLORS, LABELS):
            ax2.plot(lv, [np.nan if x is None else 100 * x for x in v], color=color, linewidth=2, marker='o', markersize=4, label=label)
        ax2.set_ylim(-3, 103)
        ax2.set_xlabel('severity dial: target CC braking demand (fraction of 7.59 m/s²)', fontsize=8, color=INK2)
        if j == 0:
            ax2.set_ylabel('fair-failure rate (%)', fontsize=8, color=INK2)
        for x in (ax, ax2):
            x.set_facecolor(SURFACE)
            for side in ('top', 'right'):
                x.spines[side].set_visible(False)
            x.grid(color=GRID, linewidth=.7)
            x.tick_params(colors=INK2, labelsize=8)
    axes[1, 0].legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper left')
    fig.suptitle('Severity controllability of the R157-referenced adversary: the dial sets how hard the careful driver must brake',
                 fontsize=10, color=INK, x=.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
