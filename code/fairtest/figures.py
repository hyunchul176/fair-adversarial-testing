"""Static figures for the B7 landscape (paper-style, light surface).

Palette: reference categorical slots 1-3 (#2a78d6 blue, #eb6834 orange, #1baf7a aqua),
validated all-pairs light (worst CVD dE 9.2, normal 24.0); aqua < 3:1 contrast, so every
series is direct-labelled. Small multiples keep <= 3 series per panel. One y-axis per panel.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from fairtest.analyze_landscape import analyse  # noqa: E402
from fairtest.boundary_planner import evaluate  # noqa: E402

SERIES = ('#2a78d6', '#eb6834', '#1baf7a')
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
PANELS = ((('following_H0', 'following H0'), ('following_H0.5', 'following H0.5'), ('following_H1', 'following H1')),
          (('reference_idm_H0', 'IDM H0'), ('following_aeb_H0', 'following+AEB H0'),
           ('following_aeb_H0.5', 'following+AEB H0.5')))
THRESHOLDS = ('0', '0.25', '0.5', '1', '2', '4', 'inf')


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)


def fair_curve(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, facecolor=SURFACE)
    xs = range(len(THRESHOLDS))
    for ax, panel in zip(axes, PANELS):
        _style(ax)
        ends = []
        for color, (key, label) in zip(SERIES, panel):
            if key not in res['suts']:
                continue
            s = res['suts'][key]
            ys = [100 * s['fair_failure_cases_by_max_exceedance'][t] / max(1, s['eligible_cases']) for t in THRESHOLDS]
            ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=5)
            ends.append([ys[-1], label])
        # de-collide end labels: keep >= 6 percentage points apart, in value order
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 6)
        for y, label in ends:
            ax.annotate(label, (xs[-1] + .15, y), va='center', fontsize=9, color=INK)
        ax.set_xticks(list(xs), ['0', '.25', '.5', '1', '2', '4', 'no limit'])
        ax.set_xlim(-.3, len(THRESHOLDS) + 2.2)
        ax.set_xlabel('allowed excess beyond natural p1–p99 (IQR units)', color=INK2, fontsize=9)
    axes[0].set_ylabel('cases with a fair failure (%)', color=INK2, fontsize=9)
    axes[0].set_ylim(0, 105)
    fig.suptitle('Fair (R157-preventable) SUT failures grow as cut-ins leave the naturalistic envelope',
                 color=INK, fontsize=11, x=.02, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def method_bars(rows, res, sut_key, sut, out):
    s = res['suts'][sut_key]
    n = s['eligible_cases']
    bnd = next(r for r in evaluate(rows, severities=(1.,), exceedances=(math.inf,)) if r['rule'] == 'boundary')
    rnd = next(r for r in evaluate(rows, severities=(1.,), exceedances=(math.inf,)) if r['rule'] == 'random')
    methods = [('B5 (frozen)', s['b5_fair'], s['b5_unpreventable']),
               ('fixed 2 s brake-then-enter', s['fixed2s_fair'], s['fixed2s_unpreventable']),
               ('best single fixed plan', s['best_single_plan']['fair'],
                next((t['unpreventable'] for t in s['top_plans'] if t['name'] == s['best_single_plan']['name']), None)),
               ('random CC-preventable plan', rnd['suts'][sut_key].get('fair_failure', 0), 0),
               ('R157-boundary planner (0 SUT queries)', bnd['suts'][sut_key].get('fair_failure', 0), 0),
               ('oracle (best plan per case, SUT queried)', s['oracle_cases_with_fair_failure'], None)]
    fig, ax = plt.subplots(figsize=(9, 4.0), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis='x', color=GRID, linewidth=.8)
    ax.grid(axis='y', visible=False)
    ys = list(range(len(methods)))[::-1]
    h = .36
    for y, (name, fair, unfair) in zip(ys, methods):
        ax.barh(y + h / 2, 100 * fair / n, height=h - .04, color=SERIES[0])
        ax.text(100 * fair / n + 1, y + h / 2, f'{100 * fair / n:.0f}%', va='center', fontsize=8, color=INK)
        if unfair is not None:
            ax.barh(y - h / 2, 100 * unfair / n, height=h - .04, color=SERIES[1])
            ax.text(100 * unfair / n + 1, y - h / 2, f'{100 * unfair / n:.0f}%', va='center', fontsize=8, color=INK)
    ax.set_yticks(ys, [m[0] for m in methods], fontsize=9, color=INK)
    ax.set_xlim(0, 110)
    ax.set_xlabel(f'share of {n} eligible cases (%)', color=INK2, fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label='fair failure (CC driver avoids, SUT collides)'),
                       Patch(color=SERIES[1], label='attacker-at-fault collision (CC also collides)')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False)
    ax.set_title(f'Methods against {sut}', color=INK, fontsize=11, loc='left', pad=22)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_all(rows, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res = analyse(rows)
    fair_curve(res, outdir / 'fig_fair_curve.png')
    method_bars(rows, res, 'following_H0', 'following H0 (late recognition, 5 m/s² braking)', outdir / 'fig_methods_following_H0.png')
    if 'following_H1' in res['suts']:
        method_bars(rows, res, 'following_H1', 'following H1 (1 s anticipation)', outdir / 'fig_methods_following_H1.png')
    (outdir / 'analysis.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    return res
