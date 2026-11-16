"""B9 figure: fair-failure rate with recording-bootstrap 95% CI, per SUT and method (primary, no limit).

Dot-and-whisker small multiples (one panel per SUT, one y axis per panel). Method identity is
carried by the y-axis label (no colour legend needed); zero-query methods in slot-1 blue, queried
methods in slot-2 orange, reference rows (random, oracle) in muted ink. Values direct-labelled.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, MUTED = '#2a78d6', '#eb6834', '#8a8983'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
ROWS = (('b5', 'B5 (frozen)', BLUE), ('fixed2s', 'fixed 2 s plan', BLUE), ('random', 'random CC-preventable', MUTED),
        ('boundary', 'R157-boundary (0 queries)', BLUE), ('learned_1', 'learned selector (0 queries)', BLUE),
        ('guided_5', 'R157-guided search (5 queries)', ORANGE), ('oracle', 'oracle (all 199 queried)', MUTED))
PANELS = (('following_H0', 'following H0'), ('following_H0.5', 'following H0.5'), ('following_H1', 'following H1'),
          ('reference_idm_H0', 'IDM H0'), ('following_aeb_H0', 'following+AEB H0'), ('following_aeb_H0.5', 'following+AEB H0.5'))


def plot(report, out):
    prim = report['primary']['no_limit']
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(axis='x', color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        m = prim[key]['methods']
        ys = list(range(len(ROWS)))[::-1]
        for y, (meth, label, color) in zip(ys, ROWS):
            v = m[meth]
            lo, hi = (100 * v['fair_ci95'][0], 100 * v['fair_ci95'][1])
            ax.plot([lo, hi], [y, y], color=color, linewidth=2)
            ax.plot(100 * v['fair'], y, 'o', color=color, markersize=6)
            ax.text(min(hi, 100) + 2, y, f"{100 * v['fair']:.0f}%", va='center', fontsize=8, color=INK)
        ax.set_yticks(ys, [r[1] for r in ROWS] if ax in axes[:, 0] else [], fontsize=8, color=INK)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlim(0, 112)
        ax.set_title(f"{title}  (n={m['oracle']['n']})", color=INK, fontsize=10, loc='left')
    for ax in axes[1]:
        ax.set_xlabel('fair-failure rate, % of cases (95% recording-bootstrap CI)', color=INK2, fontsize=8)
    fig.suptitle('B9 reserved partition: fair (R157-preventable) failures found per method', color=INK,
                 fontsize=11, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    rep = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    plot(rep, sys.argv[2])
