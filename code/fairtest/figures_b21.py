"""B21 figure: who is blamed for the same collisions depends on the fault rule (G2/G10).

  python -m fairtest.figures_b21 B21_FAULT_RULES.json OUT.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RULES = (('rss', 'RSS\nreference'), ('fsm', 'FSM\nreference'), ('cc_distance', 'CC fixed-\ndistance'), ('cc_time', 'CC R157\ntime-based'),
         ('oracle_react0', 'ideal ADS\n(no delay)'), ('nuplan', 'geometric\n(nuPlan-style)'), ('oracle_t0', 'omniscient\nsolvability'))
SUTS = (('following_H0', 'following H0'), ('reference_idm_H0', 'IDM H0'), ('following_H0.5', 'following H0.5'), ('following_H1', 'following H1'),
        ('following_aeb_H0', 'AEB H0'), ('following_aeb_H0.5', 'AEB H0.5'), ('rss_sut_H0', 'longitudinal RSS'), ('_all', 'all collisions'))
BLUE = ('#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
ORANGE = ('#fde3d6', '#f9c4aa', '#f4a47f', '#ef8657', '#eb6834', '#c9521f', '#a33f14', '#7d2f0d')


def main(src, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'
    d = json.loads(Path(src).read_text(encoding='utf-8'))
    att = np.array([[100 * d['attributed'][s][r] for r, _ in RULES] for s, _ in SUTS])
    names = [r for r, _ in RULES]
    kap = np.full((len(RULES), len(RULES)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i > j:
                v = d['kappa'].get(f'{a}~{b}', d['kappa'].get(f'{b}~{a}'))
                kap[i, j] = v
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE, gridspec_kw=dict(width_ratios=(1.15, 1)))
    cm_b = LinearSegmentedColormap.from_list('seq_blue', BLUE)
    im = ax.imshow(att, cmap=cm_b, vmin=0, vmax=100, aspect='auto')
    for i in range(att.shape[0]):
        for j in range(att.shape[1]):
            ax.text(j, i, f'{att[i, j]:.0f}', ha='center', va='center', fontsize=8, color='#ffffff' if att[i, j] >= 55 else INK)
    ax.set_xticks(range(len(RULES)), [lab for _, lab in RULES], fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(SUTS)), [f'{lab} (n={d["attributed"][s]["n"]})' for s, lab in SUTS], fontsize=7.5, color=INK2)
    ax.axhline(len(SUTS) - 1.5, color=SURFACE, linewidth=3)
    ax.set_title('(a) Share of the same falsification collisions blamed on the SUT (%)', fontsize=10, color=INK, loc='left')
    cm_o = LinearSegmentedColormap.from_list('seq_orange', ORANGE)
    cm_o.set_bad(SURFACE)
    sub = kap[1:, :-1]  # lower triangle without the empty first row / last column
    bx.imshow(np.ma.masked_invalid(sub), cmap=cm_o, vmin=0, vmax=1, aspect='auto')
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if not np.isnan(sub[i, j]):
                bx.text(j, i, f'{sub[i, j]:.2f}', ha='center', va='center', fontsize=8, color='#ffffff' if sub[i, j] >= .6 else INK)
    one_line = ('RSS reference', 'FSM reference', 'CC fixed-distance', 'CC R157 time-based', 'ideal ADS (no delay)', 'geometric (nuPlan-style)', 'omniscient solvability')
    bx.set_xticks(range(len(RULES) - 1), [lab for _, lab in RULES[:-1]], fontsize=7.5, color=INK2)
    bx.set_yticks(range(len(RULES) - 1), one_line[1:], fontsize=7.5, color=INK2)
    bx.set_title("(b) Agreement between rules (Cohen's κ, all 1,573 collisions)", fontsize=10, color=INK, loc='left')
    bx.set_xlabel('omniscient solvability blames the SUT for every collision, so its κ with any rule is 0', fontsize=7, color=INK2, labelpad=8)
    for x in (ax, bx):
        x.set_facecolor(SURFACE)
        for side in ('top', 'right', 'left', 'bottom'):
            x.spines[side].set_visible(False)
        x.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02).ax.tick_params(labelsize=7, colors=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
