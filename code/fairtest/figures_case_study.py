"""Case-study figure (exposed highD case): attacker-at-fault collision (B5) vs fair failure (R157 boundary).

Rows: the two attacks on the same initial scenario. Columns: generator lateral position, bumper gap
and follower acceleration, each for the SUT run (following H0) and the R157 careful-driver run.

  python -m fairtest.figures_case_study CASE_ID PLAN_NAME OUT.png
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fairtest import runner  # noqa: E402
from fairtest.plans import Plan  # noqa: E402
from fairtest.features import Realism, simulated_features  # noqa: E402

BLUE, ORANGE, MUTED = '#2a78d6', '#eb6834', '#8a8983'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'


def parse(name):
    parts = name.split('_')
    e, dur = parts[1][1:], parts[4][1:]
    return Plan(99. if e == 'never' else float(e), float(parts[2][3:]), float(parts[3][4:]), None if dur == 'grid' else float(dur))


def series(row):
    t, y, gap, acc = [], [], [], []
    for k, r in enumerate(row['trace']):
        g, f = r['unimpacted_generator'], r['unimpacted_target']
        t.append((k + 1) * .1)
        y.append(g['y'])
        gap.append((g['x'] - f['x']) - 5.)
        acc.append(r['target_control']['applied_mps2'])
    return np.array(t), np.array(y), np.array(gap), np.array(acc)


def plot(case_id, plan_name, out):
    runner._init()
    case = runner._STATE['cases'][case_id]
    realism = Realism()
    attacks = (('B5 (rule-based risk-seeking braking-entry agent)', runner.make_policy('b5')), ('R157-boundary planner (0 SUT queries)', parse(plan_name)))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.6), facecolor=SURFACE, sharex=True)
    for i, (title, policy) in enumerate(attacks):
        runs = {}
        for target, label, color in (('following', 'SUT (following, H0)', ORANGE), ('cc_r157', 'R157 careful driver', BLUE)):
            row = runner.run_episode(case, policy, target, 0., keep_trace=True)
            runs[target] = (row, label, color)
        feats = simulated_features(runs['cc_r157'][0])
        exc = realism.score(feats)['max_exceedance'] if feats.get('cut_in') else None
        for j, ax in enumerate(axes[i]):
            ax.set_facecolor(SURFACE)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.grid(color=GRID, linewidth=.7)
            ax.tick_params(colors=INK2, labelsize=8)
        for target, (row, label, color) in runs.items():
            t, y, gap, acc = series(row)
            crash = bool(row['pair_contact'])
            if target == 'following':
                axes[i, 0].plot(t, y, color=INK2, linewidth=1.6)
            axes[i, 1].plot(t, gap, color=color, linewidth=1.8, label=label)
            axes[i, 2].plot(t, acc, color=color, linewidth=1.8, label=label)
            if crash:
                axes[i, 1].plot(t[-1], gap[-1], 'x', color=color, markersize=9, markeredgewidth=2)
        crashed = [target for target, (row, _, _) in runs.items() if row['pair_contact']]
        if crashed:
            tt = max(len(runs[c][0]['trace']) for c in crashed) * .1
            axes[i, 1].text(tt + .2, 1.5, 'both contact' if len(crashed) == 2 else 'SUT contact', color=INK, fontsize=7.5)
        axes[i, 0].axhline(2., color=MUTED, linewidth=1, linestyle='--')
        axes[i, 0].text(.1, 2.15, 'lane boundary', color=INK2, fontsize=7)
        axes[i, 0].set_ylabel('generator lateral\nposition (m)', color=INK2, fontsize=8)
        axes[i, 1].set_ylabel('bumper gap (m)', color=INK2, fontsize=8)
        axes[i, 2].set_ylabel('follower accel. (m/s²)', color=INK2, fontsize=8)
        sut_c, cc_c = bool(runs['following'][0]['pair_contact']), bool(runs['cc_r157'][0]['pair_contact'])
        verdict = ('attacker-at-fault: the careful driver also crashes' if sut_c and cc_c else
                   'fair failure: the careful driver avoids it' if sut_c else 'no SUT contact')
        real = 'realism: not scorable' if exc is None else f'realism: {exc:.1f} IQR beyond natural p1–p99'
        axes[i, 0].set_title(f'{title}\n{verdict}; {real}', color=INK, fontsize=9, loc='left')
    for ax in axes[1]:
        ax.set_xlabel('time (s)', color=INK2, fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=7.5, loc='lower left', labelcolor=INK)
    fig.suptitle(f'Same highD cut-in (exposed case {case_id}, event {case["event"]["event_id"]}): two attacks',
                 color=INK, fontsize=10, x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    plot(int(sys.argv[1]), sys.argv[2], sys.argv[3])
