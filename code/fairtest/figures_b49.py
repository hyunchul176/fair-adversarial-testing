"""B49 figure: v2j agent vs the training-free boundary-start BO (bo_warm_g4j alone), both under the one v2j success
contract, as the share of development cases with a reported failure within k SUT queries (seed-averaged per case).

  python -m fairtest.figures_b49 OUT.png --jobs fairtest_jobs B49_s1.jsonl B49_s2.jsonl B49_s3.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GROUPS = (('weak rule-based SUTs (4)', ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')),
          ('BC human SUT', ('bc_human_H0',)),
          ('learned SUTs (6)', ('drl_sut_H0', 'drl_sut@3_H0', 'drl_sut@6_H0', 'drl_sut@p1.5_H0', 'drl_sut@p2_H0', 'drl_sut@p2.5_H0')))
KS = list(range(1, 38))


def _rows(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def curves(calls, bo, suts):
    cases = sorted({c for (_, c, s) in calls if s in suts})
    agent, base = [], []
    for k in KS:
        agent.append(100 * np.mean([np.mean([calls[(x, c, s)]['found'] and calls[(x, c, s)]['queries'] <= k
                                              for x in ('s1', 's2', 's3') if (x, c, s) in calls]) for c in cases for s in suts]))
        base.append(100 * np.mean([bo.get((c, s)) is not None and bo[(c, s)] <= k for c in cases for s in suts]) if k <= 36 else np.nan)
    return agent, base


def main(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    calls = {}
    for path in a.calls:
        seed = 's' + Path(path).stem.split('_s')[-1][0]
        for r in _rows(path):
            calls[(seed, r['case_id'], r['sut'])] = r
    bo = {}
    for name in ('joint_dev_rule.jsonl', 'joint_dev_learned.jsonl'):
        for r in _rows(Path(a.jobs) / 'b46' / name):
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == 'bo_warm_g4j' and 'error' not in rec:
                    bo[(r['case_id'], s)] = rec.get('first_fair')
    AGENT, BO = '#2a78d6', '#1baf7a'  # the E3 figure's hues: our agent, GP-BO from the R157-boundary plan
    INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, (title, suts) in zip(axes, GROUPS):
        ag, bs = curves(calls, bo, suts)
        ax.step(KS, ag, where='post', color=AGENT, linewidth=2.2, label='v2j agent: shielded policy (1 query) + guarded GP-BO')
        ax.step(KS[:36], bs[:36], where='post', color=BO, linewidth=1.6, label='training-free GP-BO from the R157-boundary plan')
        close = abs(ag[-1] - bs[35]) < 6  # keep the two end labels apart
        for y, c, dy in ((ag[-1], AGENT, 5 if close else 0), (bs[35], BO, -5 if close else 0)):
            ax.annotate(f'{y:.0f}%', xy=(37 if c == AGENT else 36, y), xytext=(4, dy), textcoords='offset points',
                        fontsize=8, color=INK, va='center')
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 37], ['1', '2', '5', '10', '20', '37'])
        ax.set_xlim(.9, 48)
        ax.set_ylim(0, 100)
        ax.set_xlabel('SUT queries (log)', fontsize=8, color=INK2)
        ax.set_ylabel('cases with a reported failure, %', fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc='left')
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(color=GRID, linewidth=.7)
        ax.tick_params(colors=INK2, labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc='lower center', ncol=2)
    fig.suptitle('Same success contract (robust fair failure + G1 + G2 + natural joint support); competent SUTs: 0% for both',
                 fontsize=9, color=INK2, x=.05, ha='left')
    fig.subplots_adjust(left=.05, right=.98, top=.86, bottom=.22, wspace=.2)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    for title, suts in GROUPS:
        ag, bs = curves(calls, bo, suts)
        print(title, 'agent k1/k6/k37', [round(ag[k - 1], 1) for k in (1, 6, 37)], 'BO k1/k6/k36', [round(bs[k - 1], 1) for k in (1, 6, 36)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('calls', nargs='+')
    main(ap.parse_args())
