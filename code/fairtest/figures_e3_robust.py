"""E3 redrawn with the robust criterion (G4: front contact, CC clearance >= 0.5 m; B26): the share of cases with a robust
fair failure within k SUT queries, for the agent (shielded policy, 1 query, then bo_warm_g4) and the refinement
optimisers under the same G4 objective (GP-BO from the R157-boundary plan, cold-start GP-BO, random search).

  python -m fairtest.figures_e3_robust OUT.png --jobs fairtest_jobs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fairtest.b26_analysis import policy_stage, refine_stage

PANELS = (('following_H0', 'following H0 (weakest)'), ('following_H0.5', 'following H0.5'), ('following_H1', 'following H1 (mid)'))
ROWS = (('dev', 'development (62 cases)'), ('b9', 'B9 reserved highD (148 cases, post hoc)'))
KS = list(range(1, 38))
# the E3 figure's validated categorical slots, same entity -> same hue
COLOR = dict(agent='#2a78d6', bo_warm_g4='#1baf7a', bo_cold_g4='#e87ba4', random_g4='#008300')
LABEL = dict(agent='agent: shielded policy (1 query) + robust GP-BO from the boundary plan',
             bo_warm_g4='robust GP-BO, R157-boundary start', bo_cold_g4='robust GP-BO, cold start', random_g4='random search (robust)')


def curve(firsts):
    return [100 * np.mean([f is not None and f <= k for f in firsts]) if k <= 36 else np.nan for k in KS]


def main(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = Path(a.jobs) / 'b26'
    INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), facecolor=SURFACE)
    for row, (ds, rtitle) in zip(axes, ROWS):
        robust, _ = policy_stage(d, ds)
        seeds = sorted({k[0] for k in robust})
        opt = {m: refine_stage(d, ds, m, [str(d / f'e3g4_{{ds}}.jsonl')]) for m in ('bo_warm_g4', 'bo_cold_g4', 'random_g4')}
        for ax, (sut, title) in zip(row, PANELS):
            cases = sorted(c for (c, s) in opt['bo_warm_g4'] if s == sut)
            q = opt['bo_warm_g4']
            agent = [100 * np.mean([np.mean([robust.get((x, c, sut), False) or (k >= 2 and q[(c, sut)] is not None and 1 + q[(c, sut)] <= k)
                                             for x in seeds]) for c in cases]) for k in KS]
            ax.step(KS, agent, where='post', color=COLOR['agent'], linewidth=2.2, label=LABEL['agent'])
            for m in ('bo_warm_g4', 'bo_cold_g4', 'random_g4'):
                ax.step(KS, curve([opt[m].get((c, sut)) for c in cases]), where='post', color=COLOR[m], linewidth=1.5, label=LABEL[m])
            ax.set_xscale('log')
            ax.set_xticks([1, 2, 5, 10, 20, 37], ['1', '2', '5', '10', '20', '37'])
            ax.set_xlim(.9, 40)
            ax.set_ylim(0, 100)
            ax.set_xlabel('SUT queries (log)', fontsize=8, color=INK2)
            ax.set_ylabel('cases with a robust fair failure, %', fontsize=8, color=INK2)
            ax.set_title(f'{title} -- {rtitle}', fontsize=9.5, color=INK, loc='left')
            ax.set_facecolor(SURFACE)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.grid(color=GRID, linewidth=.7)
            ax.tick_params(colors=INK2, labelsize=8)
            print(ds, sut, 'agent k1/k6/k37', [round(agent[k - 1], 1) for k in (1, 6, 37)],
                  {m: [round(curve([opt[m].get((c, sut)) for c in cases])[k - 1], 1) for k in (1, 6, 36)] for m in opt})
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK, loc='lower center', ncol=2)
    fig.suptitle('E3 with the robust criterion (G4). The optimisers use the same robust objective and budget 36; the agent spends one extra query on its policy.',
                 fontsize=9, color=INK2, x=.04, ha='left')
    fig.subplots_adjust(left=.05, right=.98, top=.92, bottom=.14, wspace=.2, hspace=.35)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--jobs', required=True)
    main(ap.parse_args())
