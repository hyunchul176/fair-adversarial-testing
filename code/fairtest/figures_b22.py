"""B22 figure: the weakest SUT's fair-failure rate against the attacker's relative deceleration headroom
(how much slower than the SUT the generator may become): highD and B9 under the floor intervention, NGSIM
as observed (headroom bins).

  python -m fairtest.figures_b22 B22_AUTHORITY_relative.json OUT.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from fairtest.strict_analysis import EV


def median_rel(cases):
    return float(np.median([math.hypot(c['physical_initial_state'][1]['vx'], c['physical_initial_state'][1]['vy']) - 15. for c in cases]))


def main(src, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fairtest.runner import _STATE, _init
    _init()
    S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
    INK, INK2, GRID, SURFACE, MUTED = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb', '#8a8983'
    d = json.loads(Path(src).read_text(encoding='utf-8'))
    b9 = json.loads((EV / 'b9_design' / 'reserved_cohort' / 'case_pool.json').read_text(encoding='utf-8'))['cases']
    k = 'following_H0'
    hx = [1.6, 3.0, median_rel(list(_STATE['cases'].values()))]
    hy = [100 * d[r]['grid'][k]['oracle'] for r in ('highd_exposed_rel1_6', 'highd_exposed_rel3_0', 'highd_exposed_unrestricted')]
    bx_ = [1.6, median_rel(b9)]
    by_ = [100 * d[r]['grid'][k]['oracle'] for r in ('highd_b9_rel1_6', 'highd_b9_unrestricted')]
    bins = d['ngsim']['by_margin'][k]
    mids = {'0-1': .5, '1-2': 1.5, '2-3': 2.5, '3-5': 4., '5-inf': 7.}
    ng = [(mids[b], 100 * x['fair'], x['n']) for b, x in bins.items() if x['n']]
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    ax.plot(hx, hy, color=S1, marker='o', markersize=7, linewidth=2, label='highD exposed (floor intervention)')
    ax.plot(bx_, by_, color=S2, marker='s', markersize=7, linewidth=2, label='highD B9 (floor intervention)')
    ax.plot([x for x, _, _ in ng], [y for _, y, _ in ng], color=S3, marker='D', markersize=7, linewidth=2, label='NGSIM as observed (headroom bins)')
    for x, y, n in ng:
        ax.annotate(f'n={n}', (x, y), textcoords='offset points', xytext=(-10, 9), fontsize=7, color=INK2)
    ng_all = 100 * d['ngsim']['grid'][k]['oracle']
    ax.axhline(ng_all, color=MUTED, linewidth=1, linestyle=':')
    ax.text(.42, ng_all + 2, f'NGSIM overall {ng_all:.0f}%', fontsize=7, color=INK2)
    ax.set_xscale('log')
    ax.set_xticks([.5, 1, 2, 5, 10, 20], ['0.5', '1', '2', '5', '10', '20'])
    ax.set_xlim(.4, 22)
    ax.set_ylim(-3, 100)
    ax.set_xlabel('relative deceleration headroom: SUT initial speed − generator speed floor (m/s, log)', fontsize=8, color=INK2)
    ax.set_ylabel('fair failures of the weakest SUT (following H0), % of cases', fontsize=8, color=INK2)
    ax.set_title('Attack authority sets the vulnerability magnitude across datasets', fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper left')
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=.7)
    ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
