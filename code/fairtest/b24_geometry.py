"""B24 (L1): fair-failure and collision rates of the SUT ladder against the lateral-clearance geometry.

Vehicle width w on 4 m lanes gives body clearance 4 - w between lane-centred vehicles: 2.0 (highway-env default,
~ highD median 1.92), 1.75, 1.5, 1.34 m (CARLA Town04 with the Tesla box). Grid landscapes as B7/B10 (exposed
highD 266, 199 plans + B5); the default-width rows are the stored B7/B10 landscapes.

  python -m fairtest.b24_geometry summarize W OUT.json [--grid GRID.jsonl]   (server: W = 2.25 / 2.5 / 2.66)
  python -m fairtest.b24_geometry summarize 2.0 OUT.json                      (local: stored B7/B10 landscapes)
  python -m fairtest.b24_geometry report OUT.json OUT.png SUMMARY.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUT_KEYS = ('following_H0', 'reference_idm_H0', 'following_H0.5', 'following_H1', 'following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0')


def default_rows():
    from fairtest.b22_authority import slim, stream
    from fairtest.report_b7 import rows_all
    from fairtest.strict_analysis import J
    rows, _ = rows_all()
    rows = [r if r['target'] == 'cc_r157' else slim(r) for r in rows if r['target'] in ('cc_r157', 'following', 'reference_idm', 'following_aeb')]
    return rows + list(stream([J / 'b10v2-a' / 'landscape.jsonl', J / 'b10v2-b' / 'landscape.jsonl'], lambda t: t == 'rss_sut'))


def collision_oracle(rows):
    """Share of cases in which some valid grid plan makes the SUT collide (the collision-seeking grid oracle)."""
    by = {}
    for r in rows:
        name = r['policy']['name'] if isinstance(r['policy'], dict) else r['policy']
        if r['target'] != 'cc_r157' and str(name).startswith('plan_'):
            case = by.setdefault(f"{r['target']}_H{r['H']:g}", {})
            case[r['case_id']] = case.get(r['case_id'], False) or bool(r['valid'] and r['contact'])
    return {k: float(np.mean(list(v.values()))) for k, v in by.items()}


def summarize(w, out, grid_path=None):
    from fairtest import b16_analysis as B
    from fairtest.b22_authority import attack_space, stream, summary
    from fairtest.runner import _STATE, _init
    from fairtest.strict_analysis import SUT7
    _init()
    cases = dict(_STATE['cases'])
    rows = default_rows() if grid_path is None else list(stream([Path(grid_path)]))
    grid = B.grid(rows, cases, {}, SUT7, ('cc_r157',))['cc_r157']
    rep = dict(width=w, clearance=round(4. - w, 2), n_rows=len(rows), attack_space=attack_space(rows), grid=grid,
               collision_oracle=collision_oracle(rows), validity=summary(dict(cc_r157=grid)))
    Path(out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(f"== width {w:g} (clearance {4 - w:.2f} m) rows {len(rows)} valid cut-in plans {100 * rep['attack_space']:.1f}%")
    print('   fair oracle', {k: round(100 * grid[k]['oracle'], 1) for k in SUT_KEYS if k in grid})
    print('   collision oracle', {k: round(100 * rep['collision_oracle'][k], 1) for k in SUT_KEYS if k in rep['collision_oracle']})
    print('   B5 fair/unfair', {k: (round(100 * grid[k]['b5_fair'], 1), round(100 * grid[k]['b5_unfair'], 1)) for k in SUT_KEYS if k in grid})
    print('   boundary plan', {k: round(100 * grid[k]['boundary'], 1) for k in SUT_KEYS if k in grid})


def report(out, png, paths):
    reps = sorted((json.loads(Path(p).read_text(encoding='utf-8')) for p in paths), key=lambda r: -r['clearance'])
    Path(out).write_text(json.dumps({f"{r['width']:g}": r for r in reps}, indent=1), encoding='utf-8')
    plot(reps, png)


def plot(reps, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7')
    LABEL = {'following_H0': 'following H0', 'following_H0.5': 'following H0.5', 'following_H1': 'following H1',
             'following_aeb_H0': 'AEB H0', 'following_aeb_H0.5': 'AEB H0.5', 'rss_sut_H0': 'longitudinal RSS'}
    INK, INK2, GRID, SURFACE, MUTED = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb', '#8a8983'
    xs = [r['clearance'] for r in reps]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, sharey=True)
    for c, k in zip(COLORS, LABEL):
        ax.plot(xs, [100 * r['grid'][k]['oracle'] for r in reps], color=c, marker='o', markersize=6, linewidth=2, label=LABEL[k])
        bx.plot(xs, [100 * r['collision_oracle'][k] for r in reps], color=c, marker='o', markersize=6, linewidth=2, label=LABEL[k])
    for x in (ax, bx):
        x.axvspan(1.41, 2.24, color=GRID, alpha=.6, linewidth=0)
        x.axvline(1.92, color=MUTED, linewidth=1, linestyle=':')
        x.text(1.935, 99, 'highD median', fontsize=7, color=INK2, va='top')
        x.axvline(1.34, color=MUTED, linewidth=1, linestyle='--')
        x.text(1.355, 99, 'CARLA Town04\n(default car)', fontsize=7, color=INK2, va='top')
        x.text(2.235, 1, 'shaded: highD p10-p90', fontsize=7, color=INK2, ha='right', va='bottom')
        x.set_xlabel('body clearance between lane-centred vehicles (m)', fontsize=8, color=INK2)
        x.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            x.spines[side].set_visible(False)
        x.grid(color=GRID, linewidth=.7)
        x.tick_params(colors=INK2, labelsize=8)
        x.set_ylim(-3, 103)
    ax.set_ylabel('% of cases', fontsize=8, color=INK2)
    ax.set_title('(a) Fair failures (grid oracle)', fontsize=10, color=INK, loc='left')
    bx.set_title('(b) Collisions (collision-seeking grid oracle)', fontsize=10, color=INK, loc='left')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, labelcolor=INK, loc='lower center', ncol=6)
    fig.subplots_adjust(left=.06, right=.99, top=.9, bottom=.22, wspace=.08)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('summarize', 'report'))
    ap.add_argument('args', nargs='+')
    ap.add_argument('--grid')
    a = ap.parse_args()
    if a.command == 'summarize':
        summarize(float(a.args[0]), a.args[1], a.grid)
    else:
        report(a.args[0], a.args[1], a.args[2:])
