"""B13 figure: generic falsification vs the fair adversary across SUT capability (post-hoc plotting only).

Per SUT, two adjacent bars with the same optimiser and SUT budget (CMA-ES, <= 36 queries):
  left   generic falsification (collision-seeking), split into fair failures (CC R157 would avoid,
         slot-1 blue) and attacker-at-fault collisions (CC would also crash, slot-2 orange);
  right  fair adversary (cont_search: fairness-constrained), fair failures only (blue).
Whiskers: 95% case-bootstrap CI of the bar total. Panels: highD and NGSIM US-101.

  python -m fairtest.figures_b13 OUT.png --panel "title|falsify1.jsonl,...|cont1.jsonl,..." [--panel ...]
"""
from __future__ import annotations

import argparse
import json
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, MUTED = '#2a78d6', '#eb6834', '#8a8983'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
SUTS = (('following_H0', 'following\nH0'), ('reference_idm_H0', 'IDM\nH0'), ('following_H0.5', 'following\nH0.5'),
        ('following_H1', 'following\nH1'), ('following_aeb_H0', 'AEB\nH0'), ('following_aeb_H0.5', 'AEB\nH0.5'),
        ('rss_sut_H0', 'longitudinal\nRSS'))


def ci(vals, seed=20260926, boot=4000):
    v = np.asarray(vals, float)
    rng = random.Random(seed)
    idx = np.array([[rng.randrange(len(v)) for _ in range(len(v))] for _ in range(boot)])
    d = v[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load_panel(falsify_paths, cont_paths, start='falsify_centre'):
    fal, cont = {}, {}
    for i, p in enumerate(falsify_paths):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if r.get('error'):
                raise RuntimeError(r['error'])
            for k, rec in r['runs'].items():
                s, sut = k.split('|')
                if s != start:
                    continue
                best = rec['best']
                hit = bool(best and best['contact'])
                unfair = hit and rec['refs']['cc_r157']['contact']
                fal.setdefault(sut, []).append((float(hit and not unfair), float(unfair)))
    for p in cont_paths:
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if r.get('error'):
                raise RuntimeError(r['error'])
            for sut, ss in r['sut_search'].items():
                cont.setdefault(sut, []).append(float(bool(ss.get('best') and ss['best']['fair'])))
    return fal, cont


def load_panel_b14(paths):
    """B14 rows: falsify_real hits split by CC outcome (fair / attacker-at-fault); fair_real hits (fair)."""
    fal, cont = {}, {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if r.get('error'):
                raise RuntimeError(r['error'])
            for k, rec in r['runs'].items():
                mode, sut = k.split('|')
                b = rec['best']
                hit = bool(b and b['hit'])
                if mode == 'falsify_real':
                    unfair = hit and rec['refs']['cc_r157']
                    fal.setdefault(sut, []).append((float(hit and not unfair), float(unfair)))
                else:
                    cont.setdefault(sut, []).append(float(hit))
    return fal, cont


def plot(panels, out, subtitle=None):
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.0), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    w = .36
    for ax, (title, fal, cont) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(axis='y', color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        n = None
        for i, (key, _) in enumerate(SUTS):
            f = np.array(fal[key])
            fair, unfair = 100 * f[:, 0].mean(), 100 * f[:, 1].mean()
            lo, hi = ci(f.sum(axis=1))
            x = i - w / 2 - .02
            ax.bar(x, fair, w, color=BLUE, edgecolor=SURFACE, linewidth=1)
            ax.bar(x, unfair, w, bottom=fair, color=ORANGE, edgecolor=SURFACE, linewidth=1)
            ax.plot([x, x], [100 * lo, 100 * hi], color=INK2, linewidth=1)
            ax.text(x, 100 * hi + 1.5, f'{fair + unfair:.0f}', ha='center', va='bottom', fontsize=7.5, color=INK)
            c = np.array(cont[key])
            lo, hi = ci(c)
            x = i + w / 2 + .02
            ax.bar(x, 100 * c.mean(), w, color=BLUE, edgecolor=SURFACE, linewidth=1)
            ax.plot([x, x], [100 * lo, 100 * hi], color=INK2, linewidth=1)
            ax.text(x, 100 * hi + 1.5, f'{100 * c.mean():.0f}', ha='center', va='bottom', fontsize=7.5, color=INK)
            n = len(f)
        ax.set_xticks(range(len(SUTS)), [s[1] for s in SUTS], fontsize=8, color=INK)
        for i in range(len(SUTS)):
            ax.text(i - w / 2 - .02, -2.5, 'F', ha='center', va='top', fontsize=7, color=INK2)
            ax.text(i + w / 2 + .02, -2.5, 'A', ha='center', va='top', fontsize=7, color=INK2)
        ax.tick_params(axis='x', pad=13, colors=INK2, length=0)
        ax.tick_params(axis='y', colors=INK2, labelsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f'{title}  (n = {n} cases)', color=INK, fontsize=10, loc='left')
        ax.annotate('', xy=(len(SUTS) - .6, -27), xytext=(-.4, -27), xycoords='data',
                    arrowprops=dict(arrowstyle='->', color=MUTED, linewidth=.8), annotation_clip=False)
        ax.text(len(SUTS) / 2 - .5, -29, 'SUT capability increases', ha='center', va='top', fontsize=7.5, color=INK2)
    axes[0].set_ylabel('% of cases', color=INK2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.text(.01, .975, subtitle or ('F = generic falsification (collision-seeking CMA-ES)    A = fair adversary (fairness-constrained CMA-ES)'
             '    same plan space and budget: at most 36 SUT queries per case'), fontsize=8, color=INK2, va='top')
    fig.legend(handles, ['fair failure (an R157 careful driver would avoid it)',
                         'attacker-at-fault collision (the careful driver also crashes)'],
               loc='upper left', bbox_to_anchor=(.005, .94), ncol=2, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, .02, 1, .87))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--panel', action='append', required=True, help='title|falsify1,falsify2|cont1,cont2  or  title|b14_1,b14_2 with --b14')
    ap.add_argument('--b14', action='store_true', help='panels are B14 realism-constrained runs')
    a = ap.parse_args()
    panels = []
    for spec in a.panel:
        if a.b14:
            title, bp = spec.split('|')
            fal, cont = load_panel_b14(bp.split(','))
        else:
            title, fp, cp = spec.split('|')
            fal, cont = load_panel(fp.split(','), cp.split(','))
        panels.append((title, fal, cont))
    sub = ('F = plausibility-constrained falsification (collisions within the naturalistic envelope)    A = fair and realistic adversary'
           '    both: realism <= 1 IQR, at most 36 SUT queries') if a.b14 else None
    plot(panels, a.out, sub)
