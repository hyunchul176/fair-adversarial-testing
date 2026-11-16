"""B19 analysis: does the choice of test signal change the design decision?

For the perception-threshold ADS family ads@d_p (highD exposed development cases) collect
  fair-testing signals: fair adversary = grid oracle OR cont_search (any fair failure found), the
                        guarded RL fair adversary (1 SUT query), and the R157 boundary planner;
  collision-seeking signals: generic falsification collision rate, committed collision-seeking RL, B5;
  natural cost: phantom brakes per vehicle-hour (b19_natural_alarms.json);
  mechanism of every grid fair failure (re-run with traces): 'late' = the SUT's front hits the cut-in
  vehicle (generator centre >= 3.5 m ahead), 'over' = side/rear contact while the SUT brakes
  (<= -1 m/s^2 in the last second: it braked for a vehicle alongside and was struck), 'side' = other.
Design rule: d_p* = argmin T(d_p) + lambda * phantom(d_p) for a range of lambda (collisions per phantom
brake per hour); ties go to the larger d_p (fewer phantom brakes). Writes a JSON summary and a figure.

  python -m fairtest.b19_analysis mechanism OUT.jsonl --workers W
  python -m fairtest.b19_analysis analyze OUT.json OUT.png [--mechanism M.jsonl]
  python -m fairtest.b19_analysis plot OUT.json OUT.png   (figure only, from a stored analysis)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from fairtest import b9_analysis as A
import fairtest.boundary_planner as BP
from fairtest.strict_analysis import J

DPS = ('0.5', '0.7', '0.9', '1.095', '1.3', '1.6', '2.0')  # target names as run (ads@2.0, not ads@2)
LAMBDAS = (0., .001, .01, .03, .1, .3, 1., 3.)
FRONT_DX_M = 3.5
REALISTIC_EXC = 1.  # realism budget (IQR beyond natural p1..p99), as in B14


def name(dp):
    return f'ads@{dp}'


def key(dp):
    return f'{name(dp)}_H0'


SUT_FIELDS = ('case_id', 'policy', 'target', 'H', 'valid', 'contact', 'ttc_event', 'initial_event', 'outcome')


def _stream(path, keep, slim=False):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if 'error' not in r and keep(r['target']):
                    yield {k: r[k] for k in SUT_FIELDS} if slim else r


def _rows(refs=False):
    """CC rows (+ time-based CC, FSM, RSS rows when refs) and the ADS grid; ADS rows keep only outcome fields."""
    from fairtest.report_b7 import rows_all
    rows, _ = rows_all()
    rows = [r for r in rows if r['target'] == 'cc_r157']
    if refs:
        for path in (J / 'b10v2-a' / 'landscape.jsonl', J / 'b10v2-b' / 'landscape.jsonl'):
            rows += _stream(path, lambda t: t in ('ref_fsm', 'ref_rss'))
        rows += _stream(J / 'b21' / 'cctime_exposed.jsonl', lambda t: t == 'cc_r157_time')
    return rows + list(_stream(J / 'b19' / 'b19_grid.jsonl', lambda t: t.startswith('ads@'), slim=True))


REFS = ('cc_r157', 'cc_r157_time', 'ref_fsm', 'ref_rss')


def consensus_oracle(by, sut, ids):
    """Case is a fair failure if one grid plan makes the SUT collide while every reference avoids (with a cut-in)."""
    fair = dict.fromkeys(ids, False)
    for (c, p), v in by.items():
        if c not in fair or fair[c] or not p.startswith('plan_') or p == BP.NOMINAL:
            continue
        s = v.get(sut)
        if s is None or not (s['valid'] and s['contact']):
            continue
        rr = [v.get((ref, 0.)) for ref in REFS]
        fair[c] = all(r is not None and r['valid'] and not r['contact'] and r['features'].get('cut_in') for r in rr)
    return float(np.mean(list(fair.values())))


def fair_plans(rows):
    """Grid plans where ads@d_p collides (valid) and the R157 CC does not."""
    pname = lambda r: r['policy']['name'] if isinstance(r['policy'], dict) else r['policy']
    cc = {(r['case_id'], pname(r)): r for r in rows if r['target'] == 'cc_r157'}
    out = []
    for r in rows:
        if r['target'].startswith('ads@') and r['valid'] and r['contact']:
            c = cc.get((r['case_id'], pname(r)))
            if c is not None and not c['contact']:
                out.append(dict(case_id=r['case_id'], plan=pname(r), target=r['target']))
    return out


def _mech(job):
    from fairtest import runner
    runner._init()
    specs = {runner.make_policy(s).name: s for s in runner.policy_specs('all')}
    spec = specs.get(job['plan'], 'b5')
    row = runner.run_episode(runner._STATE['cases'][job['case_id']], runner.make_policy(spec), job['target'], 0., keep_trace=True)
    tr = row['trace']
    g, t = tr[-1]['generator'], tr[-1]['target']
    dx = g['x'] - t['x']
    braking = min(s['target_control']['applied_mps2'] for s in tr[-11:])
    rec = [s['target_control'].get('recognition', {}) for s in tr]
    onset = next((k * .1 for k, x in enumerate(rec) if x.get('cc_phase') not in (None, 'free')), None)
    mech = 'late' if dx >= FRONT_DX_M else 'over' if braking <= -1. else 'side'
    return dict(job, contact=bool(row['pair_contact']), dx=float(dx), dy=float(g['y'] - t['y']), min_acc_last_1s=float(braking),
                phase_at_contact=rec[-1].get('cc_phase'), perceived_s=onset, mechanism=mech)


def mechanism(a):
    from fairtest.runner import _init
    _init()
    jobs = fair_plans(_rows())
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(_mech, jobs, chunksize=2):
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(finished=len(jobs), per_target=collections.Counter(j['target'] for j in jobs))))


def analyze(a):
    from fairtest.runner import _STATE, _init
    _init()
    cases = dict(_STATE['cases'])
    rows = _rows(refs=True)
    suts = tuple((name(dp), 0.) for dp in DPS)
    A.SUTS, BP.SUTS = suts, suts
    per = A.per_case_outcomes(rows, cases, {}, 'cc_r157', math.inf)
    per_ref = {ref: A.per_case_outcomes(rows, cases, {}, ref, math.inf) for ref in REFS[1:]}
    per_real = A.per_case_outcomes(rows, cases, {}, 'cc_r157', REALISTIC_EXC)  # B14 convention: cut-in with exceedance <= 1 IQR
    by = BP.index(rows)
    cont = {json.loads(l)['case_id']: json.loads(l) for l in open(J / 'b19' / 'b19_cont.jsonl', encoding='utf-8')}
    fal = {json.loads(l)['case_id']: json.loads(l) for l in open(J / 'b19' / 'b19_falsify.jsonl', encoding='utf-8')}
    rl_f = [json.loads(l) for l in open(J / 'b19' / 'b19_rl_fair.jsonl', encoding='utf-8')]
    rl_c = [json.loads(l) for l in open(J / 'b19' / 'b19_rl_collide.jsonl', encoding='utf-8')]
    alarms = json.loads((J / 'b19_natural_alarms.json').read_text(encoding='utf-8'))['phantom_brakes']
    mech = collections.defaultdict(list)
    if a.mechanism:
        for line in open(a.mechanism, encoding='utf-8'):
            r = json.loads(line)
            if r['contact']:
                mech[r['target']].append(r)
    rep = {}
    for dp in DPS:
        k = key(dp)
        pc = per[k]
        ids = sorted(c for c, m in pc.items() if m is not None)
        oracle = np.array([pc[c]['oracle'][0] for c in ids])
        boundary = np.array([pc[c]['boundary'][0] for c in ids])
        cs = np.array([float(bool((cont[c]['sut_search'][k].get('best') or {}).get('fair'))) for c in ids])
        fair_any = np.maximum(oracle, cs)
        b5 = [by[(c, 'b5_braking_entry')].get((name(dp), 0.)) for c in ids]
        b5_coll = float(np.mean([float(r['valid'] and r['contact']) for r in b5 if r is not None]))
        f_coll = float(np.mean([float(bool((fal[c]['runs'][f'falsify_centre|{k}']['best'] or {}).get('contact'))) for c in ids]))
        rf = [r for r in rl_f if r['sut'] == k]
        rc = [r for r in rl_c if r['sut'] == k]
        rl_fair = float(np.mean([float(r['valid'] and r['contact'] and not r['refs']['cc_r157']['contact']) for r in rf])) if rf else None
        rl_coll = float(np.mean([float(r['valid'] and r['contact']) for r in rc])) if rc else None
        m = mech.get(name(dp), [])
        kinds = collections.Counter(r['mechanism'] for r in m)
        case_kind = {kind: len({r['case_id'] for r in m if r['mechanism'] == kind}) / len(ids) for kind in ('late', 'over', 'side')}
        ref_oracle = {f'oracle_{ref}': float(np.mean([per_ref[ref][k][c]['oracle'][0] for c in ids if per_ref[ref][k].get(c)])) for ref in REFS[1:]}
        rep[dp] = dict(n=len(ids), fair_adversary=float(fair_any.mean()), grid_oracle=float(oracle.mean()), boundary=float(boundary.mean()),
                       cont_search=float(cs.mean()), rl_fair=rl_fair, falsify_collision=f_coll, rl_collide_collision=rl_coll,
                       b5_collision=b5_coll, phantom_per_hour=alarms[f'{float(dp):g}']['per_vehicle_hour'],
                       **ref_oracle, oracle_consensus=consensus_oracle(by, (name(dp), 0.), ids),
                       oracle_realistic=float(np.mean([per_real[k][c]['oracle'][0] for c in ids if per_real[k].get(c)])),
                       fair_plans=len(m), fair_plans_by_mechanism=dict(kinds), cases_by_mechanism=case_kind)
    decisions = {}
    for sig in ('fair_adversary', 'grid_oracle', *(f'oracle_{ref}' for ref in REFS[1:]), 'oracle_consensus', 'oracle_realistic', 'rl_fair',
                'falsify_collision', 'rl_collide_collision', 'b5_collision'):
        decisions[sig] = {}
        for lam in LAMBDAS:
            cost = {dp: rep[dp][sig] + lam * rep[dp]['phantom_per_hour'] for dp in rep if rep[dp][sig] is not None}
            best = min(cost, key=lambda d: (cost[d], -float(d)))
            tied = sorted((d for d in cost if cost[d] <= cost[best] + 1e-12), key=float)
            decisions[sig][f'{lam:g}'] = dict(d_p=float(best), optimal_set=[float(d) for d in tied], fair_failure=rep[best]['fair_adversary'],
                                             phantom_per_hour=rep[best]['phantom_per_hour'])
    out = dict(per_threshold=rep, decisions=decisions)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    for dp, v in rep.items():
        print(f"d_p {dp:>5}  fair adv {100*v['fair_adversary']:5.1f} (grid {100*v['grid_oracle']:5.1f}, mech {v['fair_plans_by_mechanism']})  rl_fair {100*(v['rl_fair'] or 0):5.1f}"
              f"  | falsify coll {100*v['falsify_collision']:5.1f}  rl_collide {100*(v['rl_collide_collision'] or 0):5.1f}  b5 {100*v['b5_collision']:5.1f}  | phantom/h {v['phantom_per_hour']:.3f}")
        print('        grid oracle by reference', {s: round(100 * v[s], 1) for s in ('grid_oracle', *(f'oracle_{ref}' for ref in REFS[1:]), 'oracle_consensus', 'oracle_realistic')})
    for sig, d in decisions.items():
        print(sig, {lam: (x['d_p'], round(100 * x['fair_failure'], 1)) for lam, x in d.items()})
    plot(rep, decisions, a.png)


def plot(rep, decisions, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # categorical slots of the validated reference palette, assigned in fixed order in reading order:
    # (a) 1 blue, 2 orange; (b) 3 aqua, 4 yellow, 5 magenta, 6 green, 7 violet (adjacent pairs validated;
    # aqua/yellow/magenta are below 3:1 on the surface, relieved by marker shapes, the legend and the RESULT table)
    S1, S2, S3, S4, S5, S6, S7 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7'
    MUTED, INK, INK2, GRID, SURFACE = '#8a8983', '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
    dps = [float(d) for d in rep]
    pct = lambda s: [100 * (rep[d][s] or 0) for d in rep]
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE)
    ax.plot(dps, pct('falsify_collision'), color=S1, marker='o', markersize=6, linewidth=2, label='generic falsification')
    ax.plot(dps, pct('b5_collision'), color=S2, marker='^', markersize=6, linewidth=2, label='rule-based B5')
    ax.set_ylabel('SUT collisions, % of cases', fontsize=8, color=INK2)
    ax.set_title('(a) Collision-seeking signals', fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2)
    for s, c, lab, mk in (('grid_oracle', S3, 'CC, fixed-distance perception', 'o'), ('oracle_cc_r157_time', S4, 'CC, R157 time-based perception', 's'),
                          ('oracle_ref_rss', S5, 'RSS reference', 'D'), ('oracle_ref_fsm', S6, 'FSM reference', 'v'),
                          ('oracle_consensus', S7, 'all four agree', 'o')):
        bx.plot(dps, pct(s), color=c, marker=mk, markersize=6 if s != 'oracle_consensus' else 5,
                linewidth=2 if s in ('grid_oracle', 'oracle_consensus') else 1.4, label=lab)
    bx.set_ylabel('fair failures (reference avoids), % of cases', fontsize=8, color=INK2)
    bx.set_title('(b) Fair-failure signal, by reference driver', fontsize=10, color=INK, loc='left')
    bx.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3)
    bx.text(.5, 50, '← over-reaction:\nbrakes for a vehicle\nalongside, is struck', fontsize=7, color=INK2, va='top')
    bx.text(1.42, 99, 'late perception →\nrear-ends the cut-in', fontsize=7, color=INK2, va='top')
    cx.semilogy(dps, [max(1e-3, rep[d]['phantom_per_hour']) for d in rep], color=MUTED, marker='o', markersize=6, linewidth=2)
    cx.set_ylabel('phantom brakes per vehicle-hour (highD; 0 shown at 0.001)', fontsize=8, color=INK2)
    cx.set_title('(c) Cost in natural traffic', fontsize=10, color=INK, loc='left')
    for x in (ax, bx, cx):
        x.axvline(1.095, color=MUTED, linewidth=1, linestyle=':')
        x.set_xlabel('ADS perception threshold d_p (m of lateral displacement)', fontsize=8, color=INK2)
        x.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            x.spines[side].set_visible(False)
        x.grid(color=GRID, linewidth=.7)
        x.tick_params(colors=INK2, labelsize=8)
    for x in (ax, bx):
        x.set_ylim(-3, 103)
    ax.text(1.11, 101, 'R157 CC\n1.095 m', fontsize=7, color=INK2, va='top')
    bx.text(1.11, 75, 'R157 CC\n1.095 m', fontsize=7, color=INK2, va='top')
    cx.text(1.11, 2.5, 'R157 CC\n1.095 m', fontsize=7, color=INK2, va='top')
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('mechanism')
    m.add_argument('out')
    m.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    q = sub.add_parser('analyze')
    q.add_argument('out')
    q.add_argument('png')
    q.add_argument('--mechanism')
    r = sub.add_parser('plot')
    r.add_argument('analysis')
    r.add_argument('png')
    a = ap.parse_args()
    if a.cmd == 'plot':
        d = json.loads(Path(a.analysis).read_text(encoding='utf-8'))
        plot(d['per_threshold'], d['decisions'], a.png)
    else:
        mechanism(a) if a.cmd == 'mechanism' else analyze(a)
