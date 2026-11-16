"""B27 analysis (B27 design, written before training, available on request): adversarial fine-tuning of a learned SUT with fair vs
collision-seeking attackers. Variants: base (1M) and none / fair / collide (+500k). Development cases only.

Inputs in DIR (run_b27_eval.sh):
  grid_<V>.jsonl                 drl_check rows (SUT and CC per case x grid plan): contact, valid, v_ratio, harsh, min_acc
  fair_s{2,3}_<V>.jsonl          held-out fair attacks: our policy seeds 2-3 with the window shield + CC margin 0.5 m
  collide_s{2,3}_<V>.jsonl       held-out collision-seeking attacks: collide_commit seeds 2-3
  refine_<V>.jsonl               robust refinement bo_warm_g4 (budget 36)
  margin_fair.jsonl              G4 of the fair failures of the fair_s* files
Metrics (per case, then averaged):
  E-a fair-failure rate and robust rate under the held-out fair attacks (seed mean); refinement robust found <= 5 / <= 36
  E-b collision rate under the collision-seeking attacks and the attacker-at-fault share of those collisions
  E-c benign cut-ins = grid plans the CC avoids with min acceleration > -2 m/s^2: SUT harsh-braking share (<= -4),
      final/initial speed, SUT collision rate
  E-d grid fair-failure cells (SUT contact, CC avoids) and cases with any
Hypotheses: H-U1 fair < none and fair <= collide on E-a; H-U2 collide > fair on E-c harsh braking (or speed loss).
Paired case bootstrap, 10,000 resamples, seed 20260927.

  python -m fairtest.b27_analysis DIR OUT.json [--key p2]
  python -m fairtest.b27_analysis B27B_DIR OUT.json --key 0 --competent --compare B27_DIR --compare-key p2   (B27b)
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

VARIANTS = ('base', 'none', 'fair', 'collide')
N_BOOT, SEED = 10_000, 20260927


def _rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def paired(a, b):
    cases = sorted(set(a) & set(b))
    diff = np.array([a[c] - b[c] for c in cases], float)
    idx = np.random.default_rng(SEED).integers(0, len(cases), (N_BOOT, len(cases)))
    m = diff[idx].mean(1)
    return dict(diff=float(diff.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], n=len(cases))


def main(a):
    d = Path(a.dir)
    sut = f'drl_sut@{a.key}_H0'
    margin = {}
    if (d / 'margin_fair.jsonl').exists():
        for r in _rows(d / 'margin_fair.jsonl'):
            margin[(r['source'], r['case_id'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= 3.5
                                                       and r['cc_clearance'] >= .5)
    per = {}
    for v in VARIANTS:
        m = collections.defaultdict(dict)
        # E-a held-out fair attacks
        fair, robust = collections.defaultdict(list), collections.defaultdict(list)
        for s in (2, 3):
            path = d / f'fair_s{s}_{v}.jsonl'
            for r in _rows(path):
                if 'valid' not in r:
                    continue
                f = bool(r['valid'] and r['contact'] and not r['refs']['cc_r157']['contact'])
                fair[r['case_id']].append(float(f))
                robust[r['case_id']].append(float(f and margin.get((path.name, r['case_id']), False)))
        m['fair'] = {c: float(np.mean(x)) for c, x in fair.items()}
        m['robust'] = {c: float(np.mean(x)) for c, x in robust.items()}
        ref = {r['case_id']: r['runs'].get(f'bo_warm_g4|{sut}', {}) for r in _rows(d / f'refine_{v}.jsonl')}
        m['refine5'] = {c: float(x.get('first_fair') is not None and x['first_fair'] <= 5) for c, x in ref.items()}
        m['refine36'] = {c: float(x.get('first_fair') is not None) for c, x in ref.items()}
        # E-b held-out collision-seeking attacks
        col, cols_unfair = collections.defaultdict(list), []
        for s in (2, 3):
            for r in _rows(d / f'collide_s{s}_{v}.jsonl'):
                if 'valid' not in r:
                    continue
                c = bool(r['valid'] and r['contact'])
                col[r['case_id']].append(float(c))
                if c:
                    cols_unfair.append(bool(r['refs']['cc_r157']['contact']))
        m['collision'] = {c: float(np.mean(x)) for c, x in col.items()}
        # E-c / E-d grid
        rows = _rows(d / f'grid_{v}.jsonl')
        cc = {(r['case_id'], r['plan']): r for r in rows if r['target'] == 'cc_r157_H0'}
        s_rows = [r for r in rows if r['target'] == sut and r['plan'] > 0]
        benign = collections.defaultdict(list)
        speed, crash_b, cells, anyfair = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(bool)
        for r in s_rows:
            k = (r['case_id'], r['plan'])
            ref_cc = cc[k]
            is_fair = bool(r['valid'] and r['contact'] and not ref_cc['contact'])
            cells[r['case_id']].append(float(is_fair))
            anyfair[r['case_id']] |= is_fair
            if not ref_cc['contact'] and ref_cc['min_acc'] > -2.:
                benign[r['case_id']].append(float(r['harsh']))
                speed[r['case_id']].append(float(r['v_ratio']))
                crash_b[r['case_id']].append(float(r['contact']))
        m['benign_harsh'] = {c: float(np.mean(x)) for c, x in benign.items()}
        m['benign_speed'] = {c: float(np.mean(x)) for c, x in speed.items()}
        m['benign_crash'] = {c: float(np.mean(x)) for c, x in crash_b.items()}
        m['grid_fair_cells'] = {c: float(np.mean(x)) for c, x in cells.items()}
        m['grid_fair_case'] = {c: float(x) for c, x in anyfair.items()}
        per[v] = dict(metrics=m, attacker_share=(sum(cols_unfair) / len(cols_unfair)) if cols_unfair else None,
                      collisions=len(cols_unfair), benign_plans=sum(len(x) for x in benign.values()))
    summary = {v: dict({k: float(np.mean(list(x.values()))) if x else None for k, x in per[v]['metrics'].items()},
                       attacker_share=per[v]['attacker_share'], collisions=per[v]['collisions'], benign_plans=per[v]['benign_plans'])
               for v in VARIANTS}
    tests = dict(
        U1_fair_vs_none=paired(per['fair']['metrics']['fair'], per['none']['metrics']['fair']),
        U1_fair_vs_collide=paired(per['fair']['metrics']['fair'], per['collide']['metrics']['fair']),
        U1_robust_fair_vs_none=paired(per['fair']['metrics']['robust'], per['none']['metrics']['robust']),
        U2_harsh_collide_vs_fair=paired(per['collide']['metrics']['benign_harsh'], per['fair']['metrics']['benign_harsh']),
        U2_speed_collide_vs_fair=paired(per['collide']['metrics']['benign_speed'], per['fair']['metrics']['benign_speed']),
        U3_collision_collide_vs_fair=paired(per['collide']['metrics']['collision'], per['fair']['metrics']['collision']))
    verdicts = dict(
        H_U1=tests['U1_fair_vs_none']['ci'][1] < 0. and not tests['U1_fair_vs_collide']['ci'][0] > 0.,
        H_U2=tests['U2_harsh_collide_vs_fair']['ci'][0] > 0. or tests['U2_speed_collide_vs_fair']['ci'][1] < 0.)
    rep = dict(sut=sut, summary=summary, tests=tests, verdicts=verdicts)
    if a.competent:  # B27b (DESIGN section 5): the competent-SUT regime
        cols = per['base']['collisions']
        tests['U5_speed_collide_vs_fair'] = tests['U2_speed_collide_vs_fair']
        verdicts.update(
            H_U4=(per['base']['attacker_share'] >= .8) if cols >= 10 else None,
            H_U5=tests['U5_speed_collide_vs_fair']['ci'][1] < 0.,
            H_U6=all(summary[v]['fair'] <= .05 for v in VARIANTS))
    if a.compare:  # H-U7: speed cost of collision-seeking vs fair training, this regime minus the other regime
        other = _speed_delta(Path(a.compare), f'drl_sut@{a.compare_key}_H0')
        mine = _speed_delta(d, sut)
        diff = {c: mine[c] - other[c] for c in set(mine) & set(other)}
        t = paired(diff, {c: 0. for c in diff})
        tests['U7_speed_cost_this_minus_other'] = t
        verdicts['H_U7'] = t['ci'][1] < 0.
    Path(a.out).write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))


def _speed_delta(d, sut):
    """Per case: benign-cut-in final/initial speed of the collide-trained minus the fair-trained SUT."""
    out = {}
    for v in ('collide', 'fair'):
        rows = _rows(d / f'grid_{v}.jsonl')
        cc = {(r['case_id'], r['plan']): r for r in rows if r['target'] == 'cc_r157_H0'}
        sp = collections.defaultdict(list)
        for r in rows:
            if r['target'] == sut and r['plan'] > 0:
                ref = cc[(r['case_id'], r['plan'])]
                if not ref['contact'] and ref['min_acc'] > -2.:
                    sp[r['case_id']].append(float(r['v_ratio']))
        out[v] = {c: float(np.mean(x)) for c, x in sp.items()}
    return {c: out['collide'][c] - out['fair'][c] for c in set(out['collide']) & set(out['fair'])}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('out')
    ap.add_argument('--key', default='p2')
    ap.add_argument('--competent', action='store_true', help='B27b hypotheses H-U4..H-U6')
    ap.add_argument('--compare', help='directory of the other regime (H-U7)')
    ap.add_argument('--compare-key', default='p2')
    main(ap.parse_args())
