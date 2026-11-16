"""B74 (B74 design, fixed before the run, available on request): the CARS-style
reward trained with lane-change commitment (G1), review of manuscript v7 item R3 (B66 enforced G1 only at test time).

Conditions (eval files {tag}_s{seed}.jsonl of `b17_rl eval --commit [--shield-window]`, development 62 cases, 17 SUTs,
policy seeds 1-3, one SUT query):
  carsTA    cars_commit from each seed's collide_commit 750k checkpoint, 1.5M steps (the agent's initialisation and budget)
  carsTAsh  the same policy behind the agent's window shield (the CC, 0.5 m)
  carsTB    cars_commit from scratch, 2.25M steps, 15 envs (the B23 CARS-style recipe with G1 during training)
  carsTBsh  the same behind the window shield
  ours      the agent's policy (fair_guarded2 s1-3) behind the window shield, re-run (implementation check I1)
References read from the B66 directory: carsG1 (the B23 policy with G1 enforced at test time only) and cars (no G1).
Verdicts by the CC (cc_r157): collision, fair, attacker-caused, feint (lane reversal after entry), robust fair (front contact
dx >= 3.5 m and CC clearance >= 0.5 m, b23_margin), G2 (CC-replay realism exceedance <= 1 with a complete cut-in), joint
(5-NN score of the CC replay <= tau, replay step), and the success contract (robust fair, no feint, G2 and joint).
  CB1   carsTA: P1 by the CC (robust fair, following H0 minus longitudinal RSS, one query) < 50 points   (judged)
  CB1b  carsTB: the same                                                                              (judged)
  CB2   attacker-caused share of the collisions with the weak-4 ladder SUTs and longitudinal RSS, pooled  (descriptive)
  CB3   the shielded arms: P1 (robust and contract), attacker-caused collisions                          (descriptive)
  CB4   realism of the fair failures (G2, joint, contract) and feints                                    (descriptive)
  I1    ours reproduces the stored policy stage (b26/final, identical action sequences)        (implementation check)
Rates are seed-averaged case shares (62 cases); intervals are case bootstraps (10,000 resamples, seed 74) of the
seed-averaged paired case values.

  python -m fairtest.b74_analysis replay OUT.jsonl DIR [--workers W]     (server: CC replays of the contacts)
  python -m fairtest.b74_analysis analyse OUT.json DIR --b66 B66_DIR [--stored-ours B26_FINAL_DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

TAGS = ('carsTA', 'carsTAsh', 'carsTB', 'carsTBsh', 'ours')
REF_TAGS = ('carsG1', 'cars')
WEAK4 = ('following_H0', 'following_H0.5', 'following_H1', 'reference_idm_H0')
H0, RSS = 'following_H0', 'rss_sut_H0'
DELTA, FRONT = .5, 3.5
THRESHOLD = .5  # CB1: P1 below 50 points
N_BOOT, SEED = 10_000, 74


def rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def feints(actions):
    """Lane reversals after the first entry (lane goal 4 -> 0), as b23_analysis.feints."""
    n, entered, last = 0, False, 0.
    for lane, _ in actions:
        if lane == 4.:
            entered = True
        elif entered and last == 4.:
            n += 1
        last = lane
    return n


def margins(path):
    out = {}
    if Path(path).exists():
        for r in rows(path):
            out[(r['source'], r['case_id'], r['sut'])] = bool(r['sut_contact'] and not r['cc_contact'] and r['contact_dx'] >= FRONT
                                                             and r['cc_clearance'] >= DELTA)
    return out


def _replay_task(job):
    from fairtest.b45_joint_realism import _task
    return _task(job)


def replay_main(a):
    d = Path(a.dir)
    jobs = []
    for tag in TAGS:
        for s in (1, 2, 3):
            p = d / f'{tag}_s{s}.jsonl'
            if not p.exists():
                continue
            for r in rows(p):
                if r.get('valid') and r.get('contact') and r.get('actions'):
                    jobs.append((tag, s, r))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(_replay_task, jobs, chunksize=4):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(replayed=len(jobs))))


def joint_inside(path):
    """{(tag, seed, case, sut): inside the joint support} from the replay step."""
    from fairtest.b45_joint_realism import Support
    sup, out = Support(), {}
    if Path(path).exists():
        for r in rows(path):
            sc = sup.score(r['features']) if r['cut_in'] else None
            out[(r['source'], r['seed'], r['case_id'], r['sut'])] = sc is not None and sc <= sup.tau
    return out


def condition(d, tag, margin, joint):
    """per[sut][metric][seed] = {case: 0/1}; metrics as in the docstring."""
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    seeds = []
    for s in (1, 2, 3):
        path = Path(d) / f'{tag}_s{s}.jsonl'
        if not path.exists():
            continue
        seeds.append(s)
        for r in rows(path):
            if 'valid' not in r:
                continue
            col = bool(r['valid'] and r['contact'])
            refs = r.get('refs') or {}
            judged = not col or 'contact' in refs.get('cc_r157', {})
            fair = bool(col and judged and not refs['cc_r157']['contact'])
            unfair = bool(col and judged and refs['cc_r157']['contact'])
            fe = feints(r.get('actions') or []) > 0
            rob = bool(fair and margin.get((path.name, r['case_id'], r['sut']), False))
            cc = r.get('cc_replay') or {}
            g2 = bool(cc.get('cut_in') and cc.get('exc') is not None and cc['exc'] <= 1.)
            jt = bool(joint.get((tag, s, r['case_id'], r['sut']), False))
            vals = dict(collision=col, fair=fair, robust_fair=rob, attacker_caused=unfair, unjudged=col and not judged,
                        fair_feint=fair and fe, fair_g2=fair and g2, fair_joint=fair and jt,
                        contract=rob and not fe and g2 and jt)
            for k, v in vals.items():
                per[r['sut']][k][s][r['case_id']] = bool(v)
    return per, seeds


def case_values(per, sut, metric, seeds):
    """Seed-averaged value per case ({case: share})."""
    cases = sorted(set().union(*[per[sut][metric][s].keys() for s in seeds])) if seeds else []
    return {c: float(np.mean([per[sut][metric][s].get(c, False) for s in seeds])) for c in cases}


def boot_ci(vals):
    vals = np.asarray(vals, float)
    if not len(vals):
        return None
    rng = np.random.default_rng(SEED)
    m = vals[rng.integers(0, len(vals), (N_BOOT, len(vals)))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def p1(per, seeds, metric):
    a, b = case_values(per, H0, metric, seeds), case_values(per, RSS, metric, seeds)
    cases = sorted(set(a) & set(b))
    diff = [a[c] - b[c] for c in cases]
    return dict(value=round(float(np.mean(diff)), 4) if diff else None, ci=boot_ci(diff), cases=len(cases))


def summarise(per, seeds):
    suts = {}
    for sut, mets in per.items():
        suts[sut] = {k: round(float(np.mean(list(case_values(per, sut, k, seeds).values()))), 4) for k in mets}
    pool = [s for s in WEAK4 + (RSS,) if s in per]
    cols = sum(sum(per[s]['collision'][x].values()) for s in pool for x in seeds)
    unf = sum(sum(per[s]['attacker_caused'][x].values()) for s in pool for x in seeds)
    return dict(seeds=seeds, suts=suts, P1_fair=p1(per, seeds, 'fair'), P1_robust=p1(per, seeds, 'robust_fair'),
                P1_contract=p1(per, seeds, 'contract'),
                pooled_weak4_rss=dict(collisions=cols, attacker_caused=unf, share=round(unf / cols, 4) if cols else None))


def check_ours(d, stored):
    """I1: action sequences of the re-run 'ours' rows against the stored policy stage (b26/final/final_s{s}_dev.jsonl)."""
    same = diff = missing = 0
    for s in (1, 2, 3):
        new = {(r['case_id'], r['sut']): r.get('actions') for r in rows(Path(d) / f'ours_s{s}.jsonl') if 'valid' in r}
        old = {(r['case_id'], r['sut']): r.get('actions') for r in rows(Path(stored) / f'final_s{s}_dev.jsonl') if 'valid' in r}
        for k, v in old.items():
            if k not in new:
                missing += 1
            elif new[k] == v:
                same += 1
            else:
                diff += 1
    return dict(identical=same, different=diff, missing=missing, passed=diff == 0 and missing == 0)


def analyse(a):
    d = Path(a.dir)
    margin, joint = margins(d / 'margin.jsonl'), joint_inside(d / 'replay.jsonl')
    rep = dict(conditions={})
    for tag in TAGS:
        per, seeds = condition(d, tag, margin, joint)
        if seeds:
            rep['conditions'][tag] = summarise(per, seeds)
    if a.b66:
        m66 = margins(Path(a.b66) / 'margin.jsonl')
        for tag in REF_TAGS:
            per, seeds = condition(a.b66, tag, m66, {})
            if seeds:
                c = summarise(per, seeds)
                c['note'] = 'B66 stored rows; joint realism not replayed (contract not defined)'
                rep['conditions']['b66_' + tag] = c
    for name, tag in (('CB1', 'carsTA'), ('CB1b', 'carsTB')):
        c = rep['conditions'].get(tag)
        if c:
            v = c['P1_robust']['value']
            rep[name] = dict(P1_robust=v, ci=c['P1_robust']['ci'], threshold=THRESHOLD, supported=v is not None and v < THRESHOLD)
    rep['CB2'] = {t: c['pooled_weak4_rss'] for t, c in rep['conditions'].items()}
    if a.stored_ours and 'ours' in rep['conditions']:
        rep['I1'] = check_ours(d, a.stored_ours)
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    for tag, c in rep['conditions'].items():
        s = c['suts']
        h, r = s.get(H0, {}), s.get(RSS, {})
        print(f"{tag:14s} seeds {c['seeds']} | H0 col {h.get('collision', 0) * 100:5.1f} fair {h.get('fair', 0) * 100:5.1f} "
              f"robust {h.get('robust_fair', 0) * 100:5.1f} | RSS col {r.get('collision', 0) * 100:5.1f} fair {r.get('fair', 0) * 100:5.1f} "
              f"| P1 robust {c['P1_robust']['value']} contract {c['P1_contract']['value']} | attacker-caused share "
              f"{c['pooled_weak4_rss']['share']}")
    print({k: rep[k] for k in ('CB1', 'CB1b', 'I1') if k in rep})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('replay')
    p.add_argument('out')
    p.add_argument('dir')
    p.add_argument('--workers', type=int, default=24)
    q = sub.add_parser('analyse')
    q.add_argument('out')
    q.add_argument('dir')
    q.add_argument('--b66')
    q.add_argument('--stored-ours')
    a = ap.parse_args()
    replay_main(a) if a.cmd == 'replay' else analyse(a)
