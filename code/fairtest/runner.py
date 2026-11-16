"""Local development runner for B7 (exposed partitions only, never the reserved test set).

Usage (from code/, pinned venv):
  python -m fairtest.runner validate-b5            # bit-exact replay of saved B5 dev16 runs
  python -m fairtest.runner landscape OUT.jsonl [--cases N] [--workers W]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from fairtest import devboot  # noqa: F401
from fairtest.cases import EVIDENCE, build_pool
from fairtest.env import B7Env
from fairtest.features import simulated_features
from fairtest.plans import FrozenB5, Plan, grid

CONTRACT = EVIDENCE / 'b5_execution' / 'data'
B5_RAW = (EVIDENCE / 'b5_execution' / 'download-recovery' / 'completed-return'
          / 'raw-36037592392' / 'campaign' / 'trajectories')
TARGET_PANEL = (('following', 0.), ('following', .5), ('reference_idm', 0.), ('cc_r157', 0.))
EXTENDED_PANEL = TARGET_PANEL + (('following', 1.),)

_STATE = {}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _init():
    if _STATE:
        return
    bundled = os.environ.get('FAIRTEST_DATA')
    if bundled:  # server mode: frozen contracts and case pool travel inside the bundle
        root = Path(bundled)
        _STATE['base'] = _read(root / 'base_protocol.json')
        _STATE['b1'] = _read(root / 'protocol_b1.json')
        _STATE['b2'] = _read(root / 'protocol_b2_action.json')
        cases = _read(root / 'case_pool.json')['cases']
    else:
        _STATE['base'] = _read(CONTRACT / 'base_protocol.json')
        _STATE['b1'] = _read(CONTRACT / 'protocol_b1.json')
        _STATE['b2'] = _read(CONTRACT / 'native_sentinels' / 'data' / 'protocol.json')  # B2 action contract
        cases, _ = build_pool()
    _STATE['cases'] = {c['case_id']: c for c in cases}


def make_policy(spec):
    if spec == 'b5':
        return FrozenB5()
    entry, pre, post, dur = spec
    return Plan(math.inf if entry is None else entry, pre, post, dur)


def run_episode(case, policy, target, h, keep_trace=False):
    _init()
    frozen = getattr(policy, 'frozen_planner', False)
    env = B7Env(_STATE['base'], case, _STATE['b1'], _STATE['b2'], case['physical_initial_state'],
                target=target, target_h=h, durations=policy.durations(), frozen_planner=frozen)
    try:
        env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()
        done = False
        while not done:
            _, _, done, _, _ = env.step(policy.act(env))
        row = env.result()
    finally:
        env.close()
    return row if keep_trace else compact(row, case, policy, target, h)


def compact(row, case, policy, target, h):
    decisions = row['decisions']
    entry = next((d['tick'] for d in decisions if d['lateral_plan']['applied_lane_goal'] == 4.), None)
    tc = [r['target_control'] for r in row['trace']]
    out = dict(case_id=case['case_id'], event_id=case['event']['event_id'], partition=case['event']['partition'],
               recording=case['event']['recording'], dev16=case['dev16'], policy=policy.to_json(),
               target=target, H=h, steps=len(row['trace']), outcome=row['agent_outcome'],
               contact=bool(row['pair_contact']), valid=bool(row['maneuver_valid_precontact']),
               initial_ttc=row['initial_ttc'], initial_event=bool(row['initial_risk_event']),
               min_ttc=row['min_pair_ttc'], ttc_event=bool(row['exclusive_ttc_event']),
               entry_tick=entry, target_min_acc=min(c['applied_mps2'] for c in tc),
               failed_checks=row['failed_maneuver_checks'], features=simulated_features(row))
    out.update(interaction_metrics(row))
    if target.startswith(('cc_r157', 'ref_', 'oracle_')):
        rec = [c['recognition'] for c in tc]
        out['cc_perceived_at'] = next((r['perceived_at_s'] for r in rec if r['perceived_at_s'] is not None), None)
        out['cc_emergency'] = any(r['emergency'] for r in rec)
        onset = next((k for k, r in enumerate(rec) if r['emergency']), None)
        out['cc_emergency_onset_s'] = None if onset is None else onset * .1
        out['cc_drac_at_onset'] = None if onset is None else _drac(row['trace'][onset - 1] if onset else None,
                                                                   row['physical_initial_state'])
    return out


def _drac(r, initial):
    g, t = (r['generator'], r['target']) if r is not None else (initial[0], initial[1])
    gap = g['x'] - t['x'] - 5.
    closing = t['vx'] - g['vx']
    return closing * closing / (2 * gap) if closing > 0 and gap > 0 else (math.inf if closing > 0 else 0.)


def interaction_metrics(row):
    """Longitudinal surrogate safety measures while the bodies overlap laterally."""
    min_gap, max_drac, onset = math.inf, 0., None
    for k, r in enumerate(row['trace']):
        g, t = r['generator'], r['target']
        if abs(g['y'] - t['y']) < (g['width'] + t['width']) / 2. and g['x'] > t['x']:  # bodies overlap laterally
            gap = g['x'] - t['x'] - 5.
            min_gap = min(min_gap, gap)
            closing = t['vx'] - g['vx']
            if closing > 0 and gap > 0:
                max_drac = max(max_drac, closing * closing / (2 * gap))
        if onset is None and r['target_control']['applied_mps2'] <= -1.:
            onset = k * .1
    return dict(min_gap_overlap=None if math.isinf(min_gap) else min_gap, max_drac=max_drac,
                target_brake_onset_s=onset)


def _job(args):
    case_id, spec, target, h = args
    _init()
    try:
        return run_episode(_STATE['cases'][case_id], make_policy(spec), target, h)
    except Exception as e:  # recorded, never silently dropped
        return dict(case_id=case_id, policy=spec, target=target, H=h, error=f'{type(e).__name__}: {e}')


STATE_FIELDS = ('x', 'y', 'vx', 'vy', 'heading')


def reference_states(trace):
    return [[r[k][f] for k in ('generator', 'target') for f in STATE_FIELDS] for r in trace]


def b5_reference():
    """Saved server B5 trajectories (local) or their bundled state extract (server mode)."""
    bundled = os.environ.get('FAIRTEST_DATA')
    if bundled:
        return json.loads(gzip.decompress((Path(bundled) / 'b5_reference_states.json.gz').read_bytes()))
    inputs = _read(CONTRACT / 'protocol_b1.json')['inputs']
    ref = {}
    for inp in inputs:
        for hi in (0, 1):
            condition = 2 * inp['input_id'] + hi
            saved = json.load(gzip.open(B5_RAW / f'evaluation_braking_entry_0_fixed_{condition + 1:06d}.json.gz',
                                        'rt', encoding='utf-8'))
            ref[str(condition)] = dict(event_id=inp['event_id'], H=(0., .5)[hi], contact=saved['pair_contact'],
                                       states=reference_states(saved['trace']))
    return ref


def validate_b5(tolerance=1e-9):
    """Replay frozen B5 on dev16 x H0/H.5 and compare with the saved server run (B5, a lab GPU runner)."""
    _init()
    by_event = {c['event']['event_id']: c for c in _STATE['cases'].values()}
    worst, n = 0., 0
    for condition, ref in sorted(b5_reference().items(), key=lambda x: int(x[0])):
        row = run_episode(by_event[ref['event_id']], FrozenB5(), 'following', ref['H'], keep_trace=True)
        states = reference_states(row['trace'])
        if len(states) != len(ref['states']) or bool(row['pair_contact']) != bool(ref['contact']):
            raise AssertionError(f'condition {condition}: length/contact mismatch')
        d = max(abs(a - b) for sa, sb in zip(states, ref['states']) for a, b in zip(sa, sb))
        worst, n = max(worst, d), n + 1
    print(json.dumps(dict(validated_conditions=n, worst_abs_state_difference=worst, tolerance=tolerance)))
    if worst > tolerance:
        raise AssertionError(f'B5 replay drift {worst} exceeds {tolerance}')
    return worst


def landscape(out, n_cases=None, workers=None, policies=None, panel=TARGET_PANEL, case_filter=None):
    _init()
    ids = sorted(_STATE['cases'])
    if case_filter:
        ids = [i for i in ids if case_filter(_STATE['cases'][i])]
    if n_cases:
        ids = ids[:n_cases]
    specs = policies or (['b5'] + [(None if math.isinf(p.entry_s) else p.entry_s, p.pre_mps2, p.post_mps2,
                                     p.duration_s) for p in grid()])
    jobs = [(c, s, t, h) for c in ids for s in specs for (t, h) in panel]
    done = set()
    out = Path(out)
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            key = (r['case_id'], json.dumps(r['policy'] if 'error' in r else _spec_of(r['policy'])), r['target'], r['H'])
            done.add(key)
    todo = [j for j in jobs if (j[0], json.dumps(j[1]), j[2], j[3]) not in done]
    print(json.dumps(dict(cases=len(ids), policies=len(specs), targets=len(panel), jobs=len(jobs),
                          remaining=len(todo))), flush=True)
    t0 = time.time()
    with mp.Pool(workers or max(1, os.cpu_count() - 2), initializer=_init) as pool, out.open('a', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(_job, todo, chunksize=8)):
            f.write(json.dumps(r) + '\n')
            if (i + 1) % 2000 == 0:
                f.flush()
                print(json.dumps(dict(done=i + 1, of=len(todo), elapsed_s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(todo), elapsed_s=round(time.time() - t0, 1))), flush=True)


def _spec_of(policy_json):
    if policy_json.get('name') == 'b5_braking_entry':
        return 'b5'
    return (policy_json['entry_s'], policy_json['pre_mps2'], policy_json['post_mps2'], policy_json['duration_s'])


def parse_targets(text):
    out = []
    for item in text.split(','):
        name, h = item.split(':')
        out.append((name, float(h)))
    return tuple(out)


def policy_specs(kind):
    plans = [(None if math.isinf(p.entry_s) else p.entry_s, p.pre_mps2, p.post_mps2, p.duration_s) for p in grid()]
    return {'all': ['b5'] + plans, 'grid': plans, 'b5': ['b5']}[kind]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('validate-b5', 'landscape'))
    p.add_argument('out', nargs='?')
    p.add_argument('--cases', type=int)
    p.add_argument('--workers', type=int)
    p.add_argument('--targets', help='comma list name:H, e.g. following_aeb:0,following:1')
    p.add_argument('--policies', default='all', choices=('all', 'grid', 'b5'))
    p.add_argument('--shard', help='k/n or k1+k2/n: keep case ids with id %% n in {k}')
    a = p.parse_args()
    if a.command == 'validate-b5':
        validate_b5()
        sys.exit(0)
    filt = None
    if a.shard:
        ks, n = a.shard.split('/')
        keep, n = {int(k) for k in ks.split('+')}, int(n)
        filt = lambda c, keep=keep, n=n: c['case_id'] % n in keep  # noqa: E731
    landscape(a.out, a.cases, a.workers, policies=policy_specs(a.policies),
              panel=parse_targets(a.targets) if a.targets else TARGET_PANEL, case_filter=filt)
