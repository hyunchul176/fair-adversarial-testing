"""The fair adversarial agent, final version v2, as one callable tool with a certificate for every failure it reports.
(fair_agent.py is the earlier B13-era online agent: grid selector + CMA-ES; kept unchanged.)

  agent = FairAdversaryV2(policy_path)                 # a fair_guarded2 policy (seed 1, 2 or 3)
  result = agent.attack(case_id, ('following', 0.))    # -> dict: found, queries, stage, attack, certificate

Per case and SUT:
  0. boundary plan (0 SUT queries): the CC-only continuous search for the hardest plan the R157 careful driver (CC)
     still avoids (b11_search, warm-started at the grid boundary plan of the case);
  1. shielded policy (1 SUT query): the guarded RL policy (G1 commitment, G2 realism) under the window-conditional
     reference shield with a 0.5 m CC margin -- every executed attack is CC-avoidable by construction;
  2. guarded robust refinement (<= budget SUT queries, only if stage 1 gave no robust fair failure): GP-BO from the
     boundary plan; success = robust fair failure (front contact, CC clearance >= 0.5 m) with the RL guards in plan
     space (no lateral retreat > 0.2 m after the CC perception threshold, realistic cut-in <= 1 IQR);
  3. certificate of the reported failure (b29_consensus.task): the CC counterfactual, the held-out reference drivers
     (strict CC, the original time-based CC, FSM, the RSS reference driver, two solvability oracles), the CC's
     required braking and its R157 Annex 5 grade, the decision margin, the guards and the failure-mode features.
The two stages never use each other's outcome, so the result equals the exact per-case composition of B26/B29.

v2j (B46, `--joint`): one success contract for both stages -- robust fair failure + G1 + G2 (per-feature realism of the CC
replay <= 1 IQR) + the natural joint support (5-NN distance of the CC replay's six cut-in features <= the natural
leave-one-out 99th percentile, b45_joint_realism.Support). Stage 1 checks G2 and the joint support on the CC replay of the
policy's actions; stage 2 is bo_warm_g4j. A failure is reported only with an error-free certificate (`certified`);
a stage-2 success whose certificate fails is returned as found=False, status='uncertified' (W5 P1-C).
`cost` separates the SUT episodes used to search (`queries`) from verification replays and reference-driver episodes.

  python -m fairtest.fair_agent_v2 attack POLICY.zip --case 204 --sut following:0 [--budget 36] [--joint]
  python -m fairtest.fair_agent_v2 batch POLICY.zip OUT.jsonl --partition development --suts following:0,rss_sut:0 \
         [--cont STORED_B11.jsonl] --workers 8 [--joint]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from fairtest import devboot  # noqa: F401

MARGIN, BUDGET, FRONT_DX = .5, 36, 3.5
# episodes behind one certificate (b29_consensus.task: the SUT run, the CC and six held-out references) and one
# front-contact check (b23_margin._task: SUT and CC)
CERT_SUT, CERT_REF, FRONT_SUT, FRONT_REF = 1, 7, 1, 1
_AGENTS = {}
_REALISM = []


def _cc_realism(case_id, actions):
    """G2 and the joint support of a policy attack, from its CC replay (as B45; the refinement checks the same)."""
    from fairtest.b23_crossjudge import replay
    from fairtest.b45_joint_realism import Support
    from fairtest.features import Realism, simulated_features
    from fairtest.runner import _STATE
    if not _REALISM:
        _REALISM.extend([Realism(), Support()])
    real, sup = _REALISM
    f = simulated_features(replay(_STATE['cases'][case_id], actions, 'cc_r157', 0.))
    exc = real.score(f)['max_exceedance'] if f.get('cut_in') else None
    js = sup.score(f) if f.get('cut_in') else None
    return dict(exceedance=exc, joint_score=js, g2=exc is not None and exc <= 1., joint=js is not None and js <= sup.tau)


class FairAdversaryV2:
    def __init__(self, policy_path, budget=BUDGET, joint=False):
        from stable_baselines3 import PPO
        from fairtest import b17_rl
        b17_rl.SHIELD_MARGIN = MARGIN  # the shield demands a 0.5 m CC clearance (v2)
        self.policy_path, self.budget, self.joint = str(policy_path), int(budget), bool(joint)
        self.model = PPO.load(self.policy_path, device='cpu')

    @staticmethod
    def boundary_plan(case_id):
        """Stage 0: CC-only search for the hardest plan the CC avoids (no SUT query)."""
        from fairtest import runner
        from fairtest.b11_search import search, warm_start
        case = runner._STATE['cases'][case_id]
        best, _, n_cc = search(case, 'cc_r157', 0., warm_start(case_id), 36, fair_only=False)
        return (best['theta'] if best is not None and not best['cc_contact'] else None), n_cc

    def policy_stage(self, case_id, sut):
        """Stage 1: one shielded policy episode against the SUT (one SUT query)."""
        from fairtest.b17_rl import B17Env, shielded_action
        env = B17Env([case_id], 'fair_guarded', suts=[sut], label=True)
        obs, _ = env.reset(options=dict(case_id=case_id, sut=sut))
        done, last, rollouts = False, None, 0
        while not done:
            action, n, via = shielded_action(self.model, env, obs, last, window=True)
            rollouts += n
            last = (env.effective(action), via)
            obs, _, done, _, info = env.step(action)
        actions = [list(a) for a in env.actions]
        env.close()
        return dict(valid=bool(info['valid']), contact=bool(info['contact']), actions=actions, cc_rollouts=rollouts)

    def refine_stage(self, case_id, sut, boundary, panel_index=0):
        """Stage 2: guarded robust GP-BO from the boundary plan (<= budget SUT queries). panel_index keys the random
        stream exactly as b23_search does for the SUT's position in its panel."""
        from fairtest import runner
        from fairtest.b23_search import METHODS, bo_search
        case = runner._STATE['cases'][case_id]
        rng = np.random.default_rng([case_id, panel_index, METHODS.index('bo_warm')])
        best, n = bo_search(case, sut[0], float(sut[1]), self.budget, rng, x_first=boundary, robust=True, guarded=True,
                            joint=self.joint)
        found = bool(best and best['fair'])
        return dict(found=found, queries=n, theta=best['theta'] if best else None)

    def attack(self, case_id, sut, boundary=None, panel_index=0):
        from fairtest import runner
        from fairtest.b29_consensus import task as certificate_of
        runner._init()
        t0 = time.time()
        sut = (sut[0], float(sut[1]))
        key = f'{sut[0]}_H{sut[1]:g}'
        out = dict(case_id=case_id, sut=key, version='v2j' if self.joint else 'v2', found=False, certified=False, status=None,
                   queries=1, stage=None, attack=None, certificate=None)
        cost = dict(sut_search_episodes=1, sut_verification_replays=0, reference_episodes=0, shield_cc_rollouts=0)
        p = self.policy_stage(case_id, sut)
        cost['shield_cc_rollouts'] = p['cc_rollouts']
        out['policy'] = dict(valid=p['valid'], contact=p['contact'], cc_rollouts=p['cc_rollouts'])
        if p['valid'] and p['contact']:
            cert = certificate_of(('agent', 'policy', case_id, key, 'actions', p['actions']))
            cost['sut_verification_replays'] += CERT_SUT
            cost['reference_episodes'] += CERT_REF
            robust = 'error' not in cert and not cert['cc_contact'] and cert['cc_clearance'] >= MARGIN
            if robust:
                robust = _front_contact(case_id, key, p['actions'])
                cost['sut_verification_replays'] += FRONT_SUT
                cost['reference_episodes'] += FRONT_REF
            if robust and self.joint:  # v2j: the same realism contract as stage 2
                real = _cc_realism(case_id, p['actions'])
                cost['reference_episodes'] += 1
                out['policy']['realism'] = real
                robust = real['g2'] and real['joint']
            out['policy']['certificate'] = cert
            if robust:
                out.update(found=True, certified=True, status='certified', stage='policy',
                           attack=dict(how='actions', payload=p['actions']), certificate=cert)
                out.update(cost=cost, seconds=round(time.time() - t0, 1))
                return out
        if boundary is None:
            boundary, out['boundary_cc_runs'] = self.boundary_plan(case_id)
            cost['reference_episodes'] += out['boundary_cc_runs'] or 0
        r = self.refine_stage(case_id, sut, boundary, panel_index)
        out['queries'] = 1 + r['queries']
        cost['sut_search_episodes'] = out['queries']
        cost['reference_episodes'] += r['queries']  # one CC replay per refinement query
        if r['found']:
            cert = certificate_of(('agent', 'refine', case_id, key, 'theta', r['theta']))
            cost['sut_verification_replays'] += CERT_SUT
            cost['reference_episodes'] += CERT_REF
            if 'error' in cert:  # never report an uncertified failure
                out.update(stage='refine', status='uncertified', certificate=cert)
            else:
                out.update(found=True, certified=True, status='certified', stage='refine',
                           attack=dict(how='theta', payload=r['theta']), certificate=cert)
        out.update(cost=cost, seconds=round(time.time() - t0, 1))
        return out


def _front_contact(case_id, sut, actions):
    """G4 of a policy failure (front contact and CC clearance), exactly as b23_margin judges stored episodes."""
    from fairtest.b23_margin import _task
    m = _task(('agent', case_id, sut, 'actions', actions))
    return bool(m['sut_contact'] and not m['cc_contact'] and m['contact_dx'] >= FRONT_DX and m['cc_clearance'] >= MARGIN)


def _init_worker():
    from fairtest import runner
    from fairtest.b11_search import load_warm_starts
    runner._init()
    runner._STATE['warm'] = load_warm_starts()


def _batch_task(args):
    policy, case_id, sut, boundary, panel_index = args[:5]
    joint = bool(args[5]) if len(args) > 5 else False
    if (policy, joint) not in _AGENTS:
        _AGENTS[(policy, joint)] = FairAdversaryV2(policy, joint=joint)
    try:
        return _AGENTS[(policy, joint)].attack(case_id, sut, boundary=boundary, panel_index=panel_index)
    except Exception as e:  # recorded, never dropped
        return dict(case_id=case_id, sut=f'{sut[0]}_H{sut[1]:g}', error=f'{type(e).__name__}: {e}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    one = sub.add_parser('attack')
    one.add_argument('policy')
    one.add_argument('--case', type=int, required=True)
    one.add_argument('--sut', required=True, help='name:H, e.g. following:0')
    one.add_argument('--budget', type=int, default=BUDGET)
    one.add_argument('--joint', action='store_true', help='v2j: the joint-support gate and one success contract (B46)')
    many = sub.add_parser('batch')
    many.add_argument('policy')
    many.add_argument('out')
    many.add_argument('--partition', default='development')
    many.add_argument('--suts', required=True)
    many.add_argument('--cont', help='stored CC-boundary search (b11_search output) to reuse instead of recomputing')
    many.add_argument('--cases', type=int, help='first N cases only')
    many.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    many.add_argument('--joint', action='store_true', help='v2j (B46)')
    a = ap.parse_args()
    _init_worker()
    from fairtest import runner
    if a.cmd == 'attack':
        name, h = a.sut.split(':')
        print(json.dumps(FairAdversaryV2(a.policy, a.budget, joint=a.joint).attack(a.case, (name, float(h))), indent=1))
        return
    suts = [(x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(',')]
    boundary = {}
    if a.cont:
        for line in open(a.cont, encoding='utf-8'):
            r = json.loads(line)
            b = (r.get('cc_boundary') or {}).get('best')
            boundary[r['case_id']] = b['theta'] if b and not b['cc_contact'] else None
    ids = sorted(c for c, v in runner._STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    ids = ids[:a.cases] if a.cases else ids
    jobs = [(a.policy, c, s, boundary.get(c) if a.cont else None, i, a.joint) for c in ids for i, s in enumerate(suts)]
    with mp.Pool(a.workers, initializer=_init_worker) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for res in pool.imap_unordered(_batch_task, jobs):
            f.write(json.dumps(res) + '\n')
            f.flush()
    print(json.dumps(dict(finished=len(jobs))))


if __name__ == '__main__':
    main()
