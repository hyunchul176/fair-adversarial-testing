"""B17 (post-B13 development): learned RL adversaries — does the attribution problem persist for RL,
and does an attribution-rewarded RL agent game the reference model?

Environment: the B8 closed-loop adversary (2 Hz decisions; lane goal x longitudinal request; observes
the relative state and the SUT's last acceleration, never the SUT identity). One episode = one SUT
rollout. Training SUT = uniform mixture over the panel. Training cases = exposed calibration.
Rewards
  collide  +1 valid SUT contact, +0.3 valid near miss (TTC < 1.5 s), -1 invalid manoeuvre, -w realism;
           no reference-driver information (the collision-seeking RL family, e.g. the earlier agent and CARS-style);
  fair     +1 fair failure (the realised action sequence replayed against CC avoids), +0.3 fair near
           miss, -1 attacker-at-fault contact, -1 invalid, -w realism (the B8 design).
  collide_commit / fair_guarded  the same with lane-change commitment (once the policy enters, the lane
           goal stays 'enter'; no lateral feints). fair_guarded counts a fair failure only if the CC replay
           completes a cut-in within the realism limit (<= 1 IQR); CC-avoidable contacts without a realistic
           cut-in are penalised (-0.5).
  abl_none / abl_commit / abl_real (B23 E2 guard ablation, 2 x 2 with fair_guarded2 = both guards): the
           fair_guarded2 reward structure (fair failure + 0.2 x CC-demand shaping - 0.3 attacker-at-fault - invalid) with
           G1 commitment on/off and G2 realistic-cut-in requirement on/off (abl_none: neither, abl_commit: G1 only,
           abl_real: G2 only). Warm starts: collide (no commitment) or collide_commit at 750k steps.
  dynasto / cars (B23 literature baselines, `rl_baselines.py`): the Dynasto validity-aware reward (ISO 34502
           safe-distance rules, dense collision-likelihood term) and the CARS RL fine-tuning reward (progress,
           lateral proximity and closing, collision bonus, time penalty); per-decision dense rewards, no commitment,
           no reference-driver information during training.
  cars_commit (B74, review of manuscript v7, item R3): the cars reward with lane-change commitment (G1) during
           training as well, so that commitment at evaluation no longer removes actions the policy relied on.
Evaluation (deterministic policy) for every SUT: the realised actions are replayed against CC, strict
CC, FSM, RSS and the B16 oracles, exactly as the attribution replay of CARS (adversary fixed, the
target response replaced).

  python -m fairtest.b17_rl train OUTDIR --mode collide|fair --steps 600000 --envs 30 --seed 1
  python -m fairtest.b17_rl eval MODEL.zip OUT.jsonl --partition development|all --workers 30
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import rl_baselines as BL
from fairtest.b8_env import REQUESTS, B8AdversaryEnv

PANEL = (('following', 0.), ('following', .5), ('following', 1.), ('reference_idm', 0.), ('following_aeb', 0.),
         ('following_aeb', .5), ('rss_sut', 0.))
REFS = ('cc_r157', 'cc_r157_strict', 'ref_fsm', 'ref_rss', 'oracle_t0', 'oracle_react0')


MODES = ('collide', 'fair', 'collide_commit', 'fair_guarded', 'fair_guarded2', 'dynasto', 'cars', 'abl_none', 'abl_commit', 'abl_real',
         'cars_commit')
SHAPED = ('fair_guarded2', 'abl_none', 'abl_commit', 'abl_real')  # CC replay of every valid episode (boundary shaping)
DENSE = ('dynasto', 'cars', 'cars_commit')  # per-decision rewards
LABEL_FREE = ('collide', 'collide_commit', 'dynasto', 'cars', 'cars_commit')  # no CC replay needed for the training reward
COMMIT = ('collide_commit', 'fair_guarded', 'fair_guarded2', 'abl_commit', 'cars_commit')  # lane-change commitment (G1)


SHIELD_MARGIN = float(os.environ.get('FAIRTEST_SHIELD_MARGIN', 0.))  # G4 clearance the shield demands of the CC (m)
# B39 (reference ablation): the reference the shield replays; unset = the R157 CC, the verdicts keep the CC
SHIELD_REF = os.environ.get('FAIRTEST_SHIELD_REF', 'cc_r157')


def cc_safe(result):
    """CC replay admissible for the shield: no contact and, with a margin, the CC's clearance at least the margin."""
    if result['pair_contact']:
        return False
    if SHIELD_MARGIN <= 0.:
        return True
    from fairtest.b23_margin import clearance
    return clearance(result['trace']) >= SHIELD_MARGIN


class B17Env(B8AdversaryEnv):
    def __init__(self, case_ids, mode, suts=PANEL, seed=0, realism_weight=0., label=True):
        super().__init__(case_ids, sut=suts[0], seed=seed, realism_weight=realism_weight)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode, self.suts, self.label = mode, [(s, float(h)) for s, h in suts], label
        self.commit = mode in COMMIT
        self.last_cc = None

    def reset(self, *, seed=None, options=None):
        self.sut = tuple(options['sut']) if options and 'sut' in options else self.rng.choice(self.suts)
        self.last_cc = None
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.commit and self.entered:
            lane, request = self.decode(action)
            if lane != 4.:
                action = len(REQUESTS) + REQUESTS.index(request)
        if self.mode not in DENSE:
            return super().step(action)
        before = BL.pair_state(self.env)
        lane, request = self.decode(action)
        self.actions.append((lane, request))
        self.entered = self.entered or lane == 4.
        _, _, done, _, _ = self.env.step((lane, request))
        after = BL.pair_state(self.env)
        dense = BL.dynasto_step_reward(*after) if self.mode == 'dynasto' else BL.cars_step_reward(before, after)
        obs = self._obs()
        if not done:
            return obs, dense, False, False, {}
        row = self.env.result()
        self.env.close()
        self.env = None
        info = self._score(row)
        return obs, dense + info['reward'], True, False, info

    def effective(self, action):
        """(lane, request) the environment will execute for a policy action (commitment applied)."""
        lane, request = self.decode(action)
        if self.commit and self.entered and lane != 4.:
            lane = 4.
        return lane, request

    def cc_avoids_if_held(self, action):
        """Reference shield: does the CC avoid the actions so far plus `action` held to the end of the episode?"""
        return cc_safe(self.cc_replay_of(self.actions + [self.effective(action)]))

    def cc_replay_of(self, actions):
        """CC counterfactual of an action sequence, the last action held to the end (the CC-replay semantics)."""
        env = self._make(self.case, SHIELD_REF, 0.)
        try:
            env.reset()
            done = False
            for x in actions:
                if done:
                    break
                _, _, done, _, _ = env.step(x)
            while not done:
                _, _, done, _, _ = env.step(actions[-1])
            return env.result()
        finally:
            env.close()

    def least_aggressive(self, after):
        """The fallback decision after `after`: keep the lane already chosen (commitment) and request +1 m/s^2."""
        entered = self.entered or after[0] == 4.
        return (4. if entered else 0., max(REQUESTS))

    def sut_contact_possible(self, trace, n=5, a_up=3., a_down=10., margin=.3):
        """Could the SUT touch the adversary during the next n ticks, whatever it does within [-a_down, +a_up]?
        Adversary positions come from the counterfactual trace (its motion does not depend on the target)."""
        e = self.env
        g_now, t = e.sim.inner.road.vehicles
        x, y, v = float(t.position[0]), float(t.position[1]), float(t.velocity[0])
        half_l, half_w = (g_now.LENGTH + t.LENGTH) / 2 + margin, (g_now.WIDTH + t.WIDTH) / 2 + margin
        for j in range(1, n + 1):
            k = e.tick + j - 1  # trace row k = the state after tick k
            if k >= len(trace):
                break
            g = trace[k]['generator']
            tau = .1 * j
            lo, hi = x + v * tau - .5 * a_down * tau * tau, x + v * tau + .5 * a_up * tau * tau
            if abs(g['y'] - y) < half_w and g['x'] - hi < half_l and lo - g['x'] < half_l:
                return True
        return False

    def shield_admit(self, action, window):
        """(admitted, via, rollouts) for a candidate action under the strict or the window-conditional shield."""
        a = self.effective(action)
        held = self.cc_replay_of(self.actions + [a])
        if cc_safe(held):
            return True, 'held', 1
        if not window:
            return False, None, 1
        # the recover replay has no CC contact when admissible, so its trace covers a's whole window
        recover = self.cc_replay_of(self.actions + [a, self.least_aggressive(a)])
        if not cc_safe(recover) or self.sut_contact_possible(recover['trace']):
            return False, None, 2
        return True, 'recover', 2

    def _cc_replay(self):
        if self.mode in LABEL_FREE and not self.label:
            return None
        from fairtest.features import simulated_features
        env = self._make(self.case, 'cc_r157', 0.)
        try:
            env.reset()
            done = False
            for a in self.actions:
                if done:
                    break
                _, _, done, _, _ = env.step(a)
            while not done:  # the SUT episode ended by contact; continue the latched last action
                _, _, done, _, _ = env.step(self.actions[-1])
            r = env.result()
            feats = simulated_features(r)
            exc = self.realism.score(feats)['max_exceedance'] if feats.get('cut_in') else None
            from fairtest.runner import interaction_metrics
            self.last_cc = dict(contact=bool(r['pair_contact']), cut_in=bool(feats.get('cut_in')), exc=exc,
                                demand=min(1.5, interaction_metrics(r)['max_drac'] / 7.59))
            return self.last_cc['contact']
        finally:
            env.close()

    def _score(self, row):
        info = super()._score(row)
        info['sut'] = f'{self.sut[0]}_H{self.sut[1]:g}'
        valid = info['valid']
        if self.mode in SHAPED and self.last_cc is None and valid:
            self._cc_replay()  # dense shaping needs the CC replay of every valid episode
        lc = self.last_cc
        realistic = bool(lc and lc['cut_in'] and lc['exc'] is not None and lc['exc'] <= 1.)
        info['guarded_fair'] = bool(info['fair'] and realistic)
        info['cc_replay'] = lc
        info['dynasto_valid'] = False
        if self.mode in ('collide', 'collide_commit'):
            info['reward'] = float(1. * (valid and info['contact']) + .3 * (valid and info['near_miss'] and not info['contact'])
                                   - (0. if valid else 1.) - self.realism_weight * info['realism_exceedance'])
        elif self.mode == 'fair_guarded':
            info['reward'] = float(1. * info['guarded_fair'] + .3 * (info['fair_near'] and realistic) - 1. * info['unfair']
                                   - .5 * (info['fair'] and not realistic) - (0. if valid else 1.))
        elif self.mode == 'dynasto':
            info['reward'], why = BL.dynasto_terminal(row, valid, info['contact'])
            info['dynasto_valid'] = bool(valid and info['contact'] and why is None)
            info['dynasto_violation'] = why
        elif self.mode in ('cars', 'cars_commit'):
            info['reward'] = BL.cars_terminal(valid, info['contact'])
        elif self.mode in ('fair_guarded2', 'abl_real'):
            # sparse fair reward + dense R157-boundary shaping: CC-preventable realistic cut-ins earn their CC demand
            shaping = .2 * min(1., lc['demand']) if (lc and realistic and not lc['contact'] and not info['guarded_fair']) else 0.
            info['reward'] = float(1. * info['guarded_fair'] + shaping - .3 * info['unfair'] - (0. if valid else 1.))
        elif self.mode in ('abl_none', 'abl_commit'):
            # the same without the realistic-cut-in requirement (G2 off): any CC-avoided contact counts
            shaping = .2 * min(1., lc['demand']) if (lc and lc['cut_in'] and not lc['contact'] and not info['fair']) else 0.
            info['reward'] = float(1. * info['fair'] + shaping - .3 * info['unfair'] - (0. if valid else 1.))
        return info

    def replay(self, target, h=0.):
        """Replay the realised action sequence against another target (CC-replay semantics)."""
        env = self._make(self.case, target, h)
        try:
            env.reset()
            done = False
            for a in self.actions:
                if done:
                    break
                _, _, done, _, _ = env.step(a)
            while not done:
                _, _, done, _, _ = env.step(self.actions[-1])
            r = env.result()
            return dict(contact=bool(r['pair_contact']), valid=bool(r['maneuver_valid_precontact']))
        finally:
            env.close()


def make_env(case_ids, mode, seed, realism_weight, suts=PANEL):
    def thunk():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(B17Env(case_ids, mode, suts=suts, seed=seed, realism_weight=realism_weight, label=mode not in LABEL_FREE),
                       info_keywords=('contact', 'fair', 'unfair', 'guarded_fair', 'dynasto_valid'))
    return thunk


def train_suts(spec):
    """B62: training SUT list from a JSON file (list of [name, H], repeats allowed = sampling weights) or name:H,..."""
    if spec.endswith('.json'):
        return tuple((str(n), float(h)) for n, h in json.load(open(spec, encoding='utf-8'))['train_suts'])
    return tuple((x.split(':')[0], float(x.split(':')[1])) for x in spec.split(','))


def train(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from fairtest.runner import _STATE, _init
    _init()
    cal = sorted(c for c, v in _STATE['cases'].items() if v['event']['partition'] == 'calibration')
    a.out.mkdir(parents=True, exist_ok=True)
    suts = train_suts(a.train_suts) if a.train_suts else PANEL  # B62: a training SUT population (default: the panel)
    (a.out / 'train_suts.json').write_text(json.dumps([list(x) for x in suts]), encoding='utf-8') if a.train_suts else None
    vec = SubprocVecEnv([make_env(cal, a.mode, a.seed * 1000 + i, a.realism_weight, suts) for i in range(a.envs)])
    if a.init:
        model = PPO.load(a.init, env=vec, device='cpu')  # warm start (e.g. from a collision-seeking policy)
        model.set_random_seed(a.seed)
    else:
        model = PPO('MlpPolicy', vec, n_steps=128, batch_size=256, n_epochs=8, gamma=1., gae_lambda=.95,
                    learning_rate=3e-4, ent_coef=.01, clip_range=.2, seed=a.seed, verbose=0,
                    policy_kwargs=dict(net_arch=[128, 128]))
    t0, done, log = time.time(), 0, []
    while done < a.steps:
        chunk = min(a.checkpoint, a.steps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
        buf = list(model.ep_info_buffer) if model.ep_info_buffer else []
        mean = lambda k: float(np.mean([float(ep[k]) for ep in buf])) if buf and k in buf[0] else None
        entry = dict(steps=done, seconds=round(time.time() - t0, 1), mean_episode_reward=mean('r'),
                     contact_rate=mean('contact'), fair_rate=mean('fair'), unfair_rate=mean('unfair'),
                     guarded_fair_rate=mean('guarded_fair'), dynasto_valid_rate=mean('dynasto_valid'))
        log.append(entry)
        print(json.dumps(entry), flush=True)
        model.save(a.out / f'ppo_{a.mode}_s{a.seed}_{done}')
        (a.out / 'train_log.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
    vec.close()


_MODEL = {}


def shielded_action(model, env, obs, last, window=False):
    """The most preferred admissible action (see `B17Env.shield_admit`). `last` = (effective action, admitted via
    'held') of the previous decision: repeating an action admitted as 'held' needs no new check (same trajectory)."""
    import torch
    with torch.no_grad():
        probs = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs[0].numpy()
    checks, seen = 0, set()
    for a in np.argsort(-probs, kind='stable'):
        a = int(a)
        eff = env.effective(a)
        if eff in seen:
            continue
        seen.add(eff)
        if last is not None and eff == last[0] and last[1] == 'held':
            return a, checks, 'held'
        ok, via, n = env.shield_admit(a, window)
        checks += n
        if ok:
            return a, checks, via
    # no admissible action: the CC cannot avoid any held continuation; take the least aggressive one
    fallback = min(range(len(probs)), key=lambda a: (env.effective(a)[0] == 4., -env.effective(a)[1]))
    return fallback, checks, None


def _eval_task(args):
    path, cid, sut, commit = args[:4]
    shield = len(args) > 4 and args[4]  # False, 'strict' or 'window'
    from stable_baselines3 import PPO
    from fairtest.runner import _init
    _init()
    if path not in _MODEL:
        _MODEL[path] = PPO.load(path, device='cpu')
    model = _MODEL[path]
    env = B17Env([cid], 'fair_guarded' if commit else 'fair', suts=[sut], label=True)
    obs, _ = env.reset(options=dict(case_id=cid, sut=sut))
    done, last, checks, blocked, unshieldable = False, None, 0, 0, 0
    while not done:
        greedy, _ = model.predict(obs, deterministic=True)
        action = int(greedy)
        if shield:
            action, n, via = shielded_action(model, env, obs, last, window=shield == 'window')
            checks += n
            blocked += action != int(greedy)
            unshieldable += via is None
            last = (env.effective(action), via)
        obs, _, done, _, info = env.step(action)
    out = {k: info[k] for k in ('case_id', 'sut', 'valid', 'contact', 'near_miss', 'realism_exceedance', 'cut_in', 'steps')}
    out['actions'] = env.actions
    if shield:
        out['shield'] = dict(mode=shield, cc_rollouts=checks, overridden=blocked, unshieldable=unshieldable)
    out['cc_replay'] = info.get('cc_replay')
    out['guarded_fair'] = info.get('guarded_fair')
    if info['valid'] and info['contact']:
        out['refs'] = {}
        for r in REFS:
            try:
                out['refs'][r] = env.replay(r)
            except RuntimeError as e:  # a reference driver's own actuator contract: recorded, not fatal (B28 deviation D1)
                out['refs'][r] = dict(error=f'{type(e).__name__}: {e}')
    env.close()
    return out


def evaluate(a):
    from fairtest.runner import _STATE, _init
    _init()
    ids = sorted(c for c, v in _STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    panel = PANEL if not a.suts else tuple((x.split(':')[0], float(x.split(':')[1])) for x in a.suts.split(','))
    mode = 'window' if a.shield_window else ('strict' if a.shield else False)
    jobs = [(str(a.model), c, s, a.commit, mode) for c in ids for s in panel]
    t0 = time.time()
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(_eval_task, jobs, chunksize=2)):
            f.write(json.dumps(r) + '\n')
            if (i + 1) % 200 == 0:
                print(json.dumps(dict(done=i + 1, of=len(jobs), s=round(time.time() - t0, 1))), flush=True)
    print(json.dumps(dict(finished=len(jobs), s=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('out', type=Path)
    t.add_argument('--mode', required=True, choices=MODES)
    t.add_argument('--steps', type=int, default=600_000)
    t.add_argument('--envs', type=int, default=30)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--realism-weight', type=float, default=0.)
    t.add_argument('--checkpoint', type=int, default=100_000)
    t.add_argument('--init', help='warm-start from a saved PPO model')
    t.add_argument('--train-suts', help='B62: training SUTs (JSON file with train_suts, or name:H,...); default the 7-SUT panel')
    e = sub.add_parser('eval')
    e.add_argument('model', type=Path)
    e.add_argument('out')
    e.add_argument('--partition', default='development')
    e.add_argument('--commit', action='store_true', help='the model was trained with lane-change commitment')
    e.add_argument('--suts', help='comma-separated name:H list (default: the 7-SUT panel)')
    e.add_argument('--shield', action='store_true', help='reference shield: admit only actions the CC can survive if held')
    e.add_argument('--shield-window', action='store_true', help='window-conditional shield (see B17Env.shield_admit)')
    e.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else evaluate(a)
