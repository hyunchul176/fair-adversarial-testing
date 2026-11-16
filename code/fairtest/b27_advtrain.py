"""B27: are fair adversaries better training partners? Adversarial fine-tuning of a learned SUT (b25_drl).

The same base ego policy (b25_drl, with its perception latency) is fine-tuned for the same number of steps on the
same calibration cases under three conditions; only the adversarial half of the episodes differs:
  none     every episode is non-adversarial (the b25 open-loop scenario mix) -- the control for extra training
  fair     half the episodes are attacks by our agent's policy (fair_guarded2) under the reference shield (window-
           conditional, CC margin FAIRTEST_SHIELD_MARGIN): every attack is avoidable by the R157 careful driver
  collide  half the episodes are attacks by the collision-seeking policy (collide_commit), which has no reference-
           driver information: many of its attacks are unavoidable
Attackers are fixed (not co-trained) and act closed-loop against the learning ego at 2 Hz, sampling their
stochastic policy for diversity; under the shield a rejected sample is replaced by the most preferred admissible
action. The ego's reward is the b25 reward (speed tracking, harsh braking, -10 per collision).

  python -m fairtest.b27_advtrain train OUT --base BASE.zip --lag 6 --condition fair --attacker MODEL.zip \
         --steps 500000 --envs 12 --seed 1
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b25_drl import ACCELS, EgoEnv

CONDITIONS = ('none', 'fair', 'collide')


class AdvEgoEnv(EgoEnv):
    def __init__(self, seed=0, lag=0, condition='none', attacker=None, p_adv=.5, perceive_m=None):
        super().__init__(seed=seed, lag=lag, perceive_m=perceive_m)
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition, self.attacker_path, self.p_adv = condition, attacker, float(p_adv)
        self.shield = condition == 'fair'
        self.nprng = np.random.default_rng(seed)
        self.att, self.att_model, self.adv, self.last = None, None, False, None
        self.stats = collections.Counter()
        self.seed_ = seed

    def _attacker(self):
        if self.att_model is None:
            from stable_baselines3 import PPO
            self.att_model = PPO.load(self.attacker_path, device='cpu')
        return self.att_model

    def reset(self, *, seed=None, options=None):
        self._close_attack()
        if self.condition == 'none' or self.rng.random() >= self.p_adv:
            self.adv = False
            return super().reset(seed=seed, options=options)
        from fairtest.b17_rl import B17Env
        if self.env is not None:
            self.env.close()
            self.env = None
        cid = self.rng.choice(self.cases)
        if self.att is None:  # one attacker environment per worker, reset per episode
            mode = 'fair_guarded' if self.condition == 'fair' else 'collide_commit'  # both: lane-change commitment
            self.att = B17Env(self.cases, mode, suts=[('drl_ext', 0.)], seed=self.seed_, label=False)
        self.att_obs, _ = self.att.reset(options=dict(case_id=cid, sut=('drl_ext', 0.)))
        self.env, self.adv, self.last = self.att.env, True, None
        self.env.target_control.perceiver.perceive_m = self.perceive_m
        self.stats['adversarial_episodes'] += 1
        return self._ego_obs(0., 0), {}

    def _attack_action(self):
        import torch
        from fairtest.b17_rl import shielded_action
        m = self._attacker()
        with torch.no_grad():
            probs = m.policy.get_distribution(m.policy.obs_to_tensor(self.att_obs)[0]).distribution.probs[0].numpy()
        a = int(self.nprng.choice(len(probs), p=probs / probs.sum()))
        if not self.shield:
            return a
        eff = self.att.effective(a)
        if self.last is not None and eff == self.last[0] and self.last[1] == 'held':
            return a  # same held continuation as the admitted previous decision
        ok, via, n = self.att.shield_admit(a, window=True)
        self.stats['cc_rollouts'] += n
        if not ok:
            a, n2, via = shielded_action(m, self.att, self.att_obs, self.last, window=True)
            self.stats['cc_rollouts'] += n2
            self.stats['overridden'] += 1
        self.last = (self.att.effective(a), via)
        return a

    def step(self, action):
        if not self.adv:
            return super().step(action)
        request = ACCELS[int(action)]
        b = self.att
        lane, req = b.effective(self._attack_action())  # simultaneous decisions on the current state
        b.actions.append((lane, req))
        b.entered = b.entered or lane == 4.
        self.env.target_control.request = request
        _, _, done, _, _ = self.env.step((lane, req))
        reward, contact = self._reward(request)
        obs = self._ego_obs(self.env.previous_target, self.env.tick)
        if done:
            self.stats['adversarial_contacts'] += contact
            return obs, reward, True, False, dict(contact=contact)
        self.att_obs = b._obs()
        return obs, reward, False, False, {}

    def _close_attack(self):
        if self.adv and self.att is not None and self.att.env is not None:
            self.att.env.close()
            self.att.env, self.env = None, None
        self.adv = False

    def close(self):
        self._close_attack()
        super().close()


def train(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    a.out.mkdir(parents=True, exist_ok=True)

    def make(i):
        return lambda: Monitor(AdvEgoEnv(seed=a.seed * 1000 + i, lag=a.lag, condition=a.condition, attacker=a.attacker,
                                         p_adv=a.p_adv, perceive_m=a.perceive), info_keywords=('contact',))
    vec = SubprocVecEnv([make(i) for i in range(a.envs)])
    model = PPO.load(a.base, env=vec, device='cpu')
    model.set_random_seed(a.seed)
    log, done = [], 0
    while done < a.steps:
        chunk = min(a.checkpoint, a.steps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
        buf = list(model.ep_info_buffer) if model.ep_info_buffer else []
        log.append(dict(steps=done, contact_rate=float(np.mean([float(e['contact']) for e in buf])) if buf else None,
                        mean_return=float(np.mean([e['r'] for e in buf])) if buf else None))
        tag = f'p{a.perceive:g}' if a.perceive is not None else f'lag{a.lag}'
        model.save(a.out / f'ppo_adv_{a.condition}_{tag}_s{a.seed}_{done}.zip')  # explicit suffix (see b25_drl)
        print(json.dumps(log[-1]), flush=True)
    (a.out / 'train_log.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
    vec.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('out', type=Path)
    t.add_argument('--base', required=True, help='base ego policy (b25_drl checkpoint)')
    t.add_argument('--lag', type=int, default=0)
    t.add_argument('--perceive', type=float, help='lane-assignment perception threshold of the SUT (m)')
    t.add_argument('--condition', required=True, choices=CONDITIONS)
    t.add_argument('--attacker', help='attacker PPO (fair: fair_guarded2, collide: collide_commit)')
    t.add_argument('--p-adv', type=float, default=.5)
    t.add_argument('--steps', type=int, default=500_000)
    t.add_argument('--envs', type=int, default=12)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--checkpoint', type=int, default=125_000)
    train(ap.parse_args())
