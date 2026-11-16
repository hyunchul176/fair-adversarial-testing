"""B52 (B52 design, available on request): the fair_guarded2 adversary
trained against a physical-feasibility reference instead of the normative CC, all else unchanged.

The reward code is b17_rl's `fair_guarded2` (G1 commitment, G2 realism, dense boundary shaping). Only the replay that
labels an episode fair / unfair uses another reference driver:
  oracle_t0      omniscient longitudinal solvability (FREA / STRIVE type)
  oracle_react0  ideal ADS: CC perception point, zero delay, full capability (ScenePilot type)
The environment classes live here (module level) so that SB3's subprocess workers import them by name; no frozen file
changes. Evaluation uses the frozen `b17_rl eval --commit` (every collision is re-judged by the CC and the references).

  python -m fairtest.b52_feasible train OUT_DIR --ref oracle_t0 --init COLLIDE_COMMIT_750k.zip --steps 1500000 --envs 15 --seed 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b17_rl import LABEL_FREE, B17Env

REFS = ('oracle_t0', 'oracle_react0')


class _FeasEnv(B17Env):
    REF = None

    def _cc_replay(self):
        """b17_rl.B17Env._cc_replay with the feasibility reference in place of cc_r157 (same features and demand)."""
        if self.mode in LABEL_FREE and not self.label:
            return None
        from fairtest.features import simulated_features
        from fairtest.runner import interaction_metrics
        env = self._make(self.case, self.REF, 0.)
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
            self.last_cc = dict(contact=bool(r['pair_contact']), cut_in=bool(feats.get('cut_in')), exc=exc,
                                demand=min(1.5, interaction_metrics(r)['max_drac'] / 7.59), reference=self.REF)
            return self.last_cc['contact']
        finally:
            env.close()


class FeasEnvT0(_FeasEnv):
    REF = 'oracle_t0'


class FeasEnvR0(_FeasEnv):
    REF = 'oracle_react0'


CLASSES = {'oracle_t0': FeasEnvT0, 'oracle_react0': FeasEnvR0}


def make_env(cls, case_ids, mode, seed):
    def thunk():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(cls(case_ids, mode, seed=seed, label=mode not in LABEL_FREE),
                       info_keywords=('contact', 'fair', 'unfair', 'guarded_fair', 'dynasto_valid'))
    return thunk


def train(a):
    """b17_rl.train with the feasibility environment (same PPO settings, warm start, checkpoints and log)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from fairtest.runner import _STATE, _init
    _init()
    cal = sorted(c for c, v in _STATE['cases'].items() if v['event']['partition'] == 'calibration')
    a.out.mkdir(parents=True, exist_ok=True)
    cls, mode = CLASSES[a.ref], 'fair_guarded2'
    vec = SubprocVecEnv([make_env(cls, cal, mode, a.seed * 1000 + i) for i in range(a.envs)])
    model = PPO.load(a.init, env=vec, device='cpu')
    model.set_random_seed(a.seed)
    tag = 'feast0' if a.ref == 'oracle_t0' else 'feasr0'
    t0, done, log = time.time(), 0, []
    while done < a.steps:
        chunk = min(a.checkpoint, a.steps - done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done += chunk
        buf = list(model.ep_info_buffer) if model.ep_info_buffer else []
        mean = lambda k: float(np.mean([float(ep[k]) for ep in buf])) if buf and k in buf[0] else None
        entry = dict(steps=done, seconds=round(time.time() - t0, 1), reference=a.ref, mean_episode_reward=mean('r'),
                     contact_rate=mean('contact'), fair_rate=mean('fair'), unfair_rate=mean('unfair'),
                     guarded_fair_rate=mean('guarded_fair'))
        log.append(entry)
        print(json.dumps(entry), flush=True)
        model.save(a.out / f'ppo_{tag}_s{a.seed}_{done}.zip')
        (a.out / 'train_log.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
    vec.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('out', type=Path)
    t.add_argument('--ref', required=True, choices=REFS)
    t.add_argument('--init', required=True, help='the seed\'s collide_commit 750k checkpoint (as the v2j policy)')
    t.add_argument('--steps', type=int, default=1_500_000)
    t.add_argument('--envs', type=int, default=15)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--checkpoint', type=int, default=250_000)
    a = ap.parse_args()
    train(a)
