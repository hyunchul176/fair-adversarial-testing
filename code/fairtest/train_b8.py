"""Train the B8 constrained adversary with PPO (SB3) on exposed calibration cases.

Development-only. Evaluation on development cases is deterministic (argmax policy).
Realism enters as a Lagrangian cost with PID-controlled multiplier (Stooke et al. 2020),
restoring the original constrained-RL design of the earlier agent with corrected objective and cost.

Usage (from code/, pinned venv):
  python -m fairtest.train_b8 OUTDIR --sut following 0 --steps 200000 --envs 12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401


def make_env(case_ids, sut, seed, realism_weight):
    def thunk():
        from fairtest.b8_env import B8AdversaryEnv
        return B8AdversaryEnv(case_ids, sut=sut, seed=seed, realism_weight=realism_weight)
    return thunk


class PIDLagrangian:
    def __init__(self, target, kp=.05, ki=.005, kd=0., init=0.):
        self.target, self.kp, self.ki, self.kd = target, kp, ki, kd
        self.integral, self.previous, self.value = 0., 0., init

    def update(self, cost):
        error = cost - self.target
        self.integral = max(0., self.integral + error)
        derivative = error - self.previous
        self.previous = error
        self.value = max(0., self.kp * error + self.ki * self.integral + self.kd * derivative)
        return self.value


def evaluate(model, case_ids, sut):
    from fairtest.b8_env import B8AdversaryEnv
    env = B8AdversaryEnv(case_ids, sut=sut, sequential=True)
    rows = []
    for cid in case_ids:
        obs, _ = env.reset(options=dict(case_id=cid))
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
        rows.append(info)
    env.close()
    n = len(rows)
    return dict(cases=n, fair=sum(r['fair'] for r in rows), unfair=sum(r['unfair'] for r in rows),
                fair_near=sum(r['fair_near'] for r in rows), contact=sum(r['contact'] for r in rows),
                mean_realism_exceedance=float(np.mean([r['realism_exceedance'] for r in rows])) if n else None,
                rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('--sut', nargs=2, default=['following', '0'])
    ap.add_argument('--steps', type=int, default=200_000)
    ap.add_argument('--envs', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--realism-target', type=float, default=None,
                    help='mean max-exceedance budget; None disables the constraint')
    ap.add_argument('--eval-every', type=int, default=50_000)
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from fairtest.runner import _STATE, _init
    _init()
    sut = (a.sut[0], float(a.sut[1]))
    train = sorted(c for c, v in _STATE['cases'].items() if v['event']['partition'] == 'calibration')
    dev = sorted(c for c, v in _STATE['cases'].items() if v['event']['partition'] == 'development')
    a.out.mkdir(parents=True, exist_ok=True)
    vec = SubprocVecEnv([make_env(train, sut, a.seed + i, 0.) for i in range(a.envs)])
    model = PPO('MlpPolicy', vec, n_steps=128, batch_size=256, n_epochs=8, gamma=1., gae_lambda=.95,
                learning_rate=3e-4, ent_coef=.01, clip_range=.2, seed=a.seed, verbose=0,
                policy_kwargs=dict(net_arch=[128, 128]))
    lagrange = PIDLagrangian(a.realism_target) if a.realism_target is not None else None
    log, done_steps, t0 = [], 0, time.time()
    while done_steps < a.steps:
        chunk = min(a.eval_every, a.steps - done_steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        done_steps += chunk
        result = evaluate(model, dev, sut)
        if lagrange is not None:
            weight = lagrange.update(result['mean_realism_exceedance'] or 0.)
            vec.set_attr('realism_weight', weight)
        else:
            weight = 0.
        entry = dict(steps=done_steps, seconds=round(time.time() - t0, 1), realism_weight=weight,
                     **{k: v for k, v in result.items() if k != 'rows'})
        log.append(entry)
        print(json.dumps(entry), flush=True)
        model.save(a.out / f'ppo_{done_steps}')
        (a.out / 'train_log.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
        (a.out / f'eval_{done_steps}.json').write_text(json.dumps(result['rows'], indent=1), encoding='utf-8')
    vec.close()


if __name__ == '__main__':
    main()
