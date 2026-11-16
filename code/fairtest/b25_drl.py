"""B25 (C): a learned SUT -- an ego driving policy trained with PPO to drive safely through cut-ins.

The ego (the SUT's vehicle) chooses a longitudinal acceleration every 0.5 s from ACCELS; lateral control is the lane
keeper. Training scenarios are non-adversarial: a calibration case (development cases excluded) with an open-loop
adversary plan drawn from the B11 continuous box (50%), the B7 grid (30%) or the nominal plan (20%). Reward per
decision: speed tracking -0.1 |v - cruise| / cruise and -0.05 for braking below -4 m/s^2; -10 at a collision.

At test time the SUT 'drl_sut' (env.py) runs the saved policy (FAIRTEST_DRL_SUT = model path) on the same
observation at every fifth tick, holding the request in between, exactly as in training. The policy never sees an
adversarial attack, the R157 CC or any case of the development partition.

Perception latency (learned-SUT capability ladder): 'drl_sut@<k>' observes the other vehicle's relative state k ticks
(k x 0.1 s) late -- as a perception pipeline would -- while its own speed and last request are current. It is trained
with the same latency (--lag k) and loaded from FAIRTEST_DRL_SUT_AT (a path template with {k}). k = 0 is 'drl_sut'.

Lane-assignment perception: 'drl_sut@p<d>' observes the other vehicle only while it is ahead and displaced at least d m
from its own lane centre towards the ego lane -- the object-to-lane assignment of a production ACC (rule-based analogue:
ads@d_p). Otherwise its perception slot reads 'no relevant object' (a distant same-speed lead). Trained with --perceive d.

  python -m fairtest.b25_drl train OUT_DIR --steps 1000000 --envs 15 --seed 1 [--lag 6 | --perceive 1.5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np

from fairtest import devboot  # noqa: F401

ACCELS = (1., 0., -1., -2., -4., -6., -8.)
JERK, FLOOR = 15., 8.
V_MAX, DT = 40., .1  # highway-env Vehicle.MAX_SPEED; control period
_POLICIES = {}


NO_OBJECT = (1., 0., 0., 0., 0.)


def perception(env, perceive_m=None):
    """The other vehicle's state relative to the ego (the part a perception pipeline delivers, possibly late).
    perceive_m: lane-assignment threshold -- the vehicle is reported only while ahead and displaced >= perceive_m."""
    from src.evaluation.safety_metrics import VehicleState
    g, t = VehicleState.from_vehicle(env.inner.vehicle), VehicleState.from_vehicle(env.target)
    if perceive_m is not None and not (g.x > t.x and abs(g.y) >= perceive_m):
        return NO_OBJECT
    gap = (g.x - t.x) - (g.length + t.length) / 2
    return (gap / 50., (g.vx - t.vx) / 10., (g.y - t.y) / 4., g.vy / 2., float(g.x > t.x))


def variant(key):
    """(latency ticks, perception threshold m) of a drl_sut@<key> variant: '<k>' or 'p<d>'."""
    key = str(key)
    return (0, float(key[1:])) if key.startswith('p') else (int(key), None)


def compose(p, env, previous_a, cruise, tick):
    from src.evaluation.safety_metrics import VehicleState
    t = VehicleState.from_vehicle(env.target)
    return np.clip(np.array([p[0], p[1], p[2], p[3], t.vx / 40., previous_a / 5., (t.vx - cruise) / 10., p[4], tick / 120.],
                            dtype=np.float32), -5., 5.)


def ego_obs(env, previous_a, cruise, tick):
    """Ego observation from the simulator state (adversary = env.inner.vehicle, ego = env.target), no latency."""
    return compose(perception(env), env, previous_a, cruise, tick)


class Perceiver:
    """Per-tick history of the perception part; `delayed(k)` = the value k ticks before the latest pushed state."""

    def __init__(self, perceive_m=None):
        self.history, self.perceive_m = [], perceive_m

    def push(self, env):
        self.history.append(perception(env, self.perceive_m))

    def delayed(self, k, current=None):
        h = self.history + ([current] if current is not None else [])
        return h[max(0, len(h) - 1 - int(k))]


def _telemetry(applied, request, previous_a, steer):
    return dict(raw_action=[applied / 5, 0.], following_mps2=request, requested_mps2=request, post_shield_mps2=request,
                post_stop_mps2=request, previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                viability=dict(constraint_infeasible=False), shield=dict(intervened=False), recognition=dict())


def _apply(env, request, previous_a):
    from fairtest.ref_models import project
    from src.environments.hierarchical_interaction_env import lane_controller
    from src.evaluation.safety_metrics import VehicleState
    t = VehicleState.from_vehicle(env.target)
    v = math.hypot(t.vx, t.vy)
    # speed governor at the simulator's vehicle limit: beyond V_MAX highway-env clips the acceleration, which the
    # actuator-realisation contract rejects (only possible when the policy accelerates at ~40 m/s)
    request = min(request, (V_MAX - v) / DT)
    applied = float(project(request, v, previous_a, JERK, FLOOR))
    steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
    env.target.act(dict(acceleration=applied, steering=steer))
    return applied, steer


class ExternalTarget:
    """Training-time SUT controller: the request is set from outside before every decision (B7Env.step)."""

    def __init__(self, perceive_m=None):
        self.request = 0.
        self.perceiver = Perceiver(perceive_m)

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        self.perceiver.push(env)
        applied, steer = _apply(env, self.request, previous_a)
        return applied, _telemetry(applied, self.request, previous_a, steer)


class DRLSut:
    """The learned SUT: the trained ego policy, queried every fifth tick (0.5 s) as in training."""

    def __init__(self, path=None, key=None):
        self.lag, perceive_m = variant(key) if key is not None else (0, None)
        path = path or (os.environ['FAIRTEST_DRL_SUT'] if key is None else os.environ['FAIRTEST_DRL_SUT_AT'].format(k=key))
        if path not in _POLICIES:
            from stable_baselines3 import PPO
            _POLICIES[path] = PPO.load(path, device='cpu')
        self.model, self.tick, self.request = _POLICIES[path], 0, 0.
        self.perceiver = Perceiver(perceive_m)

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        self.perceiver.push(env)
        if self.tick % 5 == 0:
            obs = compose(self.perceiver.delayed(self.lag), env, previous_a, cruise, self.tick)
            action, _ = self.model.predict(obs, deterministic=True)
            self.request = ACCELS[int(action)]
        self.tick += 1
        applied, steer = _apply(env, self.request, previous_a)
        return applied, _telemetry(applied, self.request, previous_a, steer)


LIMITS = {'drl_ext': dict(acceleration=FLOOR, jerk=JERK), 'drl_sut': dict(acceleration=FLOOR, jerk=JERK)}


class EgoEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=0, lag=0, perceive_m=None):
        from fairtest.runner import _STATE, _init
        _init()
        self.lag, self.perceive_m = int(lag), perceive_m
        self.cases = sorted(c for c, v in _STATE['cases'].items() if v['event'].get('partition') != 'development')
        self.rng = random.Random(seed)
        self.observation_space = gym.spaces.Box(-5., 5., shape=(9,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACCELS))
        self.env, self.plan = None, None

    def _plan(self):
        from fairtest.b11_search import HIGH, LOW, ContinuousPlan
        from fairtest.plans import grid
        u = self.rng.random()
        if u < .5:
            return ContinuousPlan.from_theta(LOW + np.array([self.rng.random() for _ in range(5)]) * (HIGH - LOW))
        if u < .8:
            return self.rng.choice(grid()[1:])
        return grid()[0]  # nominal: never enters

    def reset(self, *, seed=None, options=None):
        from fairtest.env import B7Env
        from fairtest.runner import _STATE
        super().reset(seed=seed)
        if self.env is not None:
            self.env.close()
        case = _STATE['cases'][self.rng.choice(self.cases)]
        self.plan = self._plan()
        self.env = B7Env(_STATE['base'], case, _STATE['b1'], _STATE['b2'], case['physical_initial_state'],
                         target='drl_ext', target_h=0., durations=self.plan.durations())
        self.env.reset()  # creates the target controller (called only from step on)
        self.env.target_control.perceiver.perceive_m = self.perceive_m
        if hasattr(self.plan, 'reset'):
            self.plan.reset()
        return self._ego_obs(0., 0), {}

    def _ego_obs(self, previous_a, tick):
        """The observation at a decision: the latest state is current, earlier ones come from the tick history."""
        sim = self.env.sim
        p = self.env.target_control.perceiver.delayed(self.lag, current=perception(sim, self.perceive_m))
        return compose(p, sim, previous_a, self.env.cruise, tick)

    def _reward(self, request):
        v = float(self.env.sim.inner.road.vehicles[1].speed)
        reward = -.1 * abs(v - self.env.cruise) / max(self.env.cruise, 1.) - (.05 if request <= -4. else 0.)
        contact = self.env.outcome == 'intended_contact'
        if contact:
            reward -= 10.
        return float(reward), contact

    def step(self, action):
        request = ACCELS[int(action)]
        self.env.target_control.request = request
        _, _, done, _, _ = self.env.step(self.plan.act(self.env))
        reward, contact = self._reward(request)
        obs = self._ego_obs(self.env.previous_target, self.env.tick)
        info = dict(contact=contact) if done else {}
        return obs, reward, bool(done), False, info

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def train(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    a.out.mkdir(parents=True, exist_ok=True)
    vec = SubprocVecEnv([(lambda i=i: EgoEnv(seed=a.seed * 1000 + i, lag=a.lag, perceive_m=a.perceive)) for i in range(a.envs)])
    model = PPO('MlpPolicy', vec, n_steps=256, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=.99,
                ent_coef=.01, seed=a.seed, verbose=0, device='cpu')
    log, done_steps = [], 0
    while done_steps < a.steps:
        model.learn(total_timesteps=a.checkpoint, reset_num_timesteps=False)
        done_steps += a.checkpoint
        tag = f'_p{a.perceive:g}' if a.perceive is not None else (f'_lag{a.lag}' if a.lag else '')
        model.save(a.out / f'ppo_drl_sut{tag}_s{a.seed}_{done_steps}.zip')  # explicit suffix: 'p1.5' would read as one
        log.append(dict(steps=done_steps))
        print(json.dumps(log[-1]), flush=True)
    (a.out / 'train_log.json').write_text(json.dumps(log, indent=1), encoding='utf-8')
    vec.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('out', type=Path)
    t.add_argument('--steps', type=int, default=1_000_000)
    t.add_argument('--envs', type=int, default=15)
    t.add_argument('--seed', type=int, default=1)
    t.add_argument('--checkpoint', type=int, default=250_000)
    t.add_argument('--lag', type=int, default=0, help='perception latency in 0.1 s ticks')
    t.add_argument('--perceive', type=float, help='lane-assignment perception threshold (m)')
    train(ap.parse_args())
