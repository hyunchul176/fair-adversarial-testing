"""B8 learning environment: SUT-specific adversary with CC-preventability and realism terms.

One episode = one native SUT rollout of the policy. If the SUT outcome is risky (contact or
TTC < 1.5 s), the realised high-level action sequence is replayed open-loop against the R157 CC
driver to label preventability. The generator's low-level control depends only on its own
state, so the replayed generator trajectory is identical until the CC target's response.
Observations exclude the SUT identity, H, case id, labels and any future information.
"""
from __future__ import annotations

import math
import os
import random

import gymnasium as gym
import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.env import B7Env
from fairtest.features import Realism, simulated_features
from fairtest.runner import _STATE, _init

LANES = (0., 4.)
REQUESTS = (1., 0., -1., -2., -3.)
OBS_DIM = 16
WIDTH_RANGE = tuple(float(x) for x in os.environ['FAIRTEST_VEH_W_RANGE'].split(',')) if os.environ.get('FAIRTEST_VEH_W_RANGE') else None
# authority randomisation for training: relative deceleration headroom r ~ log-uniform(lo, hi) per episode (B22r floor)
FLOOR_REL_RANGE = tuple(float(x) for x in os.environ['FAIRTEST_FLOOR_REL_RANGE'].split(',')) if os.environ.get('FAIRTEST_FLOOR_REL_RANGE') else None


class B8AdversaryEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, case_ids, sut=('following', 0.), *, fair_reward=1., near_miss_reward=.3,
                 unfair_penalty=1., realism_weight=0., durations=(2., 3., 4., 5., 6.), seed=0,
                 sequential=False):
        super().__init__()
        _init()
        self.case_ids = list(case_ids)
        self.sut = (sut[0], float(sut[1]))
        self.fair_reward, self.near_miss_reward = fair_reward, near_miss_reward
        self.unfair_penalty, self.realism_weight = unfair_penalty, realism_weight
        self.durations = tuple(durations)
        self.rng = random.Random(seed)
        self.sequential, self.cursor = sequential, 0
        self.realism = Realism()
        self.action_space = gym.spaces.Discrete(len(LANES) * len(REQUESTS))
        self.observation_space = gym.spaces.Box(-5., 5., shape=(OBS_DIM,), dtype=np.float32)
        self.env = None
        self.vehicle_width = None

    # ------------------------------------------------------------------ helpers
    def _make(self, case, target, h):
        return B7Env(_STATE['base'], case, _STATE['b1'], _STATE['b2'], case['physical_initial_state'],
                     target=target, target_h=h, durations=self.durations, vehicle_width=self.vehicle_width,
                     floor_rel=getattr(self, 'floor_rel', None))

    @staticmethod
    def decode(action):
        return LANES[int(action) // len(REQUESTS)], REQUESTS[int(action) % len(REQUESTS)]

    def _obs(self):
        e = self.env
        g, t = [v for v in e.sim.inner.road.vehicles]
        gx, gy = float(g.position[0]), float(g.position[1])
        tx, ty = float(t.position[0]), float(t.position[1])
        gvx, gvy = float(g.velocity[0]), float(g.velocity[1])
        tvx = float(t.velocity[0])
        goal = e.lateral_planner.goal if e.lateral_planner.reference is not None else 0.
        init = e.initial
        return np.clip(np.array([
            (gx - tx) / 50., (gy - ty) / 4., gvx / 40., tvx / 40., (tvx - gvx) / 10., gvy / 2., gy / 4.,
            e.previous_a / 3., e.previous_steer / .3, e.tick / 120., goal / 4., float(self.entered),
            (init[0]['x'] - init[1]['x'] - 5.) / 50., (init[1]['vx'] - init[0]['vx']) / 10.,
            e.previous_target / 5., 1.], dtype=np.float32), -5., 5.)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.env is not None:
            self.env.close()
        if options and 'case_id' in options:
            cid = options['case_id']
        elif self.sequential:
            cid = self.case_ids[self.cursor % len(self.case_ids)]
            self.cursor += 1
        else:
            cid = self.rng.choice(self.case_ids)
        self.case = _STATE['cases'][cid]
        # geometry randomisation for training (FAIRTEST_VEH_W_RANGE='lo,hi'); the CC replay uses the same width
        self.vehicle_width = self.rng.uniform(*WIDTH_RANGE) if WIDTH_RANGE else None
        self.floor_rel = float(np.exp(self.rng.uniform(*np.log(FLOOR_REL_RANGE)))) if FLOOR_REL_RANGE else None
        self.env = self._make(self.case, *self.sut)
        self.env.reset()
        self.actions, self.entered = [], False
        return self._obs(), dict(case_id=cid)

    def step(self, action):
        lane, request = self.decode(action)
        self.actions.append((lane, request))
        self.entered = self.entered or lane == 4.
        _, _, done, _, _ = self.env.step((lane, request))
        if not done:
            return self._obs(), 0., False, False, {}
        row = self.env.result()
        obs = self._obs()
        self.env.close()
        self.env = None
        info = self._score(row)
        return obs, info['reward'], True, False, info

    def _score(self, row):
        valid = bool(row['maneuver_valid_precontact'])
        contact = bool(row['pair_contact'])
        near = bool(row['exclusive_ttc_event'])
        feats = simulated_features(row)
        real = self.realism.score(feats) if feats.get('cut_in') else None
        exceed = (real['max_exceedance'] or 0.) if real else 0.
        cc_contact = None
        if valid and (contact or near):
            cc_contact = self._cc_replay()
        fair = valid and contact and cc_contact is False
        unfair = valid and contact and cc_contact is True
        fair_near = valid and near and not contact and cc_contact is False
        reward = (self.fair_reward * fair + self.near_miss_reward * fair_near
                  - self.unfair_penalty * unfair - self.realism_weight * exceed - (0. if valid else 1.))
        return dict(reward=float(reward), case_id=self.case['case_id'], valid=valid, contact=contact,
                    near_miss=near, cc_contact=cc_contact, fair=fair, unfair=unfair, fair_near=fair_near,
                    realism_exceedance=exceed, cut_in=bool(feats.get('cut_in')), steps=len(row['trace']))

    def _cc_replay(self):
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
            return bool(env.result()['pair_contact'])
        finally:
            env.close()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
