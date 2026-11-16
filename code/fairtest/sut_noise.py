"""B50 (B50 design, available on request): a perception-noise SUT ladder,
added without changing any frozen file.

`<base>~n<sigma>` (e.g. following_aeb~n2, with the base SUT's H): the SUT controller sees the attacking (generator)
vehicle with AR(1) Gaussian noise on its longitudinal position (sigma m) and speed (sigma / 2 m/s), correlation time
0.5 s, updated at every control step (0.1 s). Lateral position, heading and the SUT's own state are exact. The noisy
values exist only while the controller is called; the simulated state is restored bit for bit.
The noise sequence is fixed by (case, SUT name, replicate): every replay of one attack sees the same noise, so the
certificate and the front-contact check reproduce it. The replicate (env FAIRTEST_NOISE_REP, default 0) draws fresh
realisations for the reproducibility test. Reference drivers (the judging CC, the shield's CC, the certificate's
references) are noise-free: fairness stays the counterfactual of the normative reference.

  from fairtest import sut_noise; sut_noise.install()   # before any worker process is created (fork inherits it)
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

TAU_S, DT, SPEED_RATIO = .5, .1, .5
PATTERN = re.compile(r'^(?P<base>.+)~n(?P<sigma>[0-9]+(?:\.[0-9]+)?)$')
NOISE_SALT = 'acarl-b50:'  # hash prefix of the noise seeds, kept unchanged so that the noise draws stay the same


def split(target):
    """('following_aeb~n2') -> ('following_aeb', 2.0); a plain target -> (target, None)."""
    m = PATTERN.match(target)
    return (m['base'], float(m['sigma'])) if m else (target, None)


class NoisyPerception:
    """Wraps a target controller: the generator's x and speed are perturbed during the call only."""

    def __init__(self, inner, sigma, seed):
        self.inner, self.sigma = inner, float(sigma)
        self.rng = np.random.default_rng(seed)
        self.rho = float(np.exp(-DT / TAU_S))
        self.ex = self.ev = None

    def __getattr__(self, name):  # attributes of the wrapped controller (only reached when missing here)
        if name == 'inner':
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _advance(self):
        zx, zv = self.rng.standard_normal(2)
        if self.ex is None:  # stationary start
            self.ex, self.ev = self.sigma * zx, SPEED_RATIO * self.sigma * zv
        else:
            s = float(np.sqrt(1. - self.rho ** 2))
            self.ex = self.rho * self.ex + s * self.sigma * zx
            self.ev = self.rho * self.ev + s * SPEED_RATIO * self.sigma * zv

    def __call__(self, sim, *args, **kwargs):
        self._advance()
        g = sim.inner.vehicle
        x0, v0 = float(g.position[0]), float(g.speed)
        g.position[0] = x0 + self.ex
        g.speed = max(0., v0 + self.ev)
        try:
            return self.inner(sim, *args, **kwargs)
        finally:
            g.position[0], g.speed = x0, v0


def noise_seed(case_id, name, replicate):
    return int(hashlib.sha256(f'{NOISE_SALT}{case_id}|{name}|{replicate}'.encode()).hexdigest()[:16], 16)


def install():
    """Patch fairtest.env.B7Env in place: accept `<base>~n<sigma>` targets. Idempotent."""
    from fairtest import env as E
    if getattr(E.B7Env, '_b50_noise', False):
        return
    init0, reset0 = E.B7Env.__init__, E.B7Env.reset

    def __init__(self, *args, target='following', **kwargs):
        base, sigma = split(target)
        self.noise_name, self.noise_sigma = (target, sigma) if sigma is not None else (None, None)
        init0(self, *args, target=base, **kwargs)

    def reset(self, *, seed=None, options=None):
        out = reset0(self, seed=seed, options=options)
        if self.noise_sigma is not None:  # sigma 0 also goes through the wrapper (C0 checks the save / restore path)
            rep = int(os.environ.get('FAIRTEST_NOISE_REP', '0'))
            self.target_control = NoisyPerception(self.target_control, self.noise_sigma,
                                                  noise_seed(self.case.get('case_id'), self.noise_name, rep))
        return out

    E.B7Env.__init__, E.B7Env.reset = __init__, reset
    E.B7Env._b50_noise = True
