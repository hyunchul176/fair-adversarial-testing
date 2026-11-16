"""CEM baselines and an exploratory variant robust to flat sparse scores.

The guarded variant is an engineering baseline, not a claimed new optimizer.
It avoids fitting arbitrary elites when scores contain no ranking signal and
reserves a fixed uniform component after a useful region is discovered.
"""
import numpy as np


class PopulationSearch:
    def __init__(self, method, seed, population, dimensions, elite_count=2, min_std=.1, uniform_count=2):
        if method not in {'random', 'cem', 'guarded_cem'}:
            raise ValueError('Unknown search method')
        if not 1 <= elite_count <= population or not 0 <= uniform_count <= population:
            raise ValueError('Invalid search counts')
        self.method, self.population, self.dimensions = method, population, dimensions
        self.elite_count, self.min_std, self.uniform_count = elite_count, min_std, uniform_count
        self.first = np.random.default_rng(seed).uniform(size=(population, dimensions))
        self.rng = np.random.default_rng(seed + 1)
        self.mean, self.std = np.full(dimensions, .5), np.full(dimensions, .3)
        self.informative = False
        self.reason = 'initial_uniform'

    def sample(self, generation):
        if generation == 0:
            return self.first.copy()
        if self.method == 'random' or (self.method == 'guarded_cem' and not self.informative):
            return self.rng.uniform(size=(self.population, self.dimensions))
        uniform_count = self.uniform_count if self.method == 'guarded_cem' else 0
        uniform = self.rng.uniform(size=(uniform_count, self.dimensions))
        guided = np.clip(self.rng.normal(self.mean, self.std,
                         size=(self.population - uniform_count, self.dimensions)), 0, 1)
        return np.concatenate([uniform, guided], axis=0)

    def update(self, batch):
        if self.method == 'random':
            return
        valid = [r for r in batch if r['admissible']]
        if self.method == 'guarded_cem':
            scores = [r['score'] for r in valid]
            self.informative = bool(scores and max(scores) > 1e-12 and max(scores) - min(scores) > 1e-12)
            if not self.informative:
                self.reason = 'no_positive_ranking_signal_keep_uniform'
                return
            valid = [r for r in valid if r['score'] > 1e-12]
        elite = sorted(valid, key=lambda r: -r['score'])[:self.elite_count]
        if elite:
            coordinates = np.asarray([r['normalized'] for r in elite])
            self.mean = .3 * self.mean + .7 * coordinates.mean(axis=0)
            self.std = np.maximum(self.min_std, .3 * self.std + .7 * coordinates.std(axis=0))
            self.reason = 'fit_positive_elites' if self.method == 'guarded_cem' else 'fit_elites_including_ties'

    def state(self):
        return dict(method=self.method, mean=self.mean.tolist(), std=self.std.tolist(),
                    informative=self.informative, reason=self.reason,
                    uniform_count=self.uniform_count if self.informative and self.method == 'guarded_cem' else None)
