"""Online fair adversary: one call attacks one SUT on one initial scenario.

This composes the evaluated components into a single agent (offline-equivalent to the frozen hybrid
method `learned_1 -> cont_search` of B13):
  0. reference sweep (0 SUT queries): the R157 CC driver runs every grid plan of the case;
  1. candidates: CC-valid, CC-collision-free cut-in plans within the realism limit;
  2. ranking: the per-SUT learned selector if one is given, else the R157 boundary rule
     (largest CC braking demand not above the severity target, then most natural);
  3. one SUT query on the top-ranked plan; stop at a fair failure;
  4. continuous refinement: CMA-ES from the continuous CC boundary (itself a CC-only search) with
     the remaining SUT budget, fairness-constrained (`b11_search.search`).
Ties are broken by plan name (order independent). `warm_start` reproduces an offline run exactly.

  from fairtest.fair_agent import FairAdversary
  record = FairAdversary(models).attack(case, 'following', 0.)
"""
from __future__ import annotations

import math

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest import runner
from fairtest.b11_search import search
from fairtest.b8_selector import feature_row
from fairtest.boundary_planner import NOMINAL, cc_demand
from fairtest.features import Realism
from fairtest.plans import grid

CC = ('cc_r157', 0.)


class FairAdversary:
    def __init__(self, models=None, budget=36, severity=1., realism_limit=math.inf, realism=None):
        self.models = models or {}
        self.budget, self.severity, self.realism_limit = int(budget), float(severity), float(realism_limit)
        self.realism = realism or Realism()

    def sweep(self, case):
        """CC reference runs of every grid plan (0 SUT queries)."""
        return {p.name: (p, runner.run_episode(case, p, *CC)) for p in grid()}

    def candidates(self, case, sweep):
        out = []
        for name, (plan, cc) in sorted(sweep.items()):
            if name == NOMINAL or not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
                continue
            exc = self.realism.score(cc['features'])['max_exceedance']
            exc = math.inf if exc is None else exc
            if exc <= self.realism_limit:
                out.append(dict(name=name, plan=plan, cc=cc, demand=cc_demand(cc), exc=exc,
                                x=feature_row(case, name, cc, min(exc, 20.))))
        return out

    def boundary_order(self, cand):
        key = lambda c: (-c['demand'], c['exc'], c['name'])
        feasible = sorted([c for c in cand if c['demand'] <= self.severity + 1e-9] or cand, key=key)
        names = {c['name'] for c in feasible}
        return feasible + sorted([c for c in cand if c['name'] not in names], key=key)

    def rank(self, cand, sut_key):
        model = self.models.get(sut_key)
        order = self.boundary_order(cand)
        if model is None or not order:
            return order, 'r157_boundary'
        prob = model.predict_proba(np.array([c['x'] for c in order], float))[:, 1]
        return [order[i] for i in np.argsort(-prob, kind='stable')], 'learned_selector'

    def attack(self, case, sut, h=0., warm_start=None):
        sut_key = f'{sut}_H{h:g}'
        sweep = self.sweep(case)
        cand = self.candidates(case, sweep)
        order, ranker = self.rank(cand, sut_key)
        rec = dict(case_id=case['case_id'], sut=sut_key, ranker=ranker, candidates=len(cand), sut_queries=0,
                   cc_sims=len(sweep), found=False, stage=None, plan=None)
        if order:
            top = order[0]
            row = runner.run_episode(case, top['plan'], sut, h)
            rec['sut_queries'] += 1
            if row['valid'] and row['contact']:
                rec.update(found=True, stage='grid_top1', plan=top['name'], cc_demand=top['demand'], realism=top['exc'])
                return rec
        bnd_plan = self.boundary_order(cand)[0]['plan'] if cand else None
        if warm_start is not None:
            x0 = np.asarray(warm_start, float)
        elif bnd_plan is not None:
            x0 = np.array([min(8., bnd_plan.entry_s), bnd_plan.pre_mps2, bnd_plan.post_mps2, bnd_plan.post_mps2,
                           2. if bnd_plan.duration_s is None else bnd_plan.duration_s])
        else:
            x0 = None
        from fairtest.b11_search import HIGH, LOW
        x0 = (LOW + HIGH) / 2 if x0 is None else x0
        bnd, _, n_cc = search(case, 'cc_r157', 0., x0, self.budget, fair_only=False)
        rec['cc_sims'] += n_cc
        start = x0 if bnd is None else np.asarray(bnd['theta'])
        best, n_sut, n_cc = search(case, sut, h, start, self.budget)
        rec['sut_queries'] += n_sut
        rec['cc_sims'] += n_cc
        if best and best['fair']:
            rec.update(found=True, stage='continuous', plan=dict(theta=best['theta']), cc_demand=best['cc_demand'])
        return rec
