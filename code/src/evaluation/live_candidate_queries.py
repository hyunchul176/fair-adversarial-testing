"""Budgeted real simulator calls; additive adapter, not part of frozen confirmation."""
from __future__ import annotations
import copy
import math
import time
from collections.abc import Callable


class LiveCandidateQueries:
    """Same returned fields as cached CandidateQueries, with measured execution cost.

    A failed simulator call consumes an attempt. No hidden retries, future
    candidate outcomes, envelope labels, or uncounted candidate precomputation.
    """
    def __init__(self, evaluator: Callable, candidate_count: int, budget: int,
                 expected_initial_hash: str, clock: Callable = time.perf_counter):
        if not 0 <= budget <= candidate_count or not expected_initial_hash:
            raise ValueError('Invalid query budget or missing initial-world identity')
        self._evaluate = evaluator
        self._clock = clock
        self._count = candidate_count
        self._budget = budget
        self._world = expected_initial_hash
        self._trace = []
        self._outcomes = {}

    @property
    def trace(self):
        return copy.deepcopy(self._trace)

    def cached_outcome(self, index):
        """Reuse an already paid result; cannot disclose an unqueried candidate."""
        if index not in self._outcomes:
            raise ValueError('Candidate has no successfully paid observation')
        return dict(self._outcomes[index])

    def query(self, index):
        if not isinstance(index, int) or not 0 <= index < self._count:
            raise ValueError('Invalid candidate index')
        if any(t['index'] == index for t in self._trace):
            raise ValueError('Duplicate candidate attempt; no hidden retries')
        if len(self._trace) >= self._budget:
            raise ValueError('Query budget exceeded')
        entry = {'index': index, 'status': 'started', 'known_rollouts': [],
                 'rollout_count_known': False}
        self._trace.append(entry)
        started = self._clock()
        try:
            result = self._evaluate(index)
            phases = [name for name in ('attack', 'nominal', 'witness') if result.get(name) is not None]
            entry['known_rollouts'] = phases
            entry['rollout_count_known'] = True
            entry['initial_hashes'] = {name: result[name].get('initial_hash') for name in phases}
            if 'attack' not in phases or 'nominal' not in phases:
                raise ValueError('Missing paired rollout')
            for name in phases:
                if result[name].get('initial_hash') != self._world:
                    raise ValueError('Simulator reset differs from observed initial world')
            delta = float(result['paired_risk_delta'])
            if not math.isfinite(delta) or not isinstance(result['admissible'], bool):
                raise ValueError('Invalid candidate outcome')
            if result['admissible'] and 'witness' not in phases:
                raise ValueError('Admissible result requires its recoverability witness')
            outcome = {'index': index, 'admissible': result['admissible'], 'paired_risk_delta': delta}
            entry.update(status='complete', admissible=outcome['admissible'],
                         rejection_reasons=list(result.get('rejection_reasons', [])),
                         simulator_steps=sum(int(result[n].get('steps', 0)) for n in phases))
            self._outcomes[index] = outcome
            return dict(outcome)
        except Exception as exc:
            entry.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            entry['wall_seconds'] = self._clock() - started

    def cost_summary(self):
        return {'attempted_candidate_queries': len(self._trace),
                'successful_candidate_queries': len(self._outcomes),
                'failed_candidate_queries': sum(t['status'] == 'failed' for t in self._trace),
                'known_rollout_count': sum(len(t['known_rollouts']) for t in self._trace),
                'unknown_rollout_count_attempts': sum(not t['rollout_count_known'] for t in self._trace),
                'known_simulator_steps': sum(t.get('simulator_steps', 0) for t in self._trace),
                'inadmissible_observations': sum(not v['admissible'] for v in self._outcomes.values()),
                'wall_seconds': sum(t['wall_seconds'] for t in self._trace)}


def feedback_evaluator(context, candidate_parameters, scene_seed, defender_kind,
                       model=None, horizon=120, vehicles_count=15):
    """Bind actual highway-env paired evaluations; simulator imports are lazy.

    Caller must run this on a lab worker and separately obtain a nominal probe
    and its initial hash. Final executions/probe costs are separate from queries.
    """
    context = copy.deepcopy(context)
    candidates = copy.deepcopy(candidate_parameters)

    def evaluate(index):
        from .multimode_feedback_probe import FeedbackManeuver, paired_feedback_result
        maneuver = FeedbackManeuver(scenario=context['scenario'],
            initial_gap=float(context['initial_gap']), generator_speed=float(context['generator_speed']),
            target_speed=float(context['target_speed']), **candidates[index])
        return paired_feedback_result(maneuver, int(scene_seed), defender_kind, model=model,
                                      horizon=horizon, vehicles_count=vehicles_count)
    return evaluate
