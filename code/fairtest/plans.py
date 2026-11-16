"""Open-loop cut-in plan family and adapters for the frozen B5 agent.

A plan requests lane goal 0 with a pre-entry longitudinal request until `entry_s`, then lane
goal 4 with a post-entry request. entry_s=0 continues the recorded lane change immediately;
entry_s=inf never enters (nominal-like). Plans never read the target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from fairtest import devboot  # noqa: F401
from src.agents.braking_entry_agent import BrakeView, BrakingEntryAgent
from src.evaluation.adversarial_b5_policy_audit import view_from_states
from src.evaluation.safety_metrics import VehicleState


@dataclass(frozen=True)
class Plan:
    entry_s: float
    pre_mps2: float
    post_mps2: float
    duration_s: float | None = None  # None: B2 grid (2..6 s, shortest feasible)

    @property
    def name(self):
        entry = 'never' if math.isinf(self.entry_s) else f'{self.entry_s:g}'
        dur = 'grid' if self.duration_s is None else f'{self.duration_s:g}'
        return f'plan_e{entry}_pre{self.pre_mps2:g}_post{self.post_mps2:g}_d{dur}'

    def durations(self):
        return (2., 3., 4., 5., 6.) if self.duration_s is None else (float(self.duration_s),)

    def act(self, env):
        if env.tick * .1 + 1e-9 >= self.entry_s:
            return (4., self.post_mps2)
        return (0., self.pre_mps2)

    def to_json(self):
        d = asdict(self)
        d['entry_s'] = None if math.isinf(self.entry_s) else self.entry_s
        d['name'] = self.name
        return d


class FrozenB5:
    """Frozen B5 braking_entry policy; requires the exact B2 planner type."""
    name = 'b5_braking_entry'
    frozen_planner = True

    def __init__(self):
        self.agent = BrakingEntryAgent()

    def durations(self):
        return (2., 3., 4., 5., 6.)

    def reset(self):
        self.agent.reset()

    def act(self, env):
        g, t = [VehicleState.from_vehicle(v) for v in env.sim.inner.road.vehicles]
        dto = BrakeView(view_from_states(env.tick, g, t, env.ys), float(env.sim.inner.vehicle.speed),
                        env.previous_a, env.previous_steer)
        action, _ = self.agent.act(dto, env.lateral_planner)
        # Exact B2 decoding: lane goal (0, 4)[a // 3], request (0, -3, +3)[a % 3].
        return ((0., 4.)[action // 3], (0., -3., 3.)[action % 3])

    def to_json(self):
        return dict(name=self.name)


def grid(entries=(0., .5, 1., 1.5, 2., 3., 4., 5., 6.), pres=(0., -1., -2., -3.),
         posts=(0., -1.5, -3.), durations=(None, 4.)):
    plans = [Plan(math.inf, 0., 0., None)]
    for d in durations:
        for e in entries:
            for post in posts:
                for pre in (pres if e > 0 else (0.,)):
                    plans.append(Plan(float(e), float(pre), float(post), d))
    return plans
