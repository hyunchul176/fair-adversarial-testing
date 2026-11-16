"""Requested TTC-risk primitives and finite preflight accounting; no simulator."""
import math
from src.evaluation.adversarial_agent_core import ttc_risk
from src.evaluation.adversarial_a2_support import BudgetLedger


def check_goal(goal):
    if isinstance(goal,bool) or not math.isfinite(goal) or not 0<goal<1:
        raise ValueError('Finite interior risk goal required')
    return float(goal)


def category(peak,goal,outcome,tolerance=.05):
    if outcome=='maneuver_invalid':return 'invalid'
    if outcome=='intended_contact':return 'contact'
    if outcome!='horizon':raise ValueError('Only terminal outcomes can be classified')
    if peak<goal-tolerance-1e-12:return 'noncontact_below_band'
    if peak>goal+tolerance+1e-12:return 'noncontact_above_band'
    return 'noncontact_hit'


class GoalAccumulator:
    def __init__(self,initial_ttc,goal):
        self.goal=check_goal(goal);self.peak=ttc_risk(initial_ttc)
        self.initial=self.utility=1-abs(self.peak-self.goal)
        self.total=0.;self.closed=False
    def advance(self,ttc,outcome='ongoing'):
        if self.closed:raise RuntimeError('Episode already ended')
        if outcome not in ('ongoing','horizon','intended_contact','maneuver_invalid'):
            raise ValueError('Unknown outcome')
        self.peak=max(self.peak,ttc_risk(ttc))
        value=(-2. if outcome=='maneuver_invalid' else -1. if outcome=='intended_contact'
               else 1-abs(self.peak-self.goal))
        reward=value-self.utility;self.total+=reward;self.utility=value
        self.closed=outcome!='ongoing'
        return reward


def release_rule(observation,depth):
    if len(observation)!=28 or depth not in (1,2):raise ValueError('28D observation and depth1/2 required')
    if not all(math.isfinite(float(x)) for x in observation):raise ValueError('Nonfinite observation')
    goal=check_goal(float(observation[27]))
    phase=list(observation[23:27])
    if any(float(x) not in (0.,1.) for x in phase) or sum(phase)!=1:
        raise ValueError('One-hot pulse phase required')
    if phase[2] or phase[3]:return 0
    if observation[22]>=goal:return 3
    return depth if phase[0] else 0


class GoalBudget(BudgetLedger):
    LIMITS={'sentinel':dict(episodes=16,actors=384,ticks=1920),
            'wrapper':dict(episodes=8,actors=192,ticks=960),
            'early_release':dict(episodes=64,actors=1536,ticks=7680),
            'causal_rules':dict(episodes=96,actors=2304,ticks=11520)}
    def reserve_action(self,phase):
        super().reserve_action(phase)
        if sum(r['ticks'] for r in self.rows.values())+5>22080:
            raise RuntimeError('Combined revision-1 physics budget exhausted')
