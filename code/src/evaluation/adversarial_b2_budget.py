"""Separate fixed B2 limits, reusing the durable B1 reservation journal engine."""
from src.evaluation.adversarial_b1_budget import NativeBudget


class B2Budget(NativeBudget):
    VERSION='adversarial-B2-fixed-support-budget-v1'
    LIMITS={'sentinel':dict(starts=16,actors=384,ticks=1920),
            'search':dict(starts=832,actors=19968,ticks=99840)}
    GLOBAL_LIMITS=dict(starts=848,actors=20352,ticks=101760)
    RUN_LIMITS={'search:42':dict(starts=832,actors=19968,ticks=99840)}
