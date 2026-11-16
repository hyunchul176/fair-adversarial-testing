"""Fixed B4 reservations; partial slots cannot be retried."""
from src.evaluation.adversarial_b1_budget import NativeBudget


class B4Budget(NativeBudget):
    VERSION='adversarial-B4-causal-entry-budget-v1'
    LIMITS={'sentinel':dict(starts=16,actors=384,ticks=1920),
            'evaluation':dict(starts=352,actors=8448,ticks=42240)}
    GLOBAL_LIMITS=dict(starts=368,actors=8832,ticks=44160)
    RUN_LIMITS={}

    def partial(self, attempt_id, reason):
        self.fail(attempt_id,'Incomplete fixed B4 episode: '+reason)
