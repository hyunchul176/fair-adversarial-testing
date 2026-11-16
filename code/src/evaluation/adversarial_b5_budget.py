"""Fixed B5 reservations; partial slots cannot be retried."""
from src.evaluation.adversarial_b1_budget import NativeBudget


class B5Budget(NativeBudget):
    VERSION='adversarial-B5-own-braking-entry-budget-v1'
    LIMITS={'sentinel':dict(starts=16,actors=384,ticks=1920),
            'evaluation':dict(starts=384,actors=9216,ticks=46080)}
    GLOBAL_LIMITS=dict(starts=400,actors=9600,ticks=48000)
    RUN_LIMITS={}

    def partial(self, attempt_id, reason):
        self.fail(attempt_id,'Incomplete fixed B5 episode: '+reason)
