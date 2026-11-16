"""B3 fixed evaluation budget, reusing the durable reservation journal engine."""
from src.evaluation.adversarial_b1_budget import NativeBudget


class B3Budget(NativeBudget):
    VERSION = 'adversarial-B3-frozen-transfer-budget-v1'
    LIMITS = {'sentinel': dict(starts=16, actors=384, ticks=1920),
              'evaluation': dict(starts=160, actors=3840, ticks=19200)}
    GLOBAL_LIMITS = dict(starts=176, actors=4224, ticks=21120)
    RUN_LIMITS = {}

    def partial(self, attempt_id, reason):
        # A prematurely closed fixed B3 slot cannot be retried. Keep its entire
        # reservation and record the incomplete termination as a failure.
        self.fail(attempt_id, 'Incomplete fixed B3 episode: ' + reason)
