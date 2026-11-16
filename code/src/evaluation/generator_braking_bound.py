"""Diagnostic actuator interventions; neither a nominal controller nor a safety claim."""
from contextlib import contextmanager

ARMS=('following_jerk','maximum_brake_jerk','maximum_brake_unlimited')

@contextmanager
def intervention(module,arm):
    """Patch only generator requests/last stage; restore even when a rollout fails.

    Dedicated sequential worker only. Target control and all world operations are
    untouched. Unlimited means removing the jerk cap, not the acceleration bound.
    """
    if arm not in ARMS:raise ValueError('Unknown arm')
    nominal,brake=module.nominal_action,module.brake_priority
    def maximum(*args,**kwargs):
        action=nominal(*args,**kwargs).copy()
        action[0]=-1.
        return action
    def unlimited(limited,guarded,shield_intervened,enabled):
        return guarded
    try:
        if arm!='following_jerk':module.nominal_action=maximum
        if arm=='maximum_brake_unlimited':module.brake_priority=unlimited
        yield
    finally:
        module.nominal_action,module.brake_priority=nominal,brake
