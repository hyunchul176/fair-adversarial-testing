"""Pure action arithmetic for an explicitly modified-controller diagnostic."""
ARMS = {'baseline': (False, False), 'brake_priority': (True, False),
        'target_lane_hold': (False, True), 'combined': (True, True)}


def brake_priority(jerk_limited, post_stop, shield_intervened, enabled):
    """Preserve stronger post-stop braking only when the shield intervened.

    This intentionally relaxes the generator's jerk constraint. It uses the
    post-stop command without changing that guard or acceleration bounds.
    It does not establish a final stop/safety guarantee for the old pipeline.
    """
    if enabled and shield_intervened and post_stop < 0:
        return min(jerk_limited, post_stop)
    return jerk_limited
