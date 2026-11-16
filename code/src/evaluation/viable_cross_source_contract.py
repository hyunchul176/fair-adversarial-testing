"""Pure contracts for held-out-source attack replay."""


FAILURE_KEYS = (
    "pair_contact",
    "background_contact",
    "unattributed_crash",
    "offroad",
    "wrong_way",
)


def complete_tape(row, horizon=120):
    """Eligibility depends only on the held-out source rollout."""
    return bool(
        row.get("valid_reset")
        and row.get("full_horizon")
        and row.get("steps") == horizon
        and len(row.get("generator_controls", ())) == horizon
    )


def reference_safe(row, minimum_progress=80.0):
    return bool(
        row.get("valid_reset")
        and row.get("full_horizon")
        and not any(row.get(key) for key in FAILURE_KEYS)
        and not row.get("contacts")
        and row.get("target_progress_m", 0.0) >= minimum_progress
    )


def paired_contact_class(source_safe, reference_safe_value):
    if source_safe and reference_safe_value:
        return "both_safe"
    if source_safe:
        return "source_only_safe"
    if reference_safe_value:
        return "reference_only_safe"
    return "neither_safe"

