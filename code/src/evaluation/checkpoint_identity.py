"""Stable identities for policy parameters and frozen training specifications."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np


TRAINING_SPEC_KEYS = (
    "phases",
    "n_envs",
    "n_steps",
    "batch_size",
    "learning_rate",
    "n_epochs",
    "ent_coef",
    "gamma",
    "gae_lambda",
    "clip_range",
    "log_std_init",
    "architecture",
)


def training_spec(protocol: Mapping) -> dict:
    """Return only fields that determine the final-v3 training procedure."""
    missing = [key for key in TRAINING_SPEC_KEYS if key not in protocol]
    if missing:
        raise ValueError(f"training protocol is missing fields: {missing}")
    return {key: protocol[key] for key in TRAINING_SPEC_KEYS}


def training_spec_sha256(protocol: Mapping) -> str:
    payload = json.dumps(
        training_spec(protocol), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_array_mapping_sha256(values: Mapping[str, object]) -> str:
    """Hash named numeric arrays without archive timestamps or container metadata."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object array is not a stable policy parameter: {name}")
        canonical_dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
        metadata = json.dumps(
            {"name": name, "dtype": canonical_dtype.str, "shape": list(canonical.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = canonical.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def policy_state_sha256(model) -> str:
    return canonical_array_mapping_sha256(model.policy.state_dict())
