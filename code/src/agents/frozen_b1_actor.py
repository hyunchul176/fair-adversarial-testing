"""Greedy, weights-only inference for the two registered reactive B1 actors.

No SB3/cloudpickle objects, optimizer, simulator, RNG or recurrent state are
loaded. The caller constructs the original causal B1 observation and explicitly
declares the B1 action family when loading. A 108D observation alone cannot
distinguish B1 from B2, whose six actions have different meanings.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from numbers import Integral
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F


VERSION = 'frozen-reactive-b1-actor-v1'
ACTION_FAMILY = 'B1'
ACTION_REQUESTS_MPS2 = (0., -3., 3.)
CHECKPOINT_SHA256 = {
    42: '3e7902bb0d59bb2f1216be3b5dd749d38c30a1bf28f97726050c6951cc257af3',
    123: '2f09162d4b06b265bf81a0fa88c3b625ab4b483be2d7e82c818d25763eeb7cf9',
}
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
_MEMBERS = frozenset(('data', 'pytorch_variables.pth', 'policy.pth',
                      'policy.optimizer.pth', '_stable_baselines3_version', 'system_info.txt'))
_SHAPES = {
    'mlp_extractor.policy_net.0.weight': (64, 108),
    'mlp_extractor.policy_net.0.bias': (64,),
    'mlp_extractor.policy_net.2.weight': (64, 64),
    'mlp_extractor.policy_net.2.bias': (64,),
    'mlp_extractor.value_net.0.weight': (64, 108),
    'mlp_extractor.value_net.0.bias': (64,),
    'mlp_extractor.value_net.2.weight': (64, 64),
    'mlp_extractor.value_net.2.bias': (64,),
    'action_net.weight': (3, 64),
    'action_net.bias': (3,),
    'value_net.weight': (1, 64),
    'value_net.bias': (1,),
}
_ACTOR_KEYS = (
    'mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.0.bias',
    'mlp_extractor.policy_net.2.weight', 'mlp_extractor.policy_net.2.bias',
    'action_net.weight', 'action_net.bias',
)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate checkpoint JSON key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Nonfinite checkpoint JSON value: ' + value)


def _validate_metadata(data, seed):
    if not isinstance(data, dict):
        raise ValueError('Checkpoint metadata must be a JSON object')
    for key, expected in (('seed', seed), ('num_timesteps', 16128), ('n_envs', 1)):
        if type(data.get(key)) is not int or data[key] != expected:
            raise ValueError('Unexpected checkpoint metadata: ' + key)
    kwargs = data.get('policy_kwargs')
    observation, action = data.get('observation_space'), data.get('action_space')
    policy = data.get('policy_class')
    if (not isinstance(kwargs, dict)
            or kwargs.get('net_arch') != {'pi': [64, 64], 'vf': [64, 64]}
            or kwargs.get('activation_fn') != "<class 'torch.nn.modules.activation.Tanh'>"
            or data.get('use_sde') is not False
            or not isinstance(policy, dict)
            or policy.get('__module__') != 'stable_baselines3.common.policies'):
        raise ValueError('Expected the registered 108-64-tanh-64-tanh-3 actor')
    if (not isinstance(observation, dict)
            or observation.get(':type:') != "<class 'gymnasium.spaces.box.Box'>"
            or observation.get('_shape') != [108]
            or observation.get('dtype') != 'float32'
            or observation.get('low_repr') != '-1.0'
            or observation.get('high_repr') != '1.0'):
        raise ValueError('Expected the normalized 108D B1 observation space')
    if (not isinstance(action, dict)
            or action.get(':type:') != "<class 'gymnasium.spaces.discrete.Discrete'>"
            or action.get('n') != '3' or action.get('start') != '0'
            or action.get('_shape') != [] or action.get('dtype') != 'int64'):
        raise ValueError('Expected B1 Discrete(3), not the B2 action family')


def _read_actor_weights(path, seed, expected_sha256):
    # Hash the exact bytes subsequently parsed, not a second read of the path.
    with Path(path).open('rb') as stream:
        raw = stream.read(_MAX_ARCHIVE_BYTES + 1)
    if len(raw) > _MAX_ARCHIVE_BYTES:
        raise ValueError('Checkpoint archive exceeds the registered size bound')
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('B1 checkpoint SHA256 mismatch')
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) != len(_MEMBERS)
                    or {entry.filename for entry in entries} != _MEMBERS
                    or sum(entry.file_size for entry in entries) > _MAX_ARCHIVE_BYTES
                    or any(entry.flag_bits & 1 for entry in entries)):
                raise ValueError('Unexpected checkpoint archive members or sizes')
            if archive.read('_stable_baselines3_version') != b'2.8.0':
                raise ValueError('Expected stable-baselines3 2.8.0 checkpoint')
            # :serialized: strings are inert JSON data. Never unpickle them.
            data = json.loads(archive.read('data'), object_pairs_hook=_json_object,
                              parse_constant=_reject_constant)
            _validate_metadata(data, seed)
            # Optimizer, pytorch_variables and cloudpickle data are not loaded.
            weights = torch.load(io.BytesIO(archive.read('policy.pth')),
                                 map_location='cpu', weights_only=True)
    except (zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Malformed B1 checkpoint archive') from exc
    if not isinstance(weights, dict) or set(weights) != set(_SHAPES):
        raise ValueError('Unexpected policy weight keys')
    for key, shape in _SHAPES.items():
        value = weights[key]
        if (type(value) is not torch.Tensor or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.layout != torch.strided
                or value.device.type != 'cpu' or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid checkpoint weight: ' + key)
    return tuple(weights[key].detach().clone().contiguous().requires_grad_(False)
                 for key in _ACTOR_KEYS)


def _observation_tensor(observation):
    if (not isinstance(observation, np.ndarray) or observation.dtype != np.float32
            or observation.shape != (108,) or not np.isfinite(observation).all()
            or np.any(observation < -1.) or np.any(observation > 1.)):
        raise ValueError('Expected one finite float32 normalized B1 observation of shape (108,)')
    masks = observation[104:]
    if (not np.all((masks == 0.) | (masks == 1.)) or masks[-1] != 1.
            or np.any(np.diff(masks) < 0.)):
        raise ValueError('B1 history masks must be a zero prefix followed by live frames')
    frames = observation[:104].reshape(4, 26)
    if np.any(frames[masks == 0.] != 0.):
        raise ValueError('Missing B1 history frames must be exactly zero padded')
    # An owned copy also supports noncontiguous/read-only numpy input safely.
    return torch.from_numpy(observation.copy()).reshape(1, 108)


@dataclass(frozen=True, init=False, repr=False, eq=False)
class FrozenB1Actor:
    """Use ``load(path, seed=42, action_family='B1')``; then ``act(obs)``.

    The caller owns causal history. This object only evaluates a frozen actor:
    no sampling, reset, learning, observation clipping or simulator access.
    Returned action IDs mean coast, brake and accelerate, in that order.
    """
    seed: int
    checkpoint_sha256: str
    action_family: str
    _weights: tuple

    def __init__(self):
        raise TypeError('Use FrozenB1Actor.load with an authenticated B1 checkpoint')

    @classmethod
    def load(cls, path, *, seed, action_family):
        if action_family != ACTION_FAMILY:
            raise ValueError('Frozen actor supports the B1 three-action family only')
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed not in CHECKPOINT_SHA256:
            raise ValueError('Only registered reactive B1 seeds 42 and 123 are supported')
        seed = int(seed)
        sha = CHECKPOINT_SHA256[seed]
        weights = _read_actor_weights(path, seed, sha)
        actor = object.__new__(cls)
        for name, value in (('seed', seed), ('checkpoint_sha256', sha),
                            ('action_family', ACTION_FAMILY), ('_weights', weights)):
            object.__setattr__(actor, name, value)
        return actor

    def _logits(self, observation):
        values = _observation_tensor(observation)
        w0, b0, w1, b1, wa, ba = self._weights
        with torch.inference_mode():
            hidden = torch.tanh(F.linear(values, w0, b0))
            hidden = torch.tanh(F.linear(hidden, w1, b1))
            logits = F.linear(hidden, wa, ba)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite frozen actor output')
            return logits

    def act(self, observation):
        """Return deterministic action 0/1/2, choosing the first index on ties."""
        return int(torch.argmax(self._logits(observation), dim=1).item())

    def probabilities(self, observation):
        """Return a fresh float32 array of the three categorical probabilities."""
        with torch.inference_mode():
            return torch.softmax(self._logits(observation), dim=1)[0].numpy().copy()
