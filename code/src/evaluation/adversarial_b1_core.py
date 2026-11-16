"""Pure Family-B action and causal-history primitives; no native simulation.

The initial-context transform applies to the complete actor/critic observation.
Historical time/request values are stored when each frame is formed, never
retroactively overwritten with a later time or request.
"""
from copy import deepcopy
import math
from numbers import Integral

import numpy as np

from src.evaluation.adversarial_agent_core import ttc_risk
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc


VERSION = 'adversarial-b1-core-v1'
REQUESTS = (0., -3., 3.)
FRAME_DIMENSIONS = 26
HISTORY_FRAMES = 4
OBSERVATION_DIMENSIONS = 108
ELAPSED_INDEX = 13
PREVIOUS_REQUEST_INDEX = 21


def acceleration_request(action):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral) or action not in range(3):
        raise ValueError('Family-B action must be an integer in [0, 2]')
    return REQUESTS[int(action)]


def causal_frame(g, t, *, acceleration, tick, lateral_y, previous_steer,
                 maximum, initial_speed, scalar_speed, previous_request):
    """Current and previously completed state only; values are unnormalized."""
    if isinstance(tick, bool) or not isinstance(tick, Integral) or tick < 0:
        raise ValueError('Nonnegative integer physics tick required')
    if len(acceleration) != 2:
        raise ValueError('Generator and target previous accelerations required')
    if previous_request not in REQUESTS:
        raise ValueError('Previous request is outside the registered action set')
    ttc = constant_velocity_ttc(VehicleState(**g), VehicleState(**t))
    values = np.asarray([
        g['x']-t['x'], g['y']-t['y'], g['vx'], g['vy'], t['vx'], t['vy'],
        math.sin(g['heading']), math.cos(g['heading']),
        math.sin(t['heading']), math.cos(t['heading']),
        *acceleration, float(tick > 0), tick*.1, lateral_y, previous_steer,
        ttc_risk(ttc), float(ttc is not None), maximum, initial_speed,
        scalar_speed, previous_request, g['length'], g['width'],
        t['length'], t['width'],
    ], dtype=np.float64)
    if values.shape != (FRAME_DIMENSIONS,) or not np.isfinite(values).all():
        raise ValueError('Invalid causal Family-B frame')
    return values


def normalize_frame(frame, divisors):
    frame, divisors = np.asarray(frame, dtype=np.float64), np.asarray(divisors, dtype=np.float64)
    if (frame.shape != (FRAME_DIMENSIONS,) or divisors.shape != frame.shape
            or not np.isfinite(frame).all() or not np.isfinite(divisors).all()
            or np.any(divisors <= 0)):
        raise ValueError('Finite 26D frame and positive divisors required')
    normalized = frame / divisors
    return np.clip(normalized, -1., 1.).astype(np.float32), (np.abs(normalized) > 1.).astype(int)


class B1ObservationHistory:
    """Episode-local 4-frame history, padding and exact actor-input audit log."""
    def __init__(self, divisors, mode='reactive'):
        if mode not in ('reactive', 'initial_context'):
            raise ValueError('Unsupported Family-B observation mode')
        # Validation without constructing any simulator objects.
        normalize_frame(np.zeros(FRAME_DIMENSIONS), divisors)
        self.divisors = np.asarray(divisors, dtype=np.float64).copy()
        self.mode = mode
        self.reset_frame = None
        self.frames = []
        self.snapshots = []
        self.clip_counts = np.zeros(FRAME_DIMENSIONS, dtype=int)
        self.raw_clip_counts = np.zeros(FRAME_DIMENSIONS, dtype=int)

    def push(self, raw_frame, tick):
        if isinstance(tick, bool) or not isinstance(tick, Integral) or tick < 0:
            raise ValueError('Nonnegative integer physics tick required')
        if self.snapshots and tick <= self.snapshots[-1]['tick']:
            raise ValueError('History snapshots must have strictly increasing ticks')
        if not self.snapshots and tick != 0:
            raise ValueError('Episode history starts at reset tick zero')
        raw_frame = np.asarray(raw_frame, dtype=np.float64).copy()
        _, raw_clips = normalize_frame(raw_frame, self.divisors)
        if self.reset_frame is None:
            self.reset_frame = raw_frame.copy()
        effective = raw_frame.copy()
        if self.mode == 'initial_context':
            effective = self.reset_frame.copy()
            effective[ELAPSED_INDEX] = raw_frame[ELAPSED_INDEX]
            effective[PREVIOUS_REQUEST_INDEX] = raw_frame[PREVIOUS_REQUEST_INDEX]
        normalized, clips = normalize_frame(effective, self.divisors)
        self.frames.append((int(tick), normalized))
        live = self.frames[-HISTORY_FRAMES:]
        missing = HISTORY_FRAMES-len(live)
        masks = [0]*missing+[1]*len(live)
        ticks = [None]*missing+[t for t, _ in live]
        flat = np.concatenate([np.zeros(missing*FRAME_DIMENSIONS, dtype=np.float32),
                               *[f for _, f in live], np.asarray(masks, dtype=np.float32)])
        if flat.shape != (OBSERVATION_DIMENSIONS,):
            raise AssertionError('Unexpected Family-B observation shape')
        self.clip_counts += clips
        self.raw_clip_counts += raw_clips
        snapshot = dict(tick=int(tick), observation=flat.tolist(), raw_frame=raw_frame.tolist(),
                        effective_frame=effective.tolist(), frame_ticks=ticks,
                        frame_masks=masks, frame_clip_counts=clips.tolist(),
                        raw_frame_clip_counts=raw_clips.tolist())
        self.snapshots.append(snapshot)
        return flat.copy(), deepcopy(snapshot)

    def current(self):
        if not self.snapshots:
            raise RuntimeError('Reset history required')
        snapshot = deepcopy(self.snapshots[-1])
        return np.asarray(snapshot['observation'], dtype=np.float32), snapshot
