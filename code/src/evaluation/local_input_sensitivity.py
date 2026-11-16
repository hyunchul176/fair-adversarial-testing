"""Pure input construction and saved-sample diagnostics; no simulator imports."""
from copy import deepcopy
import math


def perturb(case, gap_delta, speed_delta):
    out = deepcopy(case)
    p = out['event']['metrics']['p10']
    if gap_delta == 0 and speed_delta == 0:
        return out  # Preserve the exact archived center, including metadata.
    p['gap'] += gap_delta
    p['target_speed'] += speed_delta
    if not math.isfinite(p['gap']) or p['gap'] <= 0 or not 15 <= p['target_speed'] <= 35:
        raise ValueError('Perturbation outside positive-gap / qualified-speed contract')
    p['headway'] = p['gap'] / p['target_speed']
    # The case is a synthetic input, not an updated human trajectory.
    return out


def sampled_ttc(trace, stride, phase, threshold=1.5):
    if stride < 1 or not 0 <= phase < stride:
        raise ValueError('Invalid sampling stride/phase')
    samples = trace[phase::stride]
    finite = [r['pair_ttc'] for r in samples if r['pair_ttc'] is not None]
    return dict(stride=stride, phase=phase, samples=len(samples),
                finite_samples=len(finite), min_ttc=min(finite, default=None),
                threshold_crossing=any(t < threshold for t in finite))
