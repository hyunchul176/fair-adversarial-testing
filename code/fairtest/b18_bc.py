"""B18 (post-B13 development): a learned SUT — behaviour cloning of highD human followers in cut-ins.

Training data: exposed highD natural cut-ins (calibration + development windows, 25 Hz). For every
frame with the cutting-in vehicle not far behind, the features
  gap (bumper, m), closing speed (m/s), follower speed, generator speed,
  lateral offset |dy| (m) and its rate (m/s)
map to the follower's smoothed longitudinal acceleration. Model: HistGradientBoostingRegressor
(fixed seed). In simulation (`bc_human` target) the prediction is applied through the same stopping-
viable actuator as the AEB SUT (8 m/s^2, 15 m/s^3) with lane keeping.

  python -m fairtest.b18_bc train OUT_DIR      (writes bc_human.pkl + manifest; copy into the data dir)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

FEATURES = ('gap', 'closing', 'v_follower', 'v_generator', 'dy', 'dy_rate')
LIMITS = {'bc_human': dict(acceleration=8., jerk=15.)}
PARAMS = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1., random_state=0)


def samples():
    from scipy.signal import savgol_filter
    from fairtest.features import NATURAL
    NATURAL = Path(os.environ.get('FAIRTEST_NATURAL', str(NATURAL)))
    events = {}
    for line in open(NATURAL / 'events.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r.get('eligible') and r['partition'] in ('calibration', 'development'):
            events[r['event_id']] = r
    X, y, groups = [], [], []
    for line in open(NATURAL / 'windows.jsonl', encoding='utf-8'):
        w = json.loads(line)
        if w['event_id'] not in events:
            continue
        s = w['sign']
        g, t = w['generator'], w['target']
        tt = np.array(t['time'])
        if len(tt) < 25:
            continue
        tg = np.array(g['time'])
        lo, hi = max(tg[0], tt[0]), min(tg[-1], tt[-1])
        keep = (tt >= lo) & (tt <= hi)
        ts = tt[keep]
        if len(ts) < 25:
            continue
        tv = savgol_filter(np.array(t['vx'])[keep] * s, 11, 2)
        acc = savgol_filter(np.array(t['vx'])[keep] * s, 11, 2, deriv=1, delta=float(np.median(np.diff(ts))))
        gx, gy, gv = (np.interp(ts, tg, np.array(g[k])) for k in ('cx', 'cy', 'vx'))
        tx, ty = np.array(t['cx'])[keep], np.array(t['cy'])[keep]
        gap = (gx - tx) * s - (g['length'][0] + t['length'][0]) / 2
        dy = np.abs(gy - ty)
        dy_rate = np.gradient(dy, ts)
        gvs = gv * s
        for i in range(len(ts)):
            if gap[i] < -10.:
                continue
            X.append([min(gap[i], 150.), tv[i] - gvs[i], tv[i], gvs[i], dy[i], dy_rate[i]])
            y.append(acc[i])
            groups.append(w['event_id'])
    return np.array(X, float), np.array(y, float), np.array(groups)


def train(out):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    X, y, groups = samples()
    cv = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = HistGradientBoostingRegressor(**PARAMS).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        cv.append(dict(r2=float(1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)),
                       mae=float(np.mean(np.abs(y[te] - pred)))))
    model = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    blob = pickle.dumps(model, protocol=5)
    (out / 'bc_human.pkl').write_bytes(blob)
    manifest = dict(model_sha256=hashlib.sha256(blob).hexdigest(), features=FEATURES, params=PARAMS,
                    samples=int(len(y)), events=int(len(set(groups))), acc_quantiles={f'p{q}': float(np.percentile(y, q)) for q in (0.1, 1, 50, 99, 99.9)},
                    group_cv=cv, source='highD exposed natural cut-in windows (follower), 25 Hz')
    (out / 'bc_human_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'params'}, indent=1))


_MODEL = {}


def load_model():
    if 'm' not in _MODEL:
        path = Path(os.environ.get('FAIRTEST_BC_MODEL') or (Path(os.environ.get('FAIRTEST_DATA', 'fairtest/data')) / 'bc_human.pkl'))
        _MODEL['m'] = pickle.loads(path.read_bytes())
    return _MODEL['m']


class BCHumanSUT:
    jerk, floor = 15., -8.

    def __init__(self):
        self.model = load_model()
        self.prev_dy = None

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        from fairtest.cc_target import following_law
        from fairtest.ref_models import _states, project
        from src.environments.hierarchical_interaction_env import lane_controller
        g, t = _states(env)
        gap = (g.x - t.x) - (g.length + t.length) / 2
        dy = abs(g.y - t.y)
        dy_rate = 0. if self.prev_dy is None else (dy - self.prev_dy) / .1
        self.prev_dy = dy
        x = np.array([[min(max(gap, -10.), 150.), t.vx - g.vx, t.vx, g.vx, dy, dy_rate]], float)
        request = float(self.model.predict(x)[0])
        applied = float(project(request, math.hypot(t.vx, t.vy), previous_a, self.jerk, -self.floor))
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=following_law(g, t, cruise),
                             requested_mps2=request, post_shield_mps2=request, post_stop_mps2=request,
                             previous_mps2=previous_a, applied_mps2=applied, steering=steer,
                             viability=dict(constraint_infeasible=False), shield=dict(intervened=False),
                             recognition=dict(model='bc_human'))


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(sys.argv[2])
