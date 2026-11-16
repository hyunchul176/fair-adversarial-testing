"""B23 literature baselines for the RL adversary, re-implemented from the papers in the B8/B17 environment.

dynasto  Humeniuk et al., "Dynasto" (arXiv 2603.21427): validity-aware adversary in HighwayEnv.
         Safe distances (ISO 34502-inspired; tau_r = 0.2 s, |a| = 5 m/s^2, d_min 5 m longitudinal / 2 m lateral):
           d_lon = 1/2 (v_l^2/a_l - v_f^2/a_f) + v_f tau_r + d_min_lon          (a_l = a_f = -5)
           d_lat = v_lat tau_r + 1/2 a_lat tau_r^2 + (v_lat + a_lat tau_r)^2 / (2 b_lat) + d_min_lat
         Dense reward R_coll(t) = (d_safe - d_c) / d_safe if d_c < d_safe else 0 (per decision step).
         Terminal reward on a collision: +30 if valid, else -(d_safe(t_m) - d_c(t_m)) / 5 at the first violation t_m.
         A collision is invalid if, before it, the adversary
           cut in while the longitudinal gap was below d_lon            (phi_unsafe-cut-in),
           braked while ahead in the same lane with the gap below d_lon (phi_unsafe-brake),
           accelerated while behind the ego in the same lane            (phi_rear-hit).
         Operationalisation (ours, the paper gives no code-level definitions): same lane = lateral centre distance
         < 2 m (half a lane); the cut-in is the first tick the adversary enters the ego lane ahead of it; braking /
         accelerating = |a| > 0.1 m/s^2; the lateral distance is used only while the vehicles overlap longitudinally.
cars     Xiao et al., "CARS" (arXiv 2605.13751), RL fine-tuning reward (Eq. 7 / S16):
           R = w_prog R_prog + w_l R_l + w_lc R_lc + R_hit - lambda_time
           R_prog = clip((d0_lon - d1_lon) / (d0_lon + nu), -0.5, 1)
           R_l    = exp(-l1 / sigma_l) if l0 > l1 else -0.5 exp(-l1 / sigma_l)
           R_lc   = clip((l0 - l1) / (dt_eff (l0 + nu)), -0.5, 1)
         The paper does not report weights; ours: w = 1, sigma_l = 2 m, nu = 1 m, dt_eff = 0.5 s (decision period),
         R_hit = 30 on a valid collision (so that the collision dominates the accumulable shaping, as in CARS where
         collisions are the target), lambda_time = 0.05; a lateral decrease below 1 mm counts as no decrease (our
         lateral controller approaches the lane centre asymptotically). Attribution is post hoc (FSM primary), as in CARS.
Both baselines use our PPO learner, observation and budget so that only the objective differs (Dynasto used DQN).
"""
from __future__ import annotations

import math

TAU_R, A_LON, D_MIN_LON, A_LAT, B_LAT, D_MIN_LAT = .2, 5., 5., 5., 5., 2.
LANE_HALF, ACT_EPS = 2., .1
W_PROG = W_L = W_LC = 1.
SIGMA_L, NU, DT_EFF, R_HIT, LAMBDA_TIME, L_EPS = 2., 1., .5, 30., .05, 1e-3
DYNASTO_VALID, DYNASTO_SCALE = 30., 5.


def pair_state(env):
    """(generator, target) as plain dicts from a live B7Env."""
    out = []
    for v in env.sim.inner.road.vehicles:
        out.append(dict(x=float(v.position[0]), y=float(v.position[1]), vx=float(v.velocity[0]), vy=float(v.velocity[1]),
                        length=float(v.LENGTH), width=float(v.WIDTH)))
    return out[0], out[1]


def d_lon(v_follower, v_leader):
    return .5 * (v_leader ** 2 / -A_LON - v_follower ** 2 / -A_LON) + v_follower * TAU_R + D_MIN_LON


def d_lat(v_lat):
    v = max(0., v_lat)
    return v * TAU_R + .5 * A_LAT * TAU_R ** 2 + (v + A_LAT * TAU_R) ** 2 / (2 * B_LAT) + D_MIN_LAT


def _geometry(g, t):
    """Relevant (d_safe, d_c) of the pair, or None when neither longitudinal nor lateral interaction applies."""
    dy = t['y'] - g['y']
    if abs(dy) < LANE_HALF:  # same lane: longitudinal
        lead, fol = (g, t) if g['x'] >= t['x'] else (t, g)
        gap = (lead['x'] - fol['x']) - (lead['length'] + fol['length']) / 2
        return d_lon(fol['vx'], lead['vx']), gap
    if abs(g['x'] - t['x']) < (g['length'] + t['length']) / 2:  # alongside: lateral
        closing = math.copysign(1., dy) * (g['vy'] - t['vy'])
        return d_lat(closing), abs(dy) - (g['width'] + t['width']) / 2
    return None


def dynasto_step_reward(g, t):
    geo = _geometry(g, t)
    if geo is None:
        return 0.
    safe, gap = geo
    return float(min(1., (safe - gap) / safe)) if safe > 0 and gap < safe else 0.


def dynasto_validity(trace):
    """(valid, violation) over the ticks before the collision; violation = (rule, tick, d_safe, d_c)."""
    was_same = None
    for k, row in enumerate(trace):
        g, t = row['generator'], row['target']
        a_g = float(row['acceleration'][0])
        same = abs(t['y'] - g['y']) < LANE_HALF
        ahead = g['x'] > t['x']
        if same:
            if ahead:
                safe = d_lon(t['vx'], g['vx'])
                gap = (g['x'] - t['x']) - (g['length'] + t['length']) / 2
                if was_same is False and gap < safe:
                    return False, ('unsafe_cut_in', k, safe, gap)
                if a_g < -ACT_EPS and gap < safe:
                    return False, ('unsafe_brake', k, safe, gap)
            elif a_g > ACT_EPS:
                safe = d_lon(g['vx'], t['vx'])
                gap = (t['x'] - g['x']) - (g['length'] + t['length']) / 2
                return False, ('rear_hit', k, safe, gap)
        was_same = same
    return True, None


def dynasto_terminal(row, valid_manoeuvre, contact):
    if not valid_manoeuvre:
        return -1., None
    if not contact:
        return 0., None
    ok, why = dynasto_validity(row['trace'])
    if ok:
        return DYNASTO_VALID, None
    _, _, safe, gap = why
    return -max(0., safe - gap) / DYNASTO_SCALE, why


def cars_step_reward(before, after):
    (g0, t0), (g1, t1) = before, after
    d0, d1 = abs(g0['x'] - t0['x']), abs(g1['x'] - t1['x'])
    l0, l1 = abs(g0['y'] - t0['y']), abs(g1['y'] - t1['y'])
    r_prog = min(1., max(-.5, (d0 - d1) / (d0 + NU)))
    r_l = math.exp(-l1 / SIGMA_L) if l0 - l1 > L_EPS else -.5 * math.exp(-l1 / SIGMA_L)
    r_lc = min(1., max(-.5, (l0 - l1) / (DT_EFF * (l0 + NU))))
    return float(W_PROG * r_prog + W_L * r_l + W_LC * r_lc - LAMBDA_TIME)


def cars_terminal(valid_manoeuvre, contact):
    if not valid_manoeuvre:
        return -1.
    return R_HIT if contact else 0.
