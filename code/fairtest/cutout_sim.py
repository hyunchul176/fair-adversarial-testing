"""B34: the R157 cut-out scenario in a 1-D longitudinal simulator (plus the cut-out car's lateral position).

World: follower E (SUT or reference driver), its lead L1 (cuts out) and L1's lead L2 (revealed), one lane, 0.1 s steps,
12 s horizon; bumper gaps from the highD three-car snapshot (cutout_cases); speeds never negative. L1 keeps its speed
and moves 4 m sideways on a quintic in D_co seconds from t_co; its body leaves E's lane at 3 m (2 m half lane + 1 m half
width). Plan theta = (t_co, D_co, t_b, a_b, d_b): L2 brakes at a_b for d_b seconds from t_b (jerk <= 10 m/s^3). One lane
change (G1 by construction); guard G2: a_b >= -6 m/s^2. A plan in which L1 hits L2 is invalid.
Contact = E reaches L1 while L1's body is still in the lane, or reaches L2.
Visibility: L2 is seen once L1 has moved more than 0.375 m (the R157 wandering zone); before that L1 hides it.

Followers
  cc_co        R157 careful driver, cut-out (Rev.1 Annex 3 3.3.2.2): risk perception 0.4 s from L1 exceeding 0.375 m,
               then 0.75 s reaction; then emergency braking (12.65 m/s^3 to 7.59 m/s^2, held while closing on L2 or
               while L2 brakes) if THW(E->L2) < 2 s (the regulation's danger limit) or TTC(E->L2) <= 2 s while closing
               on L2 or while L2 brakes, otherwise the
               careful-driver following law on the nearest in-lane car (comfortable <= 3 m/s^2). Before perception it
               follows L1.
  cc_co_r050/_r100  reaction 0.5 / 1.0 s;  cc_co_aeb  + the AEB layer (strongest reading, as cc_dec_aeb)
  idm_d0/_d5/_d10, acc_prod, acc_prod_aeb  the B30 controllers; lead = L1 until L1's centre crosses the lane marking
               (2.0 m, the lane-assignment threshold of the cut-in ACC), then L2 (inputs delayed for idm_dX)
  rss_dec      longitudinal RSS against every visible car in the lane (L1 while its body is in the lane, L2 once
               visible), no delay, 8 m/s^2 response -- dominates the careful driver by construction
Fair failure = SUT contact while cc_co avoids the same plan; robust = cc_co keeps >= 0.5 m.

  python -m fairtest.cutout_sim CASES OUT --partition development --workers 8
"""
from __future__ import annotations

import json

import numpy as np

from fairtest import dec_sim as D

DT, STEPS = D.DT, D.STEPS
LOW = np.array([0., 3., 0., -7., .5])     # t_co, D_co, t_b, a_b, d_b
HIGH = np.array([6., 6., 6., -1., 6.])
G2_BRAKE = -6.
LANE_W, BODY_OUT, WANDER, ASSIGN = 4., 3., .375, 2.
MARGIN = D.MARGIN
SUTS = ('idm_d0', 'idm_d5', 'idm_d10', 'acc_prod', 'acc_prod_aeb', 'rss_dec')
CC_VARIANTS = ('cc_co_r050', 'cc_co_r100', 'cc_co_aeb')


def lateral(t, t_co, d_co):
    tau = min(1., max(0., (t - t_co) / d_co))
    return LANE_W * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)


class CCCutout:
    def __init__(self, reaction=.75, aeb=False):
        self.reaction, self.aeb = float(reaction), bool(aeb)
        self.perceived, self.emergency, self.aeb_on, self.a = None, False, False, 0.

    def __call__(self, t, obs, v, cruise):
        y1, (g1, v1, a1), (g2, v2, a2) = obs['y1'], obs['l1'], obs['l2']
        if self.perceived is None and y1 > WANDER:
            self.perceived = t + .4
        near = (g1, v1) if y1 < BODY_OUT and g1 < g2 else (g2, v2)
        floor, jerk = -7.59, 12.65
        if self.perceived is not None and t >= self.perceived + self.reaction:
            closing = v - v2
            ttc = g2 / closing if closing > 0 and g2 > 0 else None
            danger = (v > 0 and g2 / v < 2.) or (ttc is not None and ttc <= 2.)
            if not self.emergency and danger and (closing > 0 or a2 <= -.5):
                self.emergency = True
            elif self.emergency and closing <= 0 and a2 > -.5:
                self.emergency = False
            req = -7.59 if self.emergency else D.cc_following(near[0], v, near[1], cruise)
        elif self.perceived is not None and t >= self.perceived:
            req = min(-.4, D.cc_following(g1, v, v1, cruise))  # reaction: foot off the pedal
        else:
            req = D.cc_following(g1, v, v1, cruise)
        if self.aeb:  # strongest reading of the regulation's AEB layer (as cc_dec_aeb): TTC <= 2 s to the nearest car
            closing = v - near[1]
            if closing > 0 and near[0] > 0 and near[0] / closing <= D.AEB_TTC:
                self.aeb_on = True
            elif closing <= 0:
                self.aeb_on = False
            if self.aeb_on:
                req, jerk, floor = D.AEB_MPS2, D.AEB_JERK, D.AEB_MPS2
        self.a = D._jerk(req, self.a, jerk, floor, 2.)
        return self.a


class Assigned:
    """A B30 controller that follows L1 until L1's centre crosses the lane marking, then L2."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, t, obs, v, cruise):
        g, vl, al = obs['l1'] if obs['y1'] < ASSIGN else obs['l2']
        return self.inner(t, g, v, vl, al, cruise)


class RSSCutout:
    RHO, AACC, BMIN, BMAX, BRESP = D.RSSDec.RHO, D.RSSDec.AACC, D.RSSDec.BMIN, D.RSSDec.BMAX, D.RSSDec.BRESP

    def __init__(self):
        self.idm, self.a = D.IDMDelay(0.), 0.

    def _unsafe(self, gap, v, v_l):
        d = max(0., v * self.RHO + .5 * self.AACC * self.RHO ** 2 + (v + self.RHO * self.AACC) ** 2 / (2 * self.BMIN)
                - v_l ** 2 / (2 * self.BMAX))
        return gap < d

    def __call__(self, t, obs, v, cruise):
        y1, (g1, v1, a1), (g2, v2, a2) = obs['y1'], obs['l1'], obs['l2']
        cars = ([(g1, v1)] if y1 < BODY_OUT else []) + ([(g2, v2)] if y1 > WANDER else [])
        lead = (g1, v1, a1) if y1 < ASSIGN else (g2, v2, a2)
        req = -self.BRESP if any(self._unsafe(g, v, vl) for g, vl in cars) else self.idm(t, lead[0], v, lead[1], lead[2], cruise)
        self.a = D._jerk(req, self.a, 20., -self.BRESP, 2.)
        return self.a


def follower(name):
    if name == 'cc_co':
        return CCCutout()
    if name == 'cc_co_aeb':
        return CCCutout(aeb=True)
    if name.startswith('cc_co_r'):
        return CCCutout(int(name[-3:]) / 100.)
    if name == 'rss_dec':
        return RSSCutout()
    return Assigned(D.follower(name))


def run(case, theta, name, trace=False):
    """One episode: (valid, contact, min gap to the car E must not hit, required braking (max DRAC), trace)."""
    t_co, d_co, t_b, a_b, d_b = (float(x) for x in theta)
    ctl = follower(name)
    g1, g12 = float(case['gap']), float(case['gap2'])
    v, v1, v2 = float(case['v_follower']), float(case['v_lead']), float(case['v_lead2'])
    len1, cruise = float(case['length_lead']), v
    lead2 = D.Lead((t_b, a_b, d_b, 0., 0.))
    min_gap, drac, tr, valid, contact, t_contact = np.inf, 0., [], True, False, None
    for k in range(STEPS):
        t = k * DT
        y1 = lateral(t, t_co, d_co)
        a2 = lead2.step(t, v2)
        g2 = g1 + len1 + g12
        obs = dict(y1=y1, l1=(g1, v1, 0.), l2=(g2, v2, a2))
        a = ctl(t, obs, v, cruise)
        if v + a * DT < 0:
            a = -v / DT
        x = v * DT + .5 * a * DT * DT
        x1 = v1 * DT
        x2 = v2 * DT + .5 * a2 * DT * DT
        v, v2 = max(0., v + a * DT), max(0., v2 + a2 * DT)
        g1 += x1 - x
        g12 += x2 - x1
        y1n = lateral(t + DT, t_co, d_co)
        g2 = g1 + len1 + g12
        in_lane = y1n < BODY_OUT
        if in_lane and g12 <= 0:
            valid = False
            break
        near = g1 if in_lane and g1 < g2 else g2
        vn = v1 if in_lane and g1 < g2 else v2
        if v - vn > 0 and near > 0:
            drac = max(drac, (v - vn) ** 2 / (2 * near))
        min_gap = min(min_gap, near)
        if trace:
            tr.append((round(t + DT, 1), round(y1n, 3), round(g1, 3), round(g2, 3), round(v, 3), round(v2, 3), round(a, 3)))
        if near <= 0:
            contact, t_contact = True, t + DT
            break
    return dict(valid=valid, contact=contact, min_gap=float(min_gap), drac=float(drac), t_contact=t_contact, trace=tr)


# ------------------------------------------------------------------ search (B30 stages)
def boundary_plan(case, rng, budget=36, cc='cc_co'):
    """CC-only: the hardest valid plan (largest required braking of the careful driver) it still avoids with the margin."""
    def obj(theta):
        r = run(case, theta, cc)
        bad = (not r['valid']) or r['contact'] or r['min_gap'] < MARGIN
        return -min(r['drac'], 7.59) + (20. if bad else 0.), False, dict(cc_drac=r['drac'], cc_contact=r['contact'], valid=r['valid'])
    best, _ = D.bo(obj, rng, budget, low=np.r_[LOW[:3], G2_BRAKE, LOW[4:]], high=HIGH)
    return best['theta'] if best and best['f'] < 10 else None


def refine(case, sut, rng, budget=36, x_first=None, cc='cc_co', consensus=False):
    """The agent's refinement: first ROBUST fair failure (the CC keeps >= 0.5 m), guard G2 (a_b >= -6 m/s^2).
    consensus (B34b, post hoc): every careful-driver variant must avoid too, the primary with the margin."""
    low = np.r_[LOW[:3], G2_BRAKE, LOW[4:]]

    def obj(theta):
        s, c = run(case, theta, sut), run(case, theta, cc)
        if not (s['valid'] and c['valid']):
            return 30., False, dict(valid=False)
        others = [run(case, theta, v) for v in CC_VARIANTS] if consensus else []
        hit = c['contact'] or any(o['contact'] for o in others)
        gap = -1. if s['contact'] else s['min_gap']
        robust = bool(s['contact'] and not hit and c['min_gap'] >= MARGIN)
        f = gap + (20. if hit else 10. * max(0., MARGIN - c['min_gap'])) - (5. if robust else 0.)
        return f, robust, dict(sut_contact=s['contact'], cc_contact=c['contact'], cc_min_gap=c['min_gap'], cc_drac=c['drac'])
    return D.bo(obj, rng, budget, x_first=x_first, low=low, high=HIGH)


def collide(case, sut, rng, budget=36, cc='cc_co'):
    """Collision-seeking baseline: the same optimiser, the full authority box, no reference information."""
    def obj(theta):
        s = run(case, theta, sut)
        if not s['valid']:
            return 30., False, dict(valid=False)
        f = -1. if s['contact'] else s['min_gap']
        rec = dict(sut_contact=s['contact'])
        if s['contact']:
            c = run(case, theta, cc)
            rec.update(cc_contact=c['contact'], cc_min_gap=c['min_gap'], cc_drac=c['drac'])
        return f, s['contact'], rec
    return D.bo(obj, rng, budget, x_first=(LOW + HIGH) / 2, low=LOW, high=HIGH)


def evaluate_case(args):
    case, suts, budget = args[:3]
    consensus, with_collide = (args[3], args[4]) if len(args) > 3 else (False, True)
    cid = case['case_id']
    out = dict(case_id=cid, partition=case['partition'], recording=case['recording'], runs={})
    b = boundary_plan(case, np.random.default_rng([cid, 0]))
    out['boundary'] = b
    for i, sut in enumerate(suts):
        best, q = refine(case, sut, np.random.default_rng([cid, 1, i]), budget, x_first=b, consensus=consensus)
        rec = dict(first_robust=q if best and best['success'] else None, best=best)
        if best and best['success']:
            th = best['theta']
            rec['judged'] = {cc: (lambda r: dict(contact=r['contact'], min_gap=round(r['min_gap'], 3)))(run(case, th, cc))
                             for cc in CC_VARIANTS}
            rec['grade'] = D.grade(best['cc_drac'])
        out['runs'][f'agent|{sut}'] = rec
        if not with_collide:
            continue
        cb, cq = collide(case, sut, np.random.default_rng([cid, 2, i]), budget)
        if cb and cb['success']:
            cb['judged'] = {cc: run(case, cb['theta'], cc)['contact'] for cc in CC_VARIANTS}
        out['runs'][f'collide|{sut}'] = dict(first_collision=cq if cb and cb['success'] else None, best=cb)
    return out


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('cases')
    ap.add_argument('out')
    ap.add_argument('--partition', default='development')
    ap.add_argument('--budget', type=int, default=36)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--consensus', action='store_true', help='B34b: robust success needs every CC variant to avoid')
    ap.add_argument('--no-collide', action='store_true', help='skip the collision-seeking baseline')
    a = ap.parse_args()
    cases = [c for c in json.load(open(a.cases, encoding='utf-8'))['cases'] if a.partition == 'all' or c['partition'] == a.partition]
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(evaluate_case, [(c, SUTS, a.budget, a.consensus, not a.no_collide) for c in cases]):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(cases=len(cases))))
