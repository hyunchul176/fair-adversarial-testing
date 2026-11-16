"""B30: the R157 deceleration scenario in a 1-D longitudinal simulator, with the fair agent's search stages.

World: a lead car (the adversary, open-loop plan) and a follower (SUT or reference driver) in one lane, 0.1 s steps,
12 s horizon; bumper gap from the highD car-following snapshot (dec_cases); speeds never negative; contact = gap <= 0.
Lead plan theta = (t_b, a_b, d_b, a_pre, a_post): a_pre before t_b, a_b for d_b seconds, then a_post; lead jerk <= 10
m/s^3, acceleration <= 2 m/s^2. One braking episode only (G1: no brake-release-brake feint).

Followers
  cc_dec          R157 careful driver, deceleration scenario (primary): risk perception takes 0.4 s from the moment the
                  lead's deceleration exceeds 5 m/s^2 (the regulation's wording as quoted in the literature), then a
                  0.75 s reaction; braking then ramps (12.65 m/s^3) to 7.59 m/s^2 and is held while the lead brakes or
                  the follower is faster; otherwise the careful-driver following law of the cut-in CC (comfortable
                  deceleration <= 3 m/s^2). Checked against the regulation text (R157 original series, Annex 4
                  Appendix 3, 3.4.3 and Table 1; Rev.1 / 01 series, Annex 3, 3.3.2.3 and Table 1): identical.
  cc_dec_onset    sensitivity (stronger reference, Mattas et al. 2022 summary): perception and reaction start when the
                  lead starts decelerating (<= -0.5 m/s^2)
  cc_dec_r050/_r100  primary with a 0.5 / 1.0 s reaction
  cc_dec_aeb      B32 (post hoc): primary + the regulation's AEB layer (Figure 1 / Table 1: 0.6 s to 0.85 g; TTC_AEB and
                  the delay unspecified) in its strongest reading: whenever TTC <= 2 s, no delay, held while closing
  idm_d0/_d5/_d10 IDM (T 1.5 s, s0 2 m, a 1.5, b 2 m/s^2; braking <= 6 m/s^2, jerk <= 15) seeing the lead 0 / 0.5 / 1.0 s late
  acc_prod        production ACC (Milanes & Shladover 2014 gains, time gap 1.5 s, first-order lag 0.4 s, ISO 15622
                  limits 3.5 m/s^2 / 2.5 m/s^3), as sut_acc
  acc_prod_aeb    + AEB: TTC <= 1.5 s, 0.3 s latency, 9 m/s^2 (jerk 30) while closing
  rss_dec         longitudinal RSS (rho 0.75 s, a_accel 3, b_min 6, lead b_max 8 m/s^2): brakes at 8 m/s^2 whenever the
                  gap is below the RSS safe distance, IDM otherwise -- dominates the careful driver by construction
Fair failure = SUT contact while cc_dec avoids the same lead plan; robust = cc_dec keeps >= 0.5 m (contacts are rear-end).
R157 grade from the careful driver's required braking (max DRAC while closing): avoidable < 5, difficult 5-7.6 m/s^2.
"""
from __future__ import annotations

import json
import math

import numpy as np

DT, STEPS = .1, 120
LOW = np.array([0., -7., .5, -1.5, -1.])     # t_b, a_b, d_b, a_pre, a_post
HIGH = np.array([5., -1., 4., 1., 1.])
G2_BRAKE = -6.                               # guard: human emergency braking range (R157 deceleration scenarios)
LEAD_JERK, LEAD_UP = 10., 2.
MARGIN = .5
SUTS = ('idm_d0', 'idm_d5', 'idm_d10', 'acc_prod', 'acc_prod_aeb', 'rss_dec')
RANK = {'idm_d10': 1, 'acc_prod': 1, 'idm_d5': 2, 'idm_d0': 3, 'acc_prod_aeb': 4, 'rss_dec': 5}


def _jerk(request, prev, jerk, lo, hi):
    return float(min(hi, max(lo, prev - jerk * DT, min(prev + jerk * DT, request))))


class Lead:
    def __init__(self, theta, a0=0.):
        self.t_b, self.a_b, self.d_b, self.a_pre, self.a_post = (float(x) for x in theta)
        self.a = float(np.clip(a0, -LEAD_UP, LEAD_UP))

    def step(self, t, v):
        req = self.a_pre if t < self.t_b else (self.a_b if t < self.t_b + self.d_b else self.a_post)
        a = _jerk(req, self.a, LEAD_JERK, -9., LEAD_UP)
        if v + a * DT < 0:
            a = -v / DT
        self.a = a
        return a


def cc_following(gap, v, v_l, cruise):
    """The careful driver's following law (cc_target.following_law in 1-D), capped at 3 m/s^2 braking."""
    free = float(np.clip(.4 * (cruise - v), -5, 2))
    return max(-3., min(free, .18 * (gap - 5 - 1.8 * v) + .8 * (v_l - v)))


AEB_MPS2, AEB_JERK, AEB_TTC = -.85 * 9.81, .85 * 9.81 / .6, 2.  # cc_dec_aeb (B32)


class CCDec:
    def __init__(self, trigger='threshold', reaction=.75, aeb=False):
        self.trigger, self.reaction, self.aeb = trigger, float(reaction), bool(aeb)
        self.perceived, self.emergency, self.aeb_on, self.a = None, False, False, 0.

    def __call__(self, t, gap, v, v_l, a_l, cruise):
        if self.perceived is None:
            if self.trigger == 'onset' and a_l <= -.5:
                self.perceived = t
            elif self.trigger == 'threshold' and a_l <= -5.:
                self.perceived = t + .4
        if self.perceived is not None and t >= self.perceived + self.reaction and (a_l <= -.5 or v > v_l):
            self.emergency = True
        elif self.emergency and a_l > -.5 and v <= v_l:
            self.emergency = False
        if self.emergency:
            req, jerk = -7.59, 12.65
        elif self.perceived is not None and t >= self.perceived:
            req, jerk = min(-.4, cc_following(gap, v, v_l, cruise)), 12.65  # reaction: foot off the pedal
        else:
            req, jerk = cc_following(gap, v, v_l, cruise), 12.65
        floor = -7.59
        if self.aeb:  # independent of the driver's perception and reaction
            closing = v - v_l
            if closing > 0 and gap > 0 and gap / closing <= AEB_TTC:
                self.aeb_on = True
            elif closing <= 0:
                self.aeb_on = False
            if self.aeb_on:
                req, jerk, floor = AEB_MPS2, AEB_JERK, AEB_MPS2
        self.a = _jerk(req, self.a, jerk, floor, 2.)
        return self.a


class IDMDelay:
    T, S0, A, B, BMAX = 1.5, 2., 1.5, 2., 6.

    def __init__(self, delay):
        self.k = int(round(delay / DT))
        self.buf, self.a = [], 0.

    def __call__(self, t, gap, v, v_l, a_l, cruise):
        self.buf.append((gap, v_l))
        g, vl = self.buf[max(0, len(self.buf) - 1 - self.k)]
        s_star = self.S0 + v * self.T + v * (v - vl) / (2 * math.sqrt(self.A * self.B))
        req = self.A * (1 - (v / max(cruise, 1.)) ** 4 - (max(s_star, 0.) / max(g, .1)) ** 2)
        self.a = _jerk(req, self.a, 15., -self.BMAX, 2.)
        return self.a


class ACCProd:
    K1, K2, K0, T_H, D0, TAU = .23, .07, .4, 1.5, 2., .4

    def __init__(self, aeb=False):
        self.aeb, self.af, self.a = aeb, 0., 0.
        self.armed, self.active = None, False

    def __call__(self, t, gap, v, v_l, a_l, cruise):
        req = min(self.K0 * (cruise - v), self.K1 * (gap - self.T_H * v - self.D0) + self.K2 * (v_l - v))
        self.af += (req - self.af) * DT / self.TAU
        req, jerk, amax = self.af, 2.5, 3.5
        if self.aeb:
            closing = v - v_l
            ttc = gap / closing if closing > 0 and gap > 0 else None
            if not self.active and self.armed is None and ttc is not None and ttc <= 1.5:
                self.armed = t
            if self.armed is not None and t - self.armed >= .3 - 1e-9:
                self.active = closing > 0
                if not self.active:
                    self.armed = None
            if self.active:
                req, jerk, amax = -9., 30., 9.
        self.a = _jerk(req, self.a, jerk, -amax, 2.)
        if not self.active:
            self.af = self.a
        return self.a


class RSSDec:
    RHO, AACC, BMIN, BMAX, BRESP = .75, 3., 6., 8., 8.

    def __init__(self):
        self.idm, self.a = IDMDelay(0.), 0.

    def __call__(self, t, gap, v, v_l, a_l, cruise):
        d = max(0., v * self.RHO + .5 * self.AACC * self.RHO ** 2 + (v + self.RHO * self.AACC) ** 2 / (2 * self.BMIN)
                - v_l ** 2 / (2 * self.BMAX))
        req = -self.BRESP if gap < d else self.idm(t, gap, v, v_l, a_l, cruise)
        self.a = _jerk(req, self.a, 20., -self.BRESP, 2.)
        return self.a


def follower(name):
    if name == 'cc_dec':
        return CCDec('threshold')
    if name == 'cc_dec_onset':
        return CCDec('onset')
    if name == 'cc_dec_aeb':
        return CCDec('threshold', aeb=True)
    if name.startswith('cc_dec_r'):
        return CCDec('threshold', int(name[-3:]) / 100.)
    if name.startswith('idm_d'):
        return IDMDelay(int(name[5:]) / 10.)
    if name in ('acc_prod', 'acc_prod_aeb'):
        return ACCProd(aeb=name == 'acc_prod_aeb')
    if name == 'rss_dec':
        return RSSDec()
    raise ValueError(name)


def run(case, theta, name, trace=False):
    """One episode: (contact, min gap, careful-driver-style required braking, trace)."""
    ctl = follower(name)
    gap, v_f, v_l = float(case['gap']), float(case['v_follower']), float(case['v_lead'])
    cruise = v_f
    lead = Lead(theta, 0.)
    min_gap, drac, tr = gap, 0., []
    contact, t_contact = False, None
    for k in range(STEPS):
        t = k * DT
        a_l = lead.step(t, v_l)
        a_f = ctl(t, gap, v_f, v_l, a_l, cruise)
        if v_f + a_f * DT < 0:
            a_f = -v_f / DT
        x_l = v_l * DT + .5 * a_l * DT * DT
        x_f = v_f * DT + .5 * a_f * DT * DT
        v_l, v_f = max(0., v_l + a_l * DT), max(0., v_f + a_f * DT)
        gap += x_l - x_f
        closing = v_f - v_l
        if closing > 0 and gap > 0:
            drac = max(drac, closing * closing / (2 * gap))
        min_gap = min(min_gap, gap)
        if trace:
            tr.append((round(t + DT, 1), round(gap, 3), round(v_l, 3), round(v_f, 3), round(a_l, 3), round(a_f, 3)))
        if gap <= 0:
            contact, t_contact = True, t + DT
            break
    return dict(contact=contact, min_gap=float(min_gap), drac=float(drac), t_contact=t_contact, trace=tr)


def grade(drac):
    return 'avoidable' if drac < 5 else ('difficult' if drac < 7.6 else 'unavoidable')


# ------------------------------------------------------------------ search (GP-BO as b23_search)
def _ei(mu, sd, best):
    from scipy.stats import norm
    sd = np.maximum(sd, 1e-9)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def bo(objective, rng, budget, x_first=None, low=LOW, high=HIGH, n_init=5, n_cand=4096):
    """Minimise objective(theta) -> (f, success, record); stop at the first success. Returns (best, queries)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    scale = high - low
    init = [] if x_first is None else [np.clip((np.asarray(x_first) - low) / scale, 0, 1)]
    while len(init) < n_init:
        init.append(rng.random(5))
    X, y, best = [], [], None
    for n in range(1, budget + 1):
        if n <= len(init):
            x = init[n - 1]
        else:
            gp = GaussianProcessRegressor(ConstantKernel(1.) * Matern(length_scale=np.full(5, .3), nu=2.5) + WhiteKernel(1e-3),
                                          normalize_y=True, n_restarts_optimizer=0, random_state=0)
            gp.fit(np.array(X), np.array(y))
            xb = np.array(X)[int(np.argmin(y))]
            cand = np.vstack([rng.random((n_cand, 5)), np.clip(xb + rng.normal(0, .08, (n_cand // 4, 5)), 0, 1)])
            mu, sd = gp.predict(cand, return_std=True)
            x = cand[int(np.argmax(_ei(mu, sd, float(np.min(y)))))]
        theta = low + x * scale
        f, ok, rec = objective(theta)
        X.append(x)
        y.append(f)
        if best is None or f < best['f']:
            best = dict(rec, f=float(f), theta=[float(v) for v in theta], success=bool(ok))
        if ok:
            return best, n
    return best, budget


def boundary_plan(case, rng, budget=36, cc='cc_dec'):
    """CC-only: the hardest plan (largest required braking of the careful driver) it still avoids with the margin."""
    def obj(theta):
        r = run(case, theta, cc)
        bad = r['contact'] or r['min_gap'] < MARGIN
        return -min(r['drac'], 7.59) + (20. if bad else 0.), False, dict(cc_drac=r['drac'], cc_contact=r['contact'])
    best, _ = bo(obj, rng, budget, low=np.r_[LOW[:1], G2_BRAKE, LOW[2:]], high=HIGH)
    return best['theta'] if best and best['f'] < 10 else None


def refine(case, sut, rng, budget=36, x_first=None, guarded=True, cc='cc_dec'):
    """The agent's refinement: first ROBUST fair failure (the CC keeps >= 0.5 m), guards G2 (braking >= -6 m/s^2)."""
    low = np.r_[LOW[:1], G2_BRAKE, LOW[2:]] if guarded else LOW

    def obj(theta):
        s, c = run(case, theta, sut), run(case, theta, cc)
        gap = -1. if s['contact'] else s['min_gap']
        robust = bool(s['contact'] and not c['contact'] and c['min_gap'] >= MARGIN)
        f = gap + (20. if c['contact'] else 10. * max(0., MARGIN - c['min_gap'])) - (5. if robust else 0.)
        return f, robust, dict(sut_contact=s['contact'], cc_contact=c['contact'], cc_min_gap=c['min_gap'], cc_drac=c['drac'])
    return bo(obj, rng, budget, x_first=x_first, low=low)


def collide(case, sut, rng, budget=36, cc='cc_dec'):
    """Collision-seeking baseline: the same optimiser, the full authority box, no reference information."""
    def obj(theta):
        s = run(case, theta, sut)
        f = -1. if s['contact'] else s['min_gap']
        rec = dict(sut_contact=s['contact'])
        if s['contact']:
            c = run(case, theta, cc)
            rec.update(cc_contact=c['contact'], cc_min_gap=c['min_gap'], cc_drac=c['drac'])
        return f, s['contact'], rec
    return bo(obj, rng, budget, x_first=(LOW + HIGH) / 2)


def evaluate_case(args):
    """All SUTs for one case: boundary plan, the agent's guarded robust refinement, the collision-seeking baseline;
    each reported failure re-judged by the careful-driver variants."""
    case, suts, budget = args
    cid = case['case_id']
    out = dict(case_id=cid, partition=case['partition'], recording=case['recording'], runs={})
    b = boundary_plan(case, np.random.default_rng([cid, 0]))
    out['boundary'] = b
    for i, sut in enumerate(suts):
        best, q = refine(case, sut, np.random.default_rng([cid, 1, i]), budget, x_first=b)
        rec = dict(first_robust=q if best and best['success'] else None, best=best)
        if best and best['success']:
            th = best['theta']
            rec['judged'] = {cc: (lambda r: dict(contact=r['contact'], min_gap=round(r['min_gap'], 3)))(run(case, th, cc))
                             for cc in ('cc_dec_onset', 'cc_dec_r050', 'cc_dec_r100')}
            rec['grade'] = grade(best['cc_drac'])
        out['runs'][f'agent|{sut}'] = rec
        cb, cq = collide(case, sut, np.random.default_rng([cid, 2, i]), budget)
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
    a = ap.parse_args()
    cases = [c for c in json.load(open(a.cases, encoding='utf-8'))['cases'] if a.partition == 'all' or c['partition'] == a.partition]
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for r in pool.imap_unordered(evaluate_case, [(c, SUTS, a.budget) for c in cases]):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(cases=len(cases))))
