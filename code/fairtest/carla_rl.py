"""E7: the trained RL adversary attacks CARLA SUTs closed-loop, without retraining (one SUT query per case).

The adversary's motion is computed by a highway-env twin (the B7 generator with its quintic lateral planner and
actuator, stepped with the policy's decisions; its own target is the omniscient braker, far behind), and mirrored into
CARLA every tick. The policy observes the twin generator and the CARLA SUT (road-frame state mapped to highway-env
units: lateral scaled by 4/w, the SUT's measured acceleration), exactly the 16-D B8 observation. After the episode the
realised adversary trajectory is replayed against the CARLA-CC (`carla_replay.run`) for the fair / attacker-at-fault
verdict, as in the highway-env CC-replay semantics. Decision margin (G4) of a collision: contact_dx (adversary centre
ahead of the SUT centre at contact; >= 3.5 m front contact) and the CC's smallest box distance cc_min_clearance; the
realised adversary trajectory is kept for re-judgement (adv_traj, [x, y, vx, vy] per 0.1 s).

  python -m fairtest.carla_rl OUT.jsonl MODEL.zip --kinds ba:normal,... [--commit] [--cases 0:62] [--port 2020]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from fairtest import devboot  # noqa: F401


def observation(twin, sut_hw, sut_acc, entered):
    """The B8AdversaryEnv observation with the SUT (target) state taken from CARLA."""
    g = twin.sim.inner.road.vehicles[0]
    gx, gy = float(g.position[0]), float(g.position[1])
    gvx, gvy = float(g.velocity[0]), float(g.velocity[1])
    tx, ty, tvx = sut_hw
    goal = twin.lateral_planner.goal if twin.lateral_planner.reference is not None else 0.
    init = twin.initial
    return np.clip(np.array([
        (gx - tx) / 50., (gy - ty) / 4., gvx / 40., tvx / 40., (tvx - gvx) / 10., gvy / 2., gy / 4.,
        twin.previous_a / 3., twin.previous_steer / .3, twin.tick / 120., goal / 4., float(entered),
        (init[0]['x'] - init[1]['x'] - 5.) / 50., (init[1]['vx'] - init[0]['vx']) / 10., sut_acc / 5., 1.], dtype=np.float32), -5., 5.)


SHIELD_CC = os.environ.get('FAIRTEST_SHIELD_CC', 'cc_r157')  # the shield's CC model (cc_r157_cal: CARLA-calibrated)


def contact_possible(trace, tick, sut_hw, n=5, a_up=3., a_down=10., margin=.3, length=5., width=None):
    """Could the SUT (x, y, v in highway-env units) touch the adversary in the next n ticks, whatever it does within
    [-a_down, +a_up]? Adversary positions from a highway-env counterfactual trace (row k = state after tick k)."""
    from fairtest.env import VEH_W
    width = width if width is not None else (VEH_W or 2.)  # half-sum of the two body widths in the twin
    x, y, v = sut_hw
    for j in range(1, n + 1):
        k = tick + j - 1
        if k >= len(trace):
            break
        g = trace[k]['generator']
        tau = .1 * j
        lo, hi = x + v * tau - .5 * a_down * tau * tau, x + v * tau + .5 * a_up * tau * tau
        if abs(g['y'] - y) < width + margin and g['x'] - hi < length + margin and lo - g['x'] < length + margin:
            return True
    return False


def shielded(model, obs, case, actions, entered, commit, last, window=False, tick=0, sut_hw=None):
    """(lane, request) of the most preferred admissible action; 'held': the highway-env CC avoids it held to the end;
    'recover' (window mode): no SUT contact possible in its window and the least aggressive continuation is CC-safe."""
    import torch
    from fairtest.b17_rl import cc_safe  # honours FAIRTEST_SHIELD_MARGIN
    from fairtest.b8_env import B8AdversaryEnv, REQUESTS
    from fairtest.b23_crossjudge import replay
    with torch.no_grad():
        probs = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.probs[0].numpy()
    checks, seen, options = 0, set(), []
    for a in np.argsort(-probs, kind='stable'):
        lane, request = B8AdversaryEnv.decode(int(a))
        if commit and entered and lane != 4.:
            lane = 4.
        eff = (lane, request)
        if eff in seen:
            continue
        seen.add(eff)
        options.append(eff)
        if last is not None and eff == last[0] and last[1] == 'held':
            return eff, checks, 'held'
        checks += 1
        if cc_safe(replay(case, actions + [eff], SHIELD_CC, 0.)):
            return eff, checks, 'held'
        if window:
            fallback = (4. if (entered or lane == 4.) else 0., max(REQUESTS))
            checks += 1
            rec = replay(case, actions + [eff, fallback], SHIELD_CC, 0.)
            if cc_safe(rec) and not contact_possible(rec['trace'], tick, sut_hw):
                return eff, checks, 'recover'
    return min(options, key=lambda e: (e[0] == 4., -e[1])), checks, None  # none admissible: least aggressive


def episode(world, road, bp, base, case, model, kind, commit, shield=False):
    import carla
    from fairtest.b8_env import B8AdversaryEnv, REQUESTS
    from fairtest.carla_replay import DT, LANE_HW, START_X, _brakes, contact, is_agent, release, run, spawn_x, start_target
    from fairtest.env import B7Env
    from fairtest.runner import _STATE
    w = road.w
    t0 = case['physical_initial_state'][1]
    sut0 = dict(x=t0['x'], y=t0['y'], vx=t0['vx'], vy=t0['vy'])
    cruise = case['event']['metrics']['p10']['target_speed']
    twin = B7Env(_STATE['base'], case, _STATE['b1'], _STATE['b2'], case['physical_initial_state'], target='oracle_t0', target_h=0.,
                 durations=(2., 3., 4., 5., 6.))
    twin.reset()
    ba = is_agent(kind)
    sut = world.spawn_actor(bp, road.transform(spawn_x(kind, sut0['vx']), w))
    adv = world.spawn_actor(bp, road.transform(min(road.length - 20., START_X + 250.), 0.))
    out = dict(kind=kind)
    try:
        _brakes(sut)
        _brakes(adv)
        world.tick()
        e, ea = sut.bounding_box.extent, adv.bounding_box.extent
        dims = (2 * e.x, 2 * e.y)
        target = start_target(world, road, sut, kind, sut0, cruise, base, dims)
        xs0, _, v0 = road.state(sut)[:3]
        shift = xs0 - sut0['x']
        last_v, acc, entered, hit, t_hit, actions, k = v0, 0., False, False, None, [], 0
        last, sh_checks, sh_unsafe = None, 0, 0
        g_init = twin.initial[0]
        traj = [dict(t=0., x=g_init['x'], y=g_init['y'], vx=g_init['vx'], vy=g_init['vy'], heading=g_init['heading'])]

        def place(g):
            gx, gy, gvx, gvy = g['x'] + shift, g['y'] * w / LANE_HW, g['vx'], g['vy'] * w / LANE_HW
            gh = math.atan2(gvy, max(gvx, .1))
            adv.set_transform(road.transform(gx, gy, gh))
            adv.set_target_velocity(road.velocity(gvx, gvy, gx))
            return gx, gy, gvx, gvy, gh
        g_now = place(traj[0])  # the adversary at t = k * DT
        if not ba:
            target.v_cmd = v0
        done = False
        while not done and not hit and k < 120:
            st = road.state(sut)
            obs = observation(twin, (st[0] - shift, st[1] * LANE_HW / w, st[2]), acc, entered)
            if shield:
                sut_hw = (st[0] - shift, st[1] * LANE_HW / w, st[2])
                (lane, request), n, via = shielded(model, obs, case, [tuple(x) for x in actions], entered, commit, last,
                                                   window=shield == 'window', tick=k, sut_hw=sut_hw)
                sh_checks, sh_unsafe, last = sh_checks + n, sh_unsafe + (via is None), ((lane, request), via)
            else:
                a, _ = model.predict(obs, deterministic=True)
                lane, request = B8AdversaryEnv.decode(int(a))
                if commit and entered and lane != 4.:
                    lane = 4.
            entered = entered or lane == 4.
            actions.append((lane, request))
            before = len(twin.meter.trace)
            _, _, done, _, _ = twin.step((lane, request))
            for row in twin.meter.trace[before:]:  # row = the adversary after the tick, t = (k + 1) * DT
                g = row['generator']
                st = road.state(sut)
                if not ba:
                    target._measured = (st[2] - last_v) / DT
                target.step(k * DT, g_now, st, dims, sut, carla)
                world.tick()
                traj.append(dict(t=round((k + 1) * DT, 1), x=g['x'], y=g['y'], vx=g['vx'], vy=g['vy'], heading=g['heading']))
                g_now = place(g)
                st = road.state(sut)
                acc, last_v, k = (st[2] - last_v) / DT, st[2], k + 1
                if contact(sut, adv):
                    hit, t_hit = True, round(k * DT, 1)
                    out['contact_dx'] = round(g_now[0] - st[0], 2)
                    break
        valid = bool(twin.maneuver_valid)
        out.update(contact=hit, t_contact=t_hit, valid=valid, v0_error=round(v0 - sut0['vx'], 3), decisions=len(actions))
        if shield:
            out['shield'] = dict(mode=shield, cc_rollouts=sh_checks, unshieldable=sh_unsafe)
    finally:
        release(sut, kind, world)
        sut.destroy()
        adv.destroy()
        twin.close()
        world.tick()
    while len(traj) < 121:  # after the SUT's contact the latched last action continues (CC-replay semantics)
        last = traj[-1]
        traj.append(dict(t=round(last['t'] + .1, 1), x=last['x'] + last['vx'] * .1, y=last['y'] + last['vy'] * .1,
                         vx=last['vx'], vy=last['vy'], heading=last['heading']))
    cc_kind = 'cc:hw' if kind.endswith(':hw') else 'cc'  # the reference perceives the same geometry as the SUT
    cc = run(world, road, bp, dict(sut0=sut0, cruise=cruise, trajectory=traj), cc_kind, base) if hit else None
    out.update(cc_contact=None if cc is None else bool(cc['contact']), fair=bool(hit and valid and cc is not None and not cc['contact']),
               feints=sum(1 for i in range(1, len(actions)) if actions[i - 1][0] == 4. and actions[i][0] == 0.))
    if cc is not None:
        out['cc_min_clearance'] = cc['min_clearance']
        out['adv_traj'] = [[round(p[q], 3) for q in ('x', 'y', 'vx', 'vy')] for p in traj]
    return out


def main(a):
    from stable_baselines3 import PPO
    from fairtest.carla_replay import BLUEPRINT, Road, connect
    from fairtest.runner import _STATE, _init
    _init()
    base = _STATE['base']
    model = PPO.load(a.model, device='cpu')
    ids = sorted(c for c, v in _STATE['cases'].items() if a.partition == 'all' or v['event'].get('partition') == a.partition)
    lo, hi = (int(x) for x in a.cases.split(':')) if a.cases else (0, len(ids))
    ids = ids[lo:hi]
    if a.resume and os.path.exists(a.out):
        done = {json.loads(l)['case_id'] for l in open(a.out, encoding='utf-8') if l.strip()}
        ids = [c for c in ids if c not in done]
    world = connect(a.port)
    road = Road(world)
    bp = world.get_blueprint_library().find(BLUEPRINT)
    try:
        with open(a.out, 'a', encoding='utf-8') as f:
            for i, c in enumerate(ids):
                res = dict(case_id=c, model=os.path.basename(a.model), runs={})
                t0 = time.time()
                for kind in a.kinds.split(','):
                    try:
                        mode = 'window' if a.shield_window else ('strict' if a.shield else False)
                        res['runs'][kind] = episode(world, road, bp, base, _STATE['cases'][c], model, kind, a.commit, mode)
                    except Exception as e:  # recorded, never dropped
                        res['runs'][kind] = dict(error=f'{type(e).__name__}: {e}')
                res['seconds'] = round(time.time() - t0, 1)
                f.write(json.dumps(res) + '\n')
                f.flush()
                print(json.dumps(dict(done=i + 1, of=len(ids), case=c, seconds=res['seconds'])), flush=True)
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('model')
    ap.add_argument('--kinds', default='ladder:following:0,ladder:following_aeb:0,ladder:rss_sut:0,ba:cautious,ba:normal,ba:aggressive')
    ap.add_argument('--commit', action='store_true')
    ap.add_argument('--partition', default='development')
    ap.add_argument('--cases')
    ap.add_argument('--port', type=int, default=2020)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--shield', action='store_true', help='reference shield with the highway-env CC (see `shielded`)')
    ap.add_argument('--shield-window', action='store_true', help='window-conditional reference shield')
    main(ap.parse_args())
