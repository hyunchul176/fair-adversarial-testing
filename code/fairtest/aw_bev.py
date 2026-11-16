"""Paper-summary video (not evidence): a B54 Autoware failure drawn in CARLA 0.9.16 Town04 from above, with the same
camera and render pass as carla_bev (physics off, poses and camera in one command batch per frame).

The vehicles are posed from recorded states, not simulated in CARLA:
  aw  the Autoware episode (aw_sut trace: the ego from Autoware's planning simulator, the adversary from the attack)
  cc  the R157 CC in the highway-env counterfactual of the same attack (the reference the verdict uses)
highway-env frame -> CARLA road frame: x + (START_X - sut0 x), lateral y * lane width / 4 (SUT lane = one lane width),
heading from the velocity (adversary) or from successive positions (ego).

  python -m fairtest.aw_bev OUT_DIR B54_DIR --variant aw_r157 --case 212 --source policy_s1 [--port 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from fairtest import carla_bev as BEV
from fairtest import carla_replay as CR


def _record(b54, variant, case_id, source):
    phase = 'refine' if source == 'refine' else 'policy'
    for p in sorted(Path(b54).glob(f'{phase}_*.jsonl')):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            if r['variant'] != variant or r['case_id'] != case_id:
                continue
            if phase == 'policy' and (r['attacker'] != 'v2j' or f"policy_s{r['seed']}" != source):
                continue
            return r['result']
    raise KeyError((variant, case_id, source))


def _log(road, states, extent, sut0):
    """states: per tick dict(t, x, y, v, a, xa, ya, vxa, vya, gap, clear) in the highway-env frame -> a carla_bev log."""
    shift, w = CR.START_X - sut0['x'], road.w
    ticks, prev = [], None
    for s in states:
        xs, ys = s['x'] + shift, s['y'] * w / CR.LANE_HW
        head = 0. if prev is None else math.atan2(ys - prev[1], max(xs - prev[0], 1e-3))
        prev = (xs, ys)
        xa, ya = s['xa'] + shift, s['ya'] * w / CR.LANE_HW
        ha = math.atan2(s['vya'] * w / CR.LANE_HW, max(s['vxa'], .1))
        ticks.append(dict(t=s['t'], sut=BEV._pose(road.transform(xs, ys, head)), adv=BEV._pose(road.transform(xa, ya, ha)),
                          xs=xs, ys=ys, xa=xa, ya=ya, v_sut=s['v'], v_adv=s['vxa'], acc=s['a'], gap=s['gap'],
                          clearance=s['clear']))
    return dict(ticks=ticks, extent=extent)


def aw_states(res, adv0, sut0):
    tr = res['trace']
    out = [dict(t=0., x=sut0['x'], y=sut0['y'], v=sut0['vx'], a=0., xa=adv0['x'], ya=adv0['y'], vxa=adv0['vx'], vya=adv0['vy'],
                gap=None, clear=None)]
    for r in tr:
        x, y, v, a = r['ego'][:4]
        xa, ya, vxa, vya = r['adv']
        out.append(dict(t=r['t'], x=x, y=y, v=v, a=a or 0., xa=xa, ya=ya, vxa=vxa, vya=vya, gap=r['gap'], clear=r['clear']))
    return out


def cc_states(raw, adv0, sut0):
    out = [dict(t=0., x=sut0['x'], y=sut0['y'], v=sut0['vx'], a=0., xa=adv0['x'], ya=adv0['y'], vxa=adv0['vx'], vya=adv0['vy'],
                gap=None, clear=None)]
    for k, r in enumerate(raw['trace']):
        g, t = r['generator'], r['target']
        same = abs(g['y'] - t['y']) < 2.
        out.append(dict(t=round(.1 * (k + 1), 1), x=t['x'], y=t['y'], v=t['vx'], a=r['target_control']['applied_mps2'],
                        xa=g['x'], ya=g['y'], vxa=g['vx'], vya=g['vy'],
                        gap=(g['x'] - t['x'] - 5.) if same and g['x'] > t['x'] else None, clear=None))
    return out


def main(a):
    from fairtest.b29_consensus import _run
    from fairtest.runner import _STATE, _init
    _init()
    case = _STATE['cases'][a.case]
    res = _record(a.b54, a.variant, a.case, a.source)
    if a.source == 'refine':
        how, payload, trace_res = 'theta', res['best']['theta'], dict(trace=res['best']['aw_trace'], t_contact=res['best']['aw']['t_contact'],
                                                                     contact=True, min_clearance=res['best']['aw']['min_clearance'])
    else:
        how, payload, trace_res = 'actions', res['actions'], res
    g0, t0 = case['physical_initial_state']
    adv0 = dict(x=g0['x'], y=g0['y'], vx=g0['vx'], vy=g0['vy'])
    sut0 = dict(x=t0['x'], y=t0['y'], vx=t0['vx'], vy=t0['vy'])
    raw_cc = _run(case, how, payload, 'cc_r157')
    client, world = BEV.connect_render(a.port, a.weather)
    road = CR.Road(world)
    lib = world.get_blueprint_library()
    bps = {}
    for role, rgb in BEV.COLORS.items():
        bp = lib.find(CR.BLUEPRINT)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rgb)
        bps[role] = bp
    probe = world.spawn_actor(bps['sut'], road.transform(CR.START_X, road.w))  # the drawn body's extent (outlines)
    world.tick()
    e = probe.bounding_box.extent
    extent = dict(sut=[e.x, e.y, e.z], adv=[e.x, e.y, e.z])
    probe.destroy()
    world.tick()
    cam_attrs = tuple(tuple(kv.split('=', 1)) for kv in a.cam.split(',') if kv)
    name = f'aw_{a.variant}_{a.case}_{a.source}'
    try:
        for kind, states, hit, t_hit in (
                ('aw', aw_states(trace_res, adv0, sut0), bool(trace_res['contact']), trace_res['t_contact']),
                ('cc', cc_states(raw_cc, adv0, sut0), bool(raw_cc['pair_contact']), None)):
            log = _log(road, states, extent, sut0)
            gaps = [t['gap'] for t in log['ticks'] if t['gap'] is not None]
            log.update(contact=hit, t_contact=t_hit, min_gap=min(gaps) if gaps else None,
                       min_clearance=trace_res.get('min_clearance') if kind == 'aw' else None)
            json.dump(log, open(os.path.join(a.out, f'{name}_{kind}.poses.json'), 'w'))
            m = BEV.render(client, world, road, bps, log, 'cc' if kind == 'cc' else 'sut', os.path.join(a.out, f'{name}_{kind}'),
                           a.width, a.height, a.span, a.lead, a.fps, cam_attrs)
            print(json.dumps(dict(scene=f'{name}_{kind}', ticks=len(log['ticks']), frames=len(m['frames']), contact=hit,
                                  t_contact=t_hit)), flush=True)
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('b54')
    ap.add_argument('--variant', default='aw_r157')
    ap.add_argument('--case', type=int, required=True)
    ap.add_argument('--source', required=True, help='policy_s1 | policy_s2 | policy_s3 | refine')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--width', type=int, default=1800)
    ap.add_argument('--height', type=int, default=340)
    ap.add_argument('--span', type=float, default=80.)
    ap.add_argument('--lead', type=float, default=30.)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--cam', default='exposure_compensation=-2')
    ap.add_argument('--weather', default='ClearNoon')
    main(ap.parse_args())
