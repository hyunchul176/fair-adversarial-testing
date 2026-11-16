"""Visual replay in CARLA (not evidence): bird's-eye renders of highway-env episodes, both vehicles posed from the stored
0.1 s states with physics off, so the pictures show exactly the episodes behind the reported verdicts (2026-10-10).

carla_bev.py re-simulates the follower in CARLA physics (validation); this module only places both vehicles where
highway-env had them (longitudinal position relative to the follower's start, lateral position scaled from the 4 m
highway-env lane to the CARLA lane width) and reuses carla_bev.render for the top-down camera, frames and meta.json.

  python -m fairtest.carla_kin OUT_DIR TRACES.json [--port 2000] [--width 1800 --height 340 --span 80 --lead 30 --fps 30]
      [--cam exposure_compensation=-2]   (FAIRTEST_CARLA_BP picks the car model, as for carla_bev)
TRACES.json: viz_traces.py output (per scene: name, init, sut_trace and cc_trace with gx, gy, gvx, gvy, gh, tx, ty, tvx).
A trace row k is the state after tick k (t = (k + 1) * 0.1 s), so the episode is posed from the initial state at t = 0
followed by the rows; the contact time is that of the last row of an episode that ended in contact.
Writes OUT_DIR/<name>_sut/ and OUT_DIR/<name>_cc/ (b%05d.png, meta.json).
"""
from __future__ import annotations

import argparse
import json
import math
import os

from fairtest import carla_bev as BEV
from fairtest import carla_replay as CR


def make_log(road, init, trace, kind, extent, contact):
    """carla_bev log built from highway-env states: the initial state (t = 0), then trace row k at t = (k + 1) * 0.1 s."""
    w = road.w
    g0, f0 = init
    states = [dict(gx=g0['x'], gy=g0['y'], gvx=g0['vx'], gvy=g0['vy'], tx=f0['x'], ty=f0['y'], tvx=f0['vx'])] + trace
    x0 = states[0]['tx']
    ticks = []
    for i, r in enumerate(states):
        xs = CR.START_X + (r['tx'] - x0)
        xa = CR.START_X + (r['gx'] - x0)
        ys, ya = r['ty'] * w / CR.LANE_HW, r['gy'] * w / CR.LANE_HW
        ha = math.atan2(r['gvy'] * w / CR.LANE_HW, max(r['gvx'], .1))
        tf_s, tf_a = road.transform(xs, ys, 0.), road.transform(xa, ya, ha)
        same_lane = abs(r['gy'] - r['ty']) < 2.
        gap = (r['gx'] - r['tx']) - 5. if same_lane and r['gx'] > r['tx'] else None
        acc = 0. if i == 0 else (r['tvx'] - states[i - 1]['tvx']) / CR.DT
        ticks.append(dict(t=round(i * CR.DT, 1), sut=BEV._pose(tf_s), adv=BEV._pose(tf_a), xs=xs, ys=ys, xa=xa, ya=ya,
                          v_sut=r['tvx'], v_adv=r['gvx'], acc=acc, gap=gap, clearance=None))
    gaps = [t['gap'] for t in ticks if t['gap'] is not None]
    return dict(kind=kind, ticks=ticks, extent=extent, contact=bool(contact),
                t_contact=ticks[-1]['t'] if contact else None, min_clearance=None, min_gap=min(gaps) if gaps else None)


def main(a):
    import carla
    from fairtest.runner import _init
    _init()
    client, world = BEV.connect_render(a.port, a.weather)
    road = CR.Road(world)
    lib = world.get_blueprint_library()
    bps = {}
    for role, rgb in BEV.COLORS.items():
        bp = lib.find(CR.BLUEPRINT)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rgb)
        bps[role] = bp
    probe = world.spawn_actor(bps['sut'], carla.Transform(carla.Location(0., 0., 500.)))
    world.tick()
    e = probe.bounding_box.extent
    extent = dict(sut=[e.x, e.y, e.z], adv=[e.x, e.y, e.z])
    probe.destroy()
    world.tick()
    scenes = json.load(open(a.traces))
    cam_attrs = tuple(tuple(kv.split('=', 1)) for kv in a.cam.split(',') if kv)
    try:
        for s in scenes:
            if a.only and s['name'] not in a.only.split(','):
                continue
            for kind, tr, hit in (('sut', s['sut_trace'], s['sut_contact']), ('cc', s['cc_trace'], s['cc_contact'])):
                log = make_log(road, s['init'], tr, kind, extent, hit)
                name = f"{s['name']}_{kind}"
                m = BEV.render(client, world, road, bps, log, kind, os.path.join(a.out, name), a.width, a.height, a.span,
                               a.lead, a.fps, cam_attrs)
                print(json.dumps(dict(scene=name, ticks=len(log['ticks']), frames=len(m['frames']), contact=log['contact'],
                                      min_gap=log['min_gap'])), flush=True)
    finally:
        st = world.get_settings()
        st.synchronous_mode = False
        world.apply_settings(st)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('traces')
    ap.add_argument('--only', help='comma-separated scene names')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--width', type=int, default=1800)
    ap.add_argument('--height', type=int, default=340)
    ap.add_argument('--span', type=float, default=80.)
    ap.add_argument('--lead', type=float, default=30.)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--cam', default='', help='extra camera attributes, e.g. exposure_compensation=-2')
    ap.add_argument('--weather', default='ClearNoon')
    main(ap.parse_args())
