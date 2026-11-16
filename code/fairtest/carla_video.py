"""Explainer video (not evidence): CARLA 0.9.16 Town04 camera recordings of stored attacks against one follower.

The replay is carla_replay.run (the adversary follows its recorded highway-env trajectory kinematically; the follower,
SUT or the R157 CC, is simulated with the E7b main configuration when the FAIRTEST_CARLA_* variables are set), copied
here with rendering on and a chase camera 13 m behind and 5.5 m above the follower; carla_replay is unchanged.
Per trajectory row and kind it writes PNG frames (one per 0.1 s tick from t = 0) and meta.json (per-frame speeds, gap,
contact).

  python -m fairtest.carla_video OUT_DIR TRAJ.jsonl --kinds cc,ladder:following:0 [--port 2000] [--width 1920 --height 540]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue

from fairtest import carla_replay as CR

COLORS = dict(adv='205,35,30', sut='30,95,220', cc='25,160,70')


def connect_render(port):
    """carla_replay.connect with rendering on (the camera needs it); synchronous, fixed 0.1 s step as E7b."""
    import carla
    client = carla.Client('localhost', port)
    client.set_timeout(120.)
    CR._CLIENT, CR.TM_PORT = client, 8000 + port - 2000
    world = client.get_world()
    if not world.get_map().name.endswith('Town04'):
        world = client.load_world('Town04')
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds, s.no_rendering_mode = True, CR.DT, False
    world.apply_settings(s)
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    world.tick()
    return world


def chase(road, st, carla):
    """Camera 13 m behind and 5.5 m above the follower, between the two lanes, pitched down 16 degrees."""
    x, y = st[0] - 13., road.w / 2.
    tf = road.transform(x, y, 0.)
    return carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z + 5.5),
                           carla.Rotation(pitch=-16., yaw=tf.rotation.yaw))


def record(world, road, bps, traj_row, kind, base, out_dir, width, height, tail_s=.4):
    import carla
    os.makedirs(out_dir, exist_ok=True)
    w = road.w
    sut0, cruise, tr = traj_row['sut0'], traj_row['cruise'], traj_row['trajectory']
    sut = world.spawn_actor(bps['cc' if kind == 'cc' else 'sut'], road.transform(CR.spawn_x(kind, sut0['vx']), w))
    adv = world.spawn_actor(bps['adv'], road.transform(min(road.length - 20., CR.START_X + 250.), 0.))
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    for k, v in (('image_size_x', width), ('image_size_y', height), ('fov', 72), ('motion_blur_intensity', 0.)):
        cam_bp.set_attribute(k, str(v))
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(0., 0., 100.)))
    images = queue.Queue()
    cam.listen(images.put)
    meta = dict(kind=kind, case_id=traj_row['case_id'], sut=traj_row['sut'], frames=[])

    def drain():
        while not images.empty():
            images.get_nowait()
    try:
        CR._brakes(sut)
        CR._brakes(adv)
        world.tick()
        e, ea = sut.bounding_box.extent, adv.bounding_box.extent
        dims = (2 * e.x, 2 * e.y)
        target = CR.start_target(world, road, sut, kind, sut0, cruise, base, dims)
        drain()
        xs0, _, v0 = road.state(sut)[:3]
        shift = xs0 - sut0['x']

        def place(p):
            gx, gy = p['x'] + shift, p['y'] * w / CR.LANE_HW
            gvx, gvy = p['vx'], p['vy'] * w / CR.LANE_HW
            gh = math.atan2(gvy, max(gvx, .1))
            adv.set_transform(road.transform(gx, gy, gh))
            adv.set_target_velocity(road.velocity(gvx, gvy, gx))
            return gx, gy, gvx, gvy, gh
        g_now = place(tr[0])
        target.v_cmd = v0
        last_v, hit, after = v0, False, 0
        for k in range(CR.HORIZON):
            st = road.state(sut)
            target._measured = (st[2] - last_v) / CR.DT
            target.step(k * CR.DT, g_now, st, dims, sut, carla)
            cam.set_transform(chase(road, st, carla))
            world.tick()
            img = images.get(timeout=30.)
            gx, gy, gvx, gvy, gh = g_now = place(tr[k + 1])
            st = road.state(sut)
            last_v = st[2]
            clear = CR.clearance(sut, adv)
            same_lane = abs(gy - st[1]) < ea.y + e.y
            gap = (gx - st[0]) - (ea.x + e.x) if same_lane and gx > st[0] else None
            hit = hit or clear <= 0.
            path = os.path.join(out_dir, f'f{k:04d}.png')
            img.save_to_disk(path)
            meta['frames'].append(dict(t=round((k + 1) * CR.DT, 1), v_follower=round(st[2], 3), v_adv=round(gvx, 3),
                                       gap=None if gap is None else round(gap, 3), contact=hit))
            if hit:
                after += 1
                if after > int(tail_s / CR.DT):
                    break
        meta['contact'] = hit
    finally:
        cam.stop()
        cam.destroy()
        CR.release(sut, kind, world)
        sut.destroy()
        adv.destroy()
        world.tick()
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=0)
    return meta


def main(a):
    from fairtest.runner import _STATE, _init
    _init()
    base = _STATE['base']
    world = connect_render(a.port)
    road = CR.Road(world)
    lib = world.get_blueprint_library()
    bps = {}
    for role, rgb in COLORS.items():
        bp = lib.find(CR.BLUEPRINT)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rgb)
        bps[role] = bp
    rows = [json.loads(line) for line in open(a.traj, encoding='utf-8') if line.strip()]
    try:
        for i, row in enumerate(rows):
            tag = 'collide' if row['hwenv'].get('cc_contact') else ('fair' if row['hwenv'].get('sut_contact') else 'competent')
            for kind in a.kinds.split(','):
                if tag == 'competent' and kind.startswith('ladder:following'):
                    kind = 'ladder:rss_sut:0'
                name = f"{tag}_{row['case_id']}_{kind.replace(':', '-')}"
                m = record(world, road, bps, row, kind, base, os.path.join(a.out, name), a.width, a.height)
                print(json.dumps(dict(scene=name, frames=len(m['frames']), contact=m['contact'])), flush=True)
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('traj')
    ap.add_argument('--kinds', default='ladder:following:0,cc')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=540)
    main(ap.parse_args())
