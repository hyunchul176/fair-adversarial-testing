"""Paper-summary video (not evidence): bird's-eye (top-down) CARLA 0.9.16 Town04 recordings of stored attacks.

Two passes per scene (one trajectory row against one follower kind):
1. physics - the carla_replay.run episode (E7b main configuration when the FAIRTEST_CARLA_* variables are set; carla_replay
   is unchanged), copied here to log both vehicles' world poses after every 0.1 s tick. Like run, it stops at contact.
2. render - both vehicles, now with physics off, are posed by interpolating the logged ticks at FPS frames per second,
   and a top-down camera follows the follower. The three poses go in one command batch that is applied before the
   tick that renders the frame, and only the image of that tick is kept. (The first video moved a free chase camera
   with set_transform before world.tick(); the move landed one tick late at random, so the vehicles seemed to jump
   back and forth.)
Per scene it writes OUT_DIR/<scene>/b%05d.png and meta.json: the physics result, and per frame the time, speeds,
acceleration, gap, contact and the pixel centre and outline of both vehicles for the overlays.

  python -m fairtest.carla_bev OUT_DIR TRAJ.jsonl --scenes fair=ladder:following:0,fair=cc [--port 2000]
      [--width 1920 --height 340 --span 80 --lead 30 --fps 30] [--tag-suffix _strip]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue

import numpy as np

from fairtest import carla_replay as CR

COLORS = dict(adv='205,35,30', sut='30,95,220', cc='25,160,70')


def connect_render(port, weather='ClearNoon'):
    """Synchronous, fixed 0.1 s step as E7b, rendering on (the camera needs it; physics is unchanged)."""
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
    world.set_weather(getattr(carla.WeatherParameters, weather))
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    world.tick()
    return client, world


def _pose(tf):
    return [tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw, tf.rotation.pitch, tf.rotation.roll]


def simulate(world, road, bps, row, kind, base):
    """carla_replay.run with a pose log (per tick: time, both world poses, road-frame states, gap, clearance)."""
    import carla
    w = road.w
    sut0, cruise, tr = row['sut0'], row['cruise'], row['trajectory']
    ba = CR.is_agent(kind)
    sut = world.spawn_actor(bps['cc' if kind == 'cc' else 'sut'], road.transform(CR.spawn_x(kind, sut0['vx']), w))
    adv = world.spawn_actor(bps['adv'], road.transform(min(road.length - 20., CR.START_X + 250.), 0.))
    log = dict(kind=kind, ticks=[])
    try:
        CR._brakes(sut)
        CR._brakes(adv)
        world.tick()
        e, ea = sut.bounding_box.extent, adv.bounding_box.extent
        dims = (2 * e.x, 2 * e.y)
        log['extent'] = dict(sut=[e.x, e.y, e.z], adv=[ea.x, ea.y, ea.z])
        target = CR.start_target(world, road, sut, kind, sut0, cruise, base, dims)
        xs0, ys0, v0 = road.state(sut)[:3]
        shift = xs0 - sut0['x']

        def place(p):
            gx, gy = p['x'] + shift, p['y'] * w / CR.LANE_HW
            gvx, gvy = p['vx'], p['vy'] * w / CR.LANE_HW
            gh = math.atan2(gvy, max(gvx, .1))
            tf = road.transform(gx, gy, gh)
            adv.set_transform(tf)
            adv.set_target_velocity(road.velocity(gvx, gvy, gx))
            return (gx, gy, gvx, gvy, gh), tf
        g_now, tf0 = place(tr[0])
        if not ba:
            target.v_cmd = v0
        log['ticks'].append(dict(t=0., sut=_pose(sut.get_transform()), adv=_pose(tf0), xs=xs0, ys=ys0, xa=g_now[0],
                                 ya=g_now[1], v_sut=v0, v_adv=g_now[2], acc=0., gap=None, clearance=None))
        last_v, hit, t_hit, min_clear = v0, False, None, math.inf
        for k in range(CR.HORIZON):
            st = road.state(sut)
            if not ba:
                target._measured = (st[2] - last_v) / CR.DT
            target.step(k * CR.DT, g_now, st, dims, sut, carla)
            world.tick()
            p_adv = _pose(adv.get_transform())  # physics pose after the tick, before the next placement
            g_now, _ = place(tr[k + 1])
            gx, gy = g_now[:2]
            st = road.state(sut)
            acc = (st[2] - last_v) / CR.DT
            last_v = st[2]
            same_lane = abs(gy - st[1]) < ea.y + e.y
            gap = (gx - st[0]) - (ea.x + e.x) if same_lane and gx > st[0] else None
            clear = CR.clearance(sut, adv)
            min_clear = min(min_clear, clear)
            log['ticks'].append(dict(t=round((k + 1) * CR.DT, 1), sut=_pose(sut.get_transform()), adv=p_adv, xs=st[0],
                                     ys=st[1], xa=gx, ya=gy, v_sut=st[2], v_adv=g_now[2], acc=acc, gap=gap,
                                     clearance=clear))
            if min_clear <= 0.:
                hit, t_hit = True, round((k + 1) * CR.DT, 1)
                break
        log.update(contact=hit, t_contact=t_hit, min_clearance=round(min_clear, 3),
                   min_gap=min([t['gap'] for t in log['ticks'] if t['gap'] is not None] or [None]))
    finally:
        CR.release(sut, kind, world)
        sut.destroy()
        adv.destroy()
        world.tick()
    return log


def _lerp_pose(a, b, f):
    out = [a[i] + (b[i] - a[i]) * f for i in range(3)]
    for i in range(3, 6):  # angles in degrees, shortest way round
        d = (b[i] - a[i] + 180.) % 360. - 180.
        out.append(a[i] + d * f)
    return out


def _tf(carla, p):
    return carla.Transform(carla.Location(p[0], p[1], p[2]), carla.Rotation(yaw=p[3], pitch=p[4], roll=p[5]))


def _project(cam_tf, K, pts):
    """World points (N x 3) -> pixel (u, v) through a pinhole camera at cam_tf (UE axes: x forward, y right, z up)."""
    m = np.array(cam_tf.get_inverse_matrix())
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ m.T
    cam = np.c_[p[:, 1], -p[:, 2], p[:, 0]]
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


def render(client, world, road, bps, log, kind, out_dir, width, height, span, lead, fps, cam_attrs=(), warmup=30):
    """Pose both vehicles (physics off) and the top-down camera in one batch per frame; keep that tick's image."""
    import carla
    os.makedirs(out_dir, exist_ok=True)
    ticks = log['ticks']
    first = ticks[0]
    lifted = [(p[0], p[1], p[2] + 30. + 10. * i) + tuple(p[3:]) for i, p in enumerate((first['sut'], first['adv']))]
    sut = world.spawn_actor(bps['cc' if kind == 'cc' else 'sut'], _tf(carla, lifted[0]))  # in the air: no spawn clash
    adv = world.spawn_actor(bps['adv'], _tf(carla, lifted[1]))
    for v in (sut, adv):
        v.set_simulate_physics(False)
    fov = 2. * math.degrees(math.atan(span / 2. / 100.))  # camera 100 m above the road
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    for k, v in (('image_size_x', width), ('image_size_y', height), ('fov', fov), ('motion_blur_intensity', 0.)) + tuple(cam_attrs):
        cam_bp.set_attribute(k, str(v))
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(0., 0., 200.), carla.Rotation(pitch=-90.)))
    images = queue.Queue()
    cam.listen(images.put)
    f = width / (2. * math.tan(math.radians(fov) / 2.))
    K = np.array([[f, 0., width / 2.], [0., f, height / 2.], [0., 0., 1.]])
    meta = dict(kind=kind, fps=fps, width=width, height=height, span=span, lead=lead, fov=fov,
                contact=log['contact'], t_contact=log['t_contact'], min_gap=log['min_gap'],
                min_clearance=log['min_clearance'], frames=[])
    def poses(t):
        i = min(int(t / CR.DT), len(ticks) - 2)
        a, b = ticks[i], ticks[i + 1]
        fr = (t - a['t']) / max(b['t'] - a['t'], 1e-9)
        ps, pa = _lerp_pose(a['sut'], b['sut'], fr), _lerp_pose(a['adv'], b['adv'], fr)
        xs = a['xs'] + (b['xs'] - a['xs']) * fr
        _, yaw, z, _, _ = road._at(xs + lead)
        centre = road.transform(xs + lead, road.w / 2.)
        cam_tf = carla.Transform(carla.Location(centre.location.x, centre.location.y, float(z) + 100.),
                                 carla.Rotation(pitch=-90., yaw=math.degrees(yaw) - 90., roll=0.))
        return a, b, fr, ps, pa, cam_tf

    def shoot(ps, pa, cam_tf):
        client.apply_batch_sync([carla.command.ApplyTransform(sut.id, _tf(carla, ps)),
                                 carla.command.ApplyTransform(adv.id, _tf(carla, pa)),
                                 carla.command.ApplyTransform(cam.id, cam_tf)], True)
        frame = world.get_snapshot().frame
        while True:
            img = images.get(timeout=30.)
            if img.frame >= frame:
                return img

    try:
        world.tick()
        t_end = ticks[-1]['t']
        n = int(round(t_end * fps)) + 1
        for _ in range(warmup):  # let the auto exposure settle on the first view
            shoot(*poses(0.)[3:])
        for j in range(n):
            t = min(j / fps, t_end)
            a, b, fr, ps, pa, cam_tf = poses(t)
            img = shoot(ps, pa, cam_tf)
            img.save_to_disk(os.path.join(out_dir, f'b{j:05d}.png'))
            row = dict(t=round(t, 4), frame=img.frame)
            for name, p, ext in (('sut', ps, log['extent']['sut']), ('adv', pa, log['extent']['adv'])):
                tf = _tf(carla, p)
                top = p[2] + 2. * ext[2]
                corners = [tf.transform(carla.Location(sx * ext[0], sy * ext[1], 0.)) for sx, sy in
                           ((1, 1), (1, -1), (-1, -1), (-1, 1))]
                uv = _project(cam_tf, K, [[c.x, c.y, top] for c in corners] + [[p[0], p[1], top]])
                row[name] = dict(uv=[round(float(x), 1) for x in uv[-1]],
                                 outline=[[round(float(x), 1) for x in q] for q in uv[:4]])
            # per-frame quantities: speeds and acceleration interpolated, gap from the nearest earlier tick
            row.update(v_sut=a['v_sut'] + (b['v_sut'] - a['v_sut']) * fr, v_adv=a['v_adv'] + (b['v_adv'] - a['v_adv']) * fr,
                       acc=b['acc'] if fr > 0 else a['acc'], gap=(b if fr > .5 else a)['gap'],
                       contact=bool(log['contact'] and t >= log['t_contact'] - 1e-9))
            meta['frames'].append(row)
    finally:
        cam.stop()
        cam.destroy()
        sut.destroy()
        adv.destroy()
        world.tick()
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), indent=0)
    return meta


def main(a):
    from fairtest.runner import _STATE, _init
    _init()
    base = _STATE['base']
    client, world = connect_render(a.port, a.weather)
    cam_attrs = tuple(tuple(kv.split('=', 1)) for kv in a.cam.split(',') if kv)
    road = CR.Road(world)
    lib = world.get_blueprint_library()
    bps = {}
    for role, rgb in COLORS.items():
        bp = lib.find(CR.BLUEPRINT)
        if bp.has_attribute('color'):
            bp.set_attribute('color', rgb)
        bps[role] = bp
    rows = {}
    for line in open(a.traj, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            tag = r.get('tag') or ('collide' if r['hwenv'].get('cc_contact') else
                                   ('fair' if r['hwenv'].get('sut_contact') else 'competent'))
            rows[tag] = r
    try:
        for spec in a.scenes.split(','):
            tag, kind = spec.split('=', 1)
            base_name = f"{tag}_{rows[tag]['case_id']}_{kind.replace(':', '-')}"
            name = base_name + a.tag_suffix
            reuse = os.path.join(a.out, f'{base_name}.physics.json')
            if a.reuse_physics and os.path.exists(reuse):  # render another view of an episode already simulated
                log = json.load(open(reuse))
            else:
                log = simulate(world, road, bps, rows[tag], kind, base)
                json.dump(log, open(os.path.join(a.out, f'{name}.physics.json'), 'w'))
            m = render(client, world, road, bps, log, kind, os.path.join(a.out, name), a.width, a.height, a.span,
                       a.lead, a.fps, cam_attrs)
            print(json.dumps(dict(scene=name, ticks=len(log['ticks']), frames=len(m['frames']), contact=log['contact'],
                                  t_contact=log['t_contact'], min_gap=log['min_gap'])), flush=True)
    finally:
        s = world.get_settings()
        s.synchronous_mode = False
        world.apply_settings(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('traj')
    ap.add_argument('--scenes', required=True, help="comma-separated tag=kind; tag = the row's 'tag' or fair / competent / collide")
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=340)
    ap.add_argument('--span', type=float, default=80., help='metres along the road across the image width')
    ap.add_argument('--lead', type=float, default=30., help='camera centre ahead of the follower (m)')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--tag-suffix', default='')
    ap.add_argument('--cam', default='', help='extra camera attributes, e.g. exposure_compensation=-1,gamma=2.2')
    ap.add_argument('--weather', default='ClearNoon')
    ap.add_argument('--reuse-physics', action='store_true',
                    help='render from OUT_DIR/<scene without suffix>.physics.json when it exists (same episode, other view)')
    main(ap.parse_args())
