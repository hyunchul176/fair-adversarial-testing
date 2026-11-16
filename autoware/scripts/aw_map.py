"""B54: a straight two-lane, same-direction highway as an Autoware map (lanelet2 OSM + local projector + tiny point cloud).

Map frame (ROS ENU, x forward, y left): the SUT lane centre is y = 0 and the adversary lane centre is y = +W (the lane on
the SUT's left, as in highway-env where the adversary lane is y = 0 and the SUT lane y = 4). Lane width W = 4 m, the
highway-env width, so highway-env lateral positions map without scaling: map_y = 4 - hw_y, map_yaw = -hw_heading.
The road runs from x = X0 to X1 in lanelets of SEG metres; the shared middle boundary is dashed (adjacent lanes).

  python -m fairtest.aw_map OUT_DIR [--x0 -600 --x1 2600 --seg 50 --speed-kmh 130]
"""
from __future__ import annotations

import argparse
import os

W = 4.


def build(x0=-600., x1=2600., seg=50., step=5., speed_kmh=130.):
    nodes, ways, rels = [], [], []
    nid, wid, rid = [1], [100000], [200000]

    def node(x, y):
        i = nid[0]
        nid[0] += 1
        nodes.append((i, x, y))
        return i

    def way(ids, subtype):
        i = wid[0]
        wid[0] += 1
        ways.append((i, ids, subtype))
        return i

    xs = []
    x = x0
    while x < x1 - 1e-6:
        xs.append(x)
        x += seg
    xs.append(x1)
    # boundary y: right edge of the SUT lane (-W/2), middle (W/2), left edge of the adversary lane (3W/2)
    bounds = {'right': -W / 2., 'mid': W / 2., 'left': 1.5 * W}
    prev_end = {k: None for k in bounds}
    for a, b in zip(xs[:-1], xs[1:]):
        n = max(1, int(round((b - a) / step)))
        line = {}
        for k, y in bounds.items():
            ids = [prev_end[k]] if prev_end[k] is not None else [node(a, y)]
            ids += [node(a + (b - a) * j / n, y) for j in range(1, n + 1)]
            prev_end[k] = ids[-1]
            line[k] = way(ids, 'dashed' if k == 'mid' else 'solid')
        for left, right in (('mid', 'right'), ('left', 'mid')):  # SUT lane, then adversary lane
            i = rid[0]
            rid[0] += 1
            rels.append((i, line[left], line[right]))
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="fairtest.aw_map">']
    for i, x, y in nodes:
        out.append(f'  <node id="{i}" lat="0" lon="0"><tag k="local_x" v="{x:.3f}"/><tag k="local_y" v="{y:.3f}"/>'
                   f'<tag k="ele" v="0.0"/></node>')
    for i, ids, subtype in ways:
        out.append(f'  <way id="{i}">' + ''.join(f'<nd ref="{r}"/>' for r in ids)
                   + f'<tag k="type" v="line_thin"/><tag k="subtype" v="{subtype}"/></way>')
    for i, left, right in rels:
        out.append(f'  <relation id="{i}"><member type="way" role="left" ref="{left}"/><member type="way" role="right" ref="{right}"/>'
                   '<tag k="type" v="lanelet"/><tag k="subtype" v="road"/><tag k="location" v="urban"/>'
                   f'<tag k="one_way" v="yes"/><tag k="participant:vehicle" v="yes"/><tag k="speed_limit" v="{speed_kmh:g}"/></relation>')
    out.append('</osm>')
    return '\n'.join(out) + '\n', len(nodes), len(ways), len(rels)


def pcd(x0, x1):
    pts = [(x, y, 0.) for x in range(int(x0), int(x1) + 1, 20) for y in (-3., 0., 3., 6.)]
    head = ['# .PCD v0.7 - Point Cloud Data file format', 'VERSION 0.7', 'FIELDS x y z', 'SIZE 4 4 4', 'TYPE F F F', 'COUNT 1 1 1',
            f'WIDTH {len(pts)}', 'HEIGHT 1', 'VIEWPOINT 0 0 0 1 0 0 0', f'POINTS {len(pts)}', 'DATA ascii']
    return '\n'.join(head + [f'{x:.1f} {y:.1f} {z:.1f}' for x, y, z in pts]) + '\n'


def main(a):
    os.makedirs(a.out, exist_ok=True)
    osm, n, w, r = build(a.x0, a.x1, a.seg, speed_kmh=a.speed_kmh)
    open(os.path.join(a.out, 'lanelet2_map.osm'), 'w').write(osm)
    open(os.path.join(a.out, 'pointcloud_map.pcd'), 'w').write(pcd(a.x0, a.x1))
    open(os.path.join(a.out, 'map_projector_info.yaml'), 'w').write('projector_type: Local\n')
    print(dict(nodes=n, ways=w, lanelets=r, out=a.out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--x0', type=float, default=-600.)
    ap.add_argument('--x1', type=float, default=2600.)
    ap.add_argument('--seg', type=float, default=50.)
    ap.add_argument('--speed-kmh', type=float, default=130.)
    main(ap.parse_args())
