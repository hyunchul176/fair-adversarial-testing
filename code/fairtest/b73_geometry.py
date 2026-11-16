"""B73 (B73 design, fixed before the run, available on request): contact geometry of
the final agent's reported failures, review of manuscript v7 item R2 (are the competent-SUT contacts meaningful contacts?).

jobs     (local, stored rows only) the reported failures of the final agent v2s, composed as in the paper: the policy
         stage's failure if it is robust and realistic (G2 and joint support), else the first success of the screened
         refinement bo_warm_g4s within 36 queries. One job per distinct trajectory (a refinement plan serves every seed
         whose policy query did not succeed); the seeds it stands for are recorded. Groups: the eight competent SUTs and
         the weakest SUT (following H0, contrast); for the CommonRoad planner (B69) every reported failure.
run      (server) every job is replayed against its SUT with the full trace:
           - geometry at the contact tick: the two bodies' overlap (separating-axis test), its depth and normal (minimum
             penetration axis, oriented from the SUT to the adversary), the contact point (centroid of the overlap), the
             body region of each vehicle nearest to it (front, rear, side), the closing speed along the normal and the
             longitudinal closing speed of the paper (SUT vx minus adversary vx); the same at the first touch, found by
             interpolating both poses between the previous tick and the contact tick;
           - body-tolerance replays: highway-env's collision polygon (RoadObject.polygon, used only by the collision check)
             shrunk by s = 0.05 and 0.10 m per side, the attack unchanged: does the SUT still touch the adversary, when, and
             how fast; s = 0 through the same patch must reproduce the plain replay (check), and the SUT's trajectory before
             the original contact must be identical (the patch changes nothing but the contact test).
analyse  GA1 (judged): at least 80% of the competent-SUT reported failures, seed-weighted and pooled over the datasets,
         persist with the bodies shrunk by 0.05 m per side. GA2-GA4 descriptive (contact types, normal closing speeds,
         negative longitudinal closing, persistence at 0.10 m). I1-I3 implementation checks.

  python -m fairtest.b73_geometry jobs OUT_DIR --jobs fairtest_jobs
  python -m fairtest.b73_geometry layout-jobs OUT.jsonl RUN_DIR --dataset NAME [--g4s REFINE.jsonl] [--all-suts]
  python -m fairtest.b73_geometry run OUT.jsonl JOBS.jsonl [--workers W]        (FAIRTEST_DATA = that dataset's cases)
  python -m fairtest.b73_geometry analyse OUT.json RUN.jsonl [RUN.jsonl ...]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np

COMPETENT8 = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0',
              'drl_sut@3_H0', 'drl_sut@6_H0')
WEAKEST = 'following_H0'
K_FULL = 37  # policy query + 36 refinement queries
SHRINKS = (0.0, 0.05, 0.10)  # metres per side; 0.0 is the patch check
GA1_SHRINK, GA1_MIN = 0.05, .80


def _rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ----------------------------------------------------------------------------------------------------------- jobs
def _g4s(paths):
    """{(case, sut): (first success query, theta)} of bo_warm_g4s."""
    out = {}
    for p in paths:
        for r in _rows(p):
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == 'bo_warm_g4s' and 'error' not in rec and rec.get('first_fair') is not None:
                    out[(r['case_id'], s)] = (rec['first_fair'], rec['best']['theta'])
    return out


def _compose(dataset, policy_ok, actions, g4s, seeds, cases, suts, group_of):
    """Jobs of the reported failures; policy_ok[(seed, case, sut)] -> bool, actions[(seed, case, sut)] -> list."""
    jobs, refine_seeds = [], collections.defaultdict(list)
    for s in suts:
        for c in cases:
            for x in seeds:
                if policy_ok.get((x, c, s), False):
                    jobs.append(dict(dataset=dataset, group=group_of(s), sut=s, case_id=c, kind='policy', seeds=[x],
                                     payload=actions[(x, c, s)]))
                elif (c, s) in g4s and 1 + g4s[(c, s)][0] <= K_FULL:
                    refine_seeds[(c, s)].append(x)
    for (c, s), xs in sorted(refine_seeds.items()):
        jobs.append(dict(dataset=dataset, group=group_of(s), sut=s, case_id=c, kind='refine', seeds=xs, payload=g4s[(c, s)][1],
                         first_query=g4s[(c, s)][0]))
    return jobs


def _group(s):
    return 'competent' if s in COMPETENT8 else ('weakest' if s == WEAKEST else 'other')


def jobs_dev(jobs_dir):
    """Development: b26 final policy rows + b26 margin + b45 replay (b56_analysis.policy), refinement b56_dev g4s."""
    from fairtest.b56_analysis import policy
    robust, inside = policy(jobs_dir, 'dev')
    actions = {}
    for s in (1, 2, 3):
        for r in _rows(jobs_dir / 'b26' / 'final' / f'final_s{s}_dev.jsonl'):
            if 'valid' in r:
                actions[(f's{s}', r['case_id'], r['sut'])] = r['actions']
    ok = {k: bool(v and inside.get(k, False)) for k, v in robust.items()}
    seeds = sorted({k[0] for k in robust})
    cases = sorted({k[1] for k in robust})
    g4s = _g4s([jobs_dir / 'b56' / 'b56_dev.jsonl'])
    return _compose('dev', ok, actions, g4s, seeds, cases, COMPETENT8 + (WEAKEST,), _group)


def jobs_layout(run_dir, dataset, g4s_paths, all_suts=False):
    """B41 layout (final_s*.jsonl, margin.jsonl, policy_replay.jsonl, case_pool.json; b41_analysis.load)."""
    from fairtest.b41_analysis import load
    evals, realistic, _, pool, _ = load(Path(run_dir))
    seeds = sorted(evals['final'])
    cases = sorted(c['case_id'] for c in pool['cases'])
    ok = {(x, c, s): bool(m.get('robust') and realistic.get((x, c, s), False)) for x in seeds for (c, s), m in evals['final'][x].items()}
    actions = {}
    for x in seeds:
        for r in _rows(Path(run_dir) / f'final_s{x}.jsonl'):
            if 'valid' in r:
                actions[(x, r['case_id'], r['sut'])] = r['actions']
    suts = sorted({s for (_, _, s) in ok}) if all_suts else COMPETENT8 + (WEAKEST,)
    g4s = _g4s([Path(p) for p in g4s_paths])
    return _compose(dataset, ok, actions, g4s, seeds, cases, suts, (lambda s: 'commonroad') if all_suts else _group)


def jobs_main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    j = Path(a.jobs)
    sets = dict(dev=jobs_dev(j),
                mitra=jobs_layout(j / 'b41', 'mitra', [j / 'b56' / 'b56_mitra.jsonl']),
                highsim=jobs_layout(j / 'b70', 'highsim', [j / 'b70' / 'refine.jsonl']),
                exid=jobs_layout(j / 'b71', 'exid', [j / 'b71' / 'refine.jsonl']))
    summary = {}
    for name, jobs in sets.items():
        with open(out / f'jobs_{name}.jsonl', 'w', encoding='utf-8') as f:
            for x in jobs:
                f.write(json.dumps(x) + '\n')
        summary[name] = {g: dict(jobs=sum(1 for x in jobs if x['group'] == g), seed_level=sum(len(x['seeds']) for x in jobs if x['group'] == g))
                         for g in ('competent', 'weakest')}
    print(json.dumps(summary, indent=1))


def layout_jobs_main(a):
    jobs = jobs_layout(a.run_dir, a.dataset, a.g4s or [Path(a.run_dir) / 'refine.jsonl'], all_suts=a.all_suts)
    with open(a.out, 'w', encoding='utf-8') as f:
        for x in jobs:
            f.write(json.dumps(x) + '\n')
    print(json.dumps(dict(jobs=len(jobs), seed_level=sum(len(x['seeds']) for x in jobs))))


# ------------------------------------------------------------------------------------------------------- geometry
def corners(s, shrink=0.):
    c, n = math.cos(s['heading']), math.sin(s['heading'])
    hl, hw = s['length'] / 2 - shrink, s['width'] / 2 - shrink
    return [(s['x'] + dx * c - dy * n, s['y'] + dx * n + dy * c) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def sat(p, q):
    """(depth, nx, ny) of the minimum penetration over the four rectangle axes, or None if the bodies are apart."""
    best = None
    for poly in (p, q):
        for i in range(2):
            (x1, y1), (x2, y2) = poly[i], poly[i + 1]
            ax, ay = -(y2 - y1), x2 - x1
            nrm = math.hypot(ax, ay)
            ax, ay = ax / nrm, ay / nrm
            pa = [ax * x + ay * y for x, y in p]
            qa = [ax * x + ay * y for x, y in q]
            o = min(max(pa), max(qa)) - max(min(pa), min(qa))
            if o <= 0.:
                return None
            if best is None or o < best[0]:
                best = (o, ax, ay)
    return best


def _clip(subject, clip):
    """Sutherland-Hodgman intersection of two convex polygons (any consistent orientation)."""
    def area2(poly):
        return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
    if area2(clip) < 0:
        clip = clip[::-1]
    out = list(subject)
    for i in range(len(clip)):
        (ax, ay), (bx, by) = clip[i], clip[(i + 1) % len(clip)]
        inside = lambda p: (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) >= 0  # noqa: E731
        inp, out = out, []
        if not inp:
            break
        prev = inp[-1]
        for cur in inp:
            if inside(cur):
                if not inside(prev):
                    out.append(_cross(prev, cur, (ax, ay), (bx, by)))
                out.append(cur)
            elif inside(prev):
                out.append(_cross(prev, cur, (ax, ay), (bx, by)))
            prev = cur
    return out


def _cross(p, q, a, b):
    x1, y1, x2, y2, x3, y3, x4, y4 = *p, *q, *a, *b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return q
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def region(s, px, py):
    """Body region of vehicle s nearest to the point: front, rear or side."""
    c, n = math.cos(s['heading']), math.sin(s['heading'])
    dx, dy = px - s['x'], py - s['y']
    u, v = dx * c + dy * n, -dx * n + dy * c
    d = dict(front=s['length'] / 2 - u, rear=u + s['length'] / 2, side=s['width'] / 2 - abs(v))
    return min(d, key=d.get)


def contact_type(sut_region, adv_region):
    if sut_region == 'front' and adv_region == 'rear':
        return 'rear_end'
    if sut_region == 'front' and adv_region == 'side':
        return 'front_to_side'
    if sut_region == 'side':
        return 'side_contact'
    return 'other'


def geometry(g, t, shrink=0.):
    """Contact geometry of adversary state g and SUT state t (trace rows), or None if the bodies do not overlap."""
    p, q = corners(t, shrink), corners(g, shrink)
    s = sat(p, q)
    if s is None:
        return None
    depth, nx, ny = s
    if nx * (g['x'] - t['x']) + ny * (g['y'] - t['y']) < 0:  # orient from the SUT to the adversary
        nx, ny = -nx, -ny
    poly = _clip(p, q) or p
    px, py = float(np.mean([x for x, _ in poly])), float(np.mean([y for _, y in poly]))
    rvx, rvy = t['vx'] - g['vx'], t['vy'] - g['vy']
    sr, ar = region(t, px, py), region(g, px, py)
    return dict(depth=round(depth, 4), normal=[round(nx, 4), round(ny, 4)], normal_closing=round(rvx * nx + rvy * ny, 3),
                closing=round(t['vx'] - g['vx'], 3), lateral_rel=round(g['vy'] - t['vy'], 3), dy=round(g['y'] - t['y'], 3),
                dx=round(g['x'] - t['x'], 3), sut_region=sr, adv_region=ar, type=contact_type(sr, ar))


def _lerp(a, b, w):
    out = dict(a)
    for k in ('x', 'y', 'heading', 'vx', 'vy'):
        out[k] = a[k] + w * (b[k] - a[k])
    return out


def states(row):
    """(adversary, SUT) before highway-env's collision impact: in the tick that detects a contact the simulator also
    separates the overlapping bodies, and the trace's generator/target are the separated states (B73 deviation D1)."""
    return row.get('unimpacted_generator') or row['generator'], row.get('unimpacted_target') or row['target']


def first_touch(prev, cur):
    """Geometry at the first overlap between two ticks (linear interpolation of both poses, bisection)."""
    (g0, t0), (g1, t1) = states(prev), states(cur)
    if sat(corners(t0), corners(g0)) is not None:
        return dict(at=0., geometry=geometry(g0, t0))
    lo, hi = 0., 1.
    for _ in range(30):
        mid = (lo + hi) / 2
        if sat(corners(_lerp(t0, t1, mid)), corners(_lerp(g0, g1, mid))) is None:
            lo = mid
        else:
            hi = mid
    return dict(at=round(hi, 4), geometry=geometry(_lerp(g0, g1, hi), _lerp(t0, t1, hi)))


# ------------------------------------------------------------------------------------------------------------ run
_SHRINK = [0.]


def _patch():
    import highway_env.vehicle.objects as objects
    if getattr(objects.RoadObject, '_b73_patched', False):
        return
    orig = objects.RoadObject.polygon

    def polygon(self):
        s = _SHRINK[0]
        if s <= 0.:
            return orig(self)
        hl, hw = self.LENGTH / 2 - s, self.WIDTH / 2 - s
        points = np.array([[-hl, -hw], [-hl, +hw], [+hl, +hw], [+hl, -hw]]).T
        c, n = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[c, -n], [n, c]])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])
    objects.RoadObject.polygon = polygon
    objects.RoadObject._b73_patched = True


def _episode(case, job, name, h):
    from fairtest import runner
    from fairtest.b11_search import ContinuousPlan
    from fairtest.b23_crossjudge import replay
    if job['kind'] == 'policy':
        return replay(case, job['payload'], name, h)
    return runner.run_episode(case, ContinuousPlan.from_theta(job['payload']), name, h, keep_trace=True)


def _task(job):
    from fairtest import runner
    runner._init()
    _patch()
    case = runner._STATE['cases'][job['case_id']]
    name, h = job['sut'].rsplit('_H', 1)
    h = float(h)
    _SHRINK[0] = 0.
    out = {k: job[k] for k in ('dataset', 'group', 'sut', 'case_id', 'kind', 'seeds')}
    try:
        base = _episode(case, job, name, h)
        tr = base['trace']
        out['contact'] = bool(base['pair_contact'])
        out['ticks'] = len(tr)
        if out['contact']:
            out['geometry'] = geometry(*states(tr[-1]))
            out['first_touch'] = first_touch(tr[-2], tr[-1]) if len(tr) >= 2 else None
        out['shrink'] = {}
        for s in SHRINKS:
            _SHRINK[0] = s
            r = _episode(case, job, name, h)
            t2 = r['trace']
            n = min(len(tr), len(t2)) - 1  # ticks before the earlier end
            same = all(abs(t2[i]['target']['x'] - tr[i]['target']['x']) < 1e-9 and abs(t2[i]['generator']['x'] - tr[i]['generator']['x']) < 1e-9
                       for i in range(max(0, n)))
            rec = dict(contact=bool(r['pair_contact']), ticks=len(t2), pre_contact_identical=bool(same))
            if r['pair_contact']:
                rec['geometry'] = geometry(*states(t2[-1]), s)
                rec['closing'] = round(t2[-1]['target']['vx'] - t2[-1]['generator']['vx'], 3)
            out['shrink'][f'{s:.2f}'] = rec
        _SHRINK[0] = 0.
    except Exception as e:  # recorded, never dropped
        out['error'] = f'{type(e).__name__}: {e}'
        _SHRINK[0] = 0.
    return out


def run_main(a):
    jobs = list(_rows(a.jobs))
    with mp.Pool(a.workers) as pool, open(a.out, 'w', encoding='utf-8') as f:
        for i, r in enumerate(pool.imap_unordered(_task, jobs, chunksize=2)):
            f.write(json.dumps(r) + '\n')
            f.flush()
    print(json.dumps(dict(replayed=len(jobs))))


# -------------------------------------------------------------------------------------------------------- analyse
def analyse(a):
    rows = [r for p in a.runs for r in _rows(p)]
    rep = dict(rows=len(rows), errors=sum('error' in r for r in rows))
    ok = [r for r in rows if 'error' not in r]
    # implementation checks
    rep['I1'] = dict(contact_in_plain_replay=sum(r['contact'] for r in ok), of=len(ok))
    rep['no_overlap_at_contact'] = sum(1 for r in ok if r['contact'] and r.get('geometry') is None)
    zero = [r for r in ok if r['contact'] and r.get('geometry') is not None]
    rep['I2'] = dict(shrink0_reproduces=sum(r['shrink']['0.00']['contact'] and r['shrink']['0.00']['ticks'] == r['ticks'] for r in zero),
                     of=len(zero))
    rep['I3'] = dict(pre_contact_identical=sum(all(v['pre_contact_identical'] for v in r['shrink'].values()) for r in zero), of=len(zero))

    def w(r):
        return len(r['seeds'])

    # persistence and seed counts over every reported failure (GA1's denominator); contact geometry over those whose
    # pre-impact bodies overlap. highway-env also registers a contact one tick ahead when the bodies would overlap within
    # the next step at their current velocities (will_intersect); those rows have no overlap yet and are counted apart.
    groups = collections.defaultdict(list)
    for r in ok:
        if r['contact']:
            groups[(r['dataset'], r['group'])].append(r)
            groups[('all', r['group'])].append(r)
    rep['groups'] = {}
    for (ds, g), allr in sorted(groups.items()):
        rs = [r for r in allr if r.get('geometry') is not None]
        tot_all = sum(w(r) for r in allr)
        tot = sum(w(r) for r in rs)
        types = collections.Counter()
        for r in rs:
            types[r['geometry']['type']] += w(r)
        neg = [r for r in rs if r['geometry']['closing'] < 0]
        negtypes = collections.Counter()
        for r in neg:
            negtypes[r['geometry']['type']] += w(r)
        nc = np.repeat([r['geometry']['normal_closing'] for r in rs], [w(r) for r in rs]) if rs else np.array([np.nan])
        ft = [r['first_touch']['geometry']['normal_closing'] for r in rs if r.get('first_touch') and r['first_touch']['geometry']]
        ftw = [w(r) for r in rs if r.get('first_touch') and r['first_touch']['geometry']]
        persist = {s_: round(sum(w(r) for r in allr if r['shrink'][s_]['contact']) / tot_all, 4) for s_ in ('0.05', '0.10')}
        rep['groups'][f'{ds}|{g}'] = dict(
            jobs=len(allr), seed_level=tot_all, lookahead_contacts=dict(jobs=len(allr) - len(rs), seed_level=tot_all - tot),
            types={k: round(v / tot, 4) for k, v in types.items()} if tot else {},
            negative_closing=dict(seed_level=sum(w(r) for r in neg), types={k: round(v / max(1, sum(w(x) for x in neg)), 4) for k, v in negtypes.items()}),
            normal_closing=dict(median=round(float(np.median(nc)), 3), p10=round(float(np.percentile(nc, 10)), 3),
                                p90=round(float(np.percentile(nc, 90)), 3)),
            first_touch_normal_closing_median=round(float(np.median(np.repeat(ft, ftw))), 3) if ft else None,
            depth_median=round(float(np.median(np.repeat([r['geometry']['depth'] for r in rs], [w(r) for r in rs]))), 4) if rs else None,
            persist=persist)
    comp = rep['groups'].get('all|competent')
    if comp:
        rep['GA1'] = dict(persist=comp['persist'][f'{GA1_SHRINK:.2f}'], threshold=GA1_MIN,
                          supported=comp['persist'][f'{GA1_SHRINK:.2f}'] >= GA1_MIN)
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('rows', 'errors', 'no_overlap_at_contact', 'I1', 'I2', 'I3', 'GA1') if k in rep}))
    for k, v in rep['groups'].items():
        print(k, v['seed_level'], v['types'], 'persist', v['persist'], 'normal', v['normal_closing'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('jobs')
    p.add_argument('out')
    p.add_argument('--jobs', required=True)
    q = sub.add_parser('layout-jobs')
    q.add_argument('out')
    q.add_argument('run_dir')
    q.add_argument('--dataset', required=True)
    q.add_argument('--g4s', nargs='*')
    q.add_argument('--all-suts', action='store_true')
    r = sub.add_parser('run')
    r.add_argument('out')
    r.add_argument('jobs')
    r.add_argument('--workers', type=int, default=24)
    s = sub.add_parser('analyse')
    s.add_argument('out')
    s.add_argument('runs', nargs='+')
    a = ap.parse_args()
    dict(jobs=jobs_main, run=run_main, analyse=analyse)[a.cmd](a) if a.cmd != 'layout-jobs' else layout_jobs_main(a)
