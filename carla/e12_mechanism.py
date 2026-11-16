"""E12 post-hoc mechanism check: when is the perception threshold of the R157 C (adversary vehicle 1.095 m from the centre of its own lane) crossed under each mapping?

In CARLA, C reads the adversary vehicle's lateral position in metres as is (cc_target: displacement = |y - 0|). So for the same highway-env trajectory
  highway-env (4 m lanes):  y >= 1.095
  proportional (y' = y w/4):   y >= 1.095 * 4/w
  distance-preserving (y' = y + w - 4): y >= 1.095 + 4 - w
is where perception starts. For each trajectory it computes the three perception times and the distance (m) between the adversary vehicle and the SUT lane
centre at that time, and counts per perception-delay bin the share of trajectories that C avoided under the proportional mapping and that become C collisions under the distance-preserving one.

  python e12_mechanism.py WORK_DIR [w]      -> runs/E12_MECHANISM.json
"""
import collections
import json
import sys
from pathlib import Path

PERCEIVE = 1.095
KIND = {'following_H0': 'ladder:following:0', 'following_H0.5': 'ladder:following:0.5', 'following_H1': 'ladder:following:1',
        'reference_idm_H0': 'ladder:reference_idm:0', 'following_aeb_H0': 'ladder:following_aeb:0',
        'following_aeb_H0.5': 'ladder:following_aeb:0.5', 'rss_sut_H0': 'ladder:rss_sut:0', 'rss_lat_sut_H0': 'ladder:rss_lat_sut:0'}


def first_t(traj, y_min):
    for p in traj:
        if p['y'] >= y_min:
            return p['t'], p
    return None, None


def outcomes(paths):
    out = {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            s, c = r['carla'].get(KIND[r['sut']], {}), r['carla'].get('cc', {})
            if 'contact' in s and 'contact' in c:
                out[f"{r['source']}|{r['case_id']}|{r['sut']}"] = (bool(s['contact']), bool(c['contact']))
    return out


def main(work, w):
    rows = []
    for tag in ('dev', 'tgsim', 'mitra', 'highsim', 'exid'):
        e12 = outcomes(work.glob(f'runs/E12_{tag}_replay_*.jsonl'))
        if not e12:
            continue
        base = outcomes(work.glob('runs/E1_replay_*.jsonl') if tag == 'dev' else work.glob(f'runs/E6_{tag}_replay_*.jsonl'))
        src = work / ('runs/E1_trajectories.jsonl' if tag == 'dev' else f'runs/E6_{tag}_trajectories.jsonl')
        for line in open(src, encoding='utf-8'):
            r = json.loads(line)
            key = f"{r['source']}|{r['case_id']}|{r['sut']}"
            if key not in e12 or key not in base:
                continue
            tr = r['trajectory']
            t_hw, _ = first_t(tr, PERCEIVE)
            t_prop, p_prop = first_t(tr, PERCEIVE * 4. / w)
            t_metric, p_metric = first_t(tr, PERCEIVE + 4. - w)
            rows.append(dict(tag=tag, key=key, sut=r['sut'], t_hw=t_hw, t_prop=t_prop, t_metric=t_metric,
                             # distance between the adversary vehicle and the SUT lane centre at perception (CARLA m)
                             d_prop=None if p_prop is None else round(w - p_prop['y'] * w / 4., 3),
                             d_metric=None if p_metric is None else round(4. - p_metric['y'], 3),
                             base=base[key], e12=e12[key]))
    res = dict(w=w, n=len(rows), threshold_y=dict(hw=PERCEIVE, proportional=round(PERCEIVE * 4. / w, 3), metric=PERCEIVE + 4. - w))
    # trajectories C avoided under the proportional mapping: share that become C collisions under the distance-preserving one, per perception-delay bin (distance-preserving minus proportional, s)
    bins = collections.defaultdict(collections.Counter)
    gaps_p, gaps_m = [], []
    for x in rows:
        if x['t_prop'] is None or x['t_metric'] is None:
            continue
        gaps_p.append(x['d_prop'])
        gaps_m.append(x['d_metric'])
        if x['base'][1]:
            continue
        d = x['t_metric'] - x['t_prop']
        b = '<0.2' if d < .2 else '0.2-0.4' if d < .4 else '0.4-0.6' if d < .6 else '>=0.6'
        bins[b]['n'] += 1
        bins[b]['cc_now_hit'] += x['e12'][1]
    res['cc_flip_by_delay'] = {b: dict(v, rate=round(v['cc_now_hit'] / v['n'], 3)) for b, v in sorted(bins.items())}
    delays = sorted(x['t_metric'] - x['t_prop'] for x in rows if x['t_prop'] is not None and x['t_metric'] is not None)
    res['delay_s'] = dict(median=round(delays[len(delays) // 2], 2), p10=round(delays[len(delays) // 10], 2),
                          p90=round(delays[9 * len(delays) // 10], 2)) if delays else None
    med = lambda v: round(sorted(v)[len(v) // 2], 3) if v else None
    res['dist_to_sut_lane_centre_at_perception_m'] = dict(hw=4. - PERCEIVE, proportional=med(gaps_p), metric=med(gaps_m))
    json.dump(res, open(work / 'runs/E12_MECHANISM.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 3.5)
