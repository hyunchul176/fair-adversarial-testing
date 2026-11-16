"""B73 addendum after the review of manuscript v9 (internal review notes, items R1 and R2, available on request):
descriptive, stored B73 rows only (no simulation).

  timing   contact type and closing speed along the contact normal at the 10 Hz contact tick (pre-impact states, as reported)
           and at the interpolated first touch (b73_geometry.first_touch: linear interpolation of position, heading and speed
           between the last free tick and the contact tick, not a re-integration); counts in unique trajectories (rows) and
           seed-weighted events; the 42 competent-SUT contacts with a negative longitudinal closing speed at the tick
  front    detection when only contacts whose SUT region is the front count (the SUT's front edge is the region of its body
           nearest to the overlap), at the tick and at the first touch, within the audited set: per dataset, following H0,
           longitudinal RSS and the competent SUTs, as seed-averaged case rates over the dataset's cases (a seed-weighted event
           is one seed's reported failure of one case; a side-classified event is not replaced by another success, so these
           rates are what the audited failures alone support), and P1 = following H0 minus RSS

  python -m fairtest.b73_sensitivity OUT.json --b73 fairtest_jobs/b73
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CASES = dict(dev=62, mitra=172, highsim=60, exid=120)
COMP = ('following_aeb_H0', 'following_aeb_H0.5', 'rss_sut_H0', 'rss_lat_sut_H0', 'ads@0.5_H0', 'drl_sut_H0', 'drl_sut@3_H0',
        'drl_sut@6_H0')


def rows(d, ds):
    return [json.loads(l) for l in open(Path(d) / f'run_{ds}.jsonl', encoding='utf-8') if l.strip()]


def timing(R):
    out = {}
    for group in ('weakest', 'competent', 'commonroad'):
        sel = [r for r in R if r['group'] == group]
        tick = [r for r in sel if r.get('geometry')]
        ft = [r for r in sel if (r.get('first_touch') or {}).get('geometry')]
        w = lambda r: len(r['seeds'])  # noqa: E731
        def share(rs, key, val):
            tot = sum(w(r) for r in rs)
            return round(sum(w(r) for r in rs if key(r) == val) / tot, 4) if tot else None
        def med(rs, f):
            v = [x for r in rs for x in [f(r)] * w(r)]
            return round(float(np.median(v)), 3) if v else None
        o = dict(rows=len(sel), events=sum(w(r) for r in sel), classified_rows=len(tick), classified_events=sum(w(r) for r in tick),
                 tick=dict(rear_end=share(tick, lambda r: r['geometry']['type'], 'rear_end'),
                           side=share(tick, lambda r: r['geometry']['type'], 'side_contact'),
                           regions={f"{a}/{b}": sum(w(r) for r in tick if (r['geometry']['sut_region'], r['geometry']['adv_region']) == (a, b))
                                    for a, b in sorted({(r['geometry']['sut_region'], r['geometry']['adv_region']) for r in tick})},
                           normal_closing_median=med(tick, lambda r: r['geometry']['normal_closing'])),
                 first_touch=dict(events=sum(w(r) for r in ft),
                                  rear_end=share(ft, lambda r: r['first_touch']['geometry']['type'], 'rear_end'),
                                  side=share(ft, lambda r: r['first_touch']['geometry']['type'], 'side_contact'),
                                  normal_closing_median=med(ft, lambda r: r['first_touch']['geometry']['normal_closing'])))
        neg = [r for r in tick if r['geometry']['closing'] < 0]
        if neg:
            negf = [r for r in neg if (r.get('first_touch') or {}).get('geometry')]
            o['negative_longitudinal'] = dict(
                events=sum(w(r) for r in neg), rows=len(neg),
                tick_normal_median=med(neg, lambda r: r['geometry']['normal_closing']),
                tick_normal_positive=sum(w(r) for r in neg if r['geometry']['normal_closing'] > 0),
                first_touch_normal_median=med(negf, lambda r: r['first_touch']['geometry']['normal_closing']),
                first_touch_normal_positive=sum(w(r) for r in negf if r['first_touch']['geometry']['normal_closing'] > 0),
                persist_5cm=round(sum(w(r) for r in neg if r['shrink']['0.05']['contact']) / sum(w(r) for r in neg), 4))
        out[group] = o
    return out


def front(R, ds):
    """Seed-averaged case rates from the audited reported failures, all and SUT-front only (tick, first touch)."""
    n = CASES[ds]
    ev = defaultdict(lambda: defaultdict(float))  # (sut) -> measure -> seed-weighted events
    for r in R:
        if r['group'] == 'commonroad':
            continue
        w = len(r['seeds'])
        g = r.get('geometry')
        f = (r.get('first_touch') or {}).get('geometry')
        ev[r['sut']]['all'] += w
        ev[r['sut']]['front_tick'] += w if (g and g['sut_region'] == 'front') else 0.
        ev[r['sut']]['front_first_touch'] += w if (f and f['sut_region'] == 'front') else 0.
        ev[r['sut']]['unclassified'] += 0. if g else w
    rate = lambda s, m: round(100. * ev[s][m] / (3 * n), 2)  # noqa: E731
    out = dict(cases=n)
    for m in ('all', 'front_tick', 'front_first_touch'):
        out[m] = dict(following_H0=rate('following_H0', m), rss=rate('rss_sut_H0', m),
                      competent_mean=round(float(np.mean([rate(s, m) for s in COMP])), 2),
                      P1=round(rate('following_H0', m) - rate('rss_sut_H0', m), 2))
    out['events'] = {s: dict(ev[s]) for s in sorted(ev)}
    return out


def main(a):
    rep = dict(timing={}, front={})
    pooled = []
    for ds in CASES:
        R = rows(a.b73, ds)
        pooled += R
        rep['front'][ds] = front(R, ds)
    rep['timing']['pooled_4_datasets'] = timing(pooled)
    rep['timing']['commonroad'] = timing(rows(a.b73, 'commonroad'))['commonroad']
    json.dump(rep, open(a.out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps(rep['timing'], indent=1))
    for ds, v in rep['front'].items():
        print(ds, {m: v[m] for m in ('all', 'front_tick', 'front_first_touch')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--b73', required=True)
    main(ap.parse_args())
