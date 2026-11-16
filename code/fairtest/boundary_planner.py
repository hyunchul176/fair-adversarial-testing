"""R157-boundary planner, evaluated offline from the B7 landscape (zero SUT queries).

Idea: the R157 Competent-and-Careful (CC) driver is a public regulatory reference, not the
system under test. An adversary that plans against the CC model may therefore select, per
case and without observing the SUT, the hardest cut-in that the CC driver can still avoid.
Severity c in (0, 1] targets the CC driver's peak braking demand c * 7.59 m/s^2.

Selection uses only the CC run of each candidate plan (and the generator's own realism
features). The selected plan is then scored against each SUT from the same landscape.
"""
from __future__ import annotations

import collections
import json
import math
import random
from pathlib import Path

from fairtest.analyze_landscape import load
from fairtest.features import Realism

CC = ('cc_r157', 0.)
SUTS = (('following', 0.), ('following', .5), ('following', 1.), ('reference_idm', 0.),
        ('following_aeb', 0.), ('following_aeb', .5))
NOMINAL = 'plan_enever_pre0_post0_dgrid'
CC_MAX = 7.59


def index(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r['case_id'], r['policy']['name'])][(r['target'], r['H'])] = r
    return by


def cc_demand(cc_row):
    """Continuous CC braking demand: max DRAC while overlapping, as a fraction of 7.59 m/s^2.

    Falls back to peak applied braking for rows recorded before interaction metrics existed.
    """
    if cc_row.get('max_drac') is not None:
        return min(1.5, cc_row['max_drac'] / CC_MAX)
    return max(0., -cc_row['target_min_acc']) / CC_MAX


def candidates(by, case, realism, max_exceedance):
    out = []
    for (c, p), v in by.items():
        if c != case or not p.startswith('plan_') or p == NOMINAL or CC not in v:
            continue
        cc = v[CC]
        if not cc['valid'] or cc['contact'] or not cc['features'].get('cut_in'):
            continue
        score = realism.score(cc['features'])
        exc = score['max_exceedance'] if score['max_exceedance'] is not None else math.inf
        if exc <= max_exceedance:
            out.append((p, cc_demand(cc), exc))
    return out


def select(by, case, realism, *, severity=1., max_exceedance=math.inf, rule='boundary', rng=None):
    cand = candidates(by, case, realism, max_exceedance)
    if not cand:
        return None
    if rule == 'boundary':      # hardest CC-preventable plan not exceeding the severity target
        feasible = [x for x in cand if x[1] <= severity + 1e-9] or cand
        return max(feasible, key=lambda x: (x[1], -x[2]))[0]
    if rule == 'target':        # closest CC demand to the severity target
        return min(cand, key=lambda x: (abs(x[1] - severity), x[2]))[0]
    if rule == 'random':        # random CC-preventable plan (baseline with the same filter)
        return (rng or random).choice(cand)[0]
    raise ValueError(rule)


def evaluate(rows, *, severities=(.4, .6, .8, 1.), exceedances=(math.inf, 2., 1., .5, 0.), seed=0):
    realism = Realism()
    by = index(rows)
    cases = sorted({c for c, _ in by})
    results = []
    for exc in exceedances:
        for sev in severities:
            for rule in ('boundary', 'target', 'random'):
                if rule == 'random' and sev != severities[-1]:
                    continue
                rng = random.Random(seed)
                tally = {f'{s[0]}_H{s[1]:g}': collections.Counter() for s in SUTS}
                demands = []
                for c in cases:
                    p = select(by, c, realism, severity=sev, max_exceedance=exc, rule=rule, rng=rng)
                    for s in SUTS:
                        name = f'{s[0]}_H{s[1]:g}'
                        nom = by.get((c, NOMINAL), {}).get(s)
                        if nom is None or nom['contact'] or nom['ttc_event'] or nom['initial_event']:
                            tally[name]['excluded'] += 1
                            continue
                        tally[name]['eligible'] += 1
                        if p is None:
                            tally[name]['no_candidate'] += 1
                            continue
                        r, cc = by[(c, p)].get(s), by[(c, p)][CC]
                        if r is None:
                            continue
                        if r['valid'] and r['contact'] and not cc['contact']:
                            tally[name]['fair_failure'] += 1
                        elif r['valid'] and r['ttc_event'] and not cc['contact']:
                            tally[name]['fair_near_miss'] += 1
                    if p is not None:
                        demands.append(cc_demand(by[(c, p)][CC]))
                results.append(dict(max_exceedance=None if math.isinf(exc) else exc, severity=sev, rule=rule,
                                    mean_cc_demand=sum(demands) / len(demands) if demands else None,
                                    selected=len(demands),
                                    suts={k: dict(v) for k, v in tally.items()}))
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('landscape')
    ap.add_argument('--out')
    a = ap.parse_args()
    ok, _ = load(a.landscape)
    res = evaluate(ok)
    text = json.dumps(res, indent=1)
    print(text[:6000])
    if a.out:
        Path(a.out).write_text(text + '\n', encoding='utf-8')
