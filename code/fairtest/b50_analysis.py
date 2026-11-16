"""B50 analysis (B50 design, fixed before the run, available on request).

  C0  sigma 0 (through the wrapper) equals the B49 seed-1 records of the base SUT: found, stage, queries, cost
  N1a competent bases (AEB H0, longitudinal RSS): v2j success (k37) non-decreasing in sigma (2 pp tolerance)
  N1b at least one competent base: rate(sigma 4) - rate(sigma 0) >= 5 pp
  N2  following H0: rate(sigma 4) >= rate(sigma 0) - 10 pp
  N3  every sigma > 0 SUT with >= 10 reported failures: median reproduction probability (20 fresh noise draws) >= 0.5
  N4  no attacker-at-fault collision in the policy stage of any noisy SUT (the certificate's CC also collides)

  python -m fairtest.b50_analysis OUT.json ATTACKS.jsonl REPRO.jsonl B49_s1.jsonl
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

from fairtest.sut_noise import split

BASES = ('following', 'following_aeb', 'rss_sut')
COMPETENT = ('following_aeb', 'rss_sut')
SIGMAS = (0., 1., 2., 4.)
B, SEED = 10000, 20260928


def _rows(path):
    return [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]


def main(out, attacks, repro, b49):
    rows = _rows(attacks)
    errors = [r for r in rows if 'error' in r]
    by = {}
    for r in rows:
        if 'error' in r:
            continue
        base, sigma = split(r['sut'].rsplit('_H', 1)[0])
        by[(base, sigma, r['case_id'])] = r
    cases = sorted({k[2] for k in by})
    ref = {(r['sut'], r['case_id']): r for r in _rows(b49)}
    # C0
    same, diffs = 0, []
    for base in BASES:
        for c in cases:
            r, b = by.get((base, 0., c)), ref.get((f'{base}_H0', c))
            if r is None or b is None:
                diffs.append(dict(base=base, case_id=c, missing=True))
                continue
            got = (r['found'], r['stage'], r['queries'], r['cost'])
            want = (b['found'], b['stage'], b['queries'], b['cost'])
            same += got == want
            if got != want:
                diffs.append(dict(base=base, case_id=c, noise0=got[:3], b49=want[:3]))
    rep = dict(attacks=len(rows), errors=len(errors), error_rows=errors[:10],
               C0=dict(compared=len(BASES) * len(cases), identical=same, supported=same == len(BASES) * len(cases),
                       differences=diffs[:20]))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(cases), (B, len(cases)))
    rates = {}
    for base in BASES:
        v0 = np.array([float(by[(base, 0., c)]['found']) for c in cases])
        for s in SIGMAS:
            v = np.array([float(by[(base, s, c)]['found']) for c in cases])
            pol = np.array([float(by[(base, s, c)]['found'] and by[(base, s, c)]['stage'] == 'policy') for c in cases])
            d = v - v0
            rates[(base, s)] = float(v.mean())
            rep.setdefault('rates', {}).setdefault(base, {})[f'{s:g}'] = dict(
                k37=round(float(v.mean()), 4), ci=[round(float(x), 4) for x in np.percentile(v[idx].mean(1), [2.5, 97.5])],
                policy_k1=round(float(pol.mean()), 4), minus_sigma0=round(float(d.mean()), 4),
                minus_sigma0_ci=[round(float(x), 4) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])])
    n1a = {b: all(rates[(b, SIGMAS[i + 1])] >= rates[(b, SIGMAS[i])] - .02 for i in range(len(SIGMAS) - 1)) for b in COMPETENT}
    rep['N1a'] = dict(per_base=n1a, supported=all(n1a.values()))
    gain = {b: rates[(b, 4.)] - rates[(b, 0.)] for b in COMPETENT}
    rep['N1b'] = dict(gain=gain, supported=max(gain.values()) >= .05)
    rep['N2'] = dict(sigma0=rates[('following', 0.)], sigma4=rates[('following', 4.)],
                     supported=rates[('following', 4.)] >= rates[('following', 0.)] - .10)
    # N3 reproducibility
    rp = collections.defaultdict(list)
    rerr = 0
    for r in _rows(repro):
        if 'error' in r:
            rerr += 1
            continue
        rp[(r['sut'], r['case_id'])].append(float(r['sut_contact']))
    n3 = {}
    for base in BASES:
        for s in SIGMAS[1:]:
            key = f'{base}~n{s:g}_H0'
            probs = [float(np.mean(v)) for (k, c), v in rp.items() if k == key]
            n3[key] = dict(failures=len(probs), median=float(np.median(probs)) if probs else None,
                           share_ge_half=round(float(np.mean([p >= .5 for p in probs])), 4) if probs else None,
                           share_all=round(float(np.mean([p == 1. for p in probs])), 4) if probs else None,
                           judged=len(probs) >= 10)
    judged = [v for v in n3.values() if v['judged']]
    rep['N3'] = dict(per_sut=n3, replay_errors=rerr, supported=(all(v['median'] >= .5 for v in judged) if judged else None))
    # N4 attacker-at-fault collisions of the policy stage
    fault = []
    for (base, s, c), r in by.items():
        p = r.get('policy') or {}
        cert = p.get('certificate')
        if p.get('valid') and p.get('contact') and cert is not None and 'error' not in cert and cert['cc_contact']:
            fault.append(dict(base=base, sigma=s, case_id=c))
    rep['N4'] = dict(attacker_at_fault=len(fault), rows=fault[:20], supported=not fault)
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    print(json.dumps({k: rep[k] for k in ('attacks', 'errors', 'C0')}, indent=1)[:1200])
    for base in BASES:
        print(base, {s: (v['k37'], v['policy_k1'], v['minus_sigma0'], v['minus_sigma0_ci']) for s, v in rep['rates'][base].items()})
    print(json.dumps({k: rep[k] for k in ('N1a', 'N1b', 'N2')}, indent=1))
    print(json.dumps(rep['N3'], indent=1))
    print(json.dumps(rep['N4'], indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:5])
