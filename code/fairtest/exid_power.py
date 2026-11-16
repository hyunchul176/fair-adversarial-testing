"""exiD pre-freeze power simulation (exiD protocol draft section 5A, available on request): the probability that the point estimates pass the
P1 threshold (following H0 - longitudinal RSS >= 50 pp, final agent v2s, 37 queries) and the severe-only P1 (contacts with
closing speed >= 3 m/s only, key secondary K2), for a cohort of n cases resampled with replacement from the per-case
values of a reference dataset. Also the original-definition P2 (S14 = key secondary K1): the probability that the
longitudinal RSS success rate is <= 5%.

Per-case values follow b64_analysis (v2s and v2s sev compositions). Reference datasets: MiTra (post hoc, dense cut-ins)
and the highD development cases (in-sample). 10,000 resamples per n, seed 20261002.

  python -m fairtest.exid_power OUT.json NAME DIR B56_REFINE.jsonl POLICY_CLOSING.jsonl G4S_CLOSING.jsonl [NAME DIR ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.b41_analysis import K_FULL, load
from fairtest.b51_severity import V_MIN
from fairtest.exid_analysis import load_refine

N_GRID = (20, 40, 60, 80, 100, 150)
N_SIM, SEED = 10000, 20261002


def closings(path, kind):
    out = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r.get('kind') == kind and 'error' not in r:
            out[(r['seed'], r['case_id'], r['sut']) if kind == 'policy' else (r['case_id'], r['sut'])] = r['closing']
    return out


def per_case(d, refine, pcl_path, gcl_path):
    evals, realistic, _, pool, _ = load(Path(d))
    cases = sorted(c['case_id'] for c in pool['cases'])
    seeds = sorted(evals['final'])
    first, _, _, _ = load_refine([refine])
    pcl, gcl = closings(pcl_path, 'policy'), closings(gcl_path, 'bo_warm_g4s')

    def val(s, c, sev):
        """The failure v2s reports for the seed (policy if it succeeded, else the refinement within 37 queries); with
        sev, a reported failure closing below V_MIN counts as none (exid_extra S7: v2s stops at a light policy failure)."""
        out = []
        for x in seeds:
            m = evals['final'][x].get((c, s), {})
            if m.get('robust') and realistic.get((x, c, s), False):
                v = pcl.get((x, c, s))
                out.append(float(not sev or (v is not None and v >= V_MIN)))
                continue
            q = first['bo_warm_g4s'].get((c, s))
            ref = q is not None and 1 + q <= K_FULL
            if ref and sev:
                v = gcl.get((c, s))
                ref = v is not None and v >= V_MIN
            out.append(float(ref))
        return float(np.mean(out))
    h0 = np.array([val('following_H0', c, False) for c in cases])
    rss = np.array([val('rss_sut_H0', c, False) for c in cases])
    h0s = np.array([val('following_H0', c, True) for c in cases])
    rsss = np.array([val('rss_sut_H0', c, True) for c in cases])
    return h0, rss, h0s, rsss


def main(out, args):
    rep = dict(n_grid=N_GRID, n_sim=N_SIM, seed=SEED, v_min=V_MIN, datasets={})
    for i in range(0, len(args), 5):
        name, d, refine, pcl, gcl = args[i:i + 5]
        h0, rss, h0s, rsss = per_case(d, refine, pcl, gcl)
        rng = np.random.default_rng(SEED)
        res = dict(n_cases=len(h0), P1=float(h0.mean() - rss.mean()), P1_severe=float(h0s.mean() - rsss.mean()),
                   rss_rate=float(rss.mean()), power={})
        for n in N_GRID:
            idx = rng.integers(0, len(h0), (N_SIM, n))
            p1 = h0[idx].mean(1) - rss[idx].mean(1)
            p1s = h0s[idx].mean(1) - rsss[idx].mean(1)
            r = rss[idx].mean(1)
            res['power'][n] = dict(P1_pass=round(float((p1 >= .5).mean()), 4), P1_severe_pass=round(float((p1s >= .5).mean()), 4),
                                   both_pass=round(float(((p1 >= .5) & (p1s >= .5)).mean()), 4),
                                   S14_rss_pass=round(float((r <= .05).mean()), 4))
        rep['datasets'][name] = res
    json.dump(rep, open(out, 'w', encoding='utf-8'), indent=1)
    for name, res in rep['datasets'].items():
        print(f"== {name}: n={res['n_cases']} P1 {res['P1'] * 100:.1f} P1_severe {res['P1_severe'] * 100:.1f} RSS {res['rss_rate'] * 100:.1f}")
        for n, p in res['power'].items():
            print(f"   n={n:3d} P1 {p['P1_pass']:.3f} severe {p['P1_severe_pass']:.3f} both {p['both_pass']:.3f} S14(RSS<=5%) {p['S14_rss_pass']:.3f}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
