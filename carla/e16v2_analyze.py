"""E16v2 analysis (v2 design §3): do the C verdicts of the v2 twins (U, M, MM, T) predict the CARLA C verdicts under harness H2, and the CARLA C collision
rate of the attacks that pass the matched twin. Pairs: proportional mapping Mustang (E1, E6) ~ M, distance-preserving mapping (E12) ~ MM, Tesla (E15a) ~ T.

  python e16v2_analyze.py WORK_DIR      -> runs/E16v2_RESULT.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e11_e12_analyze import KIND, composed  # noqa: E402

TAGS = ('dev', 'tgsim', 'mitra', 'highsim', 'exid')
# CARLA conditions: name -> (replay file pattern per dataset, twin conditions to compare, matched twin)
CARLA = {
    'prop': (lambda t: 'runs/E1_replay_*.jsonl' if t == 'dev' else f'runs/E6_{t}_replay_*.jsonl', ('U', 'M'), ('M',)),
    'metric': (lambda t: f'runs/E12_{t}_replay_*.jsonl', ('U', 'MM'), ('MM',)),
    'tesla': (lambda t: 'runs/E15a_replay_*.jsonl' if t == 'dev' else None, ('U', 'M', 'T'), ('T',)),
}


def carla(work, pat):
    out = {}
    if pat is None:
        return out
    for p in work.glob(pat):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            s, c = r['carla'].get(KIND[r['sut']], {}), r['carla'].get('cc', {})
            if 'contact' in s and 'contact' in c:
                out[f"{r['source']}|{r['case_id']}|{r['sut']}"] = (bool(s['contact']), bool(c['contact']))
    return out


def main(work):
    twin = {}
    for name in ('runs/E16v2_matched.jsonl',):
        if (work / name).exists():
            for line in open(work / name, encoding='utf-8'):
                r = json.loads(line)
                twin.setdefault((r['tag'], r['key']), {}).update({k: v for k, v in r.items() if k not in ('tag', 'key')})
    res = {}
    for cname, (pat, conds, matched) in CARLA.items():
        rows = []                                     # (tag, key, sut_hit, cc_hit, {condition: (contact, clearance)})
        for tag in TAGS:
            c = carla(work, pat(tag))
            for key, (s, cc) in c.items():
                tw = twin.get((tag, key))
                if tw:
                    rows.append((tag, key, s, cc, tw))
        if not rows:
            continue
        res[cname] = {}
        for cond in conds:
            sel = [r for r in rows if cond in r[4]]
            if not sel:
                continue
            agree = sum(r[4][cond]['contact'] == r[3] for r in sel)
            cert = [r for r in sel if not r[4][cond]['contact'] and (r[4][cond]['clearance'] or 0.) >= .5]
            res[cname][cond] = dict(
                n=len(sel), agreement=round(100 * agree / len(sel), 1),
                carla_cc_hit_all=round(100 * sum(r[3] for r in sel) / len(sel), 1),
                certified=len(cert), retained=round(100 * len(cert) / len(sel), 1),
                cc_hit_certified=round(100 * sum(r[3] for r in cert) / len(cert), 1) if cert else None,
                transfer_certified=round(100 * sum(r[2] and not r[3] for r in cert) / len(cert), 1) if cert else None,
                transfer_all=round(100 * sum(r[2] and not r[3] for r in sel) / len(sel), 1))
        # per SUT (over combinations): CARLA transfer rate of the attacks that pass the matched twin, and the share retained
        for cond in matched:
            if cond not in res[cname]:
                continue
            by = {}
            for tag in TAGS:
                if not any(r[0] == tag for r in rows):
                    continue
                look = {r[1]: r for r in rows if r[0] == tag and cond in r[4]}
                for sut, key in composed(work, tag):
                    r = look.get(key)
                    if r is None:
                        continue
                    d = by.setdefault(sut, dict(n=0, cert=0, cc=0, transfer=0))
                    d['n'] += 1
                    if not r[4][cond]['contact'] and (r[4][cond]['clearance'] or 0.) >= .5:
                        d['cert'] += 1
                        d['cc'] += r[3]
                        d['transfer'] += r[2] and not r[3]
            res[cname][cond]['by_sut'] = {s: dict(n=d['n'], retained=round(100 * d['cert'] / d['n'], 1),
                                                 cc_hit_certified=round(100 * d['cc'] / d['cert'], 1) if d['cert'] else None,
                                                 transfer_certified=round(100 * d['transfer'] / d['cert'], 1) if d['cert'] else None)
                                          for s, d in by.items()}
    hyp = {}
    for cname, (_, _, matched) in CARLA.items():
        for cond in matched:
            d = res.get(cname, {}).get(cond)
            if not d:
                continue
            u = res[cname]['U']['agreement']
            hyp[f'{cond}~{cname}'] = dict(agreement=d['agreement'], unmatched=u, H1=bool(d['agreement'] >= 95. and d['agreement'] > u),
                                          cc_hit_certified=d['cc_hit_certified'], H2=bool(d['cc_hit_certified'] is not None and d['cc_hit_certified'] <= 3.))
    res['hypotheses'] = hyp
    json.dump(res, open(work / 'runs/E16v2_RESULT.json', 'w'), indent=1)
    for cname in CARLA:
        if cname not in res:
            continue
        print(f'== CARLA {cname}')
        for cond, d in res[cname].items():
            print(f"   {cond:5} n {d['n']:5d}  agreement {d['agreement']:5.1f}%  CARLA C collision (all) {d['carla_cc_hit_all']:5.1f}%  passed {d['certified']:5d} ({d['retained']:5.1f}%)"
                  f"  CARLA C collision among passed {d['cc_hit_certified']}%  transfer among passed {d['transfer_certified']}%  (transfer all {d['transfer_all']}%)")
    for k, v in hyp.items():
        print(k, v)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
