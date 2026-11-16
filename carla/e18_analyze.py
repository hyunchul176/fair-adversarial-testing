"""E18 analysis (addenda v2-4 and v2-6): width-randomised policies (wr) s1–s3 versus the final policies s1–s3.
  CARLA: Tesla (E18 vs E17-A), Mustang (E18M vs E2). Verdicts as in e2_analyze.
  highway-env (runs/E18_hwenv/): 62 development cases × width 2.0 and 2.66, no shield and window shield.

  python e18_analyze.py WORK_DIR      -> runs/E18_RESULT.json
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_analyze import COMPETENT, ORDER, load  # noqa: E402
from e17_analyze import boot  # noqa: E402


def pooled(work, names):
    runs = [load(work / f'runs/{n}.jsonl') for n in names if (work / f'runs/{n}.jsonl').exists()]
    out = {}
    for kind in ORDER:
        c = collections.Counter(v for d in runs for (case, k), v in d.items() if k == kind)
        n = sum(c.values()) - c['error']
        per_case = collections.defaultdict(list)
        for d in runs:
            for (case, k), v in d.items():
                if k == kind and v != 'error':
                    per_case[case].append(v == 'robust')
        out[kind] = dict(n=n, robust=c['robust'], fair=c['fair'], attacker=c['attacker'], none=c['none'],
                         robust_rate=round(100 * c['robust'] / n, 1) if n else None, ci=boot(per_case))
    return dict(seeds=len(runs), runs=sum(v['n'] for v in out.values()), attacker=sum(v['attacker'] for v in out.values()), by_kind=out)


def hwenv(work):
    res = {}
    for p in sorted((work / 'runs/E18_hwenv').glob('*.jsonl')):
        name = p.stem                                   # final_s1_w2.0[_shield]
        pol, seed, width = name.split('_')[:3]
        cond = f"{pol}|{width}|{'shield' if name.endswith('shield') else 'none'}"
        d = res.setdefault(cond, collections.defaultdict(collections.Counter))
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            c = d[r['sut']]
            c['n'] += 1
            c['contact'] += bool(r['contact'])
            c['fair'] += bool(r.get('guarded_fair'))
            cc = (r.get('cc_replay') or {}).get('contact')
            c['attacker'] += bool(r['contact'] and cc)
    return {k: {s: dict(v, fair_rate=round(100 * v['fair'] / v['n'], 1)) for s, v in d.items()} for k, d in res.items()}


def main(work):
    res = dict(tesla=dict(final=pooled(work, [f'E17A_rl_s{n}' for n in (1, 2, 3)]), wr=pooled(work, [f'E18_wr_rl_s{n}' for n in (1, 2, 3)])),
               mustang=dict(final=pooled(work, [f'E2_rl_s{n}' for n in (1, 2, 3)]), wr=pooled(work, [f'E18M_wr_rl_s{n}' for n in (1, 2, 3)])))
    h0 = lambda veh, pol: res[veh][pol]['by_kind']['ladder:following:0']['robust_rate']
    share = lambda veh, pol: 100 * res[veh][pol]['attacker'] / max(res[veh][pol]['runs'], 1)
    res['E18_H1'] = dict(final=h0('tesla', 'final'), wr=h0('tesla', 'wr'), attacker_share=round(share('tesla', 'wr'), 2),
                         supported=bool(h0('tesla', 'wr') is not None and h0('tesla', 'wr') >= h0('tesla', 'final') + 8. and share('tesla', 'wr') <= 1.))
    res['E18_H2'] = dict(final=h0('mustang', 'final'), wr=h0('mustang', 'wr'), attacker_share=round(share('mustang', 'wr'), 2),
                         supported=bool(h0('mustang', 'wr') is not None and h0('mustang', 'wr') >= h0('mustang', 'final') - 5. and share('mustang', 'wr') <= 1.))
    comp = {f'{veh}|{k}': res[veh]['wr']['by_kind'][k]['robust_rate'] for veh in ('tesla', 'mustang') for k in COMPETENT}
    res['E18_H3'] = dict(rates=comp, supported=all(v is not None and v <= 3. for v in comp.values()))
    if (work / 'runs/E18_hwenv').exists():
        res['hwenv'] = hwenv(work)
    json.dump(res, open(work / 'runs/E18_RESULT.json', 'w'), indent=1)
    for veh in ('tesla', 'mustang'):
        for pol in ('final', 'wr'):
            r = res[veh][pol]
            bk = r['by_kind']
            print(f"{veh:7} {pol:5} seeds {r['seeds']} runs {r['runs']:4d} at fault {r['attacker']} | H0 robust {bk['ladder:following:0']['robust_rate']}% {bk['ladder:following:0']['ci']}"
                  f" | BA {[bk[k]['robust_rate'] for k in ('ba:cautious', 'ba:normal', 'ba:aggressive')]} | competent {[bk[k]['robust_rate'] for k in COMPETENT]}")
    for h in ('E18_H1', 'E18_H2', 'E18_H3'):
        print(h, res[h])
    for cond, d in sorted(res.get('hwenv', {}).items()):
        print(f"hwenv {cond:22}", {s: v['fair_rate'] for s, v in d.items() if s in ('following_H0', 'following_H0.5', 'reference_idm_H0', 'following_aeb_H0', 'rss_sut_H0', 'bc_human_H0')},
              'attacker', sum(v['attacker'] for v in d.values()))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
