"""E12b analysis (addendum 7, post hoc): does the transfer rate recover if C in the distance-preserving replay is replaced by a C with perception threshold 0.595 m (ccp:0.595)?

  python e12b_analyze.py WORK_DIR      -> runs/E12b_RESULT.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e11_e12_analyze import KIND, composed  # noqa: E402

TAGS = ('dev', 'tgsim', 'mitra', 'highsim', 'exid')


def load(paths, c_kind):
    out = {}
    for p in paths:
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            s, c = r['carla'].get(KIND[r['sut']], {}), r['carla'].get(c_kind, {})
            out[f"{r['source']}|{r['case_id']}|{r['sut']}"] = dict(sut=None if 'contact' not in s else bool(s['contact']),
                                                                  cc=None if 'contact' not in c else bool(c['contact']))
    return out


def main(work):
    res, flip = {}, dict(n=0, recovered=0)
    for tag in TAGS:
        base = load(work.glob('runs/E1_replay_*.jsonl') if tag == 'dev' else work.glob(f'runs/E6_{tag}_replay_*.jsonl'), 'cc')
        e12 = load(work.glob(f'runs/E12_{tag}_replay_*.jsonl'), 'cc')
        e12b = load(work.glob(f'runs/E12b_{tag}_replay_*.jsonl'), 'ccp:0.595')
        if not e12b:
            continue
        for k, v in e12.items():                      # trajectories where C avoided under the proportional mapping but collided under the distance-preserving one
            b, x = base.get(k), e12b.get(k)
            if b and b['cc'] is False and v['cc'] is True and x and x['cc'] is not None:
                flip['n'] += 1
                flip['recovered'] += x['cc'] is False
        pairs = composed(work, tag)
        res[tag] = {}
        for sut in KIND:
            keys = [key for s, key in pairs if s == sut]
            if not keys:
                continue

            def rate(sut_src, cc_src):
                sel = [(sut_src[k]['sut'], cc_src[k]['cc']) for k in keys if k in sut_src and k in cc_src
                       and sut_src[k]['sut'] is not None and cc_src[k]['cc'] is not None]
                return (round(100 * sum(a and not c for a, c in sel) / len(sel), 1), len(sel)) if sel else (None, 0)
            p, n = rate(base, base)
            m, nm = rate(e12, e12)
            mb, nb = rate(e12, e12b)                  # the SUT as in the distance-preserving replay, only C's threshold moved
            res[tag][sut] = dict(n=n, proportional=p, metric=m, metric_shifted_cc=mb, n_metric=nm, n_shifted=nb)
    res['E12b_H1'] = dict(flip, share=round(100 * flip['recovered'] / flip['n'], 1) if flip['n'] else None,
                          supported=bool(flip['n'] and flip['recovered'] / flip['n'] >= .8))
    json.dump(res, open(work / 'runs/E12b_RESULT.json', 'w'), indent=1)
    for tag in TAGS:
        if tag not in res:
            continue
        print(f'== {tag}')
        for s, v in res[tag].items():
            print(f"   {s:18} n {v['n']:4d}  proportional {v['proportional']}  distance {v['metric']}  distance + shifted-threshold C {v['metric_shifted_cc']} (n {v['n_shifted']})")
    print('E12b-H1', res['E12b_H1'])


if __name__ == '__main__':
    main(Path(sys.argv[1]))
