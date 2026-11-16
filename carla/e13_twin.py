"""E13 v2 post-hoc verdict: re-judge the first collision plan found by the collision search with b23_search.admission in the matched twin M (environment variables FAIRTEST_CC_PERCEIVE, VEH_W, VEH_L)
and write E13_collide_merged.jsonl, which is E13_collide.jsonl with twin.S1 added to each line.

  FAIRTEST_CC_PERCEIVE=1.251 FAIRTEST_VEH_W=2.166 FAIRTEST_VEH_L=4.718 python e13_twin.py [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / 'code'))


def _task(r):
    if 'error' in r or not r.get('found'):
        return r
    import fairtest.b23_search as B23
    from fairtest import runner
    runner._init()
    B23.CC = ('cc_r157', 0.)
    ok, adm = B23.admission(runner._STATE['cases'][r['case_id']], r['found']['theta'])
    r.setdefault('twin', {})['S1'] = dict(admissible=bool(ok), cc_contact=bool(adm['cc']['contact']), clear=round(float(adm['clear']), 3),
                                          feint=adm['feint'], exc=None if adm['exc'] is None else round(float(adm['exc']), 3),
                                          js=None if adm['js'] is None else round(float(adm['js']), 3), pen=round(float(adm['pen']), 3))
    return r


def main(a):
    rows = [json.loads(l) for l in open(W / 'runs/E13_collide.jsonl', encoding='utf-8')]
    with mp.Pool(a.workers) as pool, open(W / 'runs/E13_collide_merged.jsonl', 'w', encoding='utf-8') as f:
        for r in pool.imap(_task, rows, chunksize=4):
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(done=len(rows))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    main(ap.parse_args())
