"""B56 S3 follow-up input: the first bo_warm_g4s success on each competent SUT (b56_analysis.COMPETENT), with its plan
theta, from a b23_search --log-queries output. Used for the development run (competent_successes.json) and, post hoc,
for MiTra (competent_successes_mitra.json, 2026-10-01); the list is then judged by b56_competent_check.py.

  python -m fairtest.b56_competent_extract b56_mitra.jsonl competent_successes_mitra.json
"""
from __future__ import annotations

import json
import sys

from fairtest.b56_analysis import COMPETENT


def main(src, out):
    rows = []
    for line in open(src, encoding='utf-8'):
        r = json.loads(line)
        for key, rec in r['runs'].items():
            method, sut = key.split('|')
            if method == 'bo_warm_g4s' and sut in COMPETENT and rec.get('first_fair') is not None:
                rows.append(dict(case_id=r['case_id'], sut=sut, n=rec['first_fair'], cc_replays=rec['cc_replays'],
                                 warm=rec.get('warm'), theta=rec['best']['theta']))
    json.dump(rows, open(out, 'w', encoding='utf-8'), indent=1)
    print(len(rows))


if __name__ == '__main__':
    main(*sys.argv[1:3])
