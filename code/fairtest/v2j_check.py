"""v2j callable (fair_agent_v2 --joint) vs the exact composition of B46 with the unified contract (W5 P1-C).

For policy seed 1: the composition says found at 1 query if the policy stage is a robust fair failure that passes the
per-feature G2 and the joint support on its CC replay, else at 1 + q if bo_warm_g4j's first success is q, else not found
(37 queries). The callable must agree on found, stage and queries for every (case, SUT) it ran.

  python -m fairtest.v2j_check CHECK.jsonl --jobs fairtest_jobs B46_rule.jsonl B46_learned.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairtest.b26_analysis import policy_stage
from fairtest.b45_joint_realism import Support


def main(a):
    jobs = Path(a.jobs)
    sup = Support()
    robust, _ = policy_stage(jobs / 'b26', 'dev')
    ok1 = {}
    for line in open(jobs / 'b45' / 'replay_dev.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['source'] == 'v2_policy' and r['seed'] == 1:
            sc = sup.score(r['features']) if r['cut_in'] else None
            ok1[(r['case_id'], r['sut'])] = (sc is not None and sc <= sup.tau and r.get('exceedance') is not None
                                             and r['exceedance'] <= 1.)
    q = {}
    for path in a.b46:
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            for key, rec in r['runs'].items():
                m, s = key.split('|')
                if m == 'bo_warm_g4j' and 'error' not in rec:
                    q[(r['case_id'], s)] = rec.get('first_fair')
    n = same = 0
    diffs = []
    for line in open(a.check, encoding='utf-8'):
        r = json.loads(line)
        if 'error' in r:
            diffs.append(dict(case_id=r['case_id'], sut=r['sut'], error=r['error']))
            continue
        c, s = r['case_id'], r['sut']
        if robust.get(('s1', c, s), False) and ok1.get((c, s), False):
            want = (True, 'policy', 1)
        elif q.get((c, s)) is not None:
            want = (True, 'refine', 1 + q[(c, s)])
        else:
            want = (False, None, 37)
        got = (bool(r['found']), r['stage'] if r['found'] else None, r['queries'])
        n += 1
        same += got == want
        if got != want:
            diffs.append(dict(case_id=c, sut=s, callable=got, composition=want, status=r.get('status')))
    print(json.dumps(dict(compared=n, identical=same, differences=diffs[:20]), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('check')
    ap.add_argument('--jobs', required=True)
    ap.add_argument('b46', nargs='+')
    main(ap.parse_args())
