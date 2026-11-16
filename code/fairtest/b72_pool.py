"""B72 (HIGH-SIM extension, B72 design, available on request): the B70 case pool
without its 60-case compute cap. The stored B70 events (extracted once, sha256 checked) go through the frozen HIGH-SIM
selection (fairtest.highsim.select_cohort: one hash-ranked participant-disjoint pass) with MAX_CASES lifted; the first
60 cases must be the B70 cases, in the same order and identical. Writes OUT_DIR/case_pool.json, the selection ledger and
the protocol files (external_bundle.pool), so the frozen B70 steps run on it unchanged.

  python -m fairtest.b72_pool EVENTS.jsonl OUT_DIR --template DATA_DIR --b70-pool B70_CASE_POOL.json --events-sha SHA256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from fairtest import external_bundle, highsim


def main(a):
    if hashlib.sha256(Path(a.events).read_bytes()).hexdigest() != a.events_sha:
        raise SystemExit('events.jsonl is not the stored B70 extraction')
    highsim.MAX_CASES = 10 ** 6  # the only change: no compute cap
    external_bundle.pool(argparse.Namespace(events=a.events, out=a.out, template=a.template, extractor='fairtest.highsim',
                                            source='HIGH-SIM I-75 (Shi, Zhao, Yao and Li 2021; CATS Lab; CC BY-SA 4.0); '
                                                   'B70 events, selection without the 60-case cap (B72)'))
    new = json.load(open(Path(a.out) / 'case_pool.json', encoding='utf-8'))['cases']
    old = json.load(open(a.b70_pool, encoding='utf-8'))['cases']
    same = len(new) >= len(old) and all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(new, old))
    print(json.dumps(dict(cases=len(new), b70_cases=len(old), first_b70_identical=same, new_cases=len(new) - len(old))))
    if not same:
        raise SystemExit('the first cases differ from the B70 pool')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('events')
    ap.add_argument('out')
    ap.add_argument('--template', required=True)
    ap.add_argument('--b70-pool', required=True)
    ap.add_argument('--events-sha', required=True)
    main(ap.parse_args())
    sys.exit(0)
