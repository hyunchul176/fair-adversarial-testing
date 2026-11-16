"""Build the data directory of an external dataset (B28 TGSIM; the B13 NGSIM steps as one tool).

  pool   EVENTS.jsonl OUT_DIR --template DATA_DIR --source TEXT --extractor MODULE
         select the cohort with the extractor's own select_cohort (hash-ranked, participant-disjoint), write
         OUT_DIR/case_pool.json {'manifest', 'cases'} and copy the protocol files from the template data dir
  warm   GRID_CC.jsonl OUT_DIR
         grid R157-boundary plan per case from a CC-only landscape (runner landscape --targets cc_r157:0),
         written as OUT_DIR/warm_starts.json for the continuous CC-boundary search (b11_search)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import shutil
from pathlib import Path

PROTOCOL_FILES = ('base_protocol.json', 'protocol_b1.json', 'protocol_b2_action.json', 'natural_reference.json',
                  'b5_reference_states.json.gz', 'bc_human.pkl')


def pool(a):
    ext = importlib.import_module(a.extractor)
    events = [json.loads(line) for line in open(a.events, encoding='utf-8') if line.strip()]
    cases, ledger = ext.select_cohort(events)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in PROTOCOL_FILES:
        shutil.copy2(Path(a.template) / name, out / name)
    exclusions = collections.Counter(e.get('exclusion') for e in events if not e.get('eligible'))
    reasons = collections.Counter(r['reason'] for r in ledger if not r['selected'])
    manifest = dict(source=a.source, candidates=len(events), extractor_eligible=sum(1 for e in events if e.get('eligible')),
                    extractor_exclusions=dict(exclusions), selection_reasons=dict(reasons), selected=len(cases),
                    selected_per_recording=dict(collections.Counter(c['event']['recording'] for c in cases)),
                    salt=getattr(ext, 'SALT', None),
                    events_sha256=hashlib.sha256(Path(a.events).read_bytes()).hexdigest())
    (out / 'case_pool.json').write_text(json.dumps(dict(manifest=manifest, cases=cases)), encoding='utf-8')
    (out / 'selection_ledger.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ledger), encoding='utf-8')
    print(json.dumps(manifest))


def warm(a):
    from fairtest.boundary_planner import index, select
    from fairtest.features import Realism
    rows = [json.loads(line) for line in open(a.grid, encoding='utf-8') if line.strip()]
    rows = [r for r in rows if 'error' not in r]
    by, realism, out = index(rows), Realism(), {}
    for c in sorted({r['case_id'] for r in rows}):
        p = select(by, c, realism, severity=1., rule='boundary')
        if p:
            parts = p.split('_')
            e, dur = parts[1][1:], parts[4][1:]
            out[c] = [99. if e == 'never' else float(e), float(parts[2][3:]), float(parts[3][4:]), None if dur == 'grid' else float(dur)]
    path = Path(a.out) / 'warm_starts.json'
    path.write_text(json.dumps({str(k): v for k, v in out.items()}, indent=0), encoding='utf-8')
    print(json.dumps(dict(cases=len({r['case_id'] for r in rows}), warm_starts=len(out))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pool')
    p.add_argument('events')
    p.add_argument('out')
    p.add_argument('--template', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--extractor', default='fairtest.tgsim')
    w = sub.add_parser('warm')
    w.add_argument('grid')
    w.add_argument('out')
    a = ap.parse_args()
    pool(a) if a.cmd == 'pool' else warm(a)
