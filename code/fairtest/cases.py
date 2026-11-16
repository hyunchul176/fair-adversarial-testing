"""Exposed-partition case pool for B7 development (calibration + development only).

Reads the 2026-09-21 frozen cut-in extraction (events.jsonl), keeps extractor-eligible
events from the already exposed partitions, and applies the unchanged B3 initial-domain
preflight to build archived two-vehicle reset states. Reserved test recordings
(11-14, 25-57) are rejected before any metadata is used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from fairtest import devboot  # noqa: F401  (import bootstrap)
from scripts.prepare_adversarial_b3_transfer import preflight, sanitize_event

EVIDENCE = Path(os.environ.get('FAIRTEST_EVIDENCE', 'evidence'))  # local evidence folder of the study (not distributed)
EVENTS = EVIDENCE / 'highd_cutin_joint' / 'result' / 'events.jsonl'
EVENTS_SHA256 = None  # recorded in the pool manifest at build time
RESERVED = frozenset(range(11, 15)) | frozenset(range(25, 58))
EXPOSED_PARTITIONS = ('calibration', 'development')
DEV16_EVENT_IDS = None  # filled lazily from the B5 selected cases, for tagging only


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_events(path=EVENTS):
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['recording'] in RESERVED:
                raise RuntimeError('Reserved recording present in development event file')
            rows.append(row)
    return rows


def dev16_ids():
    data = EVIDENCE / 'b5_execution' / 'data' / 'selected_cases.json'
    return {c['event']['event_id'] for c in json.loads(data.read_text(encoding='utf-8'))}


def archived_dev16_states():
    """Server-computed B5 reset states (platform libm can differ by 1 ulp from local)."""
    data = EVIDENCE / 'b5_execution' / 'data'
    cases = json.loads((data / 'selected_cases.json').read_text(encoding='utf-8'))
    states = json.loads((data / 'initial_states.json').read_text(encoding='utf-8'))
    inputs = json.loads((data / 'protocol_b1.json').read_text(encoding='utf-8'))['inputs']
    by_case = {c['case_id']: c['event']['event_id'] for c in cases}
    out = {}
    for inp in inputs:
        item = states[str(inp['input_id'])]
        if item['case_id'] != inp['case_id']:
            raise ValueError('Archived B5 initial identity drift')
        out[by_case[inp['case_id']]] = item['physical_initial_state']
    return out


def build_pool(path=EVENTS):
    """Return (cases, ledger). Case ids are stable: rank of event_id within sorted pool."""
    events = [e for e in load_events(path)
              if e.get('partition') in EXPOSED_PARTITIONS and e.get('eligible') is True]
    events.sort(key=lambda e: (e['recording'], e['flip_frame'], e['vehicle']))
    seen = set()
    tagged = dev16_ids()
    archived = archived_dev16_states()
    cases, ledger = [], []
    for event in events:
        if event['event_id'] in seen:
            raise ValueError('Duplicate event id ' + event['event_id'])
        seen.add(event['event_id'])
        clean = sanitize_event(event)
        check = preflight(clean)
        item = dict(event_id=clean['event_id'], partition=clean['partition'], recording=clean['recording'],
                    follower=event.get('follower'), eligible=check['eligible'], reasons=check['reasons'],
                    dev16=clean['event_id'] in tagged)
        ledger.append(item)
        if check['eligible']:
            case_id = len(cases)
            state, source = check['physical_initial_state'], 'local_preflight'
            if clean['event_id'] in archived:
                reference = archived[clean['event_id']]
                if any(abs(a[k] - b[k]) > 1e-9 for a, b in zip(state, reference) for k in a):
                    raise ValueError('Local and archived B5 reset disagree beyond 1e-9')
                state, source = reference, 'archived_b5_server'
            cases.append(dict(case_id=case_id, event=clean, preflight=check['checks'],
                              physical_initial_state=state, initial_state_source=source,
                              follower=event.get('follower'), dev16=item['dev16']))
            item['case_id'] = case_id
    return cases, ledger


def manifest(cases, ledger, path=EVENTS):
    by_partition = {}
    for c in cases:
        by_partition[c['event']['partition']] = by_partition.get(c['event']['partition'], 0) + 1
    return dict(source=str(path), source_sha256=sha256(path), extractor_eligible=len(ledger),
                initial_domain_eligible=len(cases), by_partition=by_partition,
                dev16_included=sum(1 for c in cases if c['dev16']),
                reserved_recordings_read=0)


if __name__ == '__main__':
    pool, led = build_pool()
    print(json.dumps(manifest(pool, led), indent=1))
