"""Metadata-only frozen-policy transfer cohort. Never constructs a simulator.

Write registration.json before invoking this preparer. Source windows, prior
source qualification and target outcomes are deliberately not accepted inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.observed_lateral_timing import preflight as lateral_preflight
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc

VERSION = "acarl-b3-frozen-transfer-selection-v1"
SALT = "acarl-b3-frozen-transfer-v1:"
RECORDINGS = (58, 59, 60)
QUOTAS = {58: 6, 59: 5, 60: 5}
IMMUTABLE_FIELDS = ("time", "action", "physical_bounds", "observation", "reward", "nominal")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def sanitize_event(event):
    """Allowlist initial physical metadata; never propagate outcome fields."""
    p = event["metrics"]["p10"]
    out = {k: event[k] for k in ("event_id", "partition", "recording", "vehicle", "flip_frame")}
    out["eligible"] = event["eligible"] is True
    out["times"] = [finite_number(x) for x in event["times"]]
    out["metrics"] = {
        "duration_10_90": finite_number(event["metrics"]["duration_10_90"]),
        "p10": {k: finite_number(p[k]) for k in ("generator_speed", "target_speed", "gap", "headway")},
    }
    return out


def initial_states(event):
    """Pure replica of B1 setup geometry, not a forward vehicle simulation."""
    duration = event["metrics"]["duration_10_90"]
    p = event["metrics"]["p10"]
    lo, hi = 0.0, 0.5
    for _ in range(60):
        u = (lo + hi) / 2
        if 10*u**3 - 15*u**4 + 6*u**5 < 0.1:
            lo = u
        else:
            hi = u
    tenth = (lo + hi) / 2
    total = duration / (1 - 2*tenth)
    vy = 4*30*tenth**2*(1-tenth)**2 / total
    vx = p["generator_speed"]
    g = VehicleState(2000.0, 0.4, vx, vy, math.atan2(vy, vx), 5.0, 2.0)
    t = VehicleState(2000.0-p["gap"]-5.0, 4.0, p["target_speed"], 0.0, 0.0, 5.0, 2.0)
    return g, t


def body_on_road(state):
    radius = 0.5*(state.length*abs(math.sin(state.heading)) + state.width*abs(math.cos(state.heading)))
    return state.y-radius >= -2-1e-9 and state.y+radius <= 10+1e-9


def physical_initial_checks(g, t):
    return {
        "generator_scalar_speed": 15.0 <= math.hypot(g.vx, g.vy) <= 40.0,
        "target_speed": 15.0 <= t.vx <= 35.0,
        "body_road": body_on_road(g) and body_on_road(t),
        "reset_no_overlap": constant_velocity_ttc(g, t) != 0.0,
        "reset_nonreverse": g.vx >= -0.1 and t.vx >= -0.1,
    }


def preflight(event):
    p, times = event["metrics"]["p10"], event["times"]
    duration = event["metrics"]["duration_10_90"]
    finite = duration is not None and len(times) == 3 and all(x is not None for x in times) and all(x is not None for x in p.values())
    checks = {"finite_initial_metadata": finite}
    states = None
    if finite:
        ordered = times[0] < times[1] < times[2]
        checks.update(duration=2.0 <= duration <= 8.0, ordered_times=ordered,
                      duration_consistency=abs(times[2]-times[0]-duration) <= 1e-8,
                      positive_gap=p["gap"] > 0.0)
        if duration > 0:
            g, t = initial_states(event)
            checks.update(physical_initial_checks(g, t))
            states = [asdict(g), asdict(t)]
        else:
            checks["positive_duration_for_reset"] = False
    return {"eligible": all(checks.values()), "checks": checks,
            "reasons": [k for k, passed in checks.items() if not passed],
            "physical_initial_state": states}


def select_cohort(events, *, expected_count=81, quotas=None):
    quotas = QUOTAS if quotas is None else quotas
    pool = [sanitize_event(e) for e in events if e.get("partition") == "development" and e.get("eligible") is True]
    if len(pool) != expected_count or len({e["event_id"] for e in pool}) != len(pool):
        raise ValueError("Development eligible count or event identity mismatch")
    if {e["recording"] for e in pool} != set(RECORDINGS):
        raise ValueError("Unexpected development recording")
    pool.sort(key=lambda e: (e["recording"], e["flip_frame"], e["vehicle"]))
    ledger, buckets = [], {r: [] for r in RECORDINGS}
    for case_id, event in enumerate(pool):
        prefix, vehicle, frame = event["event_id"].split(":")
        if (int(prefix), int(vehicle), int(frame)) != (event["recording"], event["vehicle"], event["flip_frame"]):
            raise ValueError("Event identity disagrees with metadata")
        item = {"case_id": case_id, "event_id": event["event_id"], "recording": event["recording"],
                "metadata_sha256": hashlib.sha256(canonical(event)).hexdigest(),
                "selection_hash": hashlib.sha256((SALT+event["event_id"]).encode()).hexdigest(),
                "preflight": preflight(event), "selected": False, "input_id": None,
                "candidate_rank": None, "eligible_rank": None}
        ledger.append(item)
        buckets[event["recording"]].append(item)
    for recording, rows in buckets.items():
        rows.sort(key=lambda x: (x["selection_hash"], x["event_id"]))
        for rank, item in enumerate(rows):
            item["candidate_rank"] = rank
        eligible = [x for x in rows if x["preflight"]["eligible"]]
        for rank, item in enumerate(eligible):
            item["eligible_rank"] = rank
        if len(eligible) < quotas[recording]:
            raise ValueError(f"Insufficient initial-domain candidates for {recording}: {len(eligible)} < {quotas[recording]}; no replacement")
        buckets[recording] = eligible
    chosen = []
    for rank in range(max(quotas.values())):
        for recording in RECORDINGS:
            if rank < quotas[recording]:
                item = buckets[recording][rank]
                item["selected"], item["input_id"] = True, len(chosen)
                chosen.append(item)
    selected_cases = [{"case_id": x["case_id"], "event": pool[x["case_id"]],
                       "preflight": deepcopy(x["preflight"]["checks"])} for x in chosen]
    inputs = []
    for input_id, case in enumerate(selected_cases):
        p = case["event"]["metrics"]["p10"]
        inputs.append({"input_id": input_id, "case_id": case["case_id"], "event_id": case["event"]["event_id"],
                       "scenario": "original", "gap_delta_m": 0.0, "speed_delta_mps": 0.0,
                       "initial_gap_m": p["gap"], "target_initial_and_cruise_mps": p["target_speed"],
                       "eligibility": "initial_domain_pass", "selection_role": "metadata_hash_balanced_development_transfer",
                       "online_maneuver_validity": "not_yet_evaluated"})
    return {"candidate_ledger": ledger, "selected_cases": selected_cases, "pilot_inputs": inputs}


def prepare(events_path, b1_data, output):
    events_path, b1_data, output = map(Path, (events_path, b1_data, output))
    registration_path = output/"registration.json"
    registration = read(registration_path)
    if registration["version"] != VERSION or not registration["registered_before_candidate_rows_opened"]:
        raise ValueError("Missing preregistration")
    if registration["admission"]["lateral_monotone_acceleration_jerk_used_for_admission"]:
        raise ValueError("Prescribed-path diagnostics cannot select this cohort")
    if sorted(p.name for p in output.iterdir()) != ["registration.json"]:
        raise ValueError("Output must contain only preregistration; immutable outputs are never overwritten")
    source_paths = {"events.jsonl": events_path, "b1/base_protocol.json": b1_data/"base_protocol.json",
                    "b1/protocol.json": b1_data/"protocol.json", "b1/selected_cases.json": b1_data/"selected_cases.json",
                    "observed_lateral_timing.py": ROOT/"src/evaluation/observed_lateral_timing.py",
                    "safety_metrics.py": ROOT/"src/evaluation/safety_metrics.py"}
    for key, path in source_paths.items():
        if sha(path) != registration["source_hashes"][key]:
            raise ValueError("Frozen source drift: "+key)
    b1 = read(b1_data/"protocol.json")
    base = read(b1_data/"base_protocol.json")
    if base.get("lateral_timing_mode") != "observed_midpoint":
        raise ValueError("Frozen observed-midpoint B1 path required")
    old_events = {x["event_id"] for x in b1["inputs"]}
    old_recordings = {int(x.split(":")[0]) for x in old_events}
    # Only this line opens candidate rows. No windows/outcome path is accepted.
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    result = select_cohort(events)
    if old_events & {x["event_id"] for x in result["pilot_inputs"]} or old_recordings & set(RECORDINGS):
        raise ValueError("B1 identity/recording overlap")
    # Freeze selected cohort before computing descriptive prescribed-path checks.
    write(output/"candidate_ledger.json", result["candidate_ledger"])
    data = output/"data"
    data.mkdir()
    write(data/"selected_cases.json", result["selected_cases"])
    write(data/"pilot_inputs.json", result["pilot_inputs"])
    shutil.copyfile(b1_data/"base_protocol.json", data/"base_protocol.json")
    write(data/"b1_immutable_contract.json", {k: b1[k] for k in IMMUTABLE_FIELDS})
    write(data/"cohort_override_metadata.json", {
        "version": VERSION, "b1_protocol_sha256": sha(b1_data/"protocol.json"),
        "base_protocol_copied_byte_exact": True,
        "inherited_base_population_metadata_is_historical": ["partition", "recordings", "registered_cases", "events_sha256", "windows_sha256", "selection_sha256", "arms", "targets", "modes"],
        "effective_partition": "development", "effective_recordings": list(RECORDINGS),
        "effective_registered_pool": 81, "effective_selected_inputs": 16,
        "effective_arm": "continue", "effective_target": "following", "effective_H": [0.0, 0.5],
        "longitudinal_windows_loaded": False, "target_outcomes_used": False,
        "registration_sha256": sha(registration_path), "source_hashes": registration["source_hashes"],
        "exposure": registration["exposure"], "no_replacements_after_native_outcomes": True,
    })
    diagnostics = []
    for case in result["selected_cases"]:
        diag = lateral_preflight(case)
        diagnostics.append({"case_id": case["case_id"], "event_id": case["event"]["event_id"],
                            "admission_role": "none; selected denominator remains fixed", "diagnostic": diag})
    write(output/"selected_path_diagnostics.json", diagnostics)
    ledger = result["candidate_ledger"]
    manifest = {
        "version": VERSION, "registration_sha256": sha(registration_path),
        "implementation_sha256": sha(__file__), "source_paths": {k: str(v.resolve()) for k, v in source_paths.items()},
        "source_hashes": registration["source_hashes"], "registered_cases": len(ledger),
        "initial_domain_eligible": sum(x["preflight"]["eligible"] for x in ledger),
        "rejection_reason_counts": dict(Counter(reason for x in ledger for reason in x["preflight"]["reasons"])),
        "selected_count": len(result["pilot_inputs"]),
        "selected_by_recording": dict(Counter(str(x["recording"]) for x in ledger if x["selected"])),
        "selected_case_ids": [x["case_id"] for x in result["pilot_inputs"]],
        "selected_event_ids": [x["event_id"] for x in result["pilot_inputs"]],
        "B1_event_overlap": [], "B1_recording_overlap": [],
        "descriptive_path_pass_count": sum(x["diagnostic"]["eligible"] for x in diagnostics),
        "descriptive_path_filter_used": False, "native_runs": 0, "longitudinal_windows_loaded": False,
        "sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    write(output/"selection_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--b1-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.events, args.b1_data, args.output), indent=2))


if __name__ == "__main__":
    main()
