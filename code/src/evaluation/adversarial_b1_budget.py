"""Family B1 durable native-work ledger. No simulator imports.

One process is the writer. Every reservation is fsynced to an append-only
journal before this API returns; ``save`` writes a human-readable snapshot.
The journal, not that snapshot, is authoritative on restart. A malformed or
truncated journal fails closed. Unresolved reservations never free budget.

Fixed episodes reserve 120 ticks/24 actor slots at start, accept per-action
``charge`` deltas, and refund unused slots only on an unambiguous completion
or partial close. Training reserves five ticks before *each* action. Failure
retains the outstanding upper bound while preserving measured costs too.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import re


class BudgetError(RuntimeError):
    """A requested native operation has no registered budget or valid state."""


class NativeBudget:
    VERSION = "adversarial-agent-B1-native-budget-v1"
    LIMITS = {
        "sentinel": dict(starts=16, actors=384, ticks=1920),
        "search": dict(starts=1344, actors=32256, ticks=161280),
        "training": dict(starts=64516, actors=64512, ticks=322560),
        "evaluation": dict(starts=320, actors=7680, ticks=38400),
    }
    GLOBAL_LIMITS = dict(starts=66196, actors=104832, ticks=524160)
    RUN_LIMITS = {
        **{f"search:{seed}": dict(starts=672, actors=16128, ticks=80640)
           for seed in (42, 123)},
        **{f"{arm}:{seed}": dict(starts=16129, actors=16128, ticks=80640)
           for arm in ("reactive", "initial_context") for seed in (42, 123)},
    }
    FIELDS = ("starts", "completed", "partial", "failed", "actors", "ticks",
              "measured_actors", "measured_ticks", "uncertain_failed_calls")

    def __init__(self, path=None, source_sha256=None, protocol_sha256=None):
        self.path = Path(path) if path is not None else None
        self.journal_path = self.path.with_name(self.path.name + ".events.jsonl") if self.path else None
        self.source_sha256 = self._hash(source_sha256)
        self.protocol_sha256 = self._hash(protocol_sha256)
        self.rows = {phase: self._row() for phase in self.LIMITS}
        self.runs = {key: self._row() for key in self.RUN_LIMITS}
        self.attempts = {}
        self.sequence = 0
        self._journal_bytes = 0
        self._poisoned = False
        if self.journal_path and self.journal_path.exists():
            self._load()
        elif self.path and self.path.exists():
            raise BudgetError("Snapshot without authoritative journal; refusing fresh budget")
        else:
            self._append(dict(op="header", version=self.VERSION,
                              source_sha256=self.source_sha256,
                              protocol_sha256=self.protocol_sha256))

    @staticmethod
    def _hash(value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("A lowercase SHA256 source and protocol identity is required")
        return value

    @classmethod
    def _row(cls):
        return dict.fromkeys(cls.FIELDS, 0)

    @staticmethod
    def _integer(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {name}")
        return value

    def _append(self, event):
        if self._poisoned:
            raise BudgetError("Ledger persistence previously failed; restart and recover")
        event = dict(event, sequence=self.sequence)
        payload = (json.dumps(event, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode("utf-8")
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # A stale second writer is detected; concurrent writers are unsupported.
                with self.journal_path.open("ab") as stream:
                    if stream.tell() != self._journal_bytes:
                        raise BudgetError("Journal changed externally; refusing concurrent writer")
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                self._poisoned = True
                raise
        self._journal_bytes += len(payload)
        self.sequence += 1

    def _load(self):
        with self.journal_path.open("rb") as stream:
            for index, raw in enumerate(stream):
                if not raw.endswith(b"\n"):
                    raise BudgetError("Truncated budget journal; manual evidence recovery required")
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeError) as exc:
                    raise BudgetError("Corrupt budget journal") from exc
                if event.pop("sequence", None) != index:
                    raise BudgetError("Non-contiguous budget journal sequence")
                if index == 0:
                    expected = dict(op="header", version=self.VERSION,
                                    source_sha256=self.source_sha256,
                                    protocol_sha256=self.protocol_sha256)
                    if event != expected:
                        raise BudgetError("Budget source/protocol identity changed")
                else:
                    self._apply(event, persist=False)
                self.sequence += 1
                self._journal_bytes += len(raw)
        if self.sequence == 0:
            raise BudgetError("Empty budget journal")

    def _totals(self):
        return {field: sum(row[field] for row in self.rows.values()) for field in self.FIELDS}

    def _increment(self, attempt, **deltas):
        targets = [self.rows[attempt["phase"]]]
        if attempt["run_key"] is not None:
            targets.append(self.runs[attempt["run_key"]])
        for row in targets:
            for name, delta in deltas.items():
                row[name] += delta

    def _check_capacity(self, phase, run_key, **deltas):
        pairs = [(self.rows[phase], self.LIMITS[phase]),
                 (self._totals(), self.GLOBAL_LIMITS)]
        if run_key is not None:
            pairs.append((self.runs[run_key], self.RUN_LIMITS[run_key]))
        if any(row[name] + delta > caps[name]
               for row, caps in pairs for name, delta in deltas.items()):
            raise BudgetError("Registered phase/run/global native budget exhausted")

    def _active(self, attempt_id):
        if attempt_id not in self.attempts:
            raise BudgetError("Unknown attempt")
        attempt = self.attempts[attempt_id]
        if attempt["status"] != "active":
            raise BudgetError("Attempt already closed")
        return attempt

    def _apply(self, event, persist=True):
        op = event["op"]
        if op == "start":
            phase, run_key, key = event["phase"], event["run_key"], event["attempt_id"]
            if phase not in self.LIMITS or not isinstance(key, str) or not key:
                raise ValueError("Unknown phase or empty attempt ID")
            if key in self.attempts:
                raise BudgetError("Attempt ID is already permanently recorded")
            allowed_runs = ({f"search:{s}" for s in (42, 123)} if phase == "search" else
                            {f"{arm}:{s}" for arm in ("reactive", "initial_context")
                             for s in (42, 123)} if phase == "training" else {None})
            if run_key not in allowed_runs:
                raise ValueError("Run key does not belong to registered phase")
            fixed = event["fixed_episode"]
            if not isinstance(fixed, bool) or fixed != (phase != "training"):
                raise ValueError("Nontraining episodes must reserve the fixed 120-tick envelope")
            deltas = dict(starts=1, ticks=120 if fixed else 0, actors=24 if fixed else 0)
            self._check_capacity(phase, run_key, **deltas)
            # Reject non-finite/non-JSON condition metadata before persisting a start.
            json.dumps(event["condition"], allow_nan=False)
            if persist:
                self._append(event)
            attempt = dict(attempt_id=key, phase=phase, run_key=run_key,
                           condition=deepcopy(event["condition"]), source_sha256=self.source_sha256,
                           protocol_sha256=self.protocol_sha256, fixed_episode=fixed,
                           status="active", pending=False, ticks=deltas["ticks"],
                           actors=deltas["actors"], measured_ticks=0, measured_actors=0,
                           uncertain_failed_calls=0)
            self.attempts[key] = attempt
            self._increment(attempt, **deltas)
            return
        attempt = self._active(event["attempt_id"])
        if op == "reserve_action":
            if attempt["fixed_episode"] or attempt["pending"] or attempt["uncertain_failed_calls"]:
                raise BudgetError("Action already reserved or fixed episode already covers it")
            if attempt["measured_actors"] >= 24:
                raise BudgetError("Episode decision horizon exhausted")
            self._check_capacity(attempt["phase"], attempt["run_key"], ticks=5, actors=1)
            if persist:
                self._append(event)
            attempt["pending"] = True
            attempt["ticks"] += 5
            attempt["actors"] += 1
            self._increment(attempt, ticks=5, actors=1)
        elif op == "charge":
            if attempt["uncertain_failed_calls"]:
                raise BudgetError("Failed execution must close before any further native work")
            ticks = self._integer(event["measured_ticks"], "measured ticks")
            actors = self._integer(event["actor_calls"], "actor calls", 1)
            failed = event["failed"]
            if not isinstance(failed, bool) or ticks > actors * 5 or (not failed and ticks < actors):
                raise ValueError("Impossible measured action/tick count")
            fixed = attempt["fixed_episode"]
            if not fixed and (not attempt["pending"] or actors != 1):
                raise BudgetError("Charge requires one previously reserved training action")
            if (attempt["measured_ticks"] + ticks > attempt["ticks"] or
                    attempt["measured_actors"] + actors > attempt["actors"]):
                raise BudgetError("Measured execution exceeds reserved episode/action capacity")
            if persist:
                self._append(event)
            refund = 0 if fixed or failed else 5 - ticks
            attempt["ticks"] -= refund
            attempt["measured_ticks"] += ticks
            attempt["measured_actors"] += actors
            attempt["uncertain_failed_calls"] += int(failed)
            attempt["pending"] = False
            self._increment(attempt, ticks=-refund, measured_ticks=ticks,
                            measured_actors=actors, uncertain_failed_calls=int(failed))
        elif op in ("complete", "partial", "fail"):
            if op != "fail" and (attempt["pending"] or attempt["uncertain_failed_calls"]):
                raise BudgetError("Uncertain execution cannot be closed as successful or partial")
            if op == "complete" and not attempt["measured_ticks"]:
                raise BudgetError("Zero-tick reset cannot count as a completed episode")
            if op != "complete" and (not isinstance(event.get("reason"), str) or not event["reason"]):
                raise ValueError("Failure/partial reason is required")
            if persist:
                self._append(event)
            dt = da = 0
            uncertainty = int(op == "fail" and (attempt["pending"] or attempt["fixed_episode"]) and
                              not attempt["uncertain_failed_calls"])
            if op != "fail" and attempt["fixed_episode"]:
                dt = attempt["measured_ticks"] - attempt["ticks"]
                da = attempt["measured_actors"] - attempt["actors"]
                attempt["ticks"] += dt
                attempt["actors"] += da
            attempt["pending"] = False
            attempt["uncertain_failed_calls"] += uncertainty
            attempt["status"] = {"complete": "completed", "partial": "partial", "fail": "failed"}[op]
            if "reason" in event:
                attempt["reason"] = event["reason"]
            self._increment(attempt, ticks=dt, actors=da,
                            uncertain_failed_calls=uncertainty, **{attempt["status"]: 1})
        else:
            raise BudgetError("Unknown ledger operation")

    def start(self, phase, attempt_id, condition, run_key=None, fixed_episode=False):
        self._apply(dict(op="start", phase=phase, attempt_id=attempt_id,
                         condition=condition, run_key=run_key, fixed_episode=fixed_episode))
        return attempt_id

    def reserve_action(self, attempt_id):
        self._apply(dict(op="reserve_action", attempt_id=attempt_id))

    def charge(self, attempt_id, measured_ticks, actor_calls=1, failed=False):
        self._apply(dict(op="charge", attempt_id=attempt_id, measured_ticks=measured_ticks,
                         actor_calls=actor_calls, failed=failed))

    def complete(self, attempt_id):
        self._apply(dict(op="complete", attempt_id=attempt_id))

    def partial(self, attempt_id, reason):
        self._apply(dict(op="partial", attempt_id=attempt_id, reason=reason))

    def fail(self, attempt_id, reason):
        self._apply(dict(op="fail", attempt_id=attempt_id, reason=reason))

    def recover_pending(self, reason="Process restarted; live simulator state unavailable"):
        """Permanently close all active attempts; retain uncertain reservations.

        A training attempt with no pending action still closes as failed: its
        trajectory is incomplete after a crash. Measured costs are retained.
        This does not restart simulation or authorize replacement study cells.
        """
        recovered = []
        for key, attempt in list(self.attempts.items()):
            if attempt["status"] == "active":
                self.fail(key, reason)
                recovered.append(key)
        self.save()
        return recovered

    def summary(self):
        counts = {status: sum(a["status"] == status for a in self.attempts.values())
                  for status in ("active", "completed", "partial", "failed")}
        return deepcopy(dict(version=self.VERSION, source_sha256=self.source_sha256,
                             protocol_sha256=self.protocol_sha256, sequence=self.sequence,
                             journal_bytes=self._journal_bytes, global_totals=self._totals(),
                             phases=self.rows, runs=self.runs, attempt_counts=counts))

    def save(self):
        if self.path:
            temporary = self.path.with_name(self.path.name + ".tmp")
            payload = json.dumps(dict(**self.summary(), attempts=self.attempts),
                                 indent=2, allow_nan=False) + "\n"
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
