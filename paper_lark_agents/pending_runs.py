from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import time
from typing import Any, Iterator


@dataclass(frozen=True)
class PendingRun:
    run_id: str
    chat_id: str
    agent: str
    route_text: str
    event_id: str
    message_id: str
    sender_id: str
    message_type: str
    chat_type: str
    event_content: str
    source_agent: str | None
    handoff_depth: int
    start_marker: str
    end_marker: str
    session_name: str
    workspace: str
    status_message_id: str | None
    card_id: str | None
    model_label: str
    effort_label: str
    timeout: int
    created_at: float


class PendingRunStore:
    def __init__(self, state_dir: Path, ttl_seconds: int = 86400):
        self.state_dir = state_dir
        self.ttl_seconds = max(60, ttl_seconds)
        self.path = state_dir / "pending_runs.jsonl"
        self.lock_path = state_dir / "pending_runs.lock"
        self.claim_dir = state_dir / "pending_run_claims"

    def start(self, **kwargs: object) -> None:
        run_id = str(kwargs.get("run_id") or "")
        if not run_id:
            raise ValueError("pending run requires run_id")
        record = {
            "type": "run",
            "created_at": time.time(),
            "timestamp": _now(),
            **kwargs,
        }
        self._append(record)

    def mark_done(self, run_id: str, status: str = "done") -> None:
        if not run_id:
            return
        kind = "failed" if status == "failed" else "done"
        self._append(
            {
                "type": kind,
                "run_id": run_id,
                "created_at": time.time(),
                "timestamp": _now(),
            }
        )
        (self.claim_dir / safe_filename(run_id)).unlink(missing_ok=True)

    def pending_for(self, agent: str) -> list[PendingRun]:
        cutoff = time.time() - self.ttl_seconds
        with self._locked():
            records = self._read_records_unlocked()
        done_ids = {
            str(record.get("run_id") or "")
            for record in records
            if record.get("type") in {"done", "failed"}
        }
        pending: list[PendingRun] = []
        for record in records:
            if record.get("type") != "run":
                continue
            run_id = str(record.get("run_id") or "")
            if not run_id or run_id in done_ids:
                continue
            if str(record.get("agent") or "") != agent:
                continue
            created_at = float(record.get("created_at") or 0)
            if created_at < cutoff:
                continue
            pending.append(record_to_pending_run(record))
        pending.sort(key=lambda item: item.created_at)
        return pending

    def has_message_run(self, chat_id: str, message_id: str, agent: str) -> bool:
        if not chat_id or not message_id or not agent:
            return False
        cutoff = time.time() - self.ttl_seconds
        with self._locked():
            records = self._read_records_unlocked()
        for record in records:
            if record.get("type") != "run":
                continue
            if float(record.get("created_at") or 0) < cutoff:
                continue
            if (
                record.get("chat_id") == chat_id
                and record.get("message_id") == message_id
                and record.get("agent") == agent
            ):
                return True
        return False

    def claim(self, run_id: str) -> bool:
        if not run_id:
            return False
        self.claim_dir.mkdir(parents=True, exist_ok=True)
        claim_path = self.claim_dir / safe_filename(run_id)
        try:
            with claim_path.open("x", encoding="utf-8") as handle:
                handle.write(_now())
        except FileExistsError:
            return False
        return True

    def clear_chat(self, chat_id: str) -> int:
        with self._locked():
            records = self._read_records_unlocked()
            removed_ids = {
                str(record.get("run_id") or "")
                for record in records
                if record.get("chat_id") == chat_id
            }
            removed_ids.discard("")
            kept = [
                record
                for record in records
                if record.get("chat_id") != chat_id
                and str(record.get("run_id") or "") not in removed_ids
            ]
            removed = len(records) - len(kept)
            if removed:
                self._write_records_unlocked(kept)
        for run_id in removed_ids:
            (self.claim_dir / safe_filename(run_id)).unlink(missing_ok=True)
        return removed

    def compact(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        with self._locked():
            records = self._read_records_unlocked()
            done_ids = {
                str(r.get("run_id") or "")
                for r in records if r.get("type") in {"done", "failed"}
            }
            kept: list[dict[str, Any]] = []
            removed = 0
            for r in records:
                created = float(r.get("created_at") or 0)
                if created < cutoff:
                    removed += 1
                    continue
                run_id = str(r.get("run_id") or "")
                if r.get("type") == "run" and run_id in done_ids:
                    removed += 1
                    continue
                kept.append(r)
            if removed:
                self._write_records_unlocked(kept)
        return removed

    def _append(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self._locked():
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _read_records_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _write_records_unlocked(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".jsonl.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def record_to_pending_run(record: dict[str, Any]) -> PendingRun:
    source_agent = record.get("source_agent")
    return PendingRun(
        run_id=str(record.get("run_id") or ""),
        chat_id=str(record.get("chat_id") or ""),
        agent=str(record.get("agent") or ""),
        route_text=str(record.get("route_text") or ""),
        event_id=str(record.get("event_id") or ""),
        message_id=str(record.get("message_id") or ""),
        sender_id=str(record.get("sender_id") or ""),
        message_type=str(record.get("message_type") or "text"),
        chat_type=str(record.get("chat_type") or "group"),
        event_content=str(record.get("event_content") or ""),
        source_agent=str(source_agent) if source_agent else None,
        handoff_depth=int(record.get("handoff_depth") or 0),
        start_marker=str(record.get("start_marker") or ""),
        end_marker=str(record.get("end_marker") or ""),
        session_name=str(record.get("session_name") or ""),
        workspace=str(record.get("workspace") or ""),
        status_message_id=str(record.get("status_message_id") or "") or None,
        card_id=str(record.get("card_id") or "") or None,
        model_label=str(record.get("model_label") or ""),
        effort_label=str(record.get("effort_label") or ""),
        timeout=int(record.get("timeout") or 0),
        created_at=float(record.get("created_at") or 0),
    )


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
