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
import uuid


@dataclass(frozen=True)
class AgentHandoff:
    handoff_id: str
    chat_id: str
    source_agent: str
    target_agent: str
    content: str
    origin_event_id: str
    origin_message_id: str
    sender_id: str
    depth: int
    created_at: float


class AgentHandoffQueue:
    def __init__(self, state_dir: Path, ttl_seconds: int = 86400):
        self.state_dir = state_dir
        self.ttl_seconds = max(60, ttl_seconds)
        self.path = state_dir / "agent_handoffs.jsonl"
        self.lock_path = state_dir / "agent_handoffs.lock"
        self.claim_dir = state_dir / "agent_handoff_claims"

    def enqueue(
        self,
        chat_id: str,
        source_agent: str,
        target_agent: str,
        content: str,
        origin_event_id: str,
        origin_message_id: str,
        sender_id: str,
        depth: int,
    ) -> str:
        handoff_id = uuid.uuid4().hex
        record = {
            "type": "handoff",
            "handoff_id": handoff_id,
            "chat_id": chat_id,
            "source_agent": source_agent,
            "target_agent": target_agent,
            "content": content,
            "origin_event_id": origin_event_id,
            "origin_message_id": origin_message_id,
            "sender_id": sender_id,
            "depth": depth,
            "created_at": time.time(),
            "timestamp": _now(),
        }
        self._append(record)
        return handoff_id

    def pending_for(self, target_agent: str) -> list[AgentHandoff]:
        cutoff = time.time() - self.ttl_seconds
        with self._locked():
            records = self._read_records_unlocked()
        done_ids = {
            str(record.get("handoff_id") or "")
            for record in records
            if record.get("type") in {"done", "failed"}
        }
        pending: list[AgentHandoff] = []
        for record in records:
            if record.get("type") != "handoff":
                continue
            handoff_id = str(record.get("handoff_id") or "")
            if not handoff_id or handoff_id in done_ids:
                continue
            if str(record.get("target_agent") or "") != target_agent:
                continue
            created_at = float(record.get("created_at") or 0)
            if created_at < cutoff:
                continue
            content = str(record.get("content") or "").strip()
            if not content:
                continue
            pending.append(
                AgentHandoff(
                    handoff_id=handoff_id,
                    chat_id=str(record.get("chat_id") or ""),
                    source_agent=str(record.get("source_agent") or ""),
                    target_agent=str(record.get("target_agent") or ""),
                    content=content,
                    origin_event_id=str(record.get("origin_event_id") or ""),
                    origin_message_id=str(record.get("origin_message_id") or ""),
                    sender_id=str(record.get("sender_id") or ""),
                    depth=int(record.get("depth") or 0),
                    created_at=created_at,
                )
            )
        pending.sort(key=lambda item: item.created_at)
        return pending

    def claim(self, handoff_id: str) -> bool:
        if not handoff_id:
            return False
        self.claim_dir.mkdir(parents=True, exist_ok=True)
        claim_path = self.claim_dir / safe_filename(handoff_id)
        try:
            with claim_path.open("x", encoding="utf-8") as handle:
                handle.write(_now())
        except FileExistsError:
            return False
        return True

    def mark_done(self, handoff_id: str, status: str = "done") -> None:
        if not handoff_id:
            return
        kind = "failed" if status == "failed" else "done"
        self._append(
            {
                "type": kind,
                "handoff_id": handoff_id,
                "created_at": time.time(),
                "timestamp": _now(),
            }
        )

    def recent_count(self, chat_id: str, window_seconds: int) -> int:
        cutoff = time.time() - max(1, window_seconds)
        with self._locked():
            records = self._read_records_unlocked()
        return sum(
            1
            for record in records
            if record.get("type") == "handoff"
            and record.get("chat_id") == chat_id
            and float(record.get("created_at") or 0) >= cutoff
        )

    def clear_chat(self, chat_id: str) -> int:
        with self._locked():
            records = self._read_records_unlocked()
            removed_ids = {
                str(record.get("handoff_id") or "")
                for record in records
                if record.get("chat_id") == chat_id
            }
            removed_ids.discard("")
            kept = [
                record
                for record in records
                if record.get("chat_id") != chat_id
                and str(record.get("handoff_id") or "") not in removed_ids
            ]
            removed = len(records) - len(kept)
            if removed:
                self._write_records_unlocked(kept)
        for handoff_id in removed_ids:
            (self.claim_dir / safe_filename(handoff_id)).unlink(missing_ok=True)
        return removed

    def _append(self, record: dict[str, Any]) -> None:
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


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
