from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


class AssistantOutbox:
    def __init__(self, state_dir: Path, ttl_seconds: int = 86400):
        self.state_dir = state_dir
        self.ttl_seconds = max(60, ttl_seconds)
        self.path = state_dir / "assistant_outbox.jsonl"

    def remember(
        self,
        chat_id: str,
        content: str,
        max_chars: int,
        agent: str,
        discussion_trigger: bool = True,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        chunks = [chunk for chunk in _chunks(content, max_chars) if _normalize(chunk)]
        records = [
            {
                "chat_id": chat_id,
                "hash": _hash(chunk),
                "agent": agent,
                "chunk_index": index,
                "chunk_count": len(chunks),
                "discussion_trigger": discussion_trigger and index == len(chunks),
                "created_at": now,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        if not records:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def remember_message_id(
        self,
        chat_id: str,
        message_id: str,
        agent: str,
        discussion_trigger: bool = False,
    ) -> None:
        if not message_id:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "chat_id": chat_id,
            "message_id": message_id,
            "agent": agent,
            "discussion_trigger": discussion_trigger,
            "created_at": time.time(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def match(self, chat_id: str, content: str) -> dict[str, Any] | None:
        normalized = _normalize(content)
        if not normalized or not self.path.exists():
            return None
        target = _hash(normalized)
        cutoff = time.time() - self.ttl_seconds
        for record in self._read_recent(cutoff):
            if record.get("chat_id") == chat_id and record.get("hash") == target:
                return record
        return None

    def match_message_id(self, chat_id: str, message_id: str) -> dict[str, Any] | None:
        if not message_id or not self.path.exists():
            return None
        cutoff = time.time() - self.ttl_seconds
        for record in self._read_recent(cutoff):
            if record.get("chat_id") == chat_id and record.get("message_id") == message_id:
                return record
        return None

    def contains(self, chat_id: str, content: str) -> bool:
        return self.match(chat_id, content) is not None

    def recent_discussion_count(self, chat_id: str, window_seconds: int) -> int:
        if not self.path.exists():
            return 0
        cutoff = time.time() - max(1, window_seconds)
        return sum(
            1
            for record in self._read_recent(cutoff)
            if record.get("chat_id") == chat_id and record.get("discussion_trigger")
        )

    def clear_chat(self, chat_id: str) -> int:
        if not self.path.exists():
            return 0
        records = self._read_all()
        kept = [record for record in records if record.get("chat_id") != chat_id]
        removed = len(records) - len(kept)
        if removed:
            self._write_all(kept)
        return removed

    def _read_recent(self, cutoff: float) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for record in self._read_all():
            created_at = float(record.get("created_at") or 0)
            if created_at >= cutoff:
                records.append(record)
        return records

    def _read_all(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not self.path.exists():
            return records
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def _write_all(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".jsonl.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(self.path)


def _normalize(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()


def _hash(content: str) -> str:
    normalized = _normalize(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _chunks(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in text.split("\n\n"):
        paragraph_len = len(paragraph) + 2
        if current and current_len + paragraph_len > size:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0
        if paragraph_len > size:
            for offset in range(0, len(paragraph), size):
                chunks.append(paragraph[offset : offset + size])
            continue
        current.append(paragraph)
        current_len += paragraph_len
    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]
