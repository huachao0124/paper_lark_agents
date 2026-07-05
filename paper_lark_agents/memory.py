from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from .lark_cli import MessageEvent


@dataclass(frozen=True)
class MemoryEntry:
    role: str
    content: str
    sender_id: str | None = None
    agent: str | None = None
    message_id: str | None = None
    event_id: str | None = None
    timestamp: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "sender_id": self.sender_id,
            "agent": self.agent,
            "message_id": self.message_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp or _now(),
        }


class ChatMemory:
    def __init__(self, state_dir: Path, max_turns: int = 24, max_chars: int = 0):
        self.state_dir = state_dir
        self.max_turns = max(2, max_turns)
        self.max_chars = max_chars

    def append_user(self, event: MessageEvent, routed_text: str) -> None:
        content = routed_text or event.content
        entries = self._read(event.chat_id)
        if entries:
            last = entries[-1]
            if last.get("role") == "user" and last.get("message_id") == event.message_id:
                return
        self._append(
            event.chat_id,
            MemoryEntry(
                role="user",
                sender_id=event.sender_id,
                message_id=event.message_id,
                event_id=event.event_id,
                content=content,
            ),
        )

    def append_assistant(self, chat_id: str, agent: str, content: str) -> None:
        # Dedup: two processes (codex + claude) share this file, and multiple
        # code paths (dispatch, handoff, followup, pending-run recovery) can
        # write the same reply. Look back over recent entries — skipping
        # content-less markers like agent_seen, which land between the two
        # writes and would defeat a last-entry-only check.
        entries = self._read(chat_id)
        for prior in reversed(entries[-6:]):
            if not str(prior.get("content") or "").strip():
                continue
            if (
                prior.get("role") == "assistant"
                and prior.get("agent") == agent
                and prior.get("content") == content
            ):
                return
            break
        self._append(
            chat_id,
            MemoryEntry(
                role="assistant",
                agent=agent,
                content=content,
            ),
        )

    def mark_agent_seen(
        self,
        chat_id: str,
        agent: str,
        *,
        message_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        """Record that an agent has received the room context without speaking.

        [NO_REPLY] should not become visible room memory, but it still means the
        agent saw the recap. This marker advances unseen_context() for that
        agent while remaining invisible in context().
        """
        entries = self._read(chat_id)
        if entries:
            last = entries[-1]
            if (
                last.get("role") == "agent_seen"
                and last.get("agent") == agent
                and last.get("message_id") == message_id
                and last.get("event_id") == event_id
            ):
                return
        self._append(
            chat_id,
            MemoryEntry(
                role="agent_seen",
                agent=agent,
                content="",
                message_id=message_id,
                event_id=event_id,
            ),
        )

    def context(self, chat_id: str, exclude_agent: str | None = None) -> str:
        entries = self._read(chat_id)[-self.max_turns :]
        if not entries:
            return "No previous discussion in this Feishu group yet."

        lines: list[str] = []
        for entry in entries:
            role = str(entry.get("role") or "unknown")
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                if exclude_agent and entry.get("agent") == exclude_agent:
                    continue
                name = str(entry.get("agent") or "assistant")
            elif role == "user":
                name = "用户"
            else:
                continue
            lines.append(f"{name}: {content}")

        return "\n\n".join(lines)

    def unseen_context(self, chat_id: str, agent: str) -> str:
        """Return only messages that happened AFTER this agent's last turn.

        Unlike context(), this avoids re-sending content the agent already
        received in its CLI session (via handoff prompts or respond_to_all).
        """
        entries = self._read(chat_id)[-self.max_turns :]
        if not entries:
            return ""
        last_own_idx = -1
        for idx in range(len(entries) - 1, -1, -1):
            if self._is_seen_boundary(entries[idx], agent):
                last_own_idx = idx
                break
        after = entries[last_own_idx + 1:] if last_own_idx >= 0 else entries
        # Drop the last user entry — it's the current message (just appended
        # by handle_event before dispatch), which will be delivered separately
        # as the turn prompt. Skip trailing content-less markers (agent_seen
        # from the other bridge process) that may land between the append and
        # this read.
        idx = len(after) - 1
        while idx >= 0 and not str(after[idx].get("content") or "").strip():
            idx -= 1
        if idx >= 0 and after[idx].get("role") == "user":
            after = after[:idx]
        lines: list[str] = []
        for entry in after:
            content = str(entry.get("content") or "").strip()
            if not content:
                continue
            role = str(entry.get("role") or "unknown")
            if role == "assistant":
                if entry.get("agent") == agent:
                    continue
                if entry.get("agent") == "bridge":
                    continue
                name = str(entry.get("agent") or "assistant")
            elif role == "user":
                name = "用户"
            else:
                continue
            lines.append(f"{name}: {content}")
        if not lines:
            return ""
        return "\n\n".join(lines)

    def has_unseen_peer_turns(self, chat_id: str, agent: str) -> bool:
        """Whether another agent replied in this chat since this agent's last turn.

        Used to decide if the agent needs a recap: even when it is the broadcast
        responder, it never sees peer replies in its own CLI session — they go
        straight to Feishu. So if Claude replied after Codex's last turn, Codex's
        session has a blind spot that the recap must fill.
        """
        entries = self._read(chat_id)
        for entry in reversed(entries):
            role = entry.get("role")
            if self._is_seen_boundary(entry, agent):
                return False  # reached this agent's last turn — no unseen peer turns
            if role == "assistant" and entry.get("agent") and entry.get("agent") != agent:
                return True   # found a peer turn before this agent's last turn
        return False

    def _is_seen_boundary(self, entry: dict[str, Any], agent: str) -> bool:
        if entry.get("agent") != agent:
            return False
        return entry.get("role") in {"assistant", "agent_seen"}

    def clear(self, chat_id: str) -> bool:
        path = self._chat_dir(chat_id) / "history.jsonl"
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed

    def _append(self, chat_id: str, entry: MemoryEntry) -> None:
        chat_dir = self._chat_dir(chat_id)
        chat_dir.mkdir(parents=True, exist_ok=True)
        path = chat_dir / "history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")

    def _read(self, chat_id: str) -> list[dict[str, Any]]:
        path = self._chat_dir(chat_id) / "history.jsonl"
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                entries.append(data)
        return entries

    def _chat_dir(self, chat_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", chat_id or "unknown")
        return self.state_dir / "chats" / safe_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
