from __future__ import annotations

import json
from pathlib import Path
import tempfile


RESPONDER_CHOICES = ("codex", "claude", "codebuddy", "both")


class ResponderError(ValueError):
    pass


def normalize_responder(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"claude code", "claudecode", "claude-code"}:
        return "claude"
    if value in {"code buddy", "code-buddy", "code_buddy"}:
        return "codebuddy"
    if value in {"all", "everyone"}:
        return "both"
    if value in RESPONDER_CHOICES:
        return value
    raise ResponderError(
        "默认 responder 只能是 codex、claude、codebuddy 或 both，例如 `/responder claude`。"
    )


class ChatResponderStore:
    """Per-chat choice of which assistant answers unaddressed group messages.

    Stored in a shared state file so the Codex and Claude bridge processes agree
    on who owns a plain (non-@-mentioned) message. The global default is used
    until a group sets an override via ``/responder``.
    """

    def __init__(self, state_dir: Path, default_responder: str = "both"):
        self.path = state_dir / "chat_responders.json"
        try:
            self.default_responder = normalize_responder(default_responder)
        except ResponderError:
            self.default_responder = "both"

    def current(self, chat_id: str) -> str:
        return self._read().get(chat_id) or self.default_responder

    def has_override(self, chat_id: str) -> bool:
        return chat_id in self._read()

    def allows(self, chat_id: str, agent: str) -> bool:
        current = self.current(chat_id)
        return current == "both" or current == agent

    def set(self, chat_id: str, raw: str) -> str:
        value = normalize_responder(raw)
        data = self._read()
        data[chat_id] = value
        self._write(data)
        return value

    def reset(self, chat_id: str) -> str:
        data = self._read()
        data.pop(chat_id, None)
        self._write(data)
        return self.default_responder

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, str] = {}
        for chat_id, value in data.items():
            try:
                result[str(chat_id)] = normalize_responder(str(value))
            except ResponderError:
                continue
        return result

    def _write(self, data: dict[str, str]) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(json.dumps(data, ensure_ascii=False, indent=2))
            tmp_path.replace(self.path)
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
