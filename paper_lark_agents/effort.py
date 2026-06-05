from __future__ import annotations

import json
from pathlib import Path
import re


EFFORT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}")


class EffortError(ValueError):
    pass


class ChatEffortStore:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "chat_efforts.json"

    def current(self, chat_id: str, agent: str) -> str | None:
        data = self._read()
        value = data.get(chat_id, {}).get(agent)
        try:
            return normalize_effort(value or "")
        except EffortError:
            pass
        return None

    def set(self, chat_id: str, agent: str, raw_level: str) -> str:
        level = normalize_effort(raw_level)
        data = self._read()
        data.setdefault(chat_id, {})[agent] = level
        self._write(data)
        return level

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, dict[str, str]] = {}
        for chat_id, values in data.items():
            if not isinstance(values, dict):
                continue
            result[str(chat_id)] = {str(agent): str(level) for agent, level in values.items()}
        return result

    def _write(self, data: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


def normalize_effort(raw_level: str) -> str:
    level = raw_level.strip().lower()
    if not level:
        raise EffortError("请提供 effort，例如 `@Codex /effort xhigh`。")
    if not EFFORT_TOKEN_RE.fullmatch(level):
        raise EffortError(
            f"effort 只能包含字母、数字、下划线或连字符，且必须以字母开头：`{raw_level}`"
        )
    return level
