from __future__ import annotations

import json
from pathlib import Path


class ModelError(ValueError):
    pass


class ChatModelStore:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "chat_models.json"

    def current(self, chat_id: str, agent: str) -> str | None:
        data = self._read()
        value = data.get(chat_id, {}).get(agent)
        if isinstance(value, str) and value.strip():
            return value
        return None

    def set(self, chat_id: str, agent: str, raw_model: str) -> str:
        model = normalize_model(raw_model)
        data = self._read()
        data.setdefault(chat_id, {})[agent] = model
        self._write(data)
        return model

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
            result[str(chat_id)] = {str(agent): str(model) for agent, model in values.items()}
        return result

    def _write(self, data: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


def normalize_model(raw_model: str) -> str:
    model = raw_model.strip()
    if not model:
        raise ModelError("请提供 model，例如 `@Claude /model opus`。")
    if "\n" in model or "\r" in model:
        raise ModelError("model 不能包含换行。")
    if len(model) > 120:
        raise ModelError("model 名称太长。")
    return model
