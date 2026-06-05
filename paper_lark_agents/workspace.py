from __future__ import annotations

import json
import os
from pathlib import Path


class WorkspaceError(ValueError):
    pass


class ChatWorkspaceStore:
    def __init__(
        self,
        state_dir: Path,
        default_workspace: Path,
        allowed_roots: tuple[Path, ...],
    ):
        self.path = state_dir / "chat_workspaces.json"
        self.default_workspace = default_workspace.expanduser().resolve()
        roots = allowed_roots or (self.default_workspace,)
        self.allowed_roots = tuple(root.expanduser().resolve() for root in roots)

    def current(self, chat_id: str) -> Path:
        value = self._read().get(chat_id)
        if not value:
            return self.default_workspace
        return Path(value).expanduser().resolve()

    def has_override(self, chat_id: str) -> bool:
        return chat_id in self._read()

    def set(self, chat_id: str, raw_path: str) -> Path:
        workspace = self.resolve(raw_path)
        data = self._read()
        data[chat_id] = str(workspace)
        self._write(data)
        return workspace

    def reset(self, chat_id: str) -> Path:
        data = self._read()
        data.pop(chat_id, None)
        self._write(data)
        return self.default_workspace

    def resolve(self, raw_path: str) -> Path:
        text = raw_path.strip()
        if not text:
            raise WorkspaceError("请给出目录，例如 `/workspace papers/project-a`。")

        path = Path(os.path.expandvars(text)).expanduser()
        if not path.is_absolute():
            path = self.default_workspace / path
        path = path.resolve()

        if not any(is_relative_to(path, root) for root in self.allowed_roots):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            raise WorkspaceError(f"目录不在允许范围内：{path}。允许范围：{roots}")
        if path.exists() and not path.is_dir():
            raise WorkspaceError(f"不是目录：{path}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allowed_roots_text(self) -> str:
        return "\n".join(f"- `{root}`" for root in self.allowed_roots)

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items()}

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
