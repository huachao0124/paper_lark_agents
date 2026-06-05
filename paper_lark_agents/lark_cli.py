from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from typing import Any, Callable

from .config import Settings, proxy_env
from .router import TaskRequest


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LarkEvent:
    event_type: str
    event_id: str
    chat_id: str
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LarkEvent":
        header = data.get("header") if isinstance(data.get("header"), dict) else {}
        event = data.get("event") if isinstance(data.get("event"), dict) else {}
        return cls(
            event_type=str(
                data.get("type")
                or data.get("event_type")
                or header.get("event_type")
                or ""
            ),
            event_id=str(data.get("event_id") or header.get("event_id") or ""),
            chat_id=str(data.get("chat_id") or event.get("chat_id") or ""),
            raw=data,
        )


@dataclass(frozen=True)
class MessageEvent:
    event_id: str
    chat_id: str
    chat_type: str
    content: str
    sender_id: str
    message_id: str
    message_type: str = ""
    create_time: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MessageEvent":
        header = data.get("header") if isinstance(data.get("header"), dict) else {}
        event = data.get("event") if isinstance(data.get("event"), dict) else {}
        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        if not message and isinstance(event.get("message"), dict):
            message = event["message"]
        content = str(data.get("content") or message.get("content") or "")
        # Feishu replaces @-mentions with placeholders like @_user_1.
        # Resolve them back to the display name so the router can detect
        # @Codex / @Claude correctly.
        mentions = message.get("mentions") or data.get("mentions")
        if isinstance(mentions, list):
            for m in mentions:
                if not isinstance(m, dict):
                    continue
                key = m.get("key") or ""
                name = m.get("name") or ""
                if key and name:
                    content = content.replace(key, f"@{name}")
        return cls(
            event_id=str(data.get("event_id") or event.get("event_id") or header.get("event_id") or ""),
            chat_id=str(data.get("chat_id") or message.get("chat_id") or ""),
            chat_type=str(data.get("chat_type") or message.get("chat_type") or ""),
            content=content,
            sender_id=str(data.get("sender_id") or sender_id_from_message_event(data) or ""),
            message_id=str(data.get("message_id") or data.get("id") or message.get("message_id") or ""),
            message_type=str(
                data.get("message_type")
                or data.get("msg_type")
                or message.get("message_type")
                or message.get("msg_type")
                or ""
            ),
            create_time=str(data.get("create_time") or message.get("create_time") or ""),
        )


class LarkCLIError(RuntimeError):
    pass


def sender_id_from_message_event(data: dict[str, Any]) -> str:
    event = data.get("event") if isinstance(data.get("event"), dict) else {}
    sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
    sender_id = sender.get("sender_id") if isinstance(sender.get("sender_id"), dict) else {}
    return str(sender_id.get("open_id") or sender.get("open_id") or "")


class LarkCLI:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _base_cmd(self) -> list[str]:
        cmd = [self.settings.lark_cli]
        if self.settings.lark_profile:
            cmd.extend(["--profile", self.settings.lark_profile])
        return cmd

    def _env(self) -> dict[str, str]:
        return proxy_env(self.settings.proxy_url, self.settings.no_proxy)

    def send_markdown(self, chat_id: str, markdown: str) -> list[dict[str, Any]]:
        results = []
        for index, chunk in enumerate(_chunks(markdown, self.settings.max_message_chars), start=1):
            key = _idempotency_key(chat_id, chunk, index)
            cmd = [
                *self._base_cmd(),
                "im",
                "+messages-send",
                "--as",
                "bot",
                "--chat-id",
                chat_id,
                "--markdown",
                chunk,
                "--idempotency-key",
                key,
            ]
            results.append(self._run_json(cmd))
        return results

    def send_image(self, chat_id: str, image_path: str) -> dict[str, Any]:
        cmd = [
            *self._base_cmd(),
            "im",
            "+messages-send",
            "--as",
            "bot",
            "--chat-id",
            chat_id,
            "--image",
            image_path,
            "--idempotency-key",
            _idempotency_key(chat_id, "image", image_path),
        ]
        return self._run_json(cmd)

    def send_file(self, chat_id: str, file_path: str) -> dict[str, Any]:
        cmd = [
            *self._base_cmd(),
            "im",
            "+messages-send",
            "--as",
            "bot",
            "--chat-id",
            chat_id,
            "--file",
            file_path,
            "--idempotency-key",
            _idempotency_key(chat_id, "file", file_path),
        ]
        return self._run_json(cmd)

    def send_card(self, chat_id: str, card: dict[str, Any]) -> dict[str, Any]:
        content = json.dumps(card, ensure_ascii=False, separators=(",", ":"))
        cmd = [
            *self._base_cmd(),
            "im",
            "+messages-send",
            "--as",
            "bot",
            "--chat-id",
            chat_id,
            "--msg-type",
            "interactive",
            "--content",
            content,
            "--idempotency-key",
            _idempotency_key(chat_id, "card", content),
        ]
        return self._run_json(cmd)

    def update_card(self, message_id: str, card: dict[str, Any]) -> dict[str, Any]:
        content = json.dumps(card, ensure_ascii=False, separators=(",", ":"))
        data = json.dumps({"content": content}, ensure_ascii=False, separators=(",", ":"))
        cmd = [
            *self._base_cmd(),
            "api",
            "PATCH",
            f"/open-apis/im/v1/messages/{message_id}",
            "--as",
            "bot",
            "--data",
            data,
        ]
        return self._run_json(cmd)

    def pin_message(self, message_id: str) -> dict[str, Any]:
        data = json.dumps({"message_id": message_id}, ensure_ascii=False, separators=(",", ":"))
        cmd = [
            *self._base_cmd(),
            "im",
            "pins",
            "create",
            "--as",
            "bot",
            "--data",
            data,
        ]
        return self._run_json(cmd)

    def create_doc(self, content: str) -> dict[str, Any]:
        cmd = [
            *self._base_cmd(),
            "docs",
            "+create",
            "--api-version",
            "v2",
            "--as",
            "bot",
            "--doc-format",
            "xml",
            "--content",
            content,
        ]
        return self._run_json(cmd)

    def update_doc(self, doc: str, content: str) -> dict[str, Any]:
        cmd = [
            *self._base_cmd(),
            "docs",
            "+update",
            "--api-version",
            "v2",
            "--as",
            "bot",
            "--doc-format",
            "xml",
            "--doc",
            doc,
            "--command",
            "overwrite",
            "--content",
            content,
        ]
        return self._run_json(cmd)

    def list_chat_tabs(self, chat_id: str) -> dict[str, Any]:
        cmd = [
            *self._base_cmd(),
            "api",
            "GET",
            f"/open-apis/im/v1/chats/{chat_id}/chat_tabs/list_tabs",
            "--as",
            "bot",
        ]
        return self._run_json(cmd)

    def create_chat_tab(self, chat_id: str, tab_name: str, doc_url: str) -> dict[str, Any]:
        data = json.dumps(
            {
                "chat_tabs": [
                    {
                        "tab_name": tab_name,
                        "tab_type": "doc",
                        "tab_content": {"doc": doc_url},
                    }
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cmd = [
            *self._base_cmd(),
            "api",
            "POST",
            f"/open-apis/im/v1/chats/{chat_id}/chat_tabs",
            "--as",
            "bot",
            "--data",
            data,
        ]
        return self._run_json(cmd)

    def update_chat_tab(self, chat_id: str, tab_id: str, tab_name: str, doc_url: str) -> dict[str, Any]:
        data = json.dumps(
            {
                "chat_tabs": [
                    {
                        "tab_id": tab_id,
                        "tab_name": tab_name,
                        "tab_type": "doc",
                        "tab_content": {"doc": doc_url},
                    }
                ]
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cmd = [
            *self._base_cmd(),
            "api",
            "POST",
            f"/open-apis/im/v1/chats/{chat_id}/chat_tabs/update_tabs",
            "--as",
            "bot",
            "--data",
            data,
        ]
        return self._run_json(cmd)

    def download_message_resource(
        self,
        message_id: str,
        file_key: str,
        resource_type: str,
        output_relative_path: str,
    ) -> Path:
        cmd = [
            *self._base_cmd(),
            "im",
            "+messages-resources-download",
            "--as",
            "bot",
            "--message-id",
            message_id,
            "--file-key",
            file_key,
            "--type",
            resource_type,
            "--output",
            output_relative_path,
        ]
        result = self._run_json(cmd)
        path = first_download_path(result)
        if path:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.settings.workspace / candidate
            return candidate.expanduser().resolve()
        expected = self.settings.workspace / output_relative_path
        if expected.exists():
            return expected.resolve()
        matches = sorted(expected.parent.glob(f"{expected.name}*"))
        for match in matches:
            if match.is_file():
                return match.resolve()
        return expected.resolve()

    def create_task(self, task: TaskRequest) -> dict[str, Any]:
        cmd = [
            *self._base_cmd(),
            "task",
            "+create",
            "--as",
            self.settings.task_as,
            "--summary",
            task.summary,
        ]
        description = task.description
        if description:
            cmd.extend(["--description", description])
        if task.assignee:
            cmd.extend(["--assignee", task.assignee])
        if task.due:
            cmd.extend(["--due", task.due])
        tasklist_id = task.tasklist_id or self.settings.tasklist_id
        if tasklist_id:
            cmd.extend(["--tasklist-id", tasklist_id])
        cmd.extend(["--idempotency-key", _idempotency_key(task.summary, description, task.due or "")])
        return self._run_json(cmd)

    def create_chat(
        self,
        name: str,
        users: str | None = None,
        bots: str | None = None,
        description: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        cmd = [
            *self._base_cmd(),
            "im",
            "+chat-create",
            "--as",
            "bot",
            "--chat-mode",
            "group",
            "--type",
            "private",
            "--name",
            name,
            "--set-bot-manager",
        ]
        if users:
            cmd.extend(["--users", users])
        if bots:
            cmd.extend(["--bots", bots])
        if description:
            cmd.extend(["--description", description])
        if dry_run:
            cmd.append("--dry-run")
        return self._run_json(cmd)

    def consume_events(self, on_event: Callable[[LarkEvent], None]) -> None:
        event_keys = self.settings.event_keys or (self.settings.event_key,)
        required_event_key = self.settings.event_key
        consumers: list[tuple[str, subprocess.Popen[str]]] = []
        for event_key in event_keys:
            try:
                consumers.append(self._start_event_consumer(event_key))
            except LarkCLIError:
                if event_key == required_event_key:
                    raise
                LOGGER.warning("optional lark event consumer %s is not available", event_key)
        if not consumers:
            raise LarkCLIError("No lark event consumers started.")
        events: queue.Queue[LarkEvent] = queue.Queue()
        for event_key, proc in consumers:
            threading.Thread(
                target=_read_events_stdout,
                args=(proc, event_key, events),
                daemon=True,
            ).start()

        LOGGER.info("listening for %s", ", ".join(event_key for event_key, _ in consumers))
        try:
            while consumers:
                try:
                    event = events.get(timeout=1)
                except queue.Empty:
                    event = None
                if event is not None:
                    on_event(event)
                for event_key, proc in list(consumers):
                    code = proc.poll()
                    if code is None:
                        continue
                    consumers.remove((event_key, proc))
                    raise LarkCLIError(f"lark event consumer {event_key} exited with code {code}.")
        finally:
            for _, proc in consumers:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()

    def _start_event_consumer(self, event_key: str) -> tuple[str, subprocess.Popen[str]]:
        cmd = [
            *self._base_cmd(),
            "event",
            "consume",
            event_key,
            "--as",
            "bot",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=self.settings.workspace,
            env=self._env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready: queue.Queue[str] = queue.Queue()
        stderr_thread = threading.Thread(
            target=_watch_stderr,
            args=(proc, ready),
            daemon=True,
        )
        stderr_thread.start()

        try:
            marker = ready.get(timeout=45)
        except queue.Empty as exc:
            proc.terminate()
            raise LarkCLIError("Timed out waiting for lark event ready marker.") from exc
        if marker != "ready":
            proc.terminate()
            raise LarkCLIError(marker)
        return event_key, proc

    def _run_json(self, cmd: list[str]) -> dict[str, Any]:
        last_detail = ""
        for attempt in range(3):
            proc = subprocess.run(
                cmd,
                cwd=self.settings.workspace,
                env=self._env(),
                text=True,
                capture_output=True,
                check=False,
            )
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            if proc.returncode == 0:
                break
            detail = stderr or stdout or f"exit code {proc.returncode}"
            last_detail = detail[-2000:]
            if attempt < 2 and _retryable_cli_error(detail):
                time.sleep(1 + attempt)
                continue
            raise LarkCLIError(last_detail)
        else:
            raise LarkCLIError(last_detail or "lark-cli failed")
        if not stdout:
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"raw": stdout}


def _watch_stderr(proc: subprocess.Popen[str], ready: queue.Queue[str]) -> None:
    assert proc.stderr is not None
    seen_ready = False
    for line in proc.stderr:
        stripped = line.strip()
        LOGGER.debug("lark event stderr: %s", stripped)
        if "[event] ready" in stripped and not seen_ready:
            seen_ready = True
            ready.put("ready")
        elif not seen_ready and (
            "Error:" in stripped
            or "permission" in stripped.lower()
            or '"type": "validation"' in stripped
            or "requires event types not subscribed" in stripped
        ):
            ready.put(stripped)
            seen_ready = True


def _read_events_stdout(
    proc: subprocess.Popen[str],
    event_key: str,
    events: queue.Queue[LarkEvent],
) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("skipping non-json %s event line: %s", event_key, line[:300])
            continue
        events.put(LarkEvent.from_json(data))


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


def _idempotency_key(*parts: object) -> str:
    joined = "\0".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]
    return f"pla-{int(time.time() // 60)}-{digest}"


def _retryable_cli_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(
        token in lowered
        for token in (
            "api call failed",
            "eof",
            "timeout",
            "connection reset",
            "connection refused",
            "message is being sent",
            "temporary",
            "network",
            "230049",
        )
    )


def first_field(data: Any, field: str) -> Any:
    if isinstance(data, dict):
        if field in data:
            return data[field]
        for value in data.values():
            found = first_field(value, field)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = first_field(value, field)
            if found is not None:
                return found
    return None


def first_download_path(data: Any) -> str | None:
    for field in ("local_path", "path", "file_path", "output", "filename", "file"):
        value = first_field(data, field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = data.get("raw") if isinstance(data, dict) else None
    if isinstance(raw, str):
        match = re.search(r"(?m)(?:saved|downloaded|path|output)[^/\n]*(/[^\n]+)", raw, re.I)
        if match:
            return match.group(1).strip()
    return None
