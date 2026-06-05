from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import fcntl
from html import escape
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator
from urllib.parse import quote


AGENTS = ("codex", "claude")


@dataclass(frozen=True)
class DashboardSnapshot:
    chat_id: str
    message_id: str | None
    pinned: bool
    doc_url: str | None
    doc_token: str | None
    tab_id: str | None
    statuses: dict[str, dict[str, Any]]
    bridge: dict[str, Any] | None


class StatusDashboardStore:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "status_dashboards.json"
        self.lock_path = state_dir / "status_dashboards.lock"

    def ensure_message_id(
        self,
        chat_id: str,
        create_message: Callable[[], str | None],
    ) -> tuple[str | None, bool]:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            message_id = str(chat.get("message_id") or "") or None
            if message_id:
                return message_id, False
            message_id = create_message()
            if not message_id:
                return None, False
            chat["message_id"] = message_id
            chat["pinned"] = False
            chat["created_at"] = time.time()
            chat["updated_at"] = time.time()
            self._write_unlocked(data)
            return message_id, True

    def ensure_status_doc(
        self,
        chat_id: str,
        create_doc: Callable[[], tuple[str | None, str | None]],
    ) -> tuple[str | None, bool]:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            doc_url = str(chat.get("doc_url") or "") or None
            if doc_url:
                return doc_url, False
            doc_url, doc_token = create_doc()
            if not doc_url:
                return None, False
            chat["doc_url"] = doc_url
            if doc_token:
                chat["doc_token"] = doc_token
            chat["updated_at"] = time.time()
            self._write_unlocked(data)
            return doc_url, True

    def ensure_status_tab(
        self,
        chat_id: str,
        create_tab: Callable[[], str | None],
    ) -> tuple[str | None, bool]:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            tab_id = str(chat.get("tab_id") or "") or None
            if tab_id:
                return tab_id, False
            tab_id = create_tab()
            if not tab_id:
                return None, False
            chat["tab_id"] = tab_id
            chat["updated_at"] = time.time()
            self._write_unlocked(data)
            return tab_id, True

    def mark_pinned(self, chat_id: str, message_id: str) -> None:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            if str(chat.get("message_id") or "") == message_id:
                chat["pinned"] = True
                chat["updated_at"] = time.time()
                self._write_unlocked(data)

    def replace_message_id(self, chat_id: str, message_id: str | None) -> None:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            if message_id:
                chat["message_id"] = message_id
                chat["pinned"] = False
            else:
                chat.pop("message_id", None)
                chat["pinned"] = False
            chat["updated_at"] = time.time()
            self._write_unlocked(data)

    def mark_status_doc(self, chat_id: str, doc_url: str, doc_token: str | None = None) -> None:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            chat["doc_url"] = doc_url
            if doc_token:
                chat["doc_token"] = doc_token
            chat["updated_at"] = time.time()
            self._write_unlocked(data)

    def mark_status_tab(self, chat_id: str, tab_id: str) -> None:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            chat["tab_id"] = tab_id
            chat["updated_at"] = time.time()
            self._write_unlocked(data)

    def update_status(
        self,
        chat_id: str,
        agent: str,
        *,
        display_name: str,
        state: str,
        detail: str,
        workspace: str,
        model: str,
        effort: str,
        started_at: float,
    ) -> DashboardSnapshot:
        now = time.time()
        record = {
            "display_name": display_name,
            "state": state,
            "detail": detail,
            "workspace": workspace,
            "model": model,
            "effort": effort,
            "started_at": started_at,
            "updated_at": now,
        }
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            if agent in AGENTS:
                statuses = chat.setdefault("statuses", {})
                if isinstance(statuses, dict):
                    statuses[agent] = record
            else:
                chat["bridge"] = record
            chat["updated_at"] = now
            self._write_unlocked(data)
            return self._snapshot_unlocked(chat_id, chat)

    def snapshot(self, chat_id: str) -> DashboardSnapshot:
        with self._locked():
            data = self._read_unlocked()
            chat = self._chat(data, chat_id)
            return self._snapshot_unlocked(chat_id, chat)

    def _snapshot_unlocked(self, chat_id: str, chat: dict[str, Any]) -> DashboardSnapshot:
        raw_statuses = chat.get("statuses")
        statuses: dict[str, dict[str, Any]] = {}
        if isinstance(raw_statuses, dict):
            for agent, value in raw_statuses.items():
                if agent in AGENTS and isinstance(value, dict):
                    statuses[str(agent)] = dict(value)
        raw_bridge = chat.get("bridge")
        bridge = dict(raw_bridge) if isinstance(raw_bridge, dict) else None
        return DashboardSnapshot(
            chat_id=chat_id,
            message_id=str(chat.get("message_id") or "") or None,
            pinned=bool(chat.get("pinned")),
            doc_url=str(chat.get("doc_url") or "") or None,
            doc_token=str(chat.get("doc_token") or "") or None,
            tab_id=str(chat.get("tab_id") or "") or None,
            statuses=statuses,
            bridge=bridge,
        )

    def _chat(self, data: dict[str, Any], chat_id: str) -> dict[str, Any]:
        chat = data.get(chat_id)
        if not isinstance(chat, dict):
            chat = {}
            data[chat_id] = chat
        return chat

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class StatusDashboardCard:
    def __init__(self, snapshot: DashboardSnapshot, dashboard_url: str | None = None):
        self.snapshot = snapshot
        self.dashboard_url = dashboard_url

    def to_card(self) -> dict[str, object]:
        return {
            "config": {
                "wide_screen_mode": True,
                "update_multi": True,
            },
            "header": {
                "template": template_for_snapshot(self.snapshot),
                "title": {
                    "tag": "plain_text",
                    "content": "AI Status",
                },
            },
            "elements": self._elements(),
        }

    def _elements(self) -> list[dict[str, object]]:
        elements: list[dict[str, object]] = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": ticker_summary(self.snapshot),
                },
            }
        ]
        if self.dashboard_url:
            elements.append(
                {
                    "tag": "action",
                    "actions": [
                        {
                            "tag": "button",
                            "text": {
                                "tag": "plain_text",
                                "content": "Open Dashboard",
                            },
                            "type": "primary",
                            "behaviors": [
                                {
                                    "type": "open_url",
                                    "default_url": applink_url(self.dashboard_url),
                                    "pc_url": applink_url(self.dashboard_url),
                                }
                            ],
                        }
                    ],
                }
            )
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"Updated {datetime.now().strftime('%H:%M:%S')}",
                    }
                ],
            }
        )
        return elements


class StatusDashboardDoc:
    def __init__(self, snapshot: DashboardSnapshot, tab_name: str = "AI 状态"):
        self.snapshot = snapshot
        self.tab_name = tab_name

    def to_xml(self) -> str:
        codex = self.snapshot.statuses.get("codex")
        claude = self.snapshot.statuses.get("claude")
        workspace = workspace_for_ticker(codex, claude, self.snapshot.bridge) or "unknown"
        return "\n".join(
            [
                f"<title>{xml_text(self.tab_name)}</title>",
                f"<h1>{xml_text(self.tab_name)}</h1>",
                '<callout emoji="📌" background-color="light-blue" border-color="blue">',
                f"<p>群：<code>{xml_text(self.snapshot.chat_id)}</code></p>",
                f"<p>更新时间：{xml_text(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>",
                f"<p>Workspace：<code>{xml_text(short_path(workspace, limit=96))}</code></p>",
                "</callout>",
                "<h2>助手状态</h2>",
                status_table_xml(self.snapshot),
                "<h2>说明</h2>",
                "<ul>",
                "<li>这个页面由 bridge 通过飞书 API 自动更新，不依赖服务器开放端口。</li>",
                "<li>群里的固定状态卡片仍然负责展示实时摘要。</li>",
                "</ul>",
            ]
        )


def ticker_summary(snapshot: DashboardSnapshot) -> str:
    codex = snapshot.statuses.get("codex")
    claude = snapshot.statuses.get("claude")
    workspace = workspace_for_ticker(codex, claude, snapshot.bridge)
    lines = [
        f"**Codex:** {agent_ticker(codex)}",
        f"**Claude:** {agent_ticker(claude)}",
    ]
    if snapshot.bridge:
        lines.insert(0, f"**Bridge:** {bridge_ticker(snapshot.bridge)}")
    if workspace:
        lines.append(f"**Workspace:** {card_safe_text(short_path(workspace))}")
    return "\n".join(lines)


def status_table_xml(snapshot: DashboardSnapshot) -> str:
    rows = []
    if snapshot.bridge:
        rows.append(status_row_xml("Bridge", snapshot.bridge))
    rows.append(status_row_xml("Codex", snapshot.statuses.get("codex")))
    rows.append(status_row_xml("Claude", snapshot.statuses.get("claude")))
    return "\n".join(
        [
            "<table>",
            '<colgroup><col span="1" width="100"/><col span="1" width="120"/><col span="1" width="220"/><col span="1" width="220"/><col span="1" width="360"/></colgroup>',
            "<thead><tr>"
            '<th background-color="light-gray">对象</th>'
            '<th background-color="light-gray">状态</th>'
            '<th background-color="light-gray">模型 / Effort</th>'
            '<th background-color="light-gray">Workspace</th>'
            '<th background-color="light-gray">详情</th>'
            "</tr></thead>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
        ]
    )


def status_row_xml(name: str, status: dict[str, Any] | None) -> str:
    if not status:
        return (
            "<tr>"
            f"<td>{xml_text(name)}</td>"
            "<td>Idle</td>"
            "<td>unknown</td>"
            "<td>unknown</td>"
            "<td>No session activity yet.</td>"
            "</tr>"
        )
    state = label_for_state(str(status.get("state") or "idle"))
    model = str(status.get("model") or "unknown")
    effort = str(status.get("effort") or "unknown")
    workspace = str(status.get("workspace") or "unknown")
    detail = short_text(str(status.get("detail") or ""), limit=420)
    elapsed = elapsed_text(status.get("started_at"))
    if elapsed != "n/a":
        detail = f"{detail} (elapsed {elapsed})"
    return (
        "<tr>"
        f"<td>{xml_text(name)}</td>"
        f"<td>{xml_text(state)}</td>"
        f"<td>{xml_text(model)}<br/>{xml_text(effort)}</td>"
        f"<td><code>{xml_text(short_path(workspace, limit=80))}</code></td>"
        f"<td>{xml_text(detail)}</td>"
        "</tr>"
    )


def agent_ticker(status: dict[str, Any] | None) -> str:
    if not status:
        return "Idle"
    state = label_for_state(str(status.get("state") or "idle"))
    elapsed = elapsed_text(status.get("started_at"))
    detail = card_safe_text(short_text(str(status.get("detail") or ""), limit=70))
    if elapsed == "n/a":
        return f"{state} - {detail}"
    return f"{state} · {elapsed} - {detail}"


def bridge_ticker(status: dict[str, Any]) -> str:
    state = label_for_state(str(status.get("state") or "idle"))
    elapsed = elapsed_text(status.get("started_at"))
    detail = card_safe_text(bridge_detail(str(status.get("detail") or "")))
    if elapsed == "n/a":
        return f"{state} - {detail}"
    return f"{state} · {elapsed} - {detail}"


def bridge_detail(detail: str) -> str:
    clean = short_text(detail, limit=160)
    lowered = clean.lower()
    if "init completed" in lowered and "init still running" not in lowered:
        return "Init completed"
    if "init still running" in lowered:
        return "Init still running"
    if "clear completed" in lowered or "cleared this group's bridge state" in lowered:
        return "Clear completed"
    if "workspace set" in lowered:
        return "Workspace set"
    return short_text(detail, limit=90)


def card_safe_text(text: str) -> str:
    return text.replace("`", "'")


def workspace_for_ticker(*statuses: dict[str, Any] | None) -> str | None:
    for status in statuses:
        if status and status.get("workspace"):
            return str(status["workspace"])
    return None


def short_path(value: str, limit: int = 56) -> str:
    if len(value) <= limit:
        return value
    parts = value.strip("/").split("/")
    if len(parts) >= 2:
        suffix = "/".join(parts[-2:])
        return ".../" + suffix
    return "..." + value[-(limit - 3) :]


def applink_url(url: str) -> str:
    encoded = quote(url, safe="")
    return (
        "https://applink.feishu.cn/client/web_url/open"
        f"?mode=sidebar-semi&max_width=1000&reload=false&url={encoded}"
    )


def agent_element(agent: str, status: dict[str, Any] | None) -> dict[str, object]:
    display = "Codex" if agent == "codex" else "Claude"
    return {
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": agent_summary(display, status),
        },
    }


def agent_summary(display: str, status: dict[str, Any] | None) -> str:
    if not status:
        return f"**{display}**\nState: Idle\nDetail: No session activity yet."
    state = label_for_state(str(status.get("state") or "idle"))
    elapsed = elapsed_text(status.get("started_at"))
    lines = [
        f"**{display}**",
        f"State: {state}",
        f"Model: {status.get('model') or 'unknown'}",
        f"Effort: {status.get('effort') or 'unknown'}",
        f"Workspace: `{status.get('workspace') or 'unknown'}`",
        f"Elapsed: {elapsed}",
        f"Detail: {short_text(str(status.get('detail') or ''))}",
    ]
    return "\n".join(lines)


def bridge_summary(status: dict[str, Any]) -> str:
    state = label_for_state(str(status.get("state") or "running"))
    detail = short_text(str(status.get("detail") or ""))
    workspace = status.get("workspace")
    if workspace:
        return f"**Bridge:** {state} - {detail}\nWorkspace: `{workspace}`"
    return f"**Bridge:** {state} - {detail}"


def template_for_snapshot(snapshot: DashboardSnapshot) -> str:
    states = [str(value.get("state") or "") for value in snapshot.statuses.values()]
    if snapshot.bridge:
        states.append(str(snapshot.bridge.get("state") or ""))
    if "failed" in states:
        return "red"
    if any(state in {"pending"} for state in states):
        return "yellow"
    if any(state in {"running"} for state in states):
        return "blue"
    if any(state in {"done"} for state in states):
        return "green"
    return "grey"


def label_for_state(state: str) -> str:
    labels = {
        "running": "Running",
        "done": "Done",
        "pending": "Pending",
        "failed": "Failed",
        "skipped": "No reply needed",
        "idle": "Idle",
    }
    return labels.get(state, state.title())


def elapsed_text(started_at: object) -> str:
    try:
        elapsed = int(time.time() - float(started_at))
    except (TypeError, ValueError):
        return "n/a"
    return f"{max(0, elapsed)}s"


def short_text(text: str, limit: int = 260) -> str:
    clean = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not clean:
        return "No detail."
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def xml_text(value: object) -> str:
    return escape(str(value), quote=False).replace("\n", "<br/>")
