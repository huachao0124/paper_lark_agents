from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

from .agent_runner import AgentError, AgentRunner, AgentStillRunning
from .tmux_runtime import TmuxReplyStillRunning
from .artifacts import ArtifactRelay
from .config import Settings
from .effort import ChatEffortStore, EffortError, normalize_effort
from .inbound_files import (
    RESOURCE_MESSAGE_TYPES,
    extract_inbound_resources,
    inbound_output_relative_path,
    post_text_without_resources,
)
from .handoff import AgentHandoff, AgentHandoffQueue
from .lark_cli import LarkCLI, LarkCLIError, LarkEvent, MessageEvent, create_lark_client, first_field
from .memory import ChatMemory
from .models import ChatModelStore, ModelError, normalize_model
from .outbox import AssistantOutbox
from .pending_runs import PendingRun, PendingRunStore
from .responders import ChatResponderStore, ResponderError
from .prompts import (
    agent_prompt,
    agent_session_context_prompt,
    agent_session_turn_prompt,
    debate_prompt,
    debate_session_turn_prompt,
    format_debate,
)
from .router import HELP_TEXT, Route, addressed_agents, route_message
from .status_card import turn_reply_card
from .status_dashboard import StatusDashboardCard, StatusDashboardDoc, StatusDashboardStore
from .workspace import ChatWorkspaceStore, WorkspaceError


LOGGER = logging.getLogger(__name__)
SUPPORTED_MESSAGE_TYPES = {"text", "post", *RESOURCE_MESSAGE_TYPES}
MESSAGE_EVENT_TYPE = "im.message.receive_v1"
BOT_ADDED_EVENT_TYPE = "im.chat.member.bot.added_v1"
BOT_DELETED_EVENT_TYPE = "im.chat.member.bot.deleted_v1"
CHAT_DISBANDED_EVENT_TYPE = "im.chat.disbanded_v1"
CHAT_CLOSED_EVENT_TYPES = {BOT_DELETED_EVENT_TYPE, CHAT_DISBANDED_EVENT_TYPE}
FEISHU_LOCAL_TZ = timezone(timedelta(hours=8))
DEBATE_BROADCAST_PREFIX = "Feishu command: /debate"


def parse_message_create_time(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        number = int(text)
        if number > 10_000_000_000:
            return number / 1000.0
        return float(number)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=FEISHU_LOCAL_TZ)
        return parsed.timestamp()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            parsed = datetime.strptime(text, fmt).replace(tzinfo=FEISHU_LOCAL_TZ)
        except ValueError:
            continue
        return parsed.timestamp()
    return None


def doc_url_from_result(data: Any) -> str | None:
    for field in ("url", "doc_url", "document_url", "share_url"):
        value = first_field(data, field)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return first_url_in_result(data)


def first_url_in_result(data: Any) -> str | None:
    if isinstance(data, dict):
        for value in data.values():
            found = first_url_in_result(value)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = first_url_in_result(value)
            if found:
                return found
    elif isinstance(data, str) and data.startswith(("http://", "https://")):
        return data
    return None


def doc_token_from_result(data: Any) -> str | None:
    for field in ("doc_token", "document_id", "document_token", "token"):
        value = first_field(data, field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def chat_tabs_from_result(data: Any) -> list[dict[str, Any]]:
    tabs = first_field(data, "chat_tabs")
    if not isinstance(tabs, list):
        return []
    return [tab for tab in tabs if isinstance(tab, dict)]


def tab_id_from_result(
    data: Any,
    *,
    tab_name: str | None = None,
    doc_url: str | None = None,
) -> str | None:
    for tab in chat_tabs_from_result(data):
        if tab_name and str(tab.get("tab_name") or "") != tab_name:
            continue
        if doc_url:
            content = tab.get("tab_content")
            if not isinstance(content, dict) or str(content.get("doc") or "") != doc_url:
                continue
        value = tab.get("tab_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = first_field(data, "tab_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class StatusHandle:
    def __init__(
        self,
        bridge: "PaperAgentBridge",
        chat_id: str,
        message_id: str | None,
        agent: str,
        agent_name: str,
        workspace: Path,
        model: str,
        effort: str,
        started_at: float,
    ):
        self.bridge = bridge
        self.chat_id = chat_id
        self.message_id = message_id
        self.agent = agent
        self.agent_name = agent_name
        self.workspace = workspace
        self.model = model
        self.effort = effort
        self.started_at = started_at

    def update(self, state: str, detail: str) -> None:
        if not self.message_id:
            return
        if self.agent in {"codex", "claude"}:
            detected_model = self.bridge.detect_session_model_label(self.chat_id, self.agent)
            if detected_model:
                self.model = detected_model
            detected_effort = self.bridge.detect_session_effort_label(self.chat_id, self.agent)
            if detected_effort:
                self.effort = detected_effort
        self.bridge.update_status_dashboard(self, state, detail)


@dataclass
class TurnCard:
    """One interactive card per agent turn: shows live activity while running,
    then is updated in place to become the final reply."""
    message_id: str
    chat_id: str
    agent: str
    agent_name: str
    model: str
    effort: str
    started_at: float
    card_id: str | None = None
    sequence: int = 0


class PaperAgentBridge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lark = create_lark_client(settings)
        self._dashboard_lark = (
            create_lark_client(replace(settings, lark_profile=settings.dashboard_lark_profile))
            if settings.dashboard_lark_profile
            else None
        )
        self.agents = AgentRunner(settings)
        self.memory = ChatMemory(
            settings.state_dir,
            max_turns=settings.memory_turns,
            max_chars=settings.memory_chars,
        )
        self.outbox = AssistantOutbox(settings.state_dir, ttl_seconds=settings.outbox_ttl_seconds)
        self.handoffs = AgentHandoffQueue(settings.state_dir, ttl_seconds=settings.outbox_ttl_seconds)
        self.workspaces = ChatWorkspaceStore(
            settings.state_dir,
            settings.workspace,
            settings.workspace_roots,
        )
        self.efforts = ChatEffortStore(settings.state_dir)
        self.models = ChatModelStore(settings.state_dir)
        self.responders = ChatResponderStore(settings.state_dir, settings.default_responder)
        self.pending_runs = PendingRunStore(
            settings.state_dir,
            ttl_seconds=settings.outbox_ttl_seconds,
        )
        self.status_dashboards = StatusDashboardStore(settings.state_dir)
        self._pending_status_updates: dict[str, float] = {}
        self._active_run_ids: set[str] = set()
        # Track active turn cards to avoid creating duplicates when a second
        # message arrives while the agent is still busy.
        self._active_turn_cards: dict[str, TurnCard] = {}
        self._turn_card_lock = threading.Lock()
        self._recent_handoff_sigs: set[str] = set()
        # Follow-up poller state: per (agent, chat_id), the transcript cursor
        # to watch for additional end_turn messages after the first reply.
        self._followup_cursors: dict[str, tuple[str, int, MessageEvent, Route, str | None, int]] = {}
        self._followup_lock = threading.Lock()

    def serve(self) -> None:
        self._load_chat_names()
        self._load_followup_cursors()
        stop_workers = threading.Event()
        handoff_worker = self.start_handoff_worker(stop_workers)
        pending_worker = self.start_pending_run_worker(stop_workers)
        followup_worker = self.start_followup_worker(stop_workers)
        try:
            self.lark.consume_events(self.handle_lark_event)
        finally:
            stop_workers.set()
            if handoff_worker:
                handoff_worker.join(timeout=2)
            if pending_worker:
                pending_worker.join(timeout=2)
            if followup_worker:
                followup_worker.join(timeout=2)

    def _load_chat_names(self) -> None:
        """Query Feishu for group names and register them with the session runtimes."""
        try:
            chats = self.lark.list_chats()
        except LarkCLIError:
            LOGGER.warning("failed to load chat names from Feishu")
            return
        for chat in chats:
            chat_id = chat.get("chat_id") or ""
            name = chat.get("name") or ""
            if chat_id and name:
                self.agents.codex_session.set_chat_label(chat_id, name)
                self.agents.claude_session.set_chat_label(chat_id, name)

    def start_handoff_worker(self, stop_event: threading.Event) -> threading.Thread | None:
        if not self.settings.enable_agent_discussion or not self.settings.direct_agent_handoff:
            return None
        if self.default_agent not in {"codex", "claude"}:
            return None
        worker = threading.Thread(
            target=self.handoff_loop,
            args=(stop_event,),
            name=f"pla-{self.default_agent}-handoffs",
            daemon=True,
        )
        worker.start()
        return worker

    def start_pending_run_worker(self, stop_event: threading.Event) -> threading.Thread | None:
        if not self.enabled_agents:
            return None
        worker = threading.Thread(
            target=self.pending_run_loop,
            args=(stop_event,),
            name=f"pla-{self.default_agent or 'all'}-pending-runs",
            daemon=True,
        )
        worker.start()
        return worker

    def pending_run_agents(self) -> tuple[str, ...]:
        if self.default_agent in {"codex", "claude"}:
            return (self.default_agent,)
        return self.enabled_agents

    def pending_run_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                for agent in self.pending_run_agents():
                    self.process_pending_runs(agent)
            except Exception:
                LOGGER.exception("pending run recovery poll failed")
            stop_event.wait(1)

    def process_pending_runs(self, agent: str) -> None:
        for run in self.pending_runs.pending_for(agent):
            self.process_pending_run(run)

    def _turn_card_from_run(self, run: PendingRun) -> TurnCard | None:
        if not run.status_message_id:
            return None
        return TurnCard(
            run.status_message_id,
            run.chat_id,
            run.agent,
            self.agent_display_name(run.agent),
            run.model_label or self.chat_model_label(run.chat_id, run.agent),
            run.effort_label or self.chat_effort_label(run.chat_id, run.agent),
            run.created_at,
            card_id=run.card_id,
        )

    def process_pending_run(self, run: PendingRun) -> None:
        if run.run_id in self._active_run_ids:
            return
        card = self._turn_card_from_run(run)
        try:
            reply = self.agents.find_session_reply(
                run.agent,
                run.chat_id,
                run.start_marker,
                run.end_marker,
            )
        except AgentError as exc:
            LOGGER.warning("pending run %s recovery failed: %s", run.run_id, exc)
            return
        if reply is None:
            # Check if the agent is still actively working — if so, keep
            # waiting regardless of age. Only expire if it's been idle too long.
            age = time.time() - run.created_at if run.created_at else 0
            max_age = (run.timeout or self.agent_timeout(run.agent)) * 2
            if age > max_age:
                still_busy = False
                try:
                    runtime = self.agents.codex_session if run.agent == "codex" else self.agents.claude_session
                    session_name = runtime.session_name(run.chat_id)
                    if runtime.session_exists(session_name):
                        screen = runtime.capture(session_name)
                        from .tmux_runtime import session_tail_busy
                        still_busy = session_tail_busy(screen)
                except Exception:
                    pass
                if not still_busy:
                    if card:
                        self._render_turn_card(card, "failed", "超时未完成，请重试。")
                        with self._turn_card_lock:
                            self._active_turn_cards.pop(f"{card.agent}:{card.chat_id}", None)
                    self.pending_runs.mark_done(run.run_id, status="timeout")
                    LOGGER.info("pending run %s timed out after %.0fs (agent idle)", run.run_id, age)
                    return
            self.update_recovered_status(run, card)
            return
        if not self.pending_runs.claim(run.run_id):
            return
        event = self.pending_run_event(run)
        route = Route("agent", text=run.route_text, agent=run.agent)
        try:
            # If the run had no card (was queued while another card was active),
            # create one now — the previous card should be finished by now.
            if not card and self.settings.send_progress:
                card = self.start_turn_card(
                    run.chat_id, run.agent,
                    run.model_label or self.chat_model_label(run.chat_id, run.agent),
                    run.effort_label or self.chat_effort_label(run.chat_id, run.agent),
                )
            if card:
                self.finalize_turn_reply(
                    card,
                    route,
                    event,
                    reply,
                    source_agent=run.source_agent,
                    handoff_depth=run.handoff_depth,
                )
            elif reply and not self.is_no_reply(reply):
                self.send_reply(
                    event,
                    route,
                    reply,
                    source_agent=run.source_agent,
                    handoff_depth=run.handoff_depth,
                )
                if self.settings.enable_memory:
                    self.memory.append_assistant(run.chat_id, run.agent, reply)
        finally:
            self.pending_runs.mark_done(run.run_id)

    def pending_run_expired(self, run: PendingRun) -> bool:
        timeout = run.timeout or self.agent_timeout(run.agent)
        return time.time() > run.created_at + max(1, timeout)

    def update_recovered_status(self, run: PendingRun, card: TurnCard | None) -> None:
        if not card or not card.message_id:
            return
        interval = max(1, self.settings.status_update_seconds)
        last = self._pending_status_updates.get(run.run_id, 0.0)
        if time.time() - last < interval:
            return
        detail = self.agents.session_progress(run.agent, run.chat_id) or (
            "仍在处理,完成后这张卡会更新成回复…"
        )
        self._render_turn_card(card, "running", detail)
        self._pending_status_updates[run.run_id] = time.time()

    def recovered_status_handle(self, run: PendingRun) -> StatusHandle | None:
        if not run.status_message_id:
            return None
        return StatusHandle(
            self,
            run.chat_id,
            run.status_message_id,
            run.agent,
            self.agent_display_name(run.agent),
            Path(run.workspace).expanduser().resolve(),
            run.model_label or self.chat_model_label(run.chat_id, run.agent),
            run.effort_label or self.chat_effort_label(run.chat_id, run.agent),
            run.created_at,
        )

    def pending_run_event(self, run: PendingRun) -> MessageEvent:
        return MessageEvent(
            event_id=run.event_id,
            chat_id=run.chat_id,
            chat_type=run.chat_type or "group",
            content=run.event_content,
            sender_id=run.sender_id,
            message_id=run.message_id,
            message_type=run.message_type or "text",
        )

    def handoff_loop(self, stop_event: threading.Event) -> None:
        assert self.default_agent is not None
        while not stop_event.is_set():
            try:
                self.process_pending_handoffs(self.default_agent)
            except Exception:
                LOGGER.exception("agent handoff poll failed")
            stop_event.wait(1)

    def process_pending_handoffs(self, target_agent: str) -> None:
        for handoff in self.handoffs.pending_for(target_agent):
            if not self.handoffs.claim(handoff.handoff_id):
                continue
            try:
                self.process_handoff(handoff)
            except Exception:
                LOGGER.exception("agent handoff %s failed", handoff.handoff_id)
                self.handoffs.mark_done(handoff.handoff_id, status="failed")
            else:
                self.handoffs.mark_done(handoff.handoff_id)

    def process_handoff(self, handoff: AgentHandoff) -> None:
        if handoff.target_agent != self.default_agent:
            return
        event = MessageEvent(
            event_id=f"handoff:{handoff.handoff_id}",
            chat_id=handoff.chat_id,
            chat_type="group",
            content=handoff.content,
            sender_id=f"assistant:{handoff.source_agent}",
            message_id=f"handoff:{handoff.handoff_id}",
            message_type="text",
        )
        route = Route("agent", text=handoff.content, agent=handoff.target_agent)  # type: ignore[arg-type]
        LOGGER.info(
            "processing handoff %s from %s to %s in %s",
            handoff.handoff_id,
            handoff.source_agent,
            handoff.target_agent,
            handoff.chat_id,
        )
        if self.settings.enable_memory:
            self.memory.append_assistant(handoff.chat_id, handoff.source_agent, handoff.content)
        reply = self.dispatch(
            route,
            event,
            source_agent=handoff.source_agent,
            handoff_depth=handoff.depth,
        )
        if reply and not self.is_no_reply(reply):
            self.send_reply(
                event,
                route,
                reply,
                source_agent=handoff.source_agent,
                handoff_depth=handoff.depth,
            )
            if self.settings.enable_memory:
                self.memory.append_assistant(handoff.chat_id, handoff.target_agent, reply)

    def handle_lark_event(self, event: LarkEvent) -> None:
        if event.event_type == BOT_ADDED_EVENT_TYPE:
            self.handle_bot_added_event(event)
            return
        if event.event_type in CHAT_CLOSED_EVENT_TYPES:
            self.handle_chat_closed_event(event)
            return
        if event.event_type in {"", MESSAGE_EVENT_TYPE}:
            # Run in a thread so long-running dispatch doesn't block event
            # consumption — new human messages arrive even while an agent turn
            # is in progress.
            msg_event = MessageEvent.from_json(event.raw)
            threading.Thread(
                target=self._handle_event_safe,
                args=(msg_event,),
                daemon=True,
                name=f"event-{msg_event.message_id[:12]}",
            ).start()
            return
        LOGGER.debug("ignoring unsupported event type %s", event.event_type)

    def _handle_event_safe(self, event: MessageEvent) -> None:
        try:
            self.handle_event(event)
        except Exception:
            LOGGER.exception("unhandled error processing event %s", event.event_id)

    def handle_bot_added_event(self, event: LarkEvent) -> None:
        chat_id = event.chat_id
        if not chat_id:
            return
        if not self._chat_allowed(chat_id):
            return
        LOGGER.info("bot added to %s; waiting for /workspace before warming sessions", chat_id)

    def handle_chat_closed_event(self, event: LarkEvent) -> None:
        chat_id = event.chat_id
        if not chat_id:
            return
        if not self._chat_allowed(chat_id):
            return
        for agent in self.enabled_agents:
            if self.agent_runtime(agent) != "session":
                continue
            try:
                self.agents.reset_session(agent, chat_id)
            except AgentError as exc:
                LOGGER.warning("failed to reset %s session for closed chat %s: %s", agent, chat_id, exc)
                continue
            LOGGER.info("reset %s session for closed chat %s on %s", agent, chat_id, event.event_type)

    def _chat_allowed(self, chat_id: str) -> bool:
        if self.settings.chat_id and chat_id != self.settings.chat_id:
            return False
        if chat_id in self.settings.chat_id_exclude:
            return False
        return True

    def handle_event(self, event: MessageEvent) -> None:
        if not self._chat_allowed(event.chat_id):
            return
        if self.settings.bot_open_id and event.sender_id == self.settings.bot_open_id:
            return
        if event.chat_type and event.chat_type != "group":
            return
        if event.message_type and event.message_type not in SUPPORTED_MESSAGE_TYPES:
            return
        if self.is_stale_message_event(event):
            return
        source_agent = self.source_agent_for_event(event)
        if source_agent == "__ignore__":
            return
        if self.is_standalone_upload(event):
            if self.settings.handle_management_commands:
                try:
                    self.handle_standalone_upload(event)
                except LarkCLIError as exc:
                    LOGGER.warning("standalone attachment download failed for %s: %s", event.message_id, exc)
                    self.send_progress_markdown(event.chat_id, f"Attachment download failed: {exc}")
            return
        try:
            event = self.prepare_event(event)
        except LarkCLIError as exc:
            LOGGER.warning("inbound attachment download failed for %s: %s", event.message_id, exc)
            self.send_progress_markdown(event.chat_id, f"Attachment download failed: {exc}")
            return

        try:
            route = route_message(
                event.content,
                self.settings.respond_to_all,
                enabled_agents=self.enabled_agents,
                bot_aliases=self.settings.bot_aliases,
                default_agent=self.default_agent,
                strict_alias=self.settings.strict_alias_routing,
            )
        except ValueError as exc:
            self.lark.send_markdown(event.chat_id, f"Task parse error: {exc}")
            return

        if route.kind == "ignore":
            return
        if route.kind in {"workspace", "clear", "responder"} and not self.settings.handle_management_commands:
            return
        if route.kind == "task" and not self.settings.enable_tasks:
            return
        if route.kind == "debate" and not self.settings.enable_debate:
            return
        if route.broadcast and source_agent is None:
            gated = self.apply_default_responder(route, event.chat_id)
            if gated is None:
                LOGGER.info(
                    "skipping broadcast in %s: not the default responder", event.chat_id
                )
                return
            route = gated
        LOGGER.info("handling %s from %s", route.kind, event.sender_id)
        try:
            reply = self.dispatch(route, event, source_agent=source_agent)
        except (AgentError, LarkCLIError) as exc:
            reply = f"Bridge error: {exc}"
        if self.settings.enable_memory and route.kind not in {"workspace", "clear", "responder", "session_command"}:
            self.memory.append_user(event, route.text or event.content)
        if reply and not self.is_no_reply(reply):
            self.send_reply(event, route, reply, source_agent=source_agent)
            if self.settings.enable_memory:
                self.memory.append_assistant(event.chat_id, self.reply_agent_name(route), reply)

    @property
    def enabled_agents(self) -> tuple[str, ...]:
        if self.settings.agent_mode == "codex":
            return ("codex",)
        if self.settings.agent_mode == "claude":
            return ("claude",)
        if self.settings.agent_mode == "tasks":
            return ()
        return ("codex", "claude")

    @property
    def default_agent(self) -> str | None:
        if self.settings.agent_mode in {"codex", "claude"}:
            return self.settings.agent_mode
        return None

    def source_agent_for_event(self, event: MessageEvent) -> str | None:
        if not self.settings.respond_to_all:
            return None
        match = self.outbox.match_message_id(event.chat_id, event.message_id)
        if not match:
            match = self.outbox.match(event.chat_id, event.content)
        if not match:
            return None
        source_agent = str(match.get("agent") or "")
        if not source_agent or source_agent == self.default_agent:
            LOGGER.info("ignoring own assistant message in %s", event.chat_id)
            return "__ignore__"
        if not self.settings.enable_agent_discussion:
            LOGGER.info("ignoring assistant message; discussion disabled in %s", event.chat_id)
            return "__ignore__"
        if not match.get("discussion_trigger"):
            LOGGER.info("ignoring non-final assistant chunk in %s", event.chat_id)
            return "__ignore__"
        recent = self.outbox.recent_discussion_count(
            event.chat_id,
            self.settings.agent_discussion_window_seconds,
        )
        if recent >= self.settings.max_agent_discussion_turns:
            LOGGER.info("ignoring assistant message; discussion cap reached in %s", event.chat_id)
            return "__ignore__"
        return source_agent

    def is_stale_message_event(self, event: MessageEvent) -> bool:
        max_age = self.settings.max_event_age_seconds
        if max_age <= 0:
            return False
        created_at = parse_message_create_time(event.create_time)
        if created_at is None:
            return False
        age = time.time() - created_at
        if age <= max_age:
            return False
        LOGGER.info(
            "ignoring stale message event %s in %s: age %.0fs exceeds %ss",
            event.message_id,
            event.chat_id,
            age,
            max_age,
        )
        return True

    def is_standalone_upload(self, event: MessageEvent) -> bool:
        if event.message_type in {"image", "file", "audio", "video"}:
            return self.settings.enable_inbound_files
        if event.message_type != "post" or not self.settings.enable_inbound_files:
            return False
        if not extract_inbound_resources(event.message_type, event.content):
            return False
        return not post_text_without_resources(event.content)

    def handle_standalone_upload(self, event: MessageEvent) -> None:
        downloaded = self.download_inbound_resources(event)
        if not downloaded:
            self.send_progress_markdown(
                event.chat_id,
                f"收到 {event.message_type}，但没有解析到可下载的资源 key。",
            )
            return
        reply = format_standalone_upload_reply(downloaded)
        self.send_progress_markdown(event.chat_id, reply)
        if self.settings.enable_memory:
            self.memory.append_assistant(event.chat_id, "bridge", reply)

    def prepare_event(self, event: MessageEvent) -> MessageEvent:
        if event.message_type not in RESOURCE_MESSAGE_TYPES:
            return event
        if not self.settings.enable_inbound_files:
            return event
        resources = extract_inbound_resources(event.message_type, event.content)
        if not resources:
            if event.message_type == "post":
                return event
            return replace(
                event,
                content=(
                    f"User sent a {event.message_type} message, but the bridge could not "
                    f"extract a downloadable resource key.\nRaw content: {event.content}"
                ),
            )
        downloaded = self.download_inbound_resources(event)
        lines: list[str] = []
        if event.message_type == "post" and event.content.strip():
            lines.append(event.content.strip())
            lines.append("")
        lines.extend([
            f"User uploaded/embedded {len(downloaded)} attachment(s) to the Feishu group.",
            "Downloaded local paths:",
        ])
        for name, path in downloaded:
            lines.append(f"- {name}: {path}")
        lines.extend(
            [
                "",
                "Use these local paths directly when reading or analyzing the uploaded files.",
            ]
        )
        return replace(event, content="\n".join(lines))

    def download_inbound_resources(self, event: MessageEvent) -> list[tuple[str, Path]]:
        workspace = self.chat_workspace(event.chat_id)
        resources = extract_inbound_resources(event.message_type, event.content)
        downloaded: list[tuple[str, Path]] = []
        for resource in resources[: self.settings.max_inbound_files]:
            output = inbound_output_relative_path(
                self.settings.workspace,
                workspace,
                event.chat_id,
                event.message_id,
                resource,
            )
            path = self.lark.download_message_resource(
                event.message_id,
                resource.key,
                resource.resource_type,
                output,
            )
            downloaded.append((resource.name, path))
        return downloaded

    def dispatch(
        self,
        route: Route,
        event: MessageEvent,
        source_agent: str | None = None,
        handoff_depth: int = 0,
    ) -> str:
        if route.kind == "help":
            return HELP_TEXT
        if route.kind == "task":
            assert route.task is not None
            result = self.lark.create_task(route.task)
            url = first_field(result, "url")
            guid = first_field(result, "guid")
            suffix = f"\n\n{url}" if url else (f"\n\nTask guid: {guid}" if guid else "")
            return f"Task created: {route.task.summary}{suffix}"
        if route.kind == "workspace":
            return self.handle_workspace_command(event.chat_id, route.text)
        if route.kind == "clear":
            return self.handle_clear_command(event.chat_id, route.text)
        if route.kind == "responder":
            return self.handle_responder_command(event.chat_id, route.text)
        if route.kind == "import_memory":
            return self.handle_import_memory(event.chat_id, route.text)
        if route.kind == "session_command":
            assert route.agent is not None
            return self.handle_session_command(route.agent, event.chat_id, route.text)
        if route.kind == "multi_agent":
            if not route.agent_texts:
                return ""
            for agent, agent_text in route.agent_texts.items():
                subroute = Route("agent", text=agent_text, agent=agent)
                try:
                    reply = self.dispatch(
                        subroute,
                        event,
                        source_agent=source_agent,
                        handoff_depth=handoff_depth,
                    )
                except (AgentError, LarkCLIError) as exc:
                    reply = f"Bridge error ({self.agent_display_name(agent)}): {exc}"
                if reply and not self.is_no_reply(reply):
                    self.send_reply(
                        event,
                        subroute,
                        reply,
                        source_agent=source_agent,
                        handoff_depth=handoff_depth,
                    )
                    if self.settings.enable_memory:
                        self.memory.append_assistant(event.chat_id, agent, reply)
            return ""
        if route.kind == "agent":
            assert route.agent is not None
            if not route.text:
                return "Please add a question after the agent command."
            if self.pending_runs.has_message_run(event.chat_id, event.message_id, route.agent):
                LOGGER.info(
                    "ignoring duplicate %s run for message %s in %s",
                    route.agent,
                    event.message_id,
                    event.chat_id,
                )
                return ""
            workspace = self.chat_workspace(event.chat_id)
            effort = self.chat_effort(event.chat_id, route.agent)
            model = self.chat_model_override(event.chat_id, route.agent)
            turn_card = None
            if self.settings.send_progress:
                turn_card = self.start_turn_card(
                    event.chat_id,
                    route.agent,
                    self.chat_model_label(event.chat_id, route.agent),
                    self.chat_effort_label(event.chat_id, route.agent),
                    force=source_agent is None,  # human message takes priority
                )
            prompt, session_context = self.build_agent_prompt(
                route.agent,
                event,
                route.text,
                source_agent,
                workspace,
            )
            run_id: str | None = None
            if self.agent_runtime(route.agent) == "session":
                run_id = uuid.uuid4().hex
                self._active_run_ids.add(run_id)
                start_marker, end_marker = self.agents.reply_markers(route.agent, run_id)
                self.pending_runs.start(
                    run_id=run_id,
                    chat_id=event.chat_id,
                    agent=route.agent,
                    route_text=route.text,
                    event_id=event.event_id,
                    message_id=event.message_id,
                    sender_id=event.sender_id,
                    message_type=event.message_type,
                    chat_type=event.chat_type,
                    event_content=event.content,
                    source_agent=source_agent,
                    handoff_depth=handoff_depth,
                    start_marker=start_marker,
                    end_marker=end_marker,
                    session_name=self.agents.session_name(route.agent, event.chat_id),
                    workspace=str(workspace),
                    status_message_id=turn_card.message_id if turn_card else None,
                    card_id=turn_card.card_id if turn_card else None,
                    model_label=self.chat_model_label(event.chat_id, route.agent),
                    effort_label=self.chat_effort_label(event.chat_id, route.agent),
                    timeout=self.agent_timeout(route.agent),
                )
            try:
                if route.agent == "codex":
                    result = self.agents.run_codex(
                        prompt,
                        event.chat_id,
                        session_context=session_context,
                        workspace=workspace,
                        model=model,
                        effort=effort,
                        progress_callback=(lambda detail: self.update_turn_card(turn_card, detail))
                        if turn_card
                        else None,
                        run_id=run_id,
                    )
                else:
                    result = self.agents.run_claude(
                        prompt,
                        event.chat_id,
                        session_context=session_context,
                        workspace=workspace,
                        model=model,
                        effort=effort,
                        progress_callback=(lambda detail: self.update_turn_card(turn_card, detail))
                        if turn_card
                        else None,
                        run_id=run_id,
                    )
            except AgentStillRunning as exc:
                if turn_card:
                    self._render_turn_card(turn_card, "running", "仍在处理,完成后这张卡会更新成回复…")
                if run_id:
                    self._active_run_ids.discard(run_id)
                LOGGER.info("%s run still running in %s: %s", route.agent, event.chat_id, exc)
                return ""
            except Exception:
                if turn_card:
                    self._render_turn_card(turn_card, "failed", "运行失败,稍后会发送错误信息。")
                if run_id:
                    self.pending_runs.mark_done(run_id, status="failed")
                    self._active_run_ids.discard(run_id)
                raise
            text = result.text
            if run_id:
                self.pending_runs.mark_done(run_id)
                self._active_run_ids.discard(run_id)
            if turn_card:
                # The progress card morphs into the reply (or "no reply"); the
                # bridge delivers it here, so handle_event must not send it again.
                self.finalize_turn_reply(
                    turn_card,
                    route,
                    event,
                    text,
                    source_agent=source_agent,
                    handoff_depth=handoff_depth,
                )
                # Claude can produce follow-up replies after a teammate finishes.
                # Register the cursor for the persistent followup_loop to watch.
                if result.transcript_path and route.agent == "claude":
                    self.register_followup_cursor(
                        route.agent, event, route,
                        result.transcript_path, result.transcript_offset,
                        source_agent, handoff_depth,
                    )
                return ""
            return text
        if route.kind == "debate":
            if not route.text:
                return "Please add a paper, claim, or question after /debate."
            if self.settings.send_progress:
                self.send_progress_markdown(event.chat_id, "Asking Codex and Claude...")
            workspace = self.chat_workspace(event.chat_id)
            codex_prompt, codex_context = self.build_debate_prompt(
                "codex",
                event,
                route.text,
                workspace,
            )
            claude_prompt, claude_context = self.build_debate_prompt(
                "claude",
                event,
                route.text,
                workspace,
            )
            codex = self.agents.run_codex(
                codex_prompt,
                event.chat_id,
                session_context=codex_context,
                workspace=workspace,
                model=self.chat_model_override(event.chat_id, "codex"),
                effort=self.chat_effort(event.chat_id, "codex"),
            ).text
            claude = self.agents.run_claude(
                claude_prompt,
                event.chat_id,
                session_context=claude_context,
                workspace=workspace,
                model=self.chat_model_override(event.chat_id, "claude"),
                effort=self.chat_effort(event.chat_id, "claude"),
            ).text
            return format_debate(codex, claude)
        return ""

    def handle_session_command(self, agent: str, chat_id: str, command: str) -> str:
        command = command.strip()
        if not command:
            return "Please provide a session command."
        workspace = self.chat_workspace(chat_id)

        try:
            effort = self.parse_effort_command(command)
            model = self.parse_model_command(command)
        except EffortError as exc:
            return str(exc)
        except ModelError as exc:
            return str(exc)
        if effort is not None:
            try:
                effort = self.efforts.set(chat_id, agent, effort)
            except EffortError as exc:
                return str(exc)
            agent_name = self.agent_display_name(agent)
            sent = self.agents.send_session_command(
                agent,
                chat_id,
                f"/effort {effort}",
                workspace=workspace,
                effort=effort,
            )
            if sent:
                return f"{agent_name} effort 已设为 `{effort}`（当前会话生效）。"
            return f"{agent_name} effort 已记为 `{effort}`，会话当前忙碌，下轮空闲时自动生效。"
        if model is not None:
            try:
                model = self.models.set(chat_id, agent, model)
            except ModelError as exc:
                return str(exc)
            sent = self.agents.send_session_command(
                agent,
                chat_id,
                f"/model {model}",
                workspace=workspace,
                effort=self.chat_effort(chat_id, agent),
            )
            agent_name = self.agent_display_name(agent)
            if sent:
                return f"{agent_name} model set to `{model}` in the current session."

        return self.run_session_command_with_status(agent, chat_id, command, workspace)

    def run_session_command_with_status(
        self,
        agent: str,
        chat_id: str,
        command: str,
        workspace: Path,
        model: str | None = None,
        effort: str | None = None,
    ) -> str:
        agent_name = self.agent_display_name(agent)
        preview = command if len(command) <= 120 else f"{command[:117]}..."
        sent = self.agents.send_session_command(
            agent,
            chat_id,
            command,
            workspace=workspace,
            model=model,
            effort=effort,
        )
        if not sent:
            return f"{agent_name} 不在 session 模式，无法发送：`{preview}`"
        # Commands that produce output (like /deep-research) need a card +
        # JSONL monitoring, just like a normal agent turn. Settings-only
        # commands (/effort, /model) are handled before reaching here.
        if self.settings.send_progress and self.agent_runtime(agent) == "session":
            return self._watch_session_command_reply(agent, chat_id, workspace, preview)
        return f"已发送到 {agent_name} 会话：`{preview}`"

    def _watch_session_command_reply(
        self, agent: str, chat_id: str, workspace: Path, preview: str,
    ) -> str:
        """Monitor the transcript for the session command's reply, with a turn
        card for live progress — same UX as a normal agent turn."""
        runtime = self.agents.codex_session if agent == "codex" else self.agents.claude_session
        session_name = runtime.session_name(chat_id)
        turn_card = self.start_turn_card(
            chat_id, agent,
            self.chat_model_label(chat_id, agent),
            self.chat_effort_label(chat_id, agent),
            force=True,  # session command is human-initiated
        )
        run_id = uuid.uuid4().hex
        start_marker, end_marker = runtime.reply_markers(run_id)
        path_str, offset = runtime.transcript_cursor(session_name, chat_id, workspace)
        runtime.store_run_cursor(session_name, run_id, path_str, offset)
        self.pending_runs.start(
            run_id=run_id, chat_id=chat_id, agent=agent,
            route_text=preview, event_id=f"cmd:{run_id}",
            message_id=f"cmd:{run_id}", sender_id="bridge",
            message_type="text", chat_type="group", event_content=preview,
            source_agent=None, handoff_depth=0,
            start_marker=start_marker, end_marker=end_marker,
            session_name=session_name, workspace=str(workspace),
            status_message_id=turn_card.message_id if turn_card else None,
            card_id=turn_card.card_id if turn_card else None,
            model_label=self.chat_model_label(chat_id, agent),
            effort_label=self.chat_effort_label(chat_id, agent),
            timeout=self.agent_timeout(agent),
        )
        try:
            text, usage, cursor = runtime.wait_for_jsonl_reply(
                session_name, chat_id, workspace, path_str, offset,
                self.agent_timeout(agent),
                progress_callback=(lambda d: self.update_turn_card(turn_card, d))
                if turn_card else None,
                progress_interval=self.settings.status_update_seconds,
            )
        except TmuxReplyStillRunning:
            if turn_card:
                self._render_turn_card(turn_card, "running", "仍在处理,完成后这张卡会更新成回复…")
            LOGGER.info("%s session command still running in %s", agent, chat_id)
            return ""
        self.pending_runs.mark_done(run_id)
        if turn_card:
            event = MessageEvent(
                event_id=f"cmd:{run_id}", chat_id=chat_id,
                chat_type="group", content=preview,
                sender_id="bridge", message_id=f"cmd:{run_id}",
                message_type="text",
            )
            route = Route("agent", text=preview, agent=agent)
            self.finalize_turn_reply(
                turn_card, route, event, text,
                source_agent=None, handoff_depth=0,
            )
            if cursor and agent == "claude":
                self.register_followup_cursor(
                    agent, event, route, cursor[0], cursor[1], None, 0,
                )
            return ""
        return text

    def parse_effort_command(self, command: str) -> str | None:
        lowered = command.lower()
        for prefix in ("/effort",):
            if lowered == prefix:
                raise EffortError("请提供 effort，例如 `@Codex /effort xhigh`。")
            if lowered.startswith(f"{prefix} "):
                return normalize_effort(command[len(prefix) :].strip())
        return None

    def parse_model_command(self, command: str) -> str | None:
        lowered = command.lower()
        for prefix in ("/model",):
            if lowered == prefix:
                raise ModelError("请提供 model，例如 `@Claude /model opus`。")
            if lowered.startswith(f"{prefix} "):
                return normalize_model(command[len(prefix) :].strip())
        return None

    def handle_clear_command(self, chat_id: str, text: str = "") -> str:
        workspace = self.chat_workspace(chat_id)
        should_init = self.clear_should_init(text)
        memory_cleared = self.memory.clear(chat_id)
        outbox_records = self.outbox.clear_chat(chat_id)
        handoff_records = self.handoffs.clear_chat(chat_id)
        pending_run_records = self.pending_runs.clear_chat(chat_id)
        reset: list[str] = []
        failed: list[str] = []
        cli_files_deleted = 0
        for agent in ("codex", "claude"):
            if self.agent_runtime(agent) != "session":
                continue
            try:
                deleted = self.agents.reset_session(agent, chat_id)
            except AgentError as exc:
                failed.append(f"{self.agent_display_name(agent)}: {exc}")
                LOGGER.warning("failed to clear %s session for %s: %s", agent, chat_id, exc)
                continue
            cli_files_deleted += len(deleted)
            reset.append(self.agent_display_name(agent))
            LOGGER.info("cleared %s session for %s", agent, chat_id)

        lines = [
            "Cleared this group's bridge state.",
            "",
            f"- Memory: {'cleared' if memory_cleared else 'already empty'}",
            f"- Outbox records: {outbox_records}",
            f"- Pending handoff records: {handoff_records}",
            f"- Pending run records: {pending_run_records}",
            f"- Sessions closed/deleted: {', '.join(reset) if reset else 'none'}",
            f"- CLI session files deleted: {cli_files_deleted}",
            f"- Workspace kept: `{workspace}`",
        ]
        if failed:
            lines.append("")
            lines.append("Session cleanup failed:")
            lines.extend(f"- {item}" for item in failed)
        if should_init:
            lines.append("")
            status = self.start_workspace_status(
                chat_id,
                workspace,
                "Clear completed. Starting fresh sessions and running /init.",
            )
            warmup = self.warm_workspace_sessions(chat_id, workspace, status)
            self.finish_workspace_status(status, self.clear_init_progress_message(warmup))
            lines.append(warmup)
        return "\n".join(lines)

    def clear_should_init(self, text: str) -> bool:
        lowered = text.strip().lower()
        return lowered in {"init", "reinit", "restart", "warm", "--init", "start"}

    def clear_init_progress_message(self, warmup: str) -> str:
        if "Init still running" in warmup or "session startup failed" in warmup.lower():
            return f"Clear completed. Init started, but some work is still pending.\n\n{warmup}"
        return f"Clear completed. Init completed.\n\n{warmup}"

    def done_status_detail(self, usage: dict | None) -> str:
        out = 0
        if isinstance(usage, dict):
            out = int(
                usage.get("output_tokens")
                or usage.get("total_token_count")
                or usage.get("total_tokens")
                or 0
            )
        if out:
            tok = f"{out / 1000:.1f}k" if out >= 1000 else str(out)
            return f"✓ 回复完成 · {tok} tok"
        return "✓ 回复完成"

    def apply_default_responder(self, route: Route, chat_id: str) -> Route | None:
        """Gate a broadcast route by the chat's default responder.

        Returns the route to run (possibly with agents filtered out), or None if
        this message should not be answered by anyone here. Only plain
        respond-to-all broadcasts reach this; @-mentions, explicit /codex|/claude
        commands, and bot-to-bot discussion are never gated.
        """
        if route.kind == "agent":
            if route.agent and self.responders.allows(chat_id, route.agent):
                return route
            return None
        if route.kind == "multi_agent" and route.agent_texts:
            allowed = {
                agent: text
                for agent, text in route.agent_texts.items()
                if self.responders.allows(chat_id, agent)
            }
            if not allowed:
                return None
            if len(allowed) != len(route.agent_texts):
                return replace(route, agent_texts=allowed)
            return route
        return route

    def handle_import_memory(self, chat_id: str, text: str) -> str:
        """Import room memory from another chat_id into the current chat.

        Usage: /import oc_xxx — copies history.jsonl from the source chat dir
        into this chat's dir so agents get the prior context on their next turn.
        """
        source_id = text.strip()
        if not source_id:
            return "用法：`/import <source_chat_id>`\n将另一个群的对话记忆导入到当前群。"
        source_dir = self.settings.state_dir / "chats" / re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)
        source_file = source_dir / "history.jsonl"
        if not source_file.exists():
            return f"找不到源群记忆：`{source_id}`\n确认 `.state/chats/{source_id}/history.jsonl` 存在。"
        target_dir = self.settings.state_dir / "chats" / re.sub(r"[^A-Za-z0-9_.-]+", "_", chat_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "history.jsonl"
        existing = target_file.read_text(encoding="utf-8") if target_file.exists() else ""
        imported = source_file.read_text(encoding="utf-8")
        with target_file.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(imported)
        n_lines = len([l for l in imported.splitlines() if l.strip()])
        return f"已从 `{source_id}` 导入 {n_lines} 条记忆到当前群。下一条消息时 agent 会收到完整上下文。"

    def handle_responder_command(self, chat_id: str, text: str) -> str:
        command = text.strip()
        lowered = command.lower()
        if not command or lowered in {"show", "current", "status"}:
            return self.describe_responder(chat_id)
        if lowered in {"reset", "default", "auto"}:
            responder = self.responders.reset(chat_id)
            return (
                f"Default responder reset to global default: **{self.responder_label(responder)}**.\n\n"
                f"{self.responder_help_note()}"
            )
        try:
            responder = self.responders.set(chat_id, command)
        except ResponderError as exc:
            return f"{exc}\n\nUsage: `/responder [codex|claude|both|reset]`"
        return (
            f"Default responder for this group set to: **{self.responder_label(responder)}**.\n\n"
            f"{self.responder_help_note()}"
        )

    def describe_responder(self, chat_id: str) -> str:
        current = self.responders.current(chat_id)
        mode = "custom" if self.responders.has_override(chat_id) else "global default"
        return (
            f"Default responder ({mode}): **{self.responder_label(current)}**\n"
            f"Global default: {self.responder_label(self.settings.default_responder)}\n\n"
            f"{self.responder_help_note()}"
        )

    def responder_label(self, responder: str) -> str:
        if responder == "both":
            return "Codex + Claude Code"
        return self.agent_display_name(responder)

    def responder_help_note(self) -> str:
        return (
            "Unaddressed group messages are answered only by the default responder. "
            "@-mention a bot to reach it directly; this setting does not affect "
            "@-mentions or bot-to-bot discussion.\n"
            "Switch with `/responder codex`, `/responder claude`, `/responder both`, "
            "or `/responder reset`."
        )

    def handle_workspace_command(self, chat_id: str, text: str) -> str:
        command = text.strip()
        lowered = command.lower()
        if not command or lowered in {"show", "pwd", "current"}:
            workspace = self.chat_workspace(chat_id)
            mode = "custom" if self.workspaces.has_override(chat_id) else "default"
            return (
                f"Current workspace ({mode}): `{workspace}`\n\n"
                "Allowed roots:\n"
                f"{self.workspaces.allowed_roots_text()}"
            )

        if lowered in {"reset", "default", "clear"}:
            workspace = self.workspaces.reset(chat_id)
            status = self.start_workspace_status(chat_id, workspace, "Workspace reset. Starting sessions.")
            warmup = self.warm_workspace_sessions(chat_id, workspace, status)
            self.finish_workspace_status(status, warmup)
            return (
                f"Workspace reset for this group: `{workspace}`\n\n"
                f"{warmup}"
            )

        if lowered.startswith("set "):
            command = command[4:].strip()

        try:
            workspace = self.workspaces.set(chat_id, command)
        except WorkspaceError as exc:
            return (
                f"Workspace not changed: {exc}\n\n"
                "Allowed roots:\n"
                f"{self.workspaces.allowed_roots_text()}"
            )

        status = self.start_workspace_status(chat_id, workspace, "Workspace saved. Starting sessions.")
        warmup = self.warm_workspace_sessions(chat_id, workspace, status)
        self.finish_workspace_status(status, warmup)
        return (
            f"Workspace set for this group: `{workspace}`\n\n"
            f"{warmup}"
        )

    def workspace_warmup_agents(self) -> tuple[str, ...]:
        return self.settings.workspace_warmup_agents or self.enabled_agents

    def warm_workspace_sessions(
        self,
        chat_id: str,
        workspace: Path,
        status: StatusHandle | None = None,
    ) -> str:
        warmed: list[str] = []
        warmed_agents: list[str] = []
        failed: list[str] = []
        for agent in self.workspace_warmup_agents():
            if self.agent_runtime(agent) != "session":
                continue
            if status:
                status.update("running", f"Starting {self.agent_display_name(agent)} session.")
            try:
                session_name = self.agents.warmup_session(
                    agent,
                    chat_id,
                    workspace=workspace,
                    model=self.chat_model_override(chat_id, agent),
                    effort=self.chat_effort(chat_id, agent),
                )
            except AgentError as exc:
                LOGGER.warning("failed to warm %s session for %s: %s", agent, chat_id, exc)
                failed.append(f"{self.agent_display_name(agent)}: {exc}")
                continue
            if session_name:
                warmed.append(f"{self.agent_display_name(agent)} (`{session_name}`)")
                warmed_agents.append(agent)
                LOGGER.info("warmed %s session for %s: %s", agent, chat_id, session_name)
                if status:
                    status.update("running", f"{self.agent_display_name(agent)} session started.")
        init_done, init_pending = self.initialize_workspace_sessions(
            chat_id,
            workspace,
            warmed_agents,
            status,
        )
        if init_done:
            warmed.append(f"Init completed: {', '.join(init_done)}")
        if init_pending:
            warmed.append(f"Init still running: {', '.join(init_pending)}")
        if warmed and not failed:
            return "Sessions started in that directory:\n" + "\n".join(f"- {item}" for item in warmed)
        if warmed and failed:
            return (
                "Sessions started in that directory:\n"
                + "\n".join(f"- {item}" for item in warmed)
                + "\n\nSession startup failed:\n"
                + "\n".join(f"- {item}" for item in failed)
            )
        if failed:
            return "Workspace saved, but session startup failed:\n" + "\n".join(
                f"- {item}" for item in failed
            )
        return "Workspace saved. No session warmup agents are configured."

    def initialize_workspace_sessions(
        self,
        chat_id: str,
        workspace: Path,
        agents: list[str],
        status: StatusHandle | None = None,
    ) -> tuple[list[str], list[str]]:
        sent_agents: list[str] = []
        completed: list[str] = []
        pending: list[str] = []
        for agent in agents:
            if status:
                status.update("running", f"Sending /init to {self.agent_display_name(agent)}.")
            try:
                sent = self.agents.send_session_command(
                    agent,
                    chat_id,
                    "/init",
                    workspace=workspace,
                    model=self.chat_model_override(chat_id, agent),
                    effort=self.chat_effort(chat_id, agent),
                )
            except AgentError as exc:
                LOGGER.warning("failed to send /init to %s session for %s: %s", agent, chat_id, exc)
                pending.append(self.agent_display_name(agent))
                continue
            if sent:
                sent_agents.append(agent)
                LOGGER.info("sent /init to %s session for %s", agent, chat_id)
        timeout = max(120, self.settings.session_command_watch_seconds)
        for agent in sent_agents:
            if status:
                status.update("running", f"Waiting for {self.agent_display_name(agent)} /init to finish.")
            try:
                ready = self.agents.wait_session_ready(agent, chat_id, timeout)
            except AgentError as exc:
                LOGGER.warning("failed waiting for /init on %s session for %s: %s", agent, chat_id, exc)
                ready = False
            display = self.agent_display_name(agent)
            if ready:
                completed.append(display)
                LOGGER.info("/init completed for %s session in %s", agent, chat_id)
                if status:
                    status.update("running", f"{display} /init completed.")
            else:
                pending.append(display)
                LOGGER.warning("/init did not finish within %ss for %s session in %s", timeout, agent, chat_id)
                if status:
                    status.update("running", f"{display} /init is still running.")
        return completed, pending

    def build_agent_prompt(
        self,
        agent: str,
        event: MessageEvent,
        text: str,
        source_agent: str | None,
        workspace: Path,
    ) -> tuple[str, str | None]:
        room_memory = self.room_memory(event.chat_id)
        if self.agent_runtime(agent) == "session":
            return (
                agent_session_turn_prompt(
                    event,
                    text,
                    source_agent=source_agent,
                    room_recap=self.room_recap_for_turn(
                        agent, event.chat_id, room_memory, source_agent, text
                    ),
                ),
                self.agent_session_context(agent, event, room_memory, workspace),
            )
        return (
            agent_prompt(
                agent,
                event,
                text,
                room_memory,
                source_agent=source_agent,
                no_reply_token=self.settings.no_reply_token,
            ),
            None,
        )

    def wants_room_recap(self, agent: str, chat_id: str, source_agent: str | None) -> bool:
        """Whether this turn should carry a recap of the recent room thread.

        Three gaps to fill:
        (1) A teammate just pulled this agent in via @-handoff.
        (2) A human addressed this agent directly but it is *not* the chat's
            broadcast responder — it has been silently skipping unaddressed
            messages and missed the thread.
        (3) The peer agent replied since this agent's last turn. Even the
            broadcast responder never sees peer replies in its own CLI session
            (they go straight to Feishu), so a recap fills that blind spot.
        """
        if source_agent:
            return True
        if not self.responders.allows(chat_id, agent):
            return True
        if self.settings.enable_memory and self.memory.has_unseen_peer_turns(chat_id, agent):
            return True
        return False

    def room_recap_for_turn(
        self,
        agent: str,
        chat_id: str,
        room_memory: str,
        source_agent: str | None,
        text: str,
    ) -> str:
        """Recent room transcript to prepend when there is a context gap.

        The shared room memory holds every speaker's turns, so it fills what the
        agent's own (sparse) CLI session history would miss. Returns "" when no
        recap is warranted or there is no prior thread.
        """
        if not self.wants_room_recap(agent, chat_id, source_agent):
            return ""
        recap = self.memory.unseen_context(chat_id, agent).rstrip()
        if not recap or recap.startswith("No previous discussion"):
            return ""
        # On a peer handoff the peer's message is delivered separately as the
        # turn content; if room memory already recorded it as the last turn,
        # drop it so it isn't shown twice. (Human turns aren't in memory yet.)
        if source_agent:
            tail = f"{source_agent}: {text}".strip()
            if recap.endswith(tail):
                recap = recap[: -len(tail)].rstrip()
        return recap

    def build_debate_prompt(
        self,
        agent: str,
        event: MessageEvent,
        text: str,
        workspace: Path,
    ) -> tuple[str, str | None]:
        room_memory = self.room_memory(event.chat_id)
        if self.agent_runtime(agent) == "session":
            return (
                debate_session_turn_prompt(event, text),
                self.agent_session_context(agent, event, room_memory, workspace),
            )
        return debate_prompt(event, text, room_memory), None

    def agent_runtime(self, agent: str) -> str:
        if agent == "codex":
            return self.settings.codex_runtime
        return self.settings.claude_runtime

    def agent_session_context(
        self,
        agent: str,
        event: MessageEvent,
        room_memory: str,
        workspace: Path,
    ) -> str:
        agent_name = "Codex" if agent == "codex" else "Claude Code"
        peer_name = "Claude" if agent == "codex" else "Codex"
        return agent_session_context_prompt(
            agent_name,
            event,
            room_memory,
            no_reply_token=self.settings.no_reply_token,
            workspace=str(workspace),
            peer_name=peer_name,
        )

    def chat_workspace(self, chat_id: str) -> Path:
        return self.workspaces.current(chat_id)

    def chat_effort(self, chat_id: str, agent: str) -> str | None:
        return self.efforts.current(chat_id, agent) or self.default_effort(agent)

    def chat_effort_label(self, chat_id: str, agent: str) -> str:
        current = self.efforts.current(chat_id, agent)
        if current:
            return f"{current} (group)"
        default = self.default_effort(agent)
        if default:
            return f"{default} (default)"
        detected = self.detect_session_effort_label(chat_id, agent)
        if detected:
            return detected
        return "CLI default (not set by bridge)"

    def default_effort(self, agent: str) -> str | None:
        if agent == "codex":
            return self.settings.codex_default_effort
        if agent == "claude":
            return self.settings.claude_default_effort
        return None

    def agent_timeout(self, agent: str) -> int:
        if agent == "codex":
            return self.settings.codex_timeout
        if agent == "claude":
            return self.settings.claude_timeout
        return max(self.settings.codex_timeout, self.settings.claude_timeout)

    def chat_model(self, chat_id: str, agent: str) -> str | None:
        return self.models.current(chat_id, agent) or self.default_model(agent)

    def chat_model_override(self, chat_id: str, agent: str) -> str | None:
        return self.models.current(chat_id, agent)

    def chat_model_label(self, chat_id: str, agent: str) -> str:
        current = self.models.current(chat_id, agent)
        if current:
            return f"{current} (group)"
        default = self.default_model(agent)
        if default:
            return f"{default} (default)"
        detected = self.detect_session_model_label(chat_id, agent)
        if detected:
            return detected
        return "CLI default (not set by bridge)"

    def detect_session_model_label(self, chat_id: str, agent: str) -> str | None:
        detected = self.agents.detect_session_model(agent, chat_id)
        if detected:
            return f"{detected} (session)"
        return None

    def detect_session_effort_label(self, chat_id: str, agent: str) -> str | None:
        detected = self.agents.detect_session_effort(agent, chat_id)
        if detected:
            return f"{detected} (session)"
        return None

    def default_model(self, agent: str) -> str | None:
        if agent == "codex":
            return self.settings.codex_model
        if agent == "claude":
            return self.settings.claude_model
        return None

    def agent_display_name(self, agent: str) -> str:
        if agent == "codex":
            return "Codex"
        if agent == "claude":
            return "Claude"
        return agent

    def room_memory(self, chat_id: str) -> str:
        if not self.settings.enable_memory:
            return ""
        return self.memory.context(chat_id)

    def reply_agent_name(self, route: Route) -> str:
        if route.kind == "agent" and route.agent:
            return route.agent
        if route.kind == "multi_agent":
            return "multi_agent"
        if route.kind == "debate":
            return "debate"
        if route.kind == "task":
            return "task"
        if route.kind == "session_command":
            return route.agent or "bridge"
        return "bridge"

    def send_bridge_markdown(
        self,
        chat_id: str,
        markdown: str,
        *,
        agent: str | None = None,
        discussion_trigger: bool = True,
    ) -> None:
        agent_name = agent or self.settings.agent_mode
        self.outbox.remember(
            chat_id,
            markdown,
            self.settings.max_message_chars,
            agent=agent_name,
            discussion_trigger=discussion_trigger,
        )
        results = self.lark.send_markdown(chat_id, markdown)
        for index, result in enumerate(results, start=1):
            self.remember_sent_message_id(
                chat_id,
                result,
                agent=agent_name,
                discussion_trigger=discussion_trigger and index == len(results),
            )

    def send_progress_markdown(self, chat_id: str, markdown: str) -> None:
        self.outbox.remember(
            chat_id,
            markdown,
            self.settings.max_message_chars,
            agent=self.settings.agent_mode,
            discussion_trigger=False,
        )
        results = self.lark.send_markdown(chat_id, markdown)
        for result in results:
            self.remember_sent_message_id(
                chat_id,
                result,
                agent=self.settings.agent_mode,
                discussion_trigger=False,
            )

    def remember_sent_message_id(
        self,
        chat_id: str,
        result: object,
        *,
        agent: str,
        discussion_trigger: bool,
    ) -> None:
        message_id = first_field(result, "message_id") or first_field(result, "id")
        if not message_id:
            return
        self.outbox.remember_message_id(
            chat_id,
            str(message_id),
            agent=agent,
            discussion_trigger=discussion_trigger,
        )

    def start_workspace_status(
        self,
        chat_id: str,
        workspace: Path,
        detail: str,
    ) -> StatusHandle | None:
        if not self.settings.send_progress:
            return None
        started_at = time.time()
        handle = StatusHandle(
            self,
            chat_id,
            None,
            "workspace",
            "Workspace",
            workspace,
            "n/a",
            "n/a",
            started_at,
        )
        message_id = self.ensure_status_dashboard_message(chat_id)
        if not message_id:
            return None
        handle.message_id = message_id
        handle.update("running", detail)
        return handle

    def finish_workspace_status(self, status: StatusHandle | None, detail: str) -> None:
        if not status:
            return
        lowered = detail.lower()
        if "failed" in lowered:
            state = "failed"
        elif "still running" in lowered or "pending" in lowered:
            state = "pending"
        else:
            state = "done"
        summary = " ".join(line.strip() for line in detail.splitlines() if line.strip())
        if len(summary) > 500:
            summary = summary[:497].rstrip() + "..."
        status.update(state, summary)

    def _supports_streaming_cards(self) -> bool:
        return hasattr(self.lark, "create_streaming_card")

    def start_turn_card(
        self, chat_id: str, agent: str, model: str, effort: str, *, force: bool = False,
    ) -> TurnCard | None:
        card_key = f"{agent}:{chat_id}"
        with self._turn_card_lock:
            old = self._active_turn_cards.get(card_key)
            if old:
                if not force:
                    return None
                if old.card_id and self._supports_streaming_cards():
                    try:
                        self.lark.finalize_streaming_card(
                            old.card_id,
                            title=f"{old.agent_name} · Interrupted",
                            template="grey",
                            sequence=old.sequence + 1,
                        )
                    except Exception:
                        pass
                self._active_turn_cards.pop(card_key, None)
            agent_name = self.agent_display_name(agent)
            started_at = time.time()
            if self._supports_streaming_cards():
                try:
                    card_id = self.lark.create_streaming_card(f"{agent_name} · Running")
                    message_id = self.lark.send_streaming_card(chat_id, card_id)
                except LarkCLIError as exc:
                    LOGGER.warning("streaming card send failed: %s", exc)
                    return None
                if not message_id:
                    return None
                self.outbox.remember_message_id(chat_id, message_id, agent=agent, discussion_trigger=False)
                turn_card = TurnCard(message_id, chat_id, agent, agent_name, model, effort, started_at, card_id=card_id, sequence=0)
                self._active_turn_cards[card_key] = turn_card
                return turn_card
            card = turn_reply_card(agent_name, "running", "✻ 思考中", model=model, effort=effort, started_at=started_at)
            try:
                result = self.lark.send_card(chat_id, card)
            except LarkCLIError as exc:
                LOGGER.warning("turn card send failed: %s", exc)
                return None
            message_id = first_field(result, "message_id") or first_field(result, "id")
            if not message_id:
                return None
            self.outbox.remember_message_id(chat_id, str(message_id), agent=agent, discussion_trigger=False)
            turn_card = TurnCard(str(message_id), chat_id, agent, agent_name, model, effort, started_at)
            self._active_turn_cards[card_key] = turn_card
            return turn_card

    def _render_turn_card(self, card: TurnCard, state: str, body: str) -> bool:
        if card.card_id and self._supports_streaming_cards():
            try:
                if state == "running":
                    card.sequence += 1
                    self.lark.stream_card_content(card.card_id, body, sequence=card.sequence)
                else:
                    footer = self._turn_card_footer(card)
                    card.sequence += 1
                    template = {"done": "green", "failed": "red", "skipped": "grey"}.get(state, "blue")
                    title = f"{card.agent_name} · {state.title()}"
                    self.lark.finalize_streaming_card(
                        card.card_id,
                        final_content=body,
                        title=title,
                        template=template,
                        sequence=card.sequence,
                        footer=footer,
                    )
            except LarkCLIError as exc:
                LOGGER.warning("streaming card update failed for %s: %s", card.card_id, exc)
                # Streaming mode may have auto-closed (10 min timeout) or
                # sequence diverged. Fall back to regular card update.
                card.card_id = None
                return self._render_turn_card(card, state, body)
            return True
        rendered = turn_reply_card(
            card.agent_name, state, body, model=card.model, effort=card.effort, started_at=card.started_at
        )
        try:
            self.lark.update_card(card.message_id, rendered)
        except LarkCLIError as exc:
            LOGGER.warning("turn card update failed for %s: %s", card.message_id, exc)
            if "schemaV2" in str(exc) or "200830" in str(exc):
                card.message_id = ""
            return False
        return True

    def _turn_card_footer(self, card: TurnCard) -> str:
        elapsed = max(0, int(time.time() - card.started_at))
        parts = []
        if card.model:
            parts.append(card.model)
        if card.effort:
            parts.append(card.effort)
        parts.append(f"{elapsed}s")
        return " · ".join(parts)

    def register_followup_cursor(
        self,
        agent: str,
        event: MessageEvent,
        route: Route,
        transcript_path: str,
        transcript_offset: int,
        source_agent: str | None,
        handoff_depth: int,
    ) -> None:
        """Register a transcript cursor for the followup_loop to watch."""
        key = f"{agent}:{event.chat_id}"
        with self._followup_lock:
            self._followup_cursors[key] = (
                transcript_path, transcript_offset, event, route, source_agent, handoff_depth,
            )
        # Persist so a bridge restart keeps watching for the teammate/subagent
        # reply, which is produced in tmux and survives the restart on its own.
        self._persist_followup_cursors()

    def _followup_state_path(self) -> Path:
        return self.settings.state_dir / "followup_cursors.json"

    def _persist_followup_cursors(self) -> None:
        with self._followup_lock:
            snapshot = dict(self._followup_cursors)
        data = {}
        for key, (path, offset, event, route, source_agent, depth) in snapshot.items():
            data[key] = {
                "transcript_path": path,
                "offset": offset,
                "event": {
                    "event_id": event.event_id, "chat_id": event.chat_id,
                    "chat_type": event.chat_type, "content": event.content,
                    "sender_id": event.sender_id, "message_id": event.message_id,
                    "message_type": event.message_type, "create_time": event.create_time,
                },
                "route": {
                    "kind": route.kind, "text": route.text,
                    "agent": route.agent, "broadcast": route.broadcast,
                },
                "source_agent": source_agent, "handoff_depth": depth,
            }
        try:
            path = self._followup_state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            LOGGER.warning("failed to persist followup cursors", exc_info=True)

    def _load_followup_cursors(self) -> None:
        path = self._followup_state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        restored: dict[str, tuple] = {}
        for key, entry in data.items():
            try:
                ev, rt = entry["event"], entry["route"]
                event = MessageEvent(
                    event_id=ev.get("event_id", ""), chat_id=ev.get("chat_id", ""),
                    chat_type=ev.get("chat_type", "group"), content=ev.get("content", ""),
                    sender_id=ev.get("sender_id", ""), message_id=ev.get("message_id", ""),
                    message_type=ev.get("message_type", ""), create_time=ev.get("create_time", ""),
                )
                route = Route(
                    kind=rt.get("kind", "agent"), text=rt.get("text", ""),
                    agent=rt.get("agent"), broadcast=bool(rt.get("broadcast", False)),
                )
                restored[key] = (
                    entry["transcript_path"], int(entry["offset"]), event, route,
                    entry.get("source_agent"), int(entry.get("handoff_depth", 0)),
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        if restored:
            with self._followup_lock:
                for key, value in restored.items():
                    self._followup_cursors.setdefault(key, value)
            LOGGER.info("restored %d followup cursor(s) from disk", len(restored))

    def start_followup_worker(self, stop_event: threading.Event) -> threading.Thread | None:
        if self.default_agent != "claude":
            return None
        worker = threading.Thread(
            target=self.followup_loop,
            args=(stop_event,),
            name=f"pla-claude-followups",
            daemon=True,
        )
        worker.start()
        return worker

    def followup_loop(self, stop_event: threading.Event) -> None:
        """Persistent loop that polls all registered transcript cursors for
        follow-up replies (e.g. subagent results after the first end_turn)."""
        sent: dict[str, set[str]] = {}  # key -> set of sent text signatures
        while not stop_event.is_set():
            stop_event.wait(3)
            with self._followup_lock:
                snapshot = dict(self._followup_cursors)
            dirty = False
            for key, (path, cur, event, route, source_agent, depth) in snapshot.items():
                try:
                    text, new_cur = self.agents.claude_session.poll_followup_reply(path, cur, timeout=0)
                except Exception:
                    continue
                # Update cursor even if no result (advances past consumed lines).
                if new_cur != cur:
                    dirty = True
                with self._followup_lock:
                    if key in self._followup_cursors:
                        old = self._followup_cursors[key]
                        self._followup_cursors[key] = (old[0], new_cur, *old[2:])
                if text is None or self.is_no_reply(text):
                    continue
                sig = text[:200]
                key_sent = sent.setdefault(key, set())
                if sig in key_sent:
                    continue
                if self.settings.enable_memory:
                    recent = self.memory.context(event.chat_id)
                    if sig in recent:
                        key_sent.add(sig)
                        continue
                key_sent.add(sig)
                agent = key.split(":", 1)[0]
                LOGGER.info("follow-up reply from %s in %s", agent, event.chat_id)
                # Follow-up replies are already complete — use a plain card
                # (not streaming) to avoid sequence issues from rapid create+finalize.
                if self.settings.send_progress:
                    agent_name = self.agent_display_name(agent)
                    footer = self._turn_card_footer(TurnCard(
                        "", event.chat_id, agent, agent_name,
                        self.chat_model_label(event.chat_id, agent),
                        self.chat_effort_label(event.chat_id, agent),
                        time.time(),
                    ))
                    done_card = turn_reply_card(
                        agent_name, "done", text[:self.settings.max_message_chars],
                        model=self.chat_model_label(event.chat_id, agent),
                        effort=self.chat_effort_label(event.chat_id, agent),
                        started_at=time.time(),
                    )
                    try:
                        result = self.lark.send_card(event.chat_id, done_card)
                        msg_id = first_field(result, "message_id")
                        LOGGER.info("follow-up card sent for %s in %s: %s", agent, event.chat_id, msg_id or "no msg_id")
                        if msg_id:
                            self.outbox.remember_message_id(event.chat_id, str(msg_id), agent=agent, discussion_trigger=False)
                    except LarkCLIError as exc:
                        LOGGER.warning("follow-up card send failed: %s", exc)
                    if self.settings.enable_memory:
                        self.memory.append_assistant(event.chat_id, agent, text)
                    if self.should_enqueue_teammate_handoff(route, text, depth):
                        self.enqueue_teammate_handoff(
                            event, source_agent=agent, reply=text,
                            inbound_source_agent=source_agent, handoff_depth=depth,
                        )
                else:
                    self.send_bridge_markdown(
                        event.chat_id, text, agent=agent, discussion_trigger=False,
                    )
                    if self.settings.enable_memory:
                        self.memory.append_assistant(event.chat_id, agent, text)
                    if self.should_enqueue_teammate_handoff(route, text, depth):
                        self.enqueue_teammate_handoff(
                            event, source_agent=agent, reply=text,
                            inbound_source_agent=source_agent, handoff_depth=depth,
                        )
            # Checkpoint advanced offsets so a restart resumes from the right
            # place and re-scans anything written while the bridge was down.
            if dirty:
                self._persist_followup_cursors()

    def update_turn_card(self, card: TurnCard | None, detail: str) -> None:
        if card:
            self._render_turn_card(card, "running", detail)

    def finalize_turn_reply(
        self,
        card: TurnCard,
        route: Route,
        event: MessageEvent,
        text: str,
        *,
        source_agent: str | None,
        handoff_depth: int,
    ) -> None:
        if self.is_no_reply(text):
            self._render_turn_card(card, "skipped", "—")
            with self._turn_card_lock:
                self._active_turn_cards.pop(f"{card.agent}:{card.chat_id}", None)
            return
        limit = self.settings.max_message_chars
        head = text if len(text) <= limit else text[:limit].rstrip() + "\n\n…（下接）"
        if not self._render_turn_card(card, "done", head):
            # Card update failed — deliver the full reply as text so it is never lost.
            self.send_bridge_markdown(card.chat_id, text, agent=route.agent, discussion_trigger=False)
        elif len(text) > limit:
            self.send_bridge_markdown(card.chat_id, text[limit:], agent=route.agent, discussion_trigger=False)
        if route.kind in {"agent", "debate"}:
            self.relay_artifacts(card.chat_id, text, self.chat_workspace(card.chat_id))
        if self.should_enqueue_teammate_handoff(route, text, handoff_depth):
            self.enqueue_teammate_handoff(
                event,
                source_agent=route.agent,  # type: ignore[arg-type]
                reply=text,
                inbound_source_agent=source_agent,
                handoff_depth=handoff_depth,
            )
        self.outbox.remember(card.chat_id, text, limit, agent=route.agent, discussion_trigger=False)
        self.outbox.remember_message_id(card.chat_id, card.message_id, agent=route.agent, discussion_trigger=False)
        if self.settings.enable_memory:
            self.memory.append_assistant(card.chat_id, route.agent, text)
        # Release the card slot so the next queued message can get its own card.
        with self._turn_card_lock:
            self._active_turn_cards.pop(f"{card.agent}:{card.chat_id}", None)

    def start_agent_status(
        self,
        chat_id: str,
        agent: str,
        workspace: Path,
        model_text: str,
        effort_text: str,
    ) -> StatusHandle | None:
        agent_name = self.agent_display_name(agent)
        started_at = time.time()
        handle = StatusHandle(
            self,
            chat_id,
            None,
            agent,
            agent_name,
            workspace,
            model_text,
            effort_text,
            started_at,
        )
        message_id = self.ensure_status_dashboard_message(chat_id)
        if not message_id:
            self.send_progress_markdown(
                chat_id,
                f"{agent_name} is processing...\n\n"
                f"- workspace: `{workspace}`\n"
                f"- model: `{model_text}`\n"
                f"- effort: `{effort_text}`",
            )
            return None
        handle.message_id = message_id
        handle.update("running", "Starting assistant session turn.")
        return handle

    def ensure_status_dashboard_message(self, chat_id: str) -> str | None:
        snapshot_for_create = self.status_dashboards.snapshot(chat_id)

        def create_message() -> str | None:
            card = StatusDashboardCard(
                snapshot_for_create,
                dashboard_url=self.dashboard_url(chat_id),
            ).to_card()
            try:
                result = self.dashboard_lark().send_card(chat_id, card)
            except LarkCLIError as exc:
                LOGGER.warning("status dashboard card send failed: %s", exc)
                return None
            message_id = first_field(result, "message_id") or first_field(result, "id")
            if not message_id:
                return None
            self.outbox.remember_message_id(
                chat_id,
                str(message_id),
                agent="dashboard",
                discussion_trigger=False,
            )
            return str(message_id)

        message_id, created = self.status_dashboards.ensure_message_id(chat_id, create_message)
        if message_id and created:
            self.pin_status_dashboard_message(chat_id, message_id)
        self.ensure_status_dashboard_tab(chat_id)
        return message_id

    def pin_status_dashboard_message(self, chat_id: str, message_id: str) -> None:
        # Disabled: pinned status cards are deprecated in favor of per-turn cards.
        pass

    def update_status_dashboard(self, handle: StatusHandle, state: str, detail: str) -> None:
        snapshot = self.status_dashboards.update_status(
            handle.chat_id,
            handle.agent,
            display_name=handle.agent_name,
            state=state,
            detail=detail,
            workspace=str(handle.workspace),
            model=handle.model,
            effort=handle.effort,
            started_at=handle.started_at,
        )
        message_id = snapshot.message_id or handle.message_id
        if not message_id:
            return
        card = StatusDashboardCard(snapshot, dashboard_url=self.dashboard_url(handle.chat_id)).to_card()
        try:
            self.dashboard_lark().update_card(message_id, card)
        except LarkCLIError as exc:
            LOGGER.warning("status dashboard update failed for %s: %s", message_id, exc)
        self.ensure_status_dashboard_tab(handle.chat_id, snapshot)

    def ensure_status_dashboard_tab(self, chat_id: str, snapshot=None) -> None:
        if not self.settings.dashboard_tab_enabled:
            return
        snapshot = snapshot or self.status_dashboards.snapshot(chat_id)
        tab_name = self.settings.dashboard_tab_name

        def create_doc() -> tuple[str | None, str | None]:
            try:
                result = self.dashboard_lark().create_doc(StatusDashboardDoc(snapshot, tab_name).to_xml())
            except LarkCLIError as exc:
                LOGGER.warning("status dashboard doc create failed for %s: %s", chat_id, exc)
                return None, None
            doc_url = doc_url_from_result(result)
            doc_token = doc_token_from_result(result)
            if not doc_url:
                LOGGER.warning("status dashboard doc create returned no URL for %s: %s", chat_id, result)
                return None, doc_token
            return doc_url, doc_token

        doc_url, _ = self.status_dashboards.ensure_status_doc(chat_id, create_doc)
        if not doc_url:
            return
        snapshot = self.status_dashboards.snapshot(chat_id)
        self.update_status_dashboard_doc(snapshot)

        def create_tab() -> str | None:
            tab_id = self.find_status_dashboard_tab_id(chat_id, tab_name)
            if tab_id:
                try:
                    self.dashboard_lark().update_chat_tab(chat_id, tab_id, tab_name, doc_url)
                except LarkCLIError as exc:
                    LOGGER.warning("status dashboard tab update failed for %s: %s", chat_id, exc)
                return tab_id
            try:
                result = self.dashboard_lark().create_chat_tab(chat_id, tab_name, doc_url)
            except LarkCLIError as exc:
                LOGGER.warning("status dashboard tab create failed for %s: %s", chat_id, exc)
                return None
            tab_id = tab_id_from_result(result, tab_name=tab_name, doc_url=doc_url)
            if not tab_id:
                LOGGER.warning("status dashboard tab create returned no tab_id for %s: %s", chat_id, result)
            return tab_id

        self.status_dashboards.ensure_status_tab(chat_id, create_tab)

    def update_status_dashboard_doc(self, snapshot) -> None:
        doc = snapshot.doc_url or snapshot.doc_token
        if not doc:
            return
        try:
            self.dashboard_lark().update_doc(
                doc,
                StatusDashboardDoc(snapshot, self.settings.dashboard_tab_name).to_xml(),
            )
        except LarkCLIError as exc:
            LOGGER.warning("status dashboard doc update failed for %s: %s", snapshot.chat_id, exc)

    def find_status_dashboard_tab_id(self, chat_id: str, tab_name: str) -> str | None:
        try:
            result = self.dashboard_lark().list_chat_tabs(chat_id)
        except LarkCLIError as exc:
            LOGGER.warning("status dashboard tab list failed for %s: %s", chat_id, exc)
            return None
        for tab in chat_tabs_from_result(result):
            if str(tab.get("tab_name") or "") == tab_name:
                tab_id = str(tab.get("tab_id") or "")
                return tab_id or None
        return None

    def dashboard_lark(self) -> LarkCLI:
        return self._dashboard_lark or self.lark

    def dashboard_url(self, chat_id: str) -> str | None:
        base_url = self.settings.dashboard_public_url
        if not base_url:
            return None
        return f"{base_url.rstrip('/')}/dashboard/{chat_id}"

    def send_reply(
        self,
        event: MessageEvent,
        route: Route,
        reply: str,
        *,
        source_agent: str | None = None,
        handoff_depth: int = 0,
    ) -> None:
        reply_agent = self.reply_agent_name(route)
        feishu_discussion_trigger = (
            route.kind == "agent"
            and not self.settings.direct_agent_handoff
            and self.settings.enable_agent_discussion
        )
        self.send_bridge_markdown(
            event.chat_id,
            reply,
            agent=reply_agent,
            discussion_trigger=feishu_discussion_trigger,
        )
        if route.kind in {"agent", "debate"}:
            self.relay_artifacts(event.chat_id, reply, self.chat_workspace(event.chat_id))
        if self.should_enqueue_teammate_handoff(route, reply, handoff_depth):
            self.enqueue_teammate_handoff(
                event,
                source_agent=route.agent,  # type: ignore[arg-type]
                reply=reply,
                inbound_source_agent=source_agent,
                handoff_depth=handoff_depth,
            )

    def should_enqueue_teammate_handoff(
        self, route: Route, reply: str, handoff_depth: int
    ) -> bool:
        if route.kind != "agent" or not route.agent:
            return False
        # /debate seeds the first exchange even if the opener doesn't address the peer.
        if handoff_depth == 0 and route.text.lstrip().startswith(DEBATE_BROADCAST_PREFIX):
            return True
        # Otherwise a bot hands the thread to its teammate only by explicitly
        # @-addressing it in the reply. The exchange self-limits: it continues
        # only while they keep @-ing each other, and the depth/window caps in
        # enqueue_teammate_handoff are the hard backstop.
        peer = "claude" if route.agent == "codex" else "codex"
        return peer in addressed_agents(reply)

    def enqueue_teammate_handoff(
        self,
        event: MessageEvent,
        *,
        source_agent: str,
        reply: str,
        inbound_source_agent: str | None,
        handoff_depth: int,
    ) -> None:
        if not self.settings.enable_agent_discussion or not self.settings.direct_agent_handoff:
            return
        if source_agent not in {"codex", "claude"}:
            return
        if inbound_source_agent == source_agent:
            return
        # Dedup: multiple code paths (finalize_turn_reply, followup_loop,
        # send_reply) can try to enqueue the same reply as a handoff.
        sig = f"{source_agent}:{event.chat_id}:{reply[:200]}"
        if sig in self._recent_handoff_sigs:
            return
        self._recent_handoff_sigs.add(sig)
        # Keep the set bounded.
        if len(self._recent_handoff_sigs) > 200:
            self._recent_handoff_sigs = set(list(self._recent_handoff_sigs)[-100:])
        next_depth = handoff_depth + 1
        if next_depth > max(0, self.settings.max_agent_discussion_turns):
            LOGGER.info("skipping handoff from %s; discussion depth cap reached", source_agent)
            return
        recent = self.handoffs.recent_count(
            event.chat_id,
            self.settings.agent_discussion_window_seconds,
        )
        if recent >= self.settings.max_agent_discussion_turns:
            LOGGER.info("skipping handoff from %s; discussion window cap reached", source_agent)
            return
        target_agent = "claude" if source_agent == "codex" else "codex"
        handoff_id = self.handoffs.enqueue(
            event.chat_id,
            source_agent=source_agent,
            target_agent=target_agent,
            content=reply,
            origin_event_id=event.event_id,
            origin_message_id=event.message_id,
            sender_id=event.sender_id,
            depth=next_depth,
        )
        LOGGER.info(
            "queued handoff %s from %s to %s in %s",
            handoff_id,
            source_agent,
            target_agent,
            event.chat_id,
        )

    def relay_artifacts(self, chat_id: str, markdown: str, workspace: Path) -> None:
        if not self.settings.enable_artifacts or self.settings.max_artifacts <= 0:
            return
        relay = ArtifactRelay(
            self.settings.workspace,
            self.settings.state_dir,
            self.settings.workspace_roots,
            max_artifacts=self.settings.max_artifacts,
        )
        try:
            artifacts = relay.collect(markdown, workspace)
        except Exception as exc:
            LOGGER.warning("artifact collection failed for %s: %s", chat_id, exc)
            return
        if not artifacts:
            return
        if self.settings.send_progress:
            names = ", ".join(artifact.name for artifact in artifacts[:4])
            suffix = "" if len(artifacts) <= 4 else f", +{len(artifacts) - 4} more"
            self.send_progress_markdown(
                chat_id,
                f"Uploading {len(artifacts)} artifact(s): {names}{suffix}",
            )
        failures: list[str] = []
        for artifact in artifacts:
            try:
                if artifact.kind == "image":
                    result = self.lark.send_image(chat_id, artifact.upload_path)
                else:
                    result = self.lark.send_file(chat_id, artifact.upload_path)
                message_id = first_field(result, "message_id")
                if message_id:
                    self.outbox.remember_message_id(
                        chat_id,
                        str(message_id),
                        agent=self.settings.agent_mode,
                        discussion_trigger=False,
                    )
            except LarkCLIError as exc:
                LOGGER.warning("artifact upload failed for %s: %s", artifact.path, exc)
                failures.append(artifact.name)
        if failures:
            self.send_progress_markdown(
                chat_id,
                "Artifact upload failed for: " + ", ".join(failures),
            )

    def is_no_reply(self, reply: str) -> bool:
        stripped = reply.strip()
        token = self.settings.no_reply_token
        return stripped == token or stripped.startswith(token + " ") or stripped.startswith(token + "—") or stripped.startswith(token + "\n")


def format_standalone_upload_reply(downloaded: list[tuple[str, Path]]) -> str:
    lines = [
        "文件已保存到服务器，本次不会转发给 Codex/Claude。",
        "",
        "本地路径：",
    ]
    for name, path in downloaded:
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "之后可以直接引用这个文件，或把上面的路径发给指定助手。",
        ]
    )
    return "\n".join(lines)
