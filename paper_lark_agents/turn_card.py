"""Unified turn card lifecycle manager.

One card per (agent, chat_id). All card creation, update, and finalization
goes through TurnCardManager. Thread-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

from .config import Settings
from .status_card import turn_reply_card

LOGGER = logging.getLogger(__name__)


@dataclass
class TurnCard:
    message_id: str
    chat_id: str
    agent: str
    agent_name: str
    model: str
    effort: str
    started_at: float
    card_id: str | None = None
    sequence: int = 0
    strategy: str = "plain"


class TurnCardManager:
    def __init__(self, lark: Any, settings: Settings):
        self._lark = lark
        self._settings = settings
        self._lock = threading.Lock()
        self._active: dict[str, TurnCard] = {}
        self._accumulated: dict[str, str] = {}

    @staticmethod
    def _key(agent: str, chat_id: str) -> str:
        return f"{agent}:{chat_id}"

    def _supports_streaming(self) -> bool:
        return hasattr(self._lark, "create_streaming_card")

    def active_for(self, agent: str, chat_id: str) -> TurnCard | None:
        with self._lock:
            return self._active.get(self._key(agent, chat_id))

    def clear_all(self) -> int:
        with self._lock:
            count = len(self._active)
            self._active.clear()
            self._accumulated.clear()
        return count

    def acquire(
        self,
        chat_id: str,
        agent: str,
        agent_name: str,
        model: str,
        effort: str,
        *,
        force: bool = False,
        strategy: str = "auto",
    ) -> TurnCard | None:
        key = self._key(agent, chat_id)
        with self._lock:
            old = self._active.get(key)
            if old:
                if not force:
                    return None
                self._active.pop(key, None)
        if old and force:
            self._interrupt_card(old)
        if strategy == "auto":
            strategy = "streaming" if self._supports_streaming() else "plain"
        card = self._create_card(chat_id, agent, agent_name, model, effort, strategy)
        if not card:
            return None
        with self._lock:
            self._active[key] = card
        return card

    def update(self, card: TurnCard, detail: str) -> None:
        if not card or not card.message_id:
            return
        with self._lock:
            card.sequence += 1
            seq = card.sequence
            strategy = card.strategy
            card_id = card.card_id
        if strategy == "streaming" and card_id:
            acc_key = card_id
            with self._lock:
                prev = self._accumulated.get(acc_key, "")
                if prev and detail not in prev:
                    combined = f"{prev}\n{detail}"
                    if len(combined) > 3000:
                        combined = combined[-3000:]
                else:
                    combined = detail
                self._accumulated[acc_key] = combined
            try:
                self._lark.stream_card_content(card_id, combined, sequence=seq)
                return
            except Exception as exc:
                LOGGER.warning("streaming update failed for %s: %s", card_id, exc)
                if not self._renew_streaming_card(card, detail, seq):
                    self._downgrade_to_plain(card)
                    new = self._create_card(
                        card.chat_id, card.agent, card.agent_name,
                        card.model, card.effort, "plain",
                    )
                    if new:
                        with self._lock:
                            card.message_id = new.message_id
                        return
        self._update_plain(card, detail)

    def finalize(self, card: TurnCard, state: str, body: str) -> bool:
        if not card:
            return False
        key = self._key(card.agent, card.chat_id)
        with self._lock:
            self._active.pop(key, None)
            if card.card_id:
                self._accumulated.pop(card.card_id, None)
        if card.card_id and card.strategy == "streaming":
            try:
                footer = self._footer(card)
                card.sequence += 1
                template = {"done": "green", "failed": "red", "skipped": "grey"}.get(state, "blue")
                title = f"{card.agent_name} · {state.title()}"
                self._lark.finalize_streaming_card(
                    card.card_id,
                    final_content=body,
                    title=title,
                    template=template,
                    sequence=card.sequence,
                    footer=footer,
                )
                return True
            except Exception as exc:
                LOGGER.warning("streaming finalize failed for %s: %s", card.card_id, exc)
                self._delete_message(card.message_id)
        return self._send_terminal_card(card, state, body)

    def send_done_card(
        self,
        chat_id: str,
        agent: str,
        agent_name: str,
        body: str,
        *,
        model: str = "",
        effort: str = "",
    ) -> str | None:
        key = self._key(agent, chat_id)
        with self._lock:
            old = self._active.pop(key, None)
        if old:
            self._interrupt_card(old)
        done = turn_reply_card(agent_name, "done", body, model=model, effort=effort, started_at=time.time())
        try:
            from .lark_cli import first_field, LarkCLIError
            result = self._lark.send_card(chat_id, done)
            msg_id = first_field(result, "message_id") or first_field(result, "id")
            return str(msg_id) if msg_id else None
        except Exception as exc:
            LOGGER.warning("send_done_card failed: %s", exc)
            return None

    # ---- internal helpers ----

    def _create_card(
        self, chat_id: str, agent: str, agent_name: str,
        model: str, effort: str, strategy: str,
    ) -> TurnCard | None:
        started_at = time.time()
        from .lark_cli import first_field, LarkCLIError
        if strategy == "streaming":
            try:
                card_id = self._lark.create_streaming_card(f"{agent_name} · Running")
                message_id = self._lark.send_streaming_card(chat_id, card_id)
                if not message_id:
                    return None
                return TurnCard(
                    message_id, chat_id, agent, agent_name,
                    model, effort, started_at,
                    card_id=card_id, sequence=0, strategy="streaming",
                )
            except Exception as exc:
                LOGGER.warning("streaming card create failed: %s", exc)
                return None
        card = turn_reply_card(agent_name, "running", "✻ 思考中", model=model, effort=effort, started_at=started_at)
        try:
            result = self._lark.send_card(chat_id, card)
            message_id = first_field(result, "message_id") or first_field(result, "id")
            if not message_id:
                return None
            return TurnCard(
                str(message_id), chat_id, agent, agent_name,
                model, effort, started_at, strategy="plain",
            )
        except Exception as exc:
            LOGGER.warning("plain card create failed: %s", exc)
            return None

    def _update_plain(self, card: TurnCard, detail: str) -> None:
        if not card.message_id:
            LOGGER.debug("skip plain update: no message_id")
            return
        rendered = turn_reply_card(
            card.agent_name, "running", detail,
            model=card.model, effort=card.effort, started_at=card.started_at,
        )
        try:
            self._lark.update_card(card.message_id, rendered)
        except Exception as exc:
            err = str(exc)
            LOGGER.warning("plain card update failed for %s: %s", card.message_id, exc)
            if "schemaV2" in err or "200830" in err or "withdrawn" in err or "230011" in err:
                card.message_id = ""

    def _send_terminal_card(self, card: TurnCard, state: str, body: str) -> bool:
        done = turn_reply_card(
            card.agent_name, state, body,
            model=card.model, effort=card.effort, started_at=card.started_at,
        )
        try:
            from .lark_cli import first_field
            result = self._lark.send_card(card.chat_id, done)
            msg_id = first_field(result, "message_id")
            if msg_id:
                card.message_id = str(msg_id)
            return True
        except Exception as exc:
            LOGGER.warning("terminal card send failed: %s", exc)
            return False

    def _renew_streaming_card(self, card: TurnCard, detail: str, seq: int) -> bool:
        old_msg = card.message_id
        try:
            new_card_id = self._lark.create_streaming_card(f"{card.agent_name} · Running")
            new_msg_id = self._lark.send_streaming_card(card.chat_id, new_card_id)
            self._delete_message(old_msg)
            with self._lock:
                card.card_id = new_card_id
                card.message_id = new_msg_id
                card.sequence = 1
                card.started_at = time.time()
                if card.card_id:
                    self._accumulated.pop(card.card_id, None)
            self._lark.stream_card_content(new_card_id, detail, sequence=1)
            return True
        except Exception:
            return False

    def _downgrade_to_plain(self, card: TurnCard) -> None:
        old_msg = card.message_id
        with self._lock:
            card.strategy = "plain"
            card.card_id = None
            card.message_id = ""
        self._delete_message(old_msg)

    def _interrupt_card(self, card: TurnCard) -> None:
        if card.card_id and card.strategy == "streaming":
            try:
                self._lark.finalize_streaming_card(
                    card.card_id,
                    title=f"{card.agent_name} · Interrupted",
                    template="grey",
                    sequence=card.sequence + 1,
                )
                return
            except Exception:
                pass
        self._delete_message(card.message_id)

    def _delete_message(self, message_id: str) -> None:
        if not message_id:
            return
        try:
            if hasattr(self._lark, '_client'):
                from lark_oapi.api.im.v1 import DeleteMessageRequest
                req = DeleteMessageRequest.builder().message_id(message_id).build()
                self._lark._client.im.v1.message.delete(req)
        except Exception:
            pass

    def _footer(self, card: TurnCard) -> str:
        elapsed = max(0, int(time.time() - card.started_at))
        parts = []
        if card.model:
            parts.append(card.model)
        if card.effort:
            parts.append(card.effort)
        parts.append(f"{elapsed}s")
        return " · ".join(parts)
