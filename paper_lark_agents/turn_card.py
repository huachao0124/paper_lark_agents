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

# A Feishu interactive card holds far more than a plain message; cap the
# accumulated body well under the hard limit so appended follow-ups stay safe.
CARD_BODY_LIMIT = 12000


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
    stream_expired: bool = False


class TurnCardManager:
    def __init__(self, lark: Any, settings: Settings):
        self._lark = lark
        self._settings = settings
        self._lock = threading.Lock()
        self._active: dict[str, TurnCard] = {}
        self._accumulated: dict[str, str] = {}
        self._creating: set[str] = set()
        # The most recently finalized "done" plain card per (agent, chat), kept
        # so follow-up replies in the same turn can be appended into it (with a
        # divider) instead of spawning separate messages. Cleared when a new
        # turn's card is acquired.
        self._done_cards: dict[str, TurnCard] = {}
        self._done_bodies: dict[str, str] = {}

    @staticmethod
    def _key(agent: str, chat_id: str) -> str:
        return f"{agent}:{chat_id}"

    def _supports_streaming(self) -> bool:
        return hasattr(self._lark, "create_streaming_card")

    def active_for(self, agent: str, chat_id: str) -> TurnCard | None:
        with self._lock:
            return self._active.get(self._key(agent, chat_id))

    def adopt(self, card: TurnCard) -> TurnCard:
        """Register a card reconstructed from persisted state (pending run
        records) as the active card for its key. If another card is already
        registered, that one wins — the caller must use it instead."""
        key = self._key(card.agent, card.chat_id)
        with self._lock:
            existing = self._active.get(key)
            if existing is not None:
                return existing
            self._active[key] = card
            return card

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
            elif key in self._creating:
                return None
            self._creating.add(key)
            # A new turn starts here — the previous turn's done card is closed
            # and must not receive this turn's follow-ups.
            self._done_cards.pop(key, None)
            self._done_bodies.pop(key, None)
        card: TurnCard | None = None
        try:
            if old and force:
                LOGGER.info("interrupting old card %s for %s", old.message_id[:12], key)
                self._interrupt_card(old)
            if strategy == "auto":
                # Plain cards can be updated indefinitely (no 10-min streaming
                # timeout, no schemaV2 mismatch on fallback). Streaming's only
                # benefit was the typewriter animation, but the timeout/recovery
                # churn outweighs it. Header colour changes via plain PATCH work
                # fine for Running→Done transitions.
                strategy = "plain"
            card = self._create_card(chat_id, agent, agent_name, model, effort, strategy)
        finally:
            # Always release the creation guard — a leaked key would block
            # every future card for this (agent, chat) until restart.
            with self._lock:
                self._creating.discard(key)
                if card:
                    self._active[key] = card
        return card

    def update(self, card: TurnCard, detail: str) -> None:
        if not card:
            return
        if not card.message_id and not self._recover_dead_card(card):
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
            if not card.stream_expired:
                try:
                    self._lark.stream_card_content(card_id, combined, sequence=seq)
                    return
                except Exception as exc:
                    LOGGER.warning("streaming update failed for %s, muting: %s", card_id, exc)
                    card.stream_expired = True
            # Streaming expired or was never used — silently skip. The card
            # freezes at its last streamed content; finalize will update it.
            return
        # Build the card body from two parts:
        #   1. replies — follow-up reply text, accumulated with dividers (sticky)
        #   2. progress — latest step header (replaced each tick)
        # This way replies are never overwritten by tool_use progress.
        if card.message_id:
            with self._lock:
                state = self._accumulated.get(card.message_id)
                if not isinstance(state, dict):
                    state = {"replies": "", "progress": ""}
                    self._accumulated[card.message_id] = state
                state["progress"] = detail
                combined = state["replies"]
                if combined and detail:
                    combined = f"{combined}\n\n---\n\n{detail}"
                elif detail:
                    combined = detail
        else:
            combined = detail
        if len(combined) > CARD_BODY_LIMIT:
            combined = combined[-CARD_BODY_LIMIT:]
        self._update_plain(card, combined)
        if not card.message_id and self._recover_dead_card(card):
            self._update_plain(card, combined)

    def finalize(self, card: TurnCard, state: str, body: str) -> bool:
        if not card:
            return False
        key = self._key(card.agent, card.chat_id)
        with self._lock:
            if self._active.get(key) is card:
                self._active.pop(key)
            if card.card_id:
                self._accumulated.pop(card.card_id, None)
            # Prepend accumulated intermediate replies to the final body.
            # The final reply usually equals the LAST intermediate message
            # (codex task_complete repeats the last agent_message) — dedup by
            # normalized suffix so it isn't shown twice.
            acc_replies = ""
            if card.message_id:
                state_data = self._accumulated.pop(card.message_id, None)
                if isinstance(state_data, dict):
                    acc_replies = state_data.get("replies", "")
            if acc_replies and body and state == "done":
                norm_acc = "".join(acc_replies.split())
                norm_body = "".join(body.split())
                if norm_acc.endswith(norm_body):
                    body = acc_replies
                else:
                    body = f"{acc_replies}\n\n---\n\n{body}"
            card.sequence += 1
        if card.card_id and card.strategy == "streaming":
            try:
                footer = self._footer(card)
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
                # Streaming messages are schemaV2; updating them with the
                # schemaV1 plain card fails. Send a terminal card instead.
                self._delete_message(card.message_id)
                card.message_id = ""
                card.card_id = None
                card.strategy = "plain"
                result = self._send_terminal_card(card, state, body)
                if result:
                    self._remember_done_card(key, card, state, body)
                return result
        # Plain card: update in place — PATCH can change both header and body.
        if card.message_id:
            LOGGER.info("finalize plain card %s -> %s (%d chars)", card.message_id[:16], state, len(body))
            done = turn_reply_card(
                card.agent_name, state, body,
                model=card.model, effort=card.effort, started_at=card.started_at,
            )
            try:
                self._lark.update_card(card.message_id, done)
                self._remember_done_card(key, card, state, body)
                return True
            except Exception as exc:
                LOGGER.warning("plain card finalize failed for %s: %s", card.message_id, exc)
        result = self._send_terminal_card(card, state, body)
        if result:
            self._remember_done_card(key, card, state, body)
        return result

    def _remember_done_card(self, key: str, card: TurnCard, state: str, body: str) -> None:
        """Remember a finalized 'done' plain card so same-turn follow-ups can be
        appended into it instead of becoming separate messages."""
        if state != "done" or not card.message_id or card.strategy == "streaming":
            return
        with self._lock:
            self._done_cards[key] = card
            self._done_bodies[key] = body

    def update_with_reply(self, card: TurnCard, reply_text: str) -> None:
        """Append reply text to the Running card's sticky replies section.

        Unlike update() (which sets the volatile progress line), this adds
        content that persists across subsequent progress ticks — so a reply
        is never overwritten by a tool_use step header."""
        if not card or not card.message_id:
            return
        with self._lock:
            state = self._accumulated.get(card.message_id)
            if not isinstance(state, dict):
                state = {"replies": "", "progress": ""}
                self._accumulated[card.message_id] = state
            if state["replies"]:
                state["replies"] = f"{state['replies']}\n\n---\n\n{reply_text}"
            else:
                state["replies"] = reply_text
            combined = state["replies"]
            if state["progress"]:
                combined = f"{combined}\n\n---\n\n{state['progress']}"
        if len(combined) > CARD_BODY_LIMIT:
            combined = combined[-CARD_BODY_LIMIT:]
        self._update_plain(card, combined)

    def append_follow_up(self, agent: str, chat_id: str, extra_text: str) -> bool:
        """Append a same-turn follow-up reply into the turn's done card, joined
        by a divider. Returns False (caller should fall back to a message) when
        there is no appendable card or the card would exceed the size limit."""
        if not extra_text:
            return False
        key = self._key(agent, chat_id)
        with self._lock:
            card = self._done_cards.get(key)
            body = self._done_bodies.get(key, "")
        if not card or not card.message_id or card.strategy == "streaming":
            return False
        new_body = f"{body}\n\n---\n\n{extra_text}" if body else extra_text
        if len(new_body) > CARD_BODY_LIMIT:
            return False
        rendered = turn_reply_card(
            card.agent_name, "done", new_body,
            model=card.model, effort=card.effort, started_at=card.started_at,
        )
        try:
            self._lark.update_card(card.message_id, rendered)
        except Exception as exc:
            LOGGER.warning("append follow-up to card %s failed: %s", card.message_id, exc)
            return False
        with self._lock:
            self._done_bodies[key] = new_body
        return True

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
                    raise RuntimeError("streaming card send returned no message_id")
                return TurnCard(
                    message_id, chat_id, agent, agent_name,
                    model, effort, started_at,
                    card_id=card_id, sequence=0, strategy="streaming",
                )
            except Exception as exc:
                LOGGER.warning("streaming card create failed, falling back to plain card: %s", exc)
                strategy = "plain"
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
        if not card.message_id or card.stream_expired:
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

    _MAX_RECOVERIES = 1  # at most one recovery per card lifetime

    def _recover_dead_card(self, card: TurnCard) -> bool:
        # Cap recovery attempts — without this, every progress tick after a
        # failed update spawns a new card, producing the "4 identical Done
        # cards" clutter.
        recoveries = getattr(card, "_recoveries", 0)
        if recoveries >= self._MAX_RECOVERIES:
            return False
        key = self._key(card.agent, card.chat_id)
        with self._lock:
            active = self._active.get(key)
            if active is not None and active is not card:
                return False
            self._active[key] = card
        new = self._create_card(
            card.chat_id, card.agent, card.agent_name,
            card.model, card.effort, "plain",
        )
        if not new:
            return False
        with self._lock:
            card.message_id = new.message_id
            card.strategy = "plain"
            card.card_id = None
        card._recoveries = recoveries + 1  # type: ignore[attr-defined]
        LOGGER.info("recovered dead card for %s:%s", card.agent, card.chat_id)
        return True

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
        old_card_id = card.card_id
        try:
            new_card_id = self._lark.create_streaming_card(f"{card.agent_name} · Running")
            new_msg_id = self._lark.send_streaming_card(card.chat_id, new_card_id)
            self._delete_message(old_msg)
            with self._lock:
                card.card_id = new_card_id
                card.message_id = new_msg_id
                card.sequence = 1
                card.started_at = time.time()
                if old_card_id:
                    self._accumulated.pop(old_card_id, None)
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
        # Mark the card dead first — a concurrent progress tick must not try to
        # stream/update this card while we're finalizing it.
        card.stream_expired = True
        if card.card_id and card.strategy == "streaming":
            try:
                self._lark.finalize_streaming_card(
                    card.card_id,
                    title=f"{card.agent_name} · Done",
                    template="green",
                    sequence=card.sequence + 1,
                )
                return
            except Exception:
                # Streaming finalize failed — delete the streaming message and
                # send a plain terminal card. Do NOT fall through to plain
                # update_card (schemaV2 card can not change schemaV1).
                self._delete_message(card.message_id)
                card.message_id = ""
                card.card_id = None
                card.strategy = "plain"
                self._send_terminal_card(card, "done", "—")
                return
        # Plain card: update in place to Done, keeping accumulated replies + progress.
        if card.message_id:
            with self._lock:
                state = self._accumulated.get(card.message_id)
                if isinstance(state, dict):
                    body = state.get("replies", "")
                    if state.get("progress"):
                        body = f"{body}\n\n---\n\n{state['progress']}" if body else state["progress"]
                elif isinstance(state, str):
                    body = state
                else:
                    body = "—"
                body = body or "—"
            done = turn_reply_card(
                card.agent_name, "done", body,
                model=card.model, effort=card.effort, started_at=card.started_at,
            )
            try:
                self._lark.update_card(card.message_id, done)
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
                resp = self._lark._client.im.v1.message.delete(req)
                if resp.success():
                    LOGGER.info("deleted message %s", message_id[:16])
                else:
                    LOGGER.warning("delete message %s failed: %s", message_id[:16], resp.msg)
        except Exception as exc:
            LOGGER.warning("delete message %s error: %s", message_id[:16], exc)

    def _footer(self, card: TurnCard) -> str:
        elapsed = max(0, int(time.time() - card.started_at))
        parts = []
        if card.model:
            parts.append(card.model)
        if card.effort:
            parts.append(card.effort)
        parts.append(f"{elapsed}s")
        return " · ".join(parts)
