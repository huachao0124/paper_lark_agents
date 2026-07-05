import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_lark_agents.app import PaperAgentBridge
from paper_lark_agents.turn_card import TurnCard, TurnCardManager
from paper_lark_agents.config import load_settings
from paper_lark_agents.lark_cli import MessageEvent
from paper_lark_agents.router import Route


class FakeLark:
    def __init__(self):
        self.cards = []
        self.updates = []
        self.markdowns = []

    def send_card(self, chat_id, card):
        message_id = f"card_{len(self.cards) + 1}"
        self.cards.append((chat_id, card))
        return {"message_id": message_id}

    def update_card(self, message_id, card):
        self.updates.append((message_id, card))
        return {}

    def send_markdown(self, chat_id, markdown):
        self.markdowns.append((chat_id, markdown))
        return [{"message_id": f"om_{len(self.markdowns)}"}]


class FailingStreamingLark(FakeLark):
    def __init__(self):
        super().__init__()
        self.streaming_cards = []
        self.streaming_messages = []
        self.streaming_updates = []

    def create_streaming_card(self, title):
        card_id = f"stream_{len(self.streaming_cards) + 1}"
        self.streaming_cards.append((card_id, title))
        return card_id

    def send_streaming_card(self, chat_id, card_id):
        message_id = f"stream_msg_{len(self.streaming_messages) + 1}"
        self.streaming_messages.append((chat_id, card_id, message_id))
        return message_id

    def stream_card_content(self, card_id, content, sequence):
        self.streaming_updates.append((card_id, content, sequence))
        raise RuntimeError("streaming mode is closed")

    def finalize_streaming_card(self, *args, **kwargs):
        raise RuntimeError("sequence number compare failed")


class StreamingCreateFailLark(FakeLark):
    def create_streaming_card(self, title):
        raise RuntimeError("cardid is invalid")


class SchemaMismatchOnceLark(FakeLark):
    def update_card(self, message_id, card):
        if message_id == "old_card":
            raise RuntimeError(
                "API error 230099: Failed to create card content, "
                "ext=ErrCode: 200830; ErrMsg: schemaV2 card can not change schemaV1;"
            )
        return super().update_card(message_id, card)


def card_text(value) -> str:
    if isinstance(value, dict):
        return "\n".join(card_text(v) if k != "content" or not isinstance(v, str) else v for k, v in value.items())
    if isinstance(value, list):
        return "\n".join(card_text(v) for v in value)
    return value if isinstance(value, str) else ""


def event() -> MessageEvent:
    return MessageEvent(
        event_id="evt", chat_id="oc_a", chat_type="group", content="q",
        sender_id="ou_user", message_id="om_user", message_type="text",
    )


class TurnCardTests(unittest.TestCase):
    def bridge(self, root: Path):
        env = {
            "PLA_WORKSPACE": str(root), "PLA_WORKSPACE_ROOTS": str(root),
            "PLA_STATE_DIR": str(root / ".state"), "PLA_AGENT_MODE": "codex",
            "PLA_SEND_PROGRESS": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            b = PaperAgentBridge(load_settings(None))
        fake = FakeLark()
        b.lark = fake
        b.cards._lark = fake
        return b

    def test_start_turn_card_sends_and_returns_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            card = b.cards.acquire("oc_a", "codex", "Codex", "gpt", "xhigh")
            self.assertIsInstance(card, TurnCard)
            self.assertEqual(card.message_id, "card_1")
            self.assertEqual(len(b.lark.cards), 1)

    def test_follow_up_appends_into_done_card_with_divider(self):
        lark = FakeLark()
        cards = TurnCardManager(lark, object())
        card = cards.acquire("oc_a", "claude", "Claude", "opus", "max")
        # Main reply finalizes the running card → done (recorded for appends).
        self.assertTrue(cards.finalize(card, "done", "第一段回复"))
        # Two same-turn follow-ups append into the same card, not new messages.
        self.assertTrue(cards.append_follow_up("claude", "oc_a", "第二段补充"))
        self.assertTrue(cards.append_follow_up("claude", "oc_a", "第三段补充"))
        last = card_text(lark.updates[-1][1])
        self.assertIn("第一段回复", last)
        self.assertIn("第二段补充", last)
        self.assertIn("第三段补充", last)
        self.assertIn("---", last)              # divider between sections
        self.assertEqual(lark.markdowns, [])    # nothing spilled to a message

    def test_follow_up_append_falls_back_past_card_limit(self):
        lark = FakeLark()
        cards = TurnCardManager(lark, object())
        card = cards.acquire("oc_a", "claude", "Claude", "opus", "max")
        cards.finalize(card, "done", "甲" * 11000)
        # Appending would exceed CARD_BODY_LIMIT → returns False so the caller
        # spills to a message instead.
        self.assertFalse(cards.append_follow_up("claude", "oc_a", "乙" * 2000))

    def test_new_turn_closes_previous_done_card_for_appends(self):
        lark = FakeLark()
        cards = TurnCardManager(lark, object())
        first = cards.acquire("oc_a", "claude", "Claude", "opus", "max")
        cards.finalize(first, "done", "第一轮")
        # A new turn's card supersedes; the old turn's follow-up must NOT append.
        cards.acquire("oc_a", "claude", "Claude", "opus", "max", force=True)
        self.assertFalse(cards.append_follow_up("claude", "oc_a", "迟到的补充"))

    def test_plain_card_updates_continuously(self):
        lark = FakeLark()
        cards = TurnCardManager(lark, object())

        card = cards.acquire("oc_a", "codex", "Codex", "gpt", "xhigh")
        # Default is plain — no streaming timeout, no schemaV2 issues.
        self.assertEqual(card.strategy, "plain")
        cards.update(card, "第一步")
        cards.update(card, "第二步")
        # Both updates went through as plain card patches.
        self.assertEqual(len(lark.updates), 2)

    def test_streaming_create_failure_falls_back_to_plain_card(self):
        lark = StreamingCreateFailLark()
        cards = TurnCardManager(lark, object())

        card = cards.acquire("oc_a", "codex", "Codex", "gpt", "xhigh")

        self.assertIsInstance(card, TurnCard)
        self.assertEqual(card.strategy, "plain")
        self.assertIsNone(card.card_id)
        self.assertEqual(card.message_id, "card_1")
        self.assertEqual(len(lark.cards), 1)
        self.assertIn("思考中", card_text(lark.cards[-1][1]))

    def test_plain_schema_mismatch_recovers_to_new_card_same_update(self):
        lark = SchemaMismatchOnceLark()
        cards = TurnCardManager(lark, object())
        card = TurnCard("old_card", "oc_a", "codex", "Codex", "gpt", "xhigh", 0.0)
        cards.adopt(card)

        cards.update(card, "步骤 1\n正在读取文件")

        self.assertEqual(card.message_id, "card_1")
        self.assertEqual(card.strategy, "plain")
        self.assertEqual(len(lark.cards), 1)
        self.assertEqual(lark.updates[-1][0], "card_1")
        self.assertIn("正在读取文件", card_text(lark.updates[-1][1]))

    def test_default_strategy_is_plain_not_streaming(self):
        lark = FailingStreamingLark()
        cards = TurnCardManager(lark, object())

        card = cards.acquire("oc_a", "codex", "Codex", "gpt", "xhigh")
        # Default strategy is plain (no streaming timeout/schemaV2 issues).
        self.assertEqual(card.strategy, "plain")
        self.assertIsNone(card.card_id)
        self.assertTrue(cards.finalize(card, "done", "答案"))
        # Finalize updates the plain card in place.
        self.assertTrue(lark.updates)
        self.assertIn("答案", card_text(lark.updates[-1][1]))

    def test_turn_card_uses_own_identity_not_dashboard_profile(self):
        # The card is the agent's own reply, so it must go through the agent's
        # own bot identity (self.lark), never the shared dashboard profile —
        # otherwise Claude's answer would appear sent by Codex.
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            dashboard = FakeLark()
            b._dashboard_lark = dashboard  # distinct identity, as in serve-duo
            card = b.cards.acquire("oc_a", "codex", "Codex", "gpt", "xhigh")
            b.cards.finalize(card, "done", "答案")
            # acquire sends 1 card, finalize updates it in place
            self.assertEqual(len(b.lark.cards), 1)        # sent under own identity
            self.assertTrue(b.lark.updates)               # finalize updates in place
            self.assertEqual(dashboard.cards, [])         # never touched the dashboard profile
            self.assertEqual(dashboard.updates, [])

    def test_finalize_morphs_card_to_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            card = TurnCard("card_1", "oc_a", "codex", "Codex", "gpt", "xhigh", 0.0)
            route = Route("agent", text="q", agent="codex")
            b.finalize_turn_reply(card, route, event(), "这是答案", source_agent=None, handoff_depth=0)
            # TurnCardManager.finalize sends a terminal card for plain strategy
            finalized = b.lark.updates or b.lark.cards
            self.assertTrue(finalized)
            self.assertIn("这是答案", card_text(finalized[-1][1]))
            self.assertIn("这是答案", b.memory.context("oc_a"))
            self.assertEqual(b.lark.markdowns, [])  # no overflow

    def test_finalize_no_reply_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            card = TurnCard("card_1", "oc_a", "codex", "Codex", "gpt", "xhigh", 0.0)
            b.finalize_turn_reply(card, Route("agent", text="q", agent="codex"), event(), "[NO_REPLY]", source_agent=None, handoff_depth=0)
            # TurnCardManager.finalize updates the card in place (or sends a new one)
            self.assertTrue(b.lark.updates or b.lark.cards)
            self.assertEqual(b.memory.context("oc_a"), "No previous discussion in this Feishu group yet.")

    def test_no_reply_with_explanation_is_still_no_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            card = TurnCard("card_1", "oc_a", "claude", "Claude", "opus", "max", 0.0)
            b.finalize_turn_reply(
                card, Route("agent", text="q", agent="claude"), event(),
                "[NO_REPLY] — pure duplicate of Codex's suggestion, already implemented.",
                source_agent=None, handoff_depth=0,
            )
            # Should be treated as no-reply, not shown as "Done" with the text.
            finalized = b.lark.updates or b.lark.cards
            self.assertTrue(finalized)
            last_card = finalized[-1][1] if finalized else {}
            self.assertNotIn("[NO_REPLY]", card_text(last_card))
            self.assertEqual(b.memory.context("oc_a"), "No previous discussion in this Feishu group yet.")

    def test_finalize_long_answer_overflows_to_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            # Cards hold up to 12000 chars; only longer replies overflow.
            long_answer = "甲" * 12500
            card = TurnCard("card_1", "oc_a", "codex", "Codex", "gpt", "xhigh", 0.0)
            b.finalize_turn_reply(card, Route("agent", text="q", agent="codex"), event(), long_answer, source_agent=None, handoff_depth=0)
            self.assertTrue(b.lark.updates or b.lark.cards)  # head in card
            self.assertEqual(len(b.lark.markdowns), 1)       # tail overflowed to a message

    def test_finalize_medium_answer_stays_in_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            # A reply that would overflow a plain message but fits in a card.
            answer = "甲" * 5000
            card = TurnCard("card_1", "oc_a", "codex", "Codex", "gpt", "xhigh", 0.0)
            b.finalize_turn_reply(card, Route("agent", text="q", agent="codex"), event(), answer, source_agent=None, handoff_depth=0)
            self.assertTrue(b.lark.updates or b.lark.cards)  # whole reply in card
            self.assertEqual(len(b.lark.markdowns), 0)       # no overflow message

    def test_finalize_at_peer_queues_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = self.bridge(Path(tmp))
            card = TurnCard("card_1", "oc_a", "codex", "Codex", "gpt", "xhigh", 0.0)
            b.finalize_turn_reply(card, Route("agent", text="q", agent="codex"), event(), "我的看法。@Claude 你怎么看?", source_agent=None, handoff_depth=0)
            self.assertEqual(len(b.handoffs.pending_for("claude")), 1)


if __name__ == "__main__":
    unittest.main()
