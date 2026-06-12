import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_lark_agents.app import PaperAgentBridge
from paper_lark_agents.turn_card import TurnCard
from paper_lark_agents.config import load_settings
from paper_lark_agents.lark_cli import MessageEvent
from paper_lark_agents.router import Route


class FakeLark:
    def __init__(self):
        self.cards = []
        self.updates = []
        self.markdowns = []

    def send_card(self, chat_id, card):
        self.cards.append((chat_id, card))
        return {"message_id": "card_1"}

    def update_card(self, message_id, card):
        self.updates.append((message_id, card))
        return {}

    def send_markdown(self, chat_id, markdown):
        self.markdowns.append((chat_id, markdown))
        return [{"message_id": f"om_{len(self.markdowns)}"}]


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
