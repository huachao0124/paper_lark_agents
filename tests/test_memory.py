import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.lark_cli import MessageEvent
from paper_lark_agents.memory import ChatMemory
from paper_lark_agents.outbox import AssistantOutbox


class MemoryTests(unittest.TestCase):
    def test_chat_memory_isolated_by_chat_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = ChatMemory(Path(tmp), max_turns=10, max_chars=2000)
            event_a = MessageEvent(
                event_id="evt_a",
                chat_id="oc_a",
                chat_type="group",
                content="@Codex paper A",
                sender_id="ou_user",
                message_id="om_a",
            )
            event_b = MessageEvent(
                event_id="evt_b",
                chat_id="oc_b",
                chat_type="group",
                content="@Codex paper B",
                sender_id="ou_user",
                message_id="om_b",
            )

            memory.append_user(event_a, "paper A")
            memory.append_assistant("oc_a", "codex", "answer A")
            memory.append_user(event_b, "paper B")

            context_a = memory.context("oc_a")
            context_b = memory.context("oc_b")

            self.assertIn("paper A", context_a)
            self.assertIn("answer A", context_a)
            self.assertNotIn("paper B", context_a)
            self.assertIn("paper B", context_b)
            self.assertNotIn("answer A", context_b)


class OutboxTests(unittest.TestCase):
    def test_outbox_recognizes_recent_assistant_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            outbox = AssistantOutbox(Path(tmp), ttl_seconds=3600)
            outbox.remember("oc_a", "Codex answer\n\nwith detail", max_chars=3500, agent="codex")

            self.assertTrue(outbox.contains("oc_a", "Codex answer with detail"))
            self.assertFalse(outbox.contains("oc_b", "Codex answer with detail"))
            self.assertFalse(outbox.contains("oc_a", "different user message"))

    def test_outbox_records_final_chunk_for_discussion(self):
        with tempfile.TemporaryDirectory() as tmp:
            outbox = AssistantOutbox(Path(tmp), ttl_seconds=3600)
            outbox.remember("oc_a", "first\n\nsecond", max_chars=7, agent="codex")

            first = outbox.match("oc_a", "first")
            second = outbox.match("oc_a", "second")

            self.assertEqual(first["agent"], "codex")
            self.assertFalse(first["discussion_trigger"])
            self.assertTrue(second["discussion_trigger"])

    def test_outbox_can_mark_progress_as_non_discussion(self):
        with tempfile.TemporaryDirectory() as tmp:
            outbox = AssistantOutbox(Path(tmp), ttl_seconds=3600)
            outbox.remember(
                "oc_a",
                "Codex is processing...",
                max_chars=3500,
                agent="codex",
                discussion_trigger=False,
            )

            record = outbox.match("oc_a", "Codex is processing...")

            self.assertIsNotNone(record)
            self.assertFalse(record["discussion_trigger"])


if __name__ == "__main__":
    unittest.main()
