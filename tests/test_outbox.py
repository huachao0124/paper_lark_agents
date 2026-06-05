import tempfile
import unittest
from pathlib import Path

from paper_lark_agents.outbox import AssistantOutbox


class OutboxTests(unittest.TestCase):
    def test_matches_recorded_message_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            outbox = AssistantOutbox(Path(tmp))

            outbox.remember_message_id("oc_1", "om_1", agent="codex")

            match = outbox.match_message_id("oc_1", "om_1")
            self.assertIsNotNone(match)
            self.assertEqual(match["agent"], "codex")
            self.assertIsNone(outbox.match_message_id("oc_1", "om_2"))


if __name__ == "__main__":
    unittest.main()
