import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.effort import ChatEffortStore, EffortError, normalize_effort


class EffortTests(unittest.TestCase):
    def test_normalize_effort(self):
        self.assertEqual(normalize_effort("XHIGH"), "xhigh")
        self.assertEqual(normalize_effort("UltraCode"), "ultracode")

    def test_rejects_invalid_effort_token(self):
        with self.assertRaises(EffortError):
            normalize_effort("ultra code")
        with self.assertRaises(EffortError):
            normalize_effort("")

    def test_chat_effort_store_is_per_chat_and_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatEffortStore(Path(tmp))

            store.set("oc_a", "codex", "xhigh")
            store.set("oc_a", "claude", "ultracode")

            self.assertEqual(store.current("oc_a", "codex"), "xhigh")
            self.assertEqual(store.current("oc_a", "claude"), "ultracode")
            self.assertIsNone(store.current("oc_b", "codex"))


if __name__ == "__main__":
    unittest.main()
