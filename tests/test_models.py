import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.models import ChatModelStore, ModelError, normalize_model


class ModelTests(unittest.TestCase):
    def test_normalize_model(self):
        self.assertEqual(normalize_model(" opus "), "opus")

    def test_rejects_empty_model(self):
        with self.assertRaises(ModelError):
            normalize_model(" ")

    def test_rejects_multiline_model(self):
        with self.assertRaises(ModelError):
            normalize_model("opus\nsonnet")

    def test_chat_model_store_is_per_chat_and_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatModelStore(Path(tmp))

            store.set("oc_a", "codex", "gpt-5.5")
            store.set("oc_a", "claude", "opus")

            self.assertEqual(store.current("oc_a", "codex"), "gpt-5.5")
            self.assertEqual(store.current("oc_a", "claude"), "opus")
            self.assertIsNone(store.current("oc_b", "codex"))


if __name__ == "__main__":
    unittest.main()
