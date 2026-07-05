import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.responders import (
    ChatResponderStore,
    ResponderError,
    normalize_responder,
)


class NormalizeResponderTests(unittest.TestCase):
    def test_canonical_values(self):
        self.assertEqual(normalize_responder("codex"), "codex")
        self.assertEqual(normalize_responder("Claude"), "claude")
        self.assertEqual(normalize_responder(" BOTH "), "both")

    def test_aliases(self):
        self.assertEqual(normalize_responder("claude code"), "claude")
        self.assertEqual(normalize_responder("all"), "both")

    def test_invalid(self):
        with self.assertRaises(ResponderError):
            normalize_responder("gemini")


class ChatResponderStoreTests(unittest.TestCase):
    def store(self, root: Path, default: str = "both") -> ChatResponderStore:
        return ChatResponderStore(root / ".state", default_responder=default)

    def test_default_used_until_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(Path(tmp), default="claude")
            self.assertEqual(store.current("oc_a"), "claude")
            self.assertFalse(store.has_override("oc_a"))
            self.assertTrue(store.allows("oc_a", "claude"))
            self.assertFalse(store.allows("oc_a", "codex"))

    def test_both_default_allows_everyone(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(Path(tmp), default="both")
            self.assertTrue(store.allows("oc_a", "codex"))
            self.assertTrue(store.allows("oc_a", "claude"))

    def test_set_and_persist_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.store(root).set("oc_a", "codex")
            # A second process reading the shared state file must agree.
            other = self.store(root)
            self.assertTrue(other.has_override("oc_a"))
            self.assertEqual(other.current("oc_a"), "codex")
            self.assertTrue(other.allows("oc_a", "codex"))
            self.assertFalse(other.allows("oc_a", "claude"))

    def test_write_does_not_reuse_fixed_tmp_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / ".state"
            state.mkdir()
            legacy_tmp = state / "chat_responders.tmp"
            legacy_tmp.write_text("sentinel", encoding="utf-8")

            self.store(root).set("oc_a", "codebuddy")

            self.assertEqual(legacy_tmp.read_text(encoding="utf-8"), "sentinel")
            self.assertEqual(self.store(root).current("oc_a"), "codebuddy")

    def test_reset_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(Path(tmp), default="codex")
            store.set("oc_a", "claude")
            self.assertEqual(store.reset("oc_a"), "codex")
            self.assertFalse(store.has_override("oc_a"))
            self.assertEqual(store.current("oc_a"), "codex")

    def test_invalid_set_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(Path(tmp))
            with self.assertRaises(ResponderError):
                store.set("oc_a", "nobody")

    def test_invalid_default_falls_back_to_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.store(Path(tmp), default="nonsense")
            self.assertEqual(store.default_responder, "both")


if __name__ == "__main__":
    unittest.main()
