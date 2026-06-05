import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.workspace import ChatWorkspaceStore, WorkspaceError


class WorkspaceStoreTests(unittest.TestCase):
    def test_set_relative_workspace_under_default_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "paper-a"
            project.mkdir()
            store = ChatWorkspaceStore(root / ".state", root, (root,))

            workspace = store.set("oc_a", "paper-a")

            self.assertEqual(workspace, project.resolve())
            self.assertEqual(store.current("oc_a"), project.resolve())
            self.assertTrue(store.has_override("oc_a"))

    def test_set_creates_missing_workspace_under_allowed_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "paper-a"
            store = ChatWorkspaceStore(root / ".state", root, (root,))

            workspace = store.set("oc_a", "paper-a")

            self.assertEqual(workspace, project.resolve())
            self.assertTrue(project.is_dir())

    def test_rejects_workspace_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            store = ChatWorkspaceStore(root / ".state", root, (root,))

            with self.assertRaises(WorkspaceError):
                store.set("oc_a", str(outside))

    def test_reset_uses_default_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "paper-a"
            project.mkdir()
            store = ChatWorkspaceStore(root / ".state", root, (root,))

            store.set("oc_a", str(project))
            workspace = store.reset("oc_a")

            self.assertEqual(workspace, root.resolve())
            self.assertEqual(store.current("oc_a"), root.resolve())
            self.assertFalse(store.has_override("oc_a"))


if __name__ == "__main__":
    unittest.main()
