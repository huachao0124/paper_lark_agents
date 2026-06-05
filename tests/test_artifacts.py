import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_lark_agents.artifacts import ArtifactRelay, extract_path_candidates


class ArtifactTests(unittest.TestCase):
    def test_extracts_only_explicit_link_paths(self):
        text = (
            "See ![plot](results/a.png), [report](/tmp/report.pdf), "
            "`/tmp/notes.md`, and code/vtl/out.csv."
        )

        candidates = extract_path_candidates(text)

        self.assertIn("results/a.png", candidates)
        self.assertIn("/tmp/report.pdf", candidates)
        self.assertNotIn("/tmp/notes.md", candidates)
        self.assertNotIn("code/vtl/out.csv", candidates)

    def test_collects_relative_artifacts_from_chat_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "code" / "vtl"
            output = workspace / "results" / "plot.png"
            output.parent.mkdir(parents=True)
            output.write_bytes(b"png")
            relay = ArtifactRelay(
                root,
                root / ".state",
                (root,),
                max_artifacts=8,
                include_tmp=False,
            )

            artifacts = relay.collect("![plot](results/plot.png)", workspace)

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].kind, "image")
            self.assertEqual(artifacts[0].upload_path, "code/vtl/results/plot.png")

    def test_collects_plain_generated_file_mentions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "code" / "vtl"
            report = workspace / "docs" / "weekly-progress-2026-06-02.md"
            report.parent.mkdir(parents=True)
            report.write_text("weekly", encoding="utf-8")
            relay = ArtifactRelay(
                root,
                root / ".state",
                (root,),
                max_artifacts=8,
                include_tmp=False,
            )

            artifacts = relay.collect(
                "已写好周报 Markdown 草稿：\n\ndocs/weekly-progress-2026-06-02.md",
                workspace,
            )

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].kind, "file")
            self.assertEqual(artifacts[0].upload_path, "code/vtl/docs/weekly-progress-2026-06-02.md")

    def test_collects_plain_file_when_generated_cue_is_near_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "code" / "vtl"
            report = workspace / "docs" / "v3_8_preview_comment_triage.md"
            report.parent.mkdir(parents=True)
            report.write_text("triage", encoding="utf-8")
            relay = ArtifactRelay(
                root,
                root / ".state",
                (root,),
                max_artifacts=8,
                include_tmp=False,
            )

            artifacts = relay.collect(
                "我把 3.8 preview 的 32 条 comment 做成了 triage 文档：\n\n"
                "docs/v3_8_preview_comment_triage.md",
                workspace,
            )

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].upload_path, "code/vtl/docs/v3_8_preview_comment_triage.md")

    def test_plain_file_far_from_generated_cue_does_not_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "code" / "vtl"
            comments = workspace / "eval_results" / "_v3_8_preview_comments.json"
            comments.parent.mkdir(parents=True)
            comments.write_text("{}", encoding="utf-8")
            relay = ArtifactRelay(
                root,
                root / ".state",
                (root,),
                max_artifacts=8,
                include_tmp=False,
            )

            artifacts = relay.collect(
                "参考输入文件：eval_results/_v3_8_preview_comments.json。\n"
                "\n"
                "这部分主要是 reviewers 的原始 comment。\n"
                "\n"
                "- Stage1 生成修：spatial memory / belief state 的桥接。\n",
                workspace,
            )

            self.assertEqual(artifacts, [])

    def test_stages_allowed_external_artifact_under_base_cwd(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external_tmp:
            root = Path(tmp)
            external_root = Path(external_tmp)
            report = external_root / "report.pdf"
            report.write_bytes(b"pdf")
            relay = ArtifactRelay(root, root / ".state", (root, external_root), max_artifacts=8)

            artifacts = relay.collect(f"Report: [report.pdf]({report})", root)

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].kind, "file")
            self.assertTrue(artifacts[0].upload_path.startswith(".state/artifacts/"))
            self.assertTrue((root / artifacts[0].upload_path).exists())

    def test_skips_paths_outside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external_tmp:
            root = Path(tmp)
            secret = Path(external_tmp) / "secret.txt"
            secret.write_text("secret", encoding="utf-8")
            relay = ArtifactRelay(
                root,
                root / ".state",
                (root,),
                max_artifacts=8,
                include_tmp=False,
            )

            artifacts = relay.collect(f"[secret.txt]({secret})", root)

            self.assertEqual(artifacts, [])

    def test_plain_project_file_mentions_do_not_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "code" / "vtl"
            docs = workspace / "docs"
            docs.mkdir(parents=True)
            (workspace / "AGENTS.md").write_text("agents", encoding="utf-8")
            (workspace / "CLAUDE.md").write_text("claude", encoding="utf-8")
            (docs / "project-guide.md").write_text("guide", encoding="utf-8")
            relay = ArtifactRelay(
                root,
                root / ".state",
                (root,),
                max_artifacts=8,
                include_tmp=False,
            )

            artifacts = relay.collect(
                "已读 AGENTS.md、CLAUDE.md、docs/project-guide.md。",
                workspace,
            )

            self.assertEqual(artifacts, [])

    def test_home_path_without_home_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relay = ArtifactRelay(root, root / ".state", (root,), max_artifacts=8)

            with patch("pathlib.Path.expanduser", side_effect=RuntimeError("no home")):
                result = relay.resolve_candidate("~/report.md", root)

            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
