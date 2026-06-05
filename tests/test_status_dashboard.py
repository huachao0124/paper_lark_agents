import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.status_dashboard import StatusDashboardDoc, StatusDashboardStore
from paper_lark_agents.app import tab_id_from_result


class StatusDashboardTests(unittest.TestCase):
    def test_store_records_status_doc_and_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatusDashboardStore(Path(tmp))

            doc_url, created_doc = store.ensure_status_doc(
                "oc_a",
                lambda: ("https://example.feishu.cn/docx/doc_a", "doc_a"),
            )
            tab_id, created_tab = store.ensure_status_tab("oc_a", lambda: "tab_a")

            snapshot = store.snapshot("oc_a")

            self.assertTrue(created_doc)
            self.assertTrue(created_tab)
            self.assertEqual(doc_url, "https://example.feishu.cn/docx/doc_a")
            self.assertEqual(tab_id, "tab_a")
            self.assertEqual(snapshot.doc_url, "https://example.feishu.cn/docx/doc_a")
            self.assertEqual(snapshot.doc_token, "doc_a")
            self.assertEqual(snapshot.tab_id, "tab_a")

    def test_doc_xml_contains_agent_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = StatusDashboardStore(Path(tmp))
            snapshot = store.update_status(
                "oc_a",
                "codex",
                display_name="Codex",
                state="running",
                detail="Reading files",
                workspace="/tmp/project",
                model="gpt-5.5 (session)",
                effort="xhigh (session)",
                started_at=1000,
            )

            xml = StatusDashboardDoc(snapshot).to_xml()

            self.assertIn("<title>AI 状态</title>", xml)
            self.assertIn("Codex", xml)
            self.assertIn("Running", xml)
            self.assertIn("gpt-5.5 (session)", xml)
            self.assertIn("Claude", xml)

    def test_tab_id_from_result_prefers_named_doc_tab(self):
        result = {
            "data": {
                "chat_tabs": [
                    {"tab_id": "message_tab", "tab_type": "message", "tab_content": {}},
                    {
                        "tab_id": "ai_tab",
                        "tab_name": "AI 状态",
                        "tab_type": "doc",
                        "tab_content": {"doc": "https://my.feishu.cn/docx/doc_a"},
                    },
                ]
            }
        }

        self.assertEqual(
            tab_id_from_result(
                result,
                tab_name="AI 状态",
                doc_url="https://my.feishu.cn/docx/doc_a",
            ),
            "ai_tab",
        )


if __name__ == "__main__":
    unittest.main()
