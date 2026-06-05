import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.inbound_files import (
    InboundResource,
    extract_inbound_resources,
    inbound_output_relative_path,
    post_text_without_resources,
    safe_filename,
)


class InboundFileTests(unittest.TestCase):
    def test_extracts_file_resource_from_json(self):
        resources = extract_inbound_resources(
            "file",
            '{"file_key":"file_v3_abc","file_name":"paper.pdf"}',
        )

        self.assertEqual(resources, [InboundResource("file_v3_abc", "file", "paper.pdf", "file")])

    def test_extracts_image_resource_from_json(self):
        resources = extract_inbound_resources("image", '{"image_key":"img_v3_abc"}')

        self.assertEqual(resources, [InboundResource("img_v3_abc", "image", "img_v3_abc", "image")])

    def test_extracts_file_resource_from_rendered_content(self):
        resources = extract_inbound_resources("file", '<file key="file_v3_abc" name="x"/>')

        self.assertEqual(resources[0].key, "file_v3_abc")
        self.assertEqual(resources[0].name, "x")

    def test_extracts_post_resources_from_rendered_content(self):
        resources = extract_inbound_resources(
            "post",
            '请看 [Image: img_v3_abc] 和 <file key="file_v3_def" name="paper.pdf"/>',
        )

        self.assertEqual(
            resources,
            [
                InboundResource("img_v3_abc", "image", "img_v3_abc", "post"),
                InboundResource("file_v3_def", "file", "paper.pdf", "post"),
            ],
        )

    def test_extracts_post_resources_from_json(self):
        resources = extract_inbound_resources(
            "post",
            (
                '{"zh_cn":{"content":[[{"tag":"text","text":"请看"},'
                '{"tag":"img","image_key":"img_v3_abc"},'
                '{"tag":"file","file_key":"file_v3_def","name":"paper.pdf"}]]}}'
            ),
        )

        self.assertEqual(
            resources,
            [
                InboundResource("img_v3_abc", "image", "img_v3_abc", "post"),
                InboundResource("file_v3_def", "file", "paper.pdf", "post"),
            ],
        )

    def test_output_path_stays_under_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace = root / "code" / "vtl"
            workspace.mkdir(parents=True)
            rel = inbound_output_relative_path(
                root,
                workspace,
                "oc_x",
                "om_y",
                InboundResource("file_v3_abc", "file", "../paper.pdf", "file"),
            )

            self.assertEqual(rel, "code/vtl/.lark_uploads/oc_x/om_y/paper.pdf")
            self.assertTrue((root / "code/vtl/.lark_uploads/oc_x/om_y").is_dir())

    def test_safe_filename(self):
        self.assertEqual(safe_filename("../a:b.pdf"), "a-b.pdf")

    def test_post_text_without_rendered_resources(self):
        self.assertEqual(
            post_text_without_resources(
                '请分析这个文件 <file key="file_v3_def" name="paper.pdf"/> [Image: img_v3_abc]'
            ),
            "请分析这个文件",
        )

    def test_post_text_without_resources_detects_resource_only(self):
        self.assertEqual(
            post_text_without_resources('<file key="file_v3_def" name="paper.pdf"/>'),
            "",
        )

    def test_post_text_without_json_resources(self):
        content = (
            '{"zh_cn":{"content":[[{"tag":"text","text":"请分析"},'
            '{"tag":"file","file_key":"file_v3_def","name":"paper.pdf"}]]}}'
        )

        self.assertEqual(post_text_without_resources(content), "请分析")


if __name__ == "__main__":
    unittest.main()
