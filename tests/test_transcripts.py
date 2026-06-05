import json
import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.transcripts import (
    activity_detail,
    claude_followup_from_lines,
    claude_reply_from_lines,
    codex_reply_from_lines,
    encode_claude_project_dir,
    find_claude_session_file,
    find_codex_rollout,
    read_new_jsonl,
    rollout_cwd,
)


def _asst(text, stop_reason="end_turn", usage=None):
    msg = {"role": "assistant", "stop_reason": stop_reason,
           "content": [{"type": "text", "text": text}] if text is not None else []}
    if usage is not None:
        msg["usage"] = usage
    return {"type": "assistant", "message": msg}


class ClaudeParseTests(unittest.TestCase):
    def test_end_turn_returns_text_and_usage(self):
        lines = [{"type": "user"}, _asst("hello", usage={"output_tokens": 5})]
        r = claude_reply_from_lines(lines)
        self.assertEqual(r.text, "hello")
        self.assertEqual(r.usage, {"output_tokens": 5})

    def test_tool_use_last_means_not_done(self):
        lines = [_asst("let me check", stop_reason="tool_use")]
        self.assertIsNone(claude_reply_from_lines(lines))

    def test_tool_then_final_text(self):
        lines = [_asst("checking", stop_reason="tool_use"), {"type": "user"}, _asst("the answer")]
        self.assertEqual(claude_reply_from_lines(lines).text, "the answer")

    def test_empty_final_falls_back_to_earlier_text(self):
        lines = [_asst("real answer", stop_reason="tool_use"), _asst(None)]
        self.assertEqual(claude_reply_from_lines(lines).text, "real answer")

    def test_no_assistant_returns_none(self):
        self.assertIsNone(claude_reply_from_lines([{"type": "user"}]))


class ClaudeFollowupTests(unittest.TestCase):
    def test_followup_after_first_reply(self):
        # Lines read AFTER the first end_turn: a second end_turn with text.
        lines = [
            _asst("checking", stop_reason="tool_use"),
            _asst("Teammate result here"),
        ]
        r = claude_followup_from_lines(lines)
        self.assertIsNotNone(r)
        self.assertEqual(r.text, "Teammate result here")

    def test_no_followup_when_still_working(self):
        lines = [_asst("mid-work", stop_reason="tool_use")]
        self.assertIsNone(claude_followup_from_lines(lines))

    def test_followup_even_with_interleaved_user_message(self):
        # New prompts (handoffs) can arrive while a subagent is still running,
        # so user messages should not prevent detecting the follow-up.
        lines = [{"type": "user"}, _asst("subagent result after handoff")]
        r = claude_followup_from_lines(lines)
        self.assertIsNotNone(r)
        self.assertEqual(r.text, "subagent result after handoff")

    def test_no_followup_on_empty(self):
        self.assertIsNone(claude_followup_from_lines([]))


class CodexParseTests(unittest.TestCase):
    def test_task_complete_returns_message(self):
        lines = [
            {"type": "response_item", "payload": {"role": "assistant", "content": [{"type": "text", "text": "x"}]}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"total": 7}}},
            {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "final answer"}},
        ]
        r = codex_reply_from_lines(lines)
        self.assertEqual(r.text, "final answer")
        self.assertEqual(r.usage, {"total": 7})

    def test_no_task_complete_means_not_done(self):
        lines = [{"type": "event_msg", "payload": {"type": "task_started"}}]
        self.assertIsNone(codex_reply_from_lines(lines))


class ActivityTests(unittest.TestCase):
    def test_claude_tool_action_steps_tokens(self):
        lines = [
            {"type": "assistant", "message": {"role": "assistant", "stop_reason": "tool_use",
             "content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "/a/b/stage2.py"}}],
             "usage": {"output_tokens": 120}}},
        ]
        d = activity_detail(lines, "claude")
        self.assertIn("读", d)            # Read label
        self.assertIn("stage2.py", d)      # basename of target
        self.assertIn("步", d)
        self.assertIn("tok", d)

    def test_claude_writing_when_no_tool(self):
        lines = [{"type": "assistant", "message": {"role": "assistant", "stop_reason": "end_turn",
                  "content": [{"type": "text", "text": "answer"}], "usage": {"output_tokens": 5}}}]
        self.assertIn("整理回复", activity_detail(lines, "claude"))

    def test_codex_exec_action_and_steps(self):
        lines = [
            {"type": "response_item", "payload": {
                "type": "function_call", "name": "exec_command",
                "arguments": "{\"cmd\":\"sed -n 1,5p docs/x.md\",\"workdir\":\"/ws\"}"}},
        ]
        d = activity_detail(lines, "codex")
        self.assertIn("exec", d)        # exec label
        self.assertIn("sed", d)         # the command
        self.assertIn("步", d)          # counted as a step

    def test_codex_latest_action_wins_and_tokens(self):
        lines = [
            {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"output_tokens": 120, "reasoning_output_tokens": 40}}}},
            {"type": "event_msg", "payload": {"type": "web_search_end", "query": "DINO baseline"}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"output_tokens": 80, "reasoning_output_tokens": 0}}}},
        ]
        d = activity_detail(lines, "codex")
        self.assertIn("搜索", d)        # web_search_end is the latest action
        self.assertIn("tok", d)         # turn output = 120+40+80 = 240 generated tokens

    def test_empty_is_thinking(self):
        self.assertEqual(activity_detail([], "claude"), "✻ 思考中")
        self.assertEqual(activity_detail([], "codex"), "✻ 思考中")


class LocatorTests(unittest.TestCase):
    def test_encode_project_dir(self):
        self.assertEqual(
            encode_claude_project_dir(Path("/apdcephfs_sgfd/share_1/code/vtl")),
            "-apdcephfs-sgfd-share-1-code-vtl",
        )

    def test_find_claude_session_file_by_uuid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proj = root / "-some-cwd"
            proj.mkdir()
            target = proj / "abc-123.jsonl"
            target.write_text("{}\n", encoding="utf-8")
            (proj / "other.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(find_claude_session_file(root, "abc-123"), target)
            self.assertIsNone(find_claude_session_file(root, "missing"))

    def test_find_codex_rollout_matches_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "06" / "04"
            day.mkdir(parents=True)
            good = day / "rollout-a.jsonl"
            good.write_text(json.dumps({"type": "session_meta", "payload": {"cwd": "/ws"}}) + "\n", encoding="utf-8")
            bad = day / "rollout-b.jsonl"
            bad.write_text(json.dumps({"type": "session_meta", "payload": {"cwd": "/other"}}) + "\n", encoding="utf-8")
            self.assertEqual(rollout_cwd(good), "/ws")
            self.assertEqual(find_codex_rollout(root, Path("/ws")), good)
            self.assertIsNone(find_codex_rollout(root, Path("/nope")))


class IncrementalReadTests(unittest.TestCase):
    def test_reads_complete_lines_and_holds_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.jsonl"
            p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
            objs, off = read_new_jsonl(p, 0)
            self.assertEqual(objs, [{"a": 1}, {"b": 2}])
            with p.open("a", encoding="utf-8") as fh:
                fh.write('{"c":3}\n{"partial"')
            objs2, off2 = read_new_jsonl(p, off)
            self.assertEqual(objs2, [{"c": 3}])
            self.assertLess(off2, p.stat().st_size)  # partial line not consumed
            # completing the partial line yields it next time
            with p.open("a", encoding="utf-8") as fh:
                fh.write(':4}\n')
            objs3, _ = read_new_jsonl(p, off2)
            self.assertEqual(objs3, [{"partial": 4}])

    def test_skips_unparseable_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.jsonl"
            p.write_text('not json\n{"ok":1}\n', encoding="utf-8")
            objs, _ = read_new_jsonl(p, 0)
            self.assertEqual(objs, [{"ok": 1}])


if __name__ == "__main__":
    unittest.main()
