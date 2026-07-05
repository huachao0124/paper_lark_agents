import json
import tempfile
from pathlib import Path
import unittest

from paper_lark_agents.transcripts import (
    activity_detail,
    activity_status,
    claude_effort_from_lines,
    claude_followup_from_lines,
    claude_reply_from_lines,
    codex_effort_from_lines,
    codex_reply_from_lines,
    encode_claude_project_dir,
    find_claude_session_file,
    find_codex_rollout,
    find_codex_rollout_by_id,
    read_new_jsonl,
    rollout_cwd,
    turn_messages,
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

    def test_find_codex_rollout_by_id_ignores_cwd(self):
        # A migrated / workspace-switched rollout keeps its original cwd, so
        # cwd matching fails; locating it by session id must still work.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "06" / "05"
            day.mkdir(parents=True)
            sid = "019e958a-5b51-7230-9fbc-fe1529d8bd5a"
            target = day / f"rollout-2026-06-05T02-08-56-{sid}.jsonl"
            target.write_text(
                json.dumps({"type": "session_meta", "payload": {"id": sid, "cwd": "/server/path"}}) + "\n",
                encoding="utf-8",
            )
            # cwd matching can't find it under the live (different) workspace.
            self.assertIsNone(find_codex_rollout(root, Path("/Users/me/code")))
            # id matching finds it regardless of cwd.
            self.assertEqual(find_codex_rollout_by_id(root, sid), target)
            self.assertIsNone(find_codex_rollout_by_id(root, "missing-id"))
            self.assertIsNone(find_codex_rollout_by_id(root, ""))

    def test_find_codex_rollout_by_id_slow_path_reads_session_meta(self):
        # Filename without the id suffix: fall back to reading session_meta.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "06" / "05"
            day.mkdir(parents=True)
            target = day / "rollout-legacy-name.jsonl"
            target.write_text(
                json.dumps({"type": "session_meta", "payload": {"id": "xyz", "cwd": "/x"}}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(find_codex_rollout_by_id(root, "xyz"), target)


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


class SubagentNarrationTests(unittest.TestCase):
    def _text(self, text, stop_reason):
        return {"type": "assistant", "message": {
            "role": "assistant", "stop_reason": stop_reason,
            "content": [{"type": "text", "text": text}]}}

    def _tool(self, name):
        return {"type": "assistant", "message": {
            "role": "assistant", "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": name, "id": "t1", "input": {}}]}}

    def test_subagent_narration_delivered_agent_tool(self):
        # Claude narrates, then launches a teammate via the "Agent" tool, then a
        # short terminal note. The narration must not be dropped.
        lines = [
            self._text("Here is my full analysis.", "tool_use"),
            self._tool("Agent"),
            {"type": "user", "message": {"role": "user", "content": []}},
            self._text("Dispatched the subagent.", "end_turn"),
        ]
        result = claude_reply_from_lines(lines)
        self.assertIsNotNone(result)
        self.assertIn("Here is my full analysis.", result.text)
        self.assertIn("Dispatched the subagent.", result.text)

    def test_task_tool_name_also_detected(self):
        lines = [
            self._text("Analysis before task.", "tool_use"),
            self._tool("Task"),
            self._text("Launched.", "end_turn"),
        ]
        self.assertIn("Analysis before task.", claude_reply_from_lines(lines).text)

    def test_normal_tool_turn_not_concatenated(self):
        # No subagent: intermediate narration stays out; only the terminal text.
        lines = [
            self._text("Let me read the file.", "tool_use"),
            self._tool("Read"),
            {"type": "user", "message": {"role": "user", "content": []}},
            self._text("Here is the answer.", "end_turn"),
        ]
        self.assertEqual(claude_reply_from_lines(lines).text, "Here is the answer.")

    def test_narration_delivered_before_terminal(self):
        # Turn is still at the subagent dispatch (no end_turn yet); the stage-1
        # narration must be delivered now, not held until the subagent finishes.
        lines = [
            self._text("Stage-1 analysis here.", "tool_use"),
            self._tool("Agent"),  # last message, non-terminal stop_reason
        ]
        result = claude_reply_from_lines(lines)
        self.assertIsNotNone(result)
        self.assertIn("Stage-1 analysis here.", result.text)

    def test_non_terminal_without_subagent_waits(self):
        # Mid-turn with a normal tool and no subagent → keep waiting (None).
        lines = [
            self._text("Let me read the file.", "tool_use"),
            self._tool("Read"),
        ]
        self.assertIsNone(claude_reply_from_lines(lines))

    def test_followup_subagent_narration_delivered(self):
        lines = [
            self._text("Follow-up analysis, launching another agent.", "tool_use"),
            self._tool("Agent"),
            self._text("Second one dispatched.", "end_turn"),
        ]
        result = claude_followup_from_lines(lines)
        self.assertIn("Follow-up analysis", result.text)


class EffortParseTests(unittest.TestCase):
    def test_codex_effort_from_turn_context(self):
        lines = [
            {"type": "session_meta", "payload": {"id": "x"}},
            {"type": "turn_context", "payload": {"model": "gpt-5.5", "effort": "xhigh"}},
        ]
        self.assertEqual(codex_effort_from_lines(lines), "xhigh")

    def test_codex_effort_uses_most_recent_turn_context(self):
        lines = [
            {"type": "turn_context", "payload": {"effort": "high"}},
            {"type": "turn_context", "payload": {"effort": "xhigh"}},
        ]
        self.assertEqual(codex_effort_from_lines(lines), "xhigh")

    def test_codex_effort_absent_returns_none(self):
        self.assertIsNone(codex_effort_from_lines([{"type": "turn_context", "payload": {}}]))

    def test_claude_effort_not_recorded(self):
        # Claude's transcript never carries an effort level.
        self.assertIsNone(claude_effort_from_lines([
            {"type": "assistant", "message": {"model": "claude-opus-4-8", "stop_reason": "end_turn"}},
        ]))


class TurnMessagesTests(unittest.TestCase):
    @staticmethod
    def _codex_msg(text):
        return {"type": "event_msg", "payload": {"type": "agent_message", "message": text}}

    def test_codex_collects_all_intermediate_messages(self):
        lines = [
            {"type": "event_msg", "payload": {"type": "task_started"}},
            self._codex_msg("先查一下库存"),
            {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command"}},
            self._codex_msg("发现问题，正在修复"),
            self._codex_msg("修好了，正在验证"),
        ]
        self.assertEqual(
            turn_messages(lines, "codex"),
            ["先查一下库存", "发现问题，正在修复", "修好了，正在验证"],
        )

    def test_codex_skips_auto_review_json(self):
        lines = [
            self._codex_msg('{"outcome":"allow"}'),
            self._codex_msg("真实回复"),
            self._codex_msg('{"risk_level":"low","outcome":"allow","rationale":"x"}'),
        ]
        self.assertEqual(turn_messages(lines, "codex"), ["真实回复"])

    def test_claude_collects_assistant_texts(self):
        lines = [
            _asst("第一段叙述", stop_reason=None),
            _asst("最终回复"),
        ]
        self.assertEqual(turn_messages(lines, "claude"), ["第一段叙述", "最终回复"])

    def test_codex_activity_status_line(self):
        lines = [
            {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": '{"cmd":"ls"}'}},
        ]
        status = activity_status(lines, "codex")
        self.assertIn("步", status)


if __name__ == "__main__":
    unittest.main()
