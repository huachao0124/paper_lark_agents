import unittest
import json
import tempfile
from pathlib import Path
from shlex import quote as shlex_quote
from types import SimpleNamespace
from unittest.mock import patch

from paper_lark_agents.tmux_runtime import (
    TmuxSessionRuntime,
    claude_feedback_prompt_visible,
    extract_mismatched_marked_reply_after_prompt,
    extract_marked_reply,
    has_codex_effort_arg,
    has_model_arg,
    normalize_terminal_text,
    parse_session_effort,
    parse_session_model,
    run_id_from_marker,
    session_ready_for_current_input,
    session_ready_for_input,
    summarize_session_progress,
    without_existing_cwd_args,
    with_codex_effort_arg,
    with_model_arg,
)


class ResumeLaunchTests(unittest.TestCase):
    def rt(self, agent):
        settings = SimpleNamespace(state_dir=Path("/tmp"), codex_cmd="codex", claude_cmd="claude")
        return TmuxSessionRuntime(settings, agent)

    def test_claude_fresh_uses_session_id(self):
        launch, uid = self.rt("claude").build_launch_command(["claude", "--permission-mode", "auto"], None)
        self.assertIn("--session-id", launch)
        self.assertEqual(launch[launch.index("--session-id") + 1], uid)
        self.assertNotIn("--resume", launch)

    def test_claude_resume_uses_resume_flag(self):
        launch, uid = self.rt("claude").build_launch_command(["claude", "--permission-mode", "auto"], "uuid-1")
        self.assertEqual(launch[-2:], ["--resume", "uuid-1"])
        self.assertEqual(uid, "uuid-1")
        self.assertNotIn("--session-id", launch)

    def test_codex_fresh_is_plain(self):
        launch, uid = self.rt("codex").build_launch_command(["codex", "--no-alt-screen", "-C", "/ws"], None)
        self.assertEqual(launch, ["codex", "--no-alt-screen", "-C", "/ws"])
        self.assertIsNone(uid)

    def test_codex_resume_inserts_subcommand_and_drops_cwd(self):
        launch, uid = self.rt("codex").build_launch_command(
            ["env", "X=1", "codex", "--no-alt-screen", "-C", "/ws"], "sid-9"
        )
        self.assertEqual(uid, "sid-9")
        self.assertEqual(launch[:5], ["env", "X=1", "codex", "resume", "sid-9"])
        self.assertNotIn("-C", launch)
        self.assertIn("--no-alt-screen", launch)


class JsonlRecoveryTests(unittest.TestCase):
    def runtime(self, root, agent="claude"):
        settings = SimpleNamespace(state_dir=root, no_reply_token="[NO_REPLY]")
        return TmuxSessionRuntime(settings, agent)

    def test_run_id_from_marker(self):
        self.assertEqual(run_id_from_marker("PLA_REPLY_START_abc"), "abc")
        self.assertEqual(run_id_from_marker("PLA_REPLY_END_abc"), "abc")
        self.assertEqual(run_id_from_marker("abc"), "abc")

    def test_cursor_roundtrip_and_find_marked_reply_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.runtime(root)
            chat_id = "oc_x"
            session_name = runtime.session_name(chat_id)
            tf = root / "t.jsonl"
            tf.write_text(
                json.dumps({
                    "type": "assistant",
                    "message": {"role": "assistant", "stop_reason": "end_turn",
                                "content": [{"type": "text", "text": "recovered answer"}]},
                }) + "\n",
                encoding="utf-8",
            )
            runtime.store_run_cursor(session_name, "run1", str(tf), 0)
            self.assertEqual(runtime.read_run_cursor(session_name, "run1"), (str(tf), 0))
            self.assertEqual(
                runtime.find_marked_reply(chat_id, "PLA_REPLY_START_run1", "PLA_REPLY_END_run1"),
                "recovered answer",
            )

    def test_unknown_run_id_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.runtime(Path(tmp))
            self.assertIsNone(
                runtime.find_marked_reply("oc_x", "PLA_REPLY_START_missing", "PLA_REPLY_END_missing")
            )

    def test_offset_at_eof_means_not_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.runtime(root)
            session_name = runtime.session_name("oc_x")
            tf = root / "t.jsonl"
            tf.write_text(
                json.dumps({"type": "assistant", "message": {"role": "assistant",
                            "stop_reason": "end_turn", "content": [{"type": "text", "text": "old"}]}}) + "\n",
                encoding="utf-8",
            )
            # cursor starts at EOF -> nothing appended since -> no reply yet
            runtime.store_run_cursor(session_name, "run2", str(tf), tf.stat().st_size)
            self.assertIsNone(
                runtime.find_marked_reply("oc_x", "PLA_REPLY_START_run2", "PLA_REPLY_END_run2")
            )

    def test_codex_recovery_uses_task_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.runtime(root, agent="codex")
            session_name = runtime.session_name("oc_y")
            tf = root / "rollout.jsonl"
            tf.write_text(
                json.dumps({"type": "event_msg", "payload": {"type": "task_complete",
                            "last_agent_message": "codex done"}}) + "\n",
                encoding="utf-8",
            )
            runtime.store_run_cursor(session_name, "runc", str(tf), 0)
            self.assertEqual(
                runtime.find_marked_reply("oc_y", "PLA_REPLY_START_runc", "PLA_REPLY_END_runc"),
                "codex done",
            )


class TmuxRuntimeTests(unittest.TestCase):
    def test_extract_marked_reply_uses_markers_on_own_lines(self):
        text = """Prompt mentions PLA_REPLY_START_abc and PLA_REPLY_END_abc inline.

PLA_REPLY_START_abc
Useful reply.
PLA_REPLY_END_abc
"""
        self.assertEqual(
            extract_marked_reply(text, "PLA_REPLY_START_abc", "PLA_REPLY_END_abc"),
            "Useful reply.",
        )

    def test_claude_feedback_prompt_visible_in_current_tail(self):
        text = """Older output.
● How is Claude doing this session? (optional)
  1: Bad    2: Fine   3: Good   0: Dismiss
❯
"""
        self.assertTrue(claude_feedback_prompt_visible(text))

    def test_claude_feedback_prompt_ignores_old_scrollback(self):
        text = """● How is Claude doing this session? (optional)
  1: Bad    2: Fine   3: Good   0: Dismiss
"""
        text += "\n".join(f"later line {index}" for index in range(20))
        self.assertFalse(claude_feedback_prompt_visible(text))

    def test_extract_marked_reply_takes_latest(self):
        text = """PLA_REPLY_START_x
old
PLA_REPLY_END_x
PLA_REPLY_START_x
new
PLA_REPLY_END_x
"""
        self.assertEqual(extract_marked_reply(text, "PLA_REPLY_START_x", "PLA_REPLY_END_x"), "new")

    def test_extract_marked_reply_ignores_inline_prompt_markers(self):
        text = "Use PLA_REPLY_START_x then PLA_REPLY_END_x."
        self.assertIsNone(extract_marked_reply(text, "PLA_REPLY_START_x", "PLA_REPLY_END_x"))

    def test_extract_marked_reply_accepts_tui_bullet_prefix(self):
        text = """● PLA_REPLY_START_x
reply from tui
  PLA_REPLY_END_x
"""
        self.assertEqual(
            extract_marked_reply(text, "PLA_REPLY_START_x", "PLA_REPLY_END_x"),
            "reply from tui",
        )

    def test_extract_marked_reply_uses_nearest_start_before_end(self):
        text = """PLA_REPLY_START_x
stale redraw
Message:
Feishu message:
●PLA_REPLY_START_x
clean final reply
PLA_REPLY_END_x
"""
        self.assertEqual(
            extract_marked_reply(text, "PLA_REPLY_START_x", "PLA_REPLY_END_x"),
            "clean final reply",
        )

    def test_extract_mismatched_reply_after_current_prompt(self):
        text = """PLA_REPLY_START_old
old reply
PLA_REPLY_END_old

Output protocol for this turn:
- Always put the literal start marker PLA_REPLY_START_expected on its own line.
- Then put the literal end marker PLA_REPLY_END_expected on its own line.

Message:
Feishu message

● PLA_REPLY_START_stale
current reply with reused marker
PLA_REPLY_END_stale
"""
        self.assertEqual(
            extract_mismatched_marked_reply_after_prompt(
                text,
                "PLA_REPLY_START_expected",
                "PLA_REPLY_END_expected",
            ),
            "current reply with reused marker",
        )

    def test_extract_mismatched_reply_requires_current_prompt(self):
        text = """PLA_REPLY_START_stale
old reply
PLA_REPLY_END_stale
"""
        self.assertIsNone(
            extract_mismatched_marked_reply_after_prompt(
                text,
                "PLA_REPLY_START_expected",
                "PLA_REPLY_END_expected",
            )
        )

    def test_no_reply_is_extracted_from_marked_body(self):
        text = """PLA_REPLY_START_x
[NO_REPLY]
PLA_REPLY_END_x
"""
        self.assertEqual(
            extract_marked_reply(text, "PLA_REPLY_START_x", "PLA_REPLY_END_x"),
            "[NO_REPLY]",
        )

    def test_wrap_prompt_only_includes_session_context_when_requested(self):
        settings = SimpleNamespace(no_reply_token="[NO_REPLY]")
        runtime = TmuxSessionRuntime(settings, "codex")

        warm = runtime.wrap_prompt(
            "Feishu message",
            include_context=True,
            session_context="Session setup once.",
        )
        lean = runtime.wrap_prompt(
            "Feishu message",
            include_context=False,
            session_context="Session setup once.",
        )

        self.assertIn("Session setup once.", warm)
        self.assertNotIn("Session setup once.", lean)
        self.assertIn("Feishu message", warm)
        self.assertIn("Feishu message", lean)
        # No reply markers are injected anymore.
        self.assertNotIn("PLA_REPLY_START", warm)
        self.assertNotIn("PLA_REPLY_END", lean)

    def test_without_existing_cwd_args_removes_old_codex_cwd(self):
        self.assertEqual(
            without_existing_cwd_args(("--no-alt-screen", "-C", "/old", "--cd=/older")),
            ["--no-alt-screen"],
        )

    def test_with_model_arg_adds_model_when_absent(self):
        self.assertEqual(
            with_model_arg(["--no-alt-screen"], "gpt-5.5", "-m"),
            ["--no-alt-screen", "-m", "gpt-5.5"],
        )

    def test_with_model_arg_keeps_existing_model(self):
        args = ["--model", "opus"]
        self.assertTrue(has_model_arg(args))
        self.assertEqual(with_model_arg(args, "sonnet", "--model"), args)

    def test_with_codex_effort_arg_adds_config_override(self):
        self.assertEqual(
            with_codex_effort_arg(["--no-alt-screen"], "xhigh"),
            ["--no-alt-screen", "-c", 'model_reasoning_effort="xhigh"'],
        )

    def test_with_codex_effort_arg_keeps_existing_override(self):
        args = ["-c", 'model_reasoning_effort="high"']
        self.assertTrue(has_codex_effort_arg(args))
        self.assertEqual(with_codex_effort_arg(args, "xhigh"), args)

    def test_codex_command_uses_agent_proxy_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                codex_cmd="codex",
                codex_session_args=("--no-alt-screen",),
                codex_model=None,
                codex_default_effort=None,
                agent_proxy_url="http://127.0.0.1:7899",
                no_proxy="localhost",
            )
            runtime = TmuxSessionRuntime(settings, "codex")

            command = runtime.command(root)

            self.assertEqual(command[:2], ["env", "http_proxy=http://127.0.0.1:7899"])
            self.assertIn("codex", command)
            self.assertIn("-C", command)

    def test_claude_command_uses_agent_proxy_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                claude_cmd="claude",
                claude_session_args=("--permission-mode", "auto"),
                claude_model=None,
                agent_proxy_url="http://127.0.0.1:7899",
                no_proxy="localhost",
            )
            runtime = TmuxSessionRuntime(settings, "claude")

            command = runtime.command(root)

            self.assertEqual(command[:2], ["env", "http_proxy=http://127.0.0.1:7899"])
            self.assertIn("claude", command)
            self.assertIn("--permission-mode", command)

    def test_session_command_matches_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
            )
            runtime = TmuxSessionRuntime(settings, "codex")
            runtime.write_metadata("pla-codex-oc", "oc", ["codex", "--effort", "high"], root)

            self.assertTrue(
                runtime.session_command_matches("pla-codex-oc", ["codex", "--effort", "high"])
            )
            self.assertFalse(
                runtime.session_command_matches("pla-codex-oc", ["codex", "--effort", "max"])
            )

    def test_session_effort_matches_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
            )
            runtime = TmuxSessionRuntime(settings, "claude")

            runtime.mark_session_effort("pla-claude-oc", "max")

            self.assertTrue(runtime.session_effort_matches("pla-claude-oc", "max"))
            self.assertFalse(runtime.session_effort_matches("pla-claude-oc", "xhigh"))

    def test_apply_codex_effort_pastes_session_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
            )
            runtime = TmuxSessionRuntime(settings, "codex")
            calls = []
            runtime.paste_and_submit = lambda session_name, text: calls.append((session_name, text))  # type: ignore[method-assign]

            with patch("paper_lark_agents.tmux_runtime.time.sleep"):
                runtime.apply_effort_if_needed("pla-codex-oc", "ultracode")

            self.assertEqual(calls, [("pla-codex-oc", "/effort ultracode")])
            self.assertTrue(runtime.session_effort_matches("pla-codex-oc", "ultracode"))

    def test_send_session_command_creates_session_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                session_startup_wait=0,
                codex_session_args=(),
                codex_model=None,
                codex_cmd="codex",
            )
            runtime = TmuxSessionRuntime(settings, "codex")
            calls: list[tuple[str, object]] = []

            runtime.session_exists = lambda session_name: False  # type: ignore[method-assign]
            runtime.ensure_session = lambda session_name, chat_id, workspace, model=None, effort=None: calls.append(  # type: ignore[method-assign]
                ("ensure", session_name, chat_id, workspace, model, effort)
            ) or True
            runtime.paste_and_submit = lambda session_name, text: calls.append(  # type: ignore[method-assign]
                ("paste", session_name, text)
            )

            self.assertTrue(runtime.send_session_command("oc", "/help", workspace=root))
            self.assertEqual(calls[0][0], "ensure")
            self.assertEqual(calls[0][1:4], ("pla-codex-oc", "oc", root))
            self.assertEqual(calls[1], ("paste", "pla-codex-oc", "/help"))

    def test_reset_session_removes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                codex_state_dir=root / ".codex",
                claude_state_dir=root / ".claude",
            )
            runtime = TmuxSessionRuntime(settings, "codex")
            runtime.write_metadata("pla-codex-oc", "oc", ["codex"], root)
            runtime.kill_session = lambda session_name: None  # type: ignore[method-assign]

            runtime.reset_session("oc")

            self.assertFalse((root / ".state" / "tmux" / "pla-codex-oc.json").exists())

    def test_reset_session_removes_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                codex_state_dir=root / ".codex",
                claude_state_dir=root / ".claude",
            )
            runtime = TmuxSessionRuntime(settings, "codex")
            transcript = runtime.transcript_path("pla-codex-oc")
            transcript.parent.mkdir(parents=True)
            transcript.write_text("old terminal output", encoding="utf-8")
            runtime.kill_session = lambda session_name: None  # type: ignore[method-assign]

            runtime.reset_session("oc")

            self.assertFalse(transcript.exists())

    def test_reset_session_deletes_matching_codex_session_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chat_id = "oc_1234567890abcdef"
            codex_dir = root / ".codex"
            session_dir = codex_dir / "sessions" / "2026" / "06" / "02"
            session_dir.mkdir(parents=True)
            matching = session_dir / "rollout-2026-06-02T14-00-00-019eaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            unrelated = session_dir / "rollout-2026-06-02T14-00-01-019effff-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            matching.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "session_meta",
                                "payload": {
                                    "id": "019eaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                                    "timestamp": "2026-06-02T06:00:00Z",
                                    "cwd": str(root),
                                },
                            }
                        ),
                        json.dumps({"text": f"chat_id: {chat_id}"}),
                    ]
                ),
                encoding="utf-8",
            )
            unrelated.write_text(
                json.dumps({"type": "session_meta", "payload": {"cwd": str(root)}}),
                encoding="utf-8",
            )
            history = codex_dir / "history.jsonl"
            history.write_text(
                "\n".join(
                    [
                        json.dumps({"session_id": "019eaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "text": chat_id}),
                        json.dumps({"session_id": "keep", "text": "other"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                codex_state_dir=codex_dir,
                claude_state_dir=root / ".claude",
            )
            runtime = TmuxSessionRuntime(settings, "codex")
            runtime.write_metadata(f"pla-codex-{chat_id}", chat_id, ["codex"], root)
            runtime.kill_session = lambda session_name: None  # type: ignore[method-assign]

            deleted = runtime.reset_session(chat_id)

            self.assertIn(matching.resolve(), deleted)
            self.assertFalse(matching.exists())
            self.assertTrue(unrelated.exists())
            self.assertEqual(history.read_text(encoding="utf-8").strip(), json.dumps({"session_id": "keep", "text": "other"}))

    def test_reset_session_deletes_matching_claude_session_jsonl_and_process_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chat_id = "oc_1234567890abcdef"
            claude_dir = root / ".claude"
            project_dir = claude_dir / "projects" / "-workspace"
            project_dir.mkdir(parents=True)
            session_id = "11111111-2222-3333-4444-555555555555"
            matching = project_dir / f"{session_id}.jsonl"
            unrelated = project_dir / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            matching.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "mode", "sessionId": session_id}),
                        json.dumps({"type": "user", "cwd": str(root), "message": {"content": f"chat_id: {chat_id}"}}),
                    ]
                ),
                encoding="utf-8",
            )
            unrelated.write_text(json.dumps({"type": "mode", "sessionId": "other"}), encoding="utf-8")
            subdir = project_dir / session_id
            subdir.mkdir()
            (subdir / "subagents").mkdir()
            (subdir / "subagents" / "agent.jsonl").write_text("{}", encoding="utf-8")
            process_dir = claude_dir / "sessions"
            process_dir.mkdir()
            process_state = process_dir / "123.json"
            process_state.write_text(json.dumps({"sessionId": session_id, "cwd": str(root)}), encoding="utf-8")
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                codex_state_dir=root / ".codex",
                claude_state_dir=claude_dir,
            )
            runtime = TmuxSessionRuntime(settings, "claude")
            runtime.write_metadata(f"pla-claude-{chat_id}", chat_id, ["claude"], root)
            runtime.kill_session = lambda session_name: None  # type: ignore[method-assign]

            deleted = runtime.reset_session(chat_id)

            self.assertIn(matching.resolve(), deleted)
            self.assertIn(process_state.resolve(), deleted)
            self.assertIn(subdir.resolve(), deleted)
            self.assertFalse(matching.exists())
            self.assertFalse(process_state.exists())
            self.assertFalse(subdir.exists())
            self.assertTrue(unrelated.exists())

    def test_paste_and_submit_uses_bracketed_paste(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
            )
            runtime = TmuxSessionRuntime(settings, "codex")

            with patch("subprocess.run") as run:
                runtime.paste_and_submit("pla-codex-oc", "line 1\nline 2")

            commands = [call.args[0] for call in run.call_args_list]
            self.assertIn(["tmux", "send-keys", "-t", "pla-codex-oc", "C-u"], commands)
            self.assertIn(["tmux", "paste-buffer", "-p", "-t", "pla-codex-oc"], commands)

    def test_capture_with_transcript_includes_tail_for_scrolled_claude_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                session_capture_lines=20000,
            )
            runtime = TmuxSessionRuntime(settings, "claude")
            transcript = runtime.transcript_path("pla-claude-oc")
            transcript.parent.mkdir(parents=True)
            transcript.write_text(
                "\x1b[31mPLA_REPLY_START_x\x1b[0m\nfull reply\nPLA_REPLY_END_x\n",
                encoding="utf-8",
            )

            with patch("subprocess.run", return_value=SimpleNamespace(stdout="PLA_REPLY_END_x\n")):
                captured = runtime.capture_with_transcript("pla-claude-oc")

            self.assertEqual(extract_marked_reply(captured, "PLA_REPLY_START_x", "PLA_REPLY_END_x"), "full reply")

    def test_capture_ignores_transcript_for_progress_cleanliness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                session_capture_lines=20000,
            )
            runtime = TmuxSessionRuntime(settings, "claude")
            transcript = runtime.transcript_path("pla-claude-oc")
            transcript.parent.mkdir(parents=True)
            transcript.write_text("noisy redraw", encoding="utf-8")

            with patch("subprocess.run", return_value=SimpleNamespace(stdout="clean pane\n")):
                captured = runtime.capture("pla-claude-oc")

            self.assertEqual(captured, "clean pane\n")

    def test_wait_for_reply_prefers_complete_transcript_before_combined_fallback(self):
        settings = SimpleNamespace(no_reply_token="[NO_REPLY]")
        runtime = TmuxSessionRuntime(settings, "claude")
        runtime.capture = lambda session_name: "PLA_REPLY_END_x\n"  # type: ignore[method-assign]
        runtime.read_transcript_tail = lambda session_name: (  # type: ignore[method-assign]
            "PLA_REPLY_START_x\n"
            "clean reply\n"
            "PLA_REPLY_END_x\n"
            "PLA_REPLY_START_x\n"
            "unfinished redraw\n"
        )

        reply = runtime.wait_for_reply(
            "pla-claude-oc",
            "PLA_REPLY_START_x",
            "PLA_REPLY_END_x",
            timeout=1,
        )

        self.assertEqual(reply, "clean reply")

    def test_ensure_transcript_pipe_appends_to_session_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
            )
            runtime = TmuxSessionRuntime(settings, "claude")

            with patch("subprocess.run") as run:
                runtime.ensure_transcript_pipe("pla-claude-oc")

            command = run.call_args.args[0]
            self.assertEqual(command[:4], ["tmux", "pipe-pane", "-o", "-t"])
            self.assertEqual(command[4], "pla-claude-oc")
            self.assertIn("cat >>", command[5])
            self.assertIn("pla-claude-oc.log", command[5])

    def test_configure_history_limit_sets_tmux_global_option(self):
        settings = SimpleNamespace(session_history_limit=50000, no_reply_token="[NO_REPLY]")
        runtime = TmuxSessionRuntime(settings, "claude")

        with patch("subprocess.run") as run:
            runtime.configure_history_limit()

        self.assertEqual(
            run.call_args.args[0],
            [
                "tmux",
                "set-option",
                "-g",
                "history-limit",
                "50000",
            ],
        )

    def test_configure_window_size_resizes_tmux_window(self):
        settings = SimpleNamespace(session_columns=120, session_rows=80, no_reply_token="[NO_REPLY]")
        runtime = TmuxSessionRuntime(settings, "claude")

        with patch("subprocess.run") as run:
            runtime.configure_window_size("pla-claude-oc")

        self.assertEqual(
            run.call_args.args[0],
            [
                "tmux",
                "resize-pane",
                "-t",
                "pla-claude-oc",
                "-x",
                "120",
                "-y",
                "80",
            ],
        )

    def test_ensure_session_sets_size_on_new_tmux_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(
                state_dir=root / ".state",
                workspace=root,
                no_reply_token="[NO_REPLY]",
                session_history_limit=0,
                session_columns=132,
                session_rows=44,
                codex_session_args=(),
                codex_model=None,
                codex_default_effort=None,
                codex_cmd="codex",
                agent_proxy_url=None,
                no_proxy=None,
            )
            runtime = TmuxSessionRuntime(settings, "codex")
            runtime.session_exists = lambda session_name: False  # type: ignore[method-assign]
            runtime.configure_window_size = lambda session_name: None  # type: ignore[method-assign]
            runtime.ensure_transcript_pipe = lambda session_name: None  # type: ignore[method-assign]
            runtime.wait_for_session_ready = lambda session_name: None  # type: ignore[method-assign]
            runtime.refresh_detected_session_labels = lambda session_name: None  # type: ignore[method-assign]
            runtime.apply_startup_commands = lambda session_name: None  # type: ignore[method-assign]

            with patch("subprocess.run") as run:
                runtime.ensure_session("pla-codex-oc", "oc", root)

            self.assertEqual(
                run.call_args.args[0],
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-x",
                    "132",
                    "-y",
                    "44",
                    "-s",
                    "pla-codex-oc",
                    "-c",
                    str(root),
                    "codex -C " + shlex_quote(str(root)),
                ],
            )

    def test_apply_startup_commands_pastes_codex_commands(self):
        settings = SimpleNamespace(
            codex_startup_commands=("/permissions auto-review", ""),
            no_reply_token="[NO_REPLY]",
        )
        runtime = TmuxSessionRuntime(settings, "codex")
        calls: list[tuple[str, str]] = []
        runtime.paste_and_submit = lambda session_name, text: calls.append(  # type: ignore[method-assign]
            (session_name, text)
        )

        with patch("time.sleep"):
            runtime.apply_startup_commands("pla-codex-oc")

        self.assertEqual(calls, [("pla-codex-oc", "/permissions auto-review")])

    def test_apply_startup_commands_skips_non_codex(self):
        settings = SimpleNamespace(
            codex_startup_commands=("/permissions auto-review",),
            no_reply_token="[NO_REPLY]",
        )
        runtime = TmuxSessionRuntime(settings, "claude")
        calls: list[tuple[str, str]] = []
        runtime.paste_and_submit = lambda session_name, text: calls.append(  # type: ignore[method-assign]
            (session_name, text)
        )

        runtime.apply_startup_commands("pla-claude-oc")

        self.assertEqual(calls, [])

    def test_session_ready_for_input_detects_claude_prompt(self):
        self.assertTrue(session_ready_for_input('❯ Try "fix lint errors"', "claude"))
        self.assertTrue(session_ready_for_input("⏵⏵ don't ask on", "claude"))
        self.assertFalse(session_ready_for_input("Welcome back", "claude"))

    def test_session_ready_for_current_input_ignores_old_prompts_during_compact(self):
        text = """
❯ old prompt
✢ Compacting conversation… (1m 7s)
  ▰▰▰▰▰▱ 53%
  ⏵⏵ auto mode on (shift+tab to cycle) · esc to interrupt
"""

        self.assertFalse(session_ready_for_current_input(text, "claude"))

    def test_session_ready_for_current_input_uses_tail_prompt(self):
        text = """
❯ old prompt
✻ Baked for 29s
────────────────
❯
  ⏵⏵ auto mode on (shift+tab to cycle) · ← for agents
"""

        self.assertTrue(session_ready_for_current_input(text, "claude"))

    def test_session_ready_for_input_detects_codex_prompt(self):
        self.assertTrue(session_ready_for_input("› Message", "codex"))
        self.assertTrue(session_ready_for_input("98% context left", "codex"))
        self.assertFalse(session_ready_for_input("OpenAI Codex", "codex"))

    def test_parse_codex_model_from_footer(self):
        text = "› Write tests\n\n  gpt-5.5 xhigh · /workspace/project\n"

        self.assertEqual(parse_session_model(text, "codex"), "gpt-5.5")

    def test_parse_explicit_model_line(self):
        text = "State: Running\nModel: `claude-opus-4.1`\n"

        self.assertEqual(parse_session_model(text, "claude"), "claude-opus-4.1")

    def test_parse_model_ignores_prose_mentions(self):
        text = "Do nothing with this model; keep the session running.\n"

        self.assertIsNone(parse_session_model(text, "claude"))

    def test_parse_claude_tui_model_and_effort(self):
        text = """
 ▐▛███▜▌   Claude Code v2.1.159
▝▜█████▛▘  Opus 4.8 (1M context) with xhigh effort · Claude Max
  ▘▘ ▝▝    /workspace/project
"""

        self.assertEqual(parse_session_model(text, "claude"), "Opus 4.8 (1M context)")
        self.assertEqual(parse_session_effort(text, "claude"), "xhigh")

    def test_parse_claude_model_ignores_thinking_status(self):
        text = """
almost done thinking with xhigh effort
 ▐▛███▜▌   Claude Code v2.1.159
▝▜█████▛▘  Opus 4.8 (1M context) with xhigh effort · Claude Max
"""

        self.assertEqual(parse_session_model(text, "claude"), "Opus 4.8 (1M context)")

    def test_parse_claude_effort_prefers_command_result(self):
        text = """
▝▜█████▛▘  Opus 4.8 (1M context) with xhigh effort · Claude Max
❯ /effort ultracode
  ⎿  Set effort level to ultracode (this session only): xhigh + dynamic workflow orchestration
"""

        self.assertEqual(parse_session_effort(text, "claude"), "ultracode")

    def test_parse_codex_footer_effort(self):
        text = "› Write tests\n\n  gpt-5.5 xhigh · /workspace/project\n"

        self.assertEqual(parse_session_effort(text, "codex"), "xhigh")

    def test_parse_custom_effort_token(self):
        codex_text = "› Build\n\n  gpt-5.5 ultracode · /workspace/project\n"
        claude_text = "Opus 4.8 (1M context) with UltraCode effort · Claude Max"

        self.assertEqual(parse_session_effort(codex_text, "codex"), "ultracode")
        self.assertEqual(parse_session_effort(claude_text, "claude"), "ultracode")

    def test_parse_explicit_effort_line(self):
        text = "State: Running\nEffort: `max`\n"

        self.assertEqual(parse_session_effort(text, "claude"), "max")

    def test_summarize_session_progress_prefers_recent_action(self):
        text = """Output protocol for this turn:
Message:
• Explored
  └ Read project-guide.md

  gpt-5.5 xhigh · /workspace/project
"""

        self.assertEqual(summarize_session_progress(text, "codex"), "└ Read project-guide.md")

    def test_summarize_session_progress_skips_output_protocol_bullets(self):
        text = """● Reading repository
- Then put the literal end marker PLA_REPLY_END_abc on its own line.
- Do not put either marker anywhere else.
Message:
"""

        self.assertEqual(summarize_session_progress(text, "claude"), "● Reading repository")

    def test_summarize_session_progress_prefers_command_result(self):
        text = """✻ Churned for 52s
❯ /effort ultracode
  ⎿  Set effort level to ultracode (this session only): xhigh + dynamic workflow orchestration
"""

        self.assertEqual(
            summarize_session_progress(text, "claude"),
            "⎿ Set effort level to ultracode (this session only): xhigh + dynamic workflow orchestration",
        )

    def test_normalize_terminal_text_strips_ansi_and_carriage_returns(self):
        text = "\x1b[31mred\x1b[0m\rnext"

        self.assertEqual(normalize_terminal_text(text), "red\nnext")


if __name__ == "__main__":
    unittest.main()
