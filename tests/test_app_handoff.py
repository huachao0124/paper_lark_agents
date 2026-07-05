import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

from paper_lark_agents.agent_runner import AgentStillRunning
from paper_lark_agents.app import PaperAgentBridge
from paper_lark_agents.config import load_settings
from paper_lark_agents.lark_cli import MessageEvent
from paper_lark_agents.router import Route


class FakeLark:
    def __init__(self):
        self.markdowns = []

    def send_markdown(self, chat_id, markdown):
        self.markdowns.append((chat_id, markdown))
        return [{"message_id": f"om_fake_{len(self.markdowns)}"}]


class FakeAgents:
    def __init__(self):
        self.codex_prompts = []
        self.claude_prompts = []
        self.codebuddy_prompts = []
        self.recovered_reply = None

    def run_codex(
        self,
        prompt,
        chat_id=None,
        session_context=None,
        workspace=None,
        model=None,
        effort=None,
        progress_callback=None,
        run_id=None,
    ):
        self.codex_prompts.append(prompt)
        return SimpleNamespace(text="Codex follow-up")

    def run_claude(
        self,
        prompt,
        chat_id=None,
        session_context=None,
        workspace=None,
        model=None,
        effort=None,
        progress_callback=None,
        run_id=None,
    ):
        self.claude_prompts.append(prompt)
        return SimpleNamespace(text="Claude follow-up")

    def run_codebuddy(
        self,
        prompt,
        chat_id=None,
        session_context=None,
        workspace=None,
        model=None,
        effort=None,
        progress_callback=None,
        run_id=None,
    ):
        self.codebuddy_prompts.append(prompt)
        return SimpleNamespace(text="CodeBuddy follow-up")

    def detect_session_model(self, agent, chat_id):
        return None

    def detect_session_effort(self, agent, chat_id):
        return None

    def reply_markers(self, agent, run_id):
        return f"PLA_REPLY_START_{run_id}", f"PLA_REPLY_END_{run_id}"

    def session_name(self, agent, chat_id):
        return f"pla-{agent}-{chat_id}"

    def find_session_reply(self, agent, chat_id, start_marker, end_marker):
        return self.recovered_reply

    def find_session_reply(self, agent, chat_id, start_marker, end_marker):
        return self.recovered_reply

    def runtime_for(self, agent):
        return _FakeRuntime(self.recovered_reply)


class _FakeRuntime:
    def __init__(self, reply):
        self._reply = reply

    def session_name(self, chat_id):
        return f"pla-fake-{chat_id}"

    def resolve_transcript_path(self, session_name, chat_id, workspace):
        if self._reply is not None:
            return self._fake_path()
        return None

    def parse_transcript_reply(self, lines):
        if self._reply:
            return SimpleNamespace(text=self._reply)
        return None

    def _fake_path(self):
        p = Path(tempfile.gettempdir()) / "fake_transcript.jsonl"
        if not p.exists():
            p.write_text("{}\n")
        return p


class StillRunningAgents(FakeAgents):
    def run_claude(
        self,
        prompt,
        chat_id=None,
        session_context=None,
        workspace=None,
        model=None,
        effort=None,
        progress_callback=None,
        run_id=None,
    ):
        raise AgentStillRunning("Claude is still running")


class NoReplyPromptAgents(FakeAgents):
    def run_claude(
        self,
        prompt,
        chat_id=None,
        session_context=None,
        workspace=None,
        model=None,
        effort=None,
        progress_callback=None,
        run_id=None,
    ):
        self.claude_prompts.append(prompt)
        return SimpleNamespace(text="[NO_REPLY]")


class NoReplyRuntime:
    def __init__(self, transcript_path: Path, offset: int = 0):
        self.transcript_path = transcript_path
        self.offset = offset

    def session_name(self, chat_id):
        return f"pla-claude-{chat_id}"

    def session_exists(self, session_name):
        return True

    def capture(self, session_name):
        return "❯"

    def read_run_cursor(self, session_name, run_id):
        return (str(self.transcript_path), self.offset)


class NoReplyAgents(FakeAgents):
    def __init__(self, runtime: NoReplyRuntime):
        super().__init__()
        self.runtime = runtime

    def runtime_for(self, agent):
        return self.runtime


class AppHandoffTests(unittest.TestCase):
    def test_codebuddy_starts_handoff_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codebuddy"))
            stop = threading.Event()
            with patch.object(bridge, "handoff_loop") as loop:
                worker = bridge.start_handoff_worker(stop)
                if worker:
                    worker.join(timeout=1)
            self.assertIsNotNone(worker)
            loop.assert_called_once()

    def test_gpt_pro_does_not_start_handoff_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "gpt-pro"))
            stop = threading.Event()

            worker = bridge.start_handoff_worker(stop)

            self.assertIsNone(worker)

    def test_plain_reply_does_not_queue_direct_handoff_to_teammate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()
            event = self.event()
            route = Route("agent", text="question", agent="codex")

            bridge.send_reply(event, route, "Codex answer")

            self.assertEqual(bridge.handoffs.pending_for("claude"), [])
            self.assertEqual(bridge.handoffs.pending_for("codex"), [])

    def test_at_peer_in_reply_queues_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()
            event = self.event()
            route = Route("agent", text="question", agent="codex")

            # Codex pulls Claude into the thread by @-addressing it.
            bridge.send_reply(event, route, "我觉得用 A 方案。@Claude 你怎么看 B 的风险?")

            pending = bridge.handoffs.pending_for("claude")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].source_agent, "codex")
            self.assertEqual(pending[0].target_agent, "claude")
            self.assertEqual(bridge.handoffs.pending_for("codex"), [])

    def test_at_peer_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()

            bridge.send_reply(self.event(), Route("agent", text="q", agent="codex"), "ok @claude 看下")

            self.assertEqual(len(bridge.handoffs.pending_for("claude")), 1)

    def test_codex_can_handoff_to_codebuddy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()

            bridge.send_reply(
                self.event(),
                Route("agent", text="q", agent="codex"),
                "我看完了。@CodeBuddy 你接着检查一下实现风险。",
            )

            pending = bridge.handoffs.pending_for("codebuddy")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].source_agent, "codex")
            self.assertEqual(pending[0].target_agent, "codebuddy")
            self.assertEqual(bridge.handoffs.pending_for("claude"), [])

    def test_codebuddy_can_handoff_to_codex(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codebuddy"))
            bridge.lark = FakeLark()

            bridge.send_reply(
                self.event(),
                Route("agent", text="q", agent="codebuddy"),
                "@Codex 我这里建议你复核一下测试覆盖。",
            )

            pending = bridge.handoffs.pending_for("codex")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].source_agent, "codebuddy")
            self.assertEqual(pending[0].target_agent, "codex")

    def test_explicit_at_can_return_to_inbound_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codebuddy"))
            bridge.lark = FakeLark()

            bridge.send_reply(
                self.event(),
                Route("agent", text="q", agent="codebuddy"),
                "@Codex 收到，我建议你复核一下。",
                source_agent="codex",
                handoff_depth=1,
            )

            pending = bridge.handoffs.pending_for("codex")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].source_agent, "codebuddy")
            self.assertEqual(pending[0].target_agent, "codex")
            self.assertEqual(pending[0].depth, 2)

    def test_multiple_mentions_queue_multiple_handoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()

            bridge.send_reply(
                self.event(),
                Route("agent", text="q", agent="codex"),
                "@Claude @CodeBuddy 你们分别看一下。",
            )

            self.assertEqual(len(bridge.handoffs.pending_for("claude")), 1)
            self.assertEqual(len(bridge.handoffs.pending_for("codebuddy")), 1)

    def test_gpt_pro_can_handoff_outbound_to_teammates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "gpt-pro"))
            bridge.lark = FakeLark()

            bridge.send_reply(
                self.event(),
                Route("agent", text="q", agent="gpt-pro"),
                "@Codex @Claude @CodeBuddy 你们分别复核一下。",
            )

            self.assertEqual(len(bridge.handoffs.pending_for("codex")), 1)
            self.assertEqual(len(bridge.handoffs.pending_for("claude")), 1)
            self.assertEqual(len(bridge.handoffs.pending_for("codebuddy")), 1)
            self.assertEqual(bridge.handoffs.pending_for("gpt-pro"), [])

    def test_gpt_pro_is_not_a_handoff_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()

            bridge.send_reply(
                self.event(),
                Route("agent", text="q", agent="codex"),
                "@GPT-Pro 你来看一下。",
            )

            self.assertEqual(bridge.handoffs.pending_for("gpt-pro"), [])

    def test_passing_mention_without_at_does_not_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()

            # Mentions the name but does not address it -> no handoff.
            bridge.send_reply(self.event(), Route("agent", text="q", agent="codex"), "这点和 Claude 之前说的一致")

            self.assertEqual(bridge.handoffs.pending_for("claude"), [])

    def test_at_self_does_not_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()

            # Codex addressing "Codex" is not a handoff to the peer.
            bridge.send_reply(self.event(), Route("agent", text="q", agent="codex"), "@Codex 提醒自己一下")

            self.assertEqual(bridge.handoffs.pending_for("claude"), [])
            self.assertEqual(bridge.handoffs.pending_for("codex"), [])

    def test_debate_reply_queues_direct_handoff_to_teammate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()
            event = self.event()
            route = Route(
                "agent",
                text="Feishu command: /debate\nPrompt:\nquestion",
                agent="codex",
            )

            bridge.send_reply(event, route, "Codex answer")

            pending = bridge.handoffs.pending_for("claude")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].source_agent, "codex")
            self.assertEqual(pending[0].target_agent, "claude")
            self.assertEqual(pending[0].content, "Codex answer")
            self.assertEqual(bridge.handoffs.pending_for("codex"), [])

    def test_target_process_consumes_handoff_and_sends_as_own_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = PaperAgentBridge(self.settings(root, "codex"))
            codex.lark = FakeLark()
            event = self.event()
            codex.send_reply(
                event,
                Route(
                    "agent",
                    text="Feishu command: /debate\nPrompt:\nquestion",
                    agent="codex",
                ),
                "Codex answer",
            )

            claude = PaperAgentBridge(self.settings(root, "claude", max_turns=1))
            fake_lark = FakeLark()
            fake_agents = FakeAgents()
            claude.lark = fake_lark
            claude.agents = fake_agents

            claude.process_pending_handoffs("claude")
            import time; time.sleep(0.5)  # handoff dispatch runs in a thread

            self.assertIn("[codex]", fake_agents.claude_prompts[0])
            self.assertEqual(fake_lark.markdowns, [("oc_a", "Claude follow-up")])
            self.assertEqual(claude.handoffs.pending_for("claude"), [])
            self.assertEqual(claude.handoffs.pending_for("codex"), [])

    def test_codebuddy_process_consumes_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = PaperAgentBridge(self.settings(root, "codex"))
            codex.lark = FakeLark()
            codex.send_reply(
                self.event(),
                Route("agent", text="q", agent="codex"),
                "@CodeBuddy 请接着看。",
            )

            codebuddy = PaperAgentBridge(self.settings(root, "codebuddy", max_turns=1))
            fake_lark = FakeLark()
            fake_agents = FakeAgents()
            codebuddy.lark = fake_lark
            codebuddy.agents = fake_agents

            codebuddy.process_pending_handoffs("codebuddy")
            import time; time.sleep(0.5)

            self.assertIn("[codex]", fake_agents.codebuddy_prompts[0])
            self.assertEqual(fake_lark.markdowns, [("oc_a", "CodeBuddy follow-up")])
            self.assertEqual(codebuddy.handoffs.pending_for("codebuddy"), [])

    def test_handoff_prompt_includes_room_thread_claude_missed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Codex pulls Claude in after a human<->codex thread.
            codex = PaperAgentBridge(self.settings(root, "codex"))
            codex.lark = FakeLark()
            human = MessageEvent(
                event_id="evt_h", chat_id="oc_a", chat_type="group",
                content="Is the ablation convincing?", sender_id="ou_user",
                message_id="om_h", message_type="text",
            )
            # The human's question + codex's prior reply land in shared memory,
            # exactly as the normal flow records them.
            codex.memory.append_user(human, "Is the ablation convincing?")
            codex.memory.append_assistant("oc_a", "codex", "Mostly, but the seed variance worries me.")
            codex.send_reply(
                self.event(),
                Route("agent", text="q", agent="codex"),
                "@Claude can you sanity-check the seed-variance concern?",
            )

            claude = PaperAgentBridge(self.settings(root, "claude", max_turns=6))
            fake_agents = FakeAgents()
            claude.lark = FakeLark()
            claude.agents = fake_agents
            claude.process_pending_handoffs("claude")
            import time; time.sleep(0.5)

            prompt = fake_agents.claude_prompts[0]
            # Claude now sees the human's original ask and codex's earlier turn,
            # even though it never processed those messages itself.
            self.assertIn("Is the ablation convincing?", prompt)
            self.assertIn("seed variance", prompt)
            self.assertIn("[codex]", prompt)

    def test_human_turn_has_no_recap_when_no_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Default responder is "both", and no peer turns since claude's
            # last turn — no gap, no recap needed.
            bridge = PaperAgentBridge(self.settings(root, "claude"))
            bridge.memory.append_assistant("oc_a", "claude", "claude answered")
            prompt, _ = bridge.build_agent_prompt(
                "claude", self.event(), "hello", None, root
            )
            self.assertNotIn("---\n\n", prompt)

    def test_human_at_addressing_out_of_loop_agent_gets_recap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "claude"))
            # Codex owns broadcasts here, so Claude has been skipping the thread.
            bridge.responders.set("oc_a", "codex")
            bridge.memory.append_assistant("oc_a", "codex", "codex handled this solo")
            # A human now @-addresses Claude directly (source_agent is None).
            prompt, _ = bridge.build_agent_prompt(
                "claude", self.event(), "@claude what do you think?", None, root
            )
            self.assertIn("---\n\n", prompt)
            self.assertIn("codex handled this solo", prompt)

    def test_broadcast_responder_gets_recap_when_peer_replied_since_last_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Codex is the broadcast responder — normally no recap.
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.responders.set("oc_a", "codex")
            # Codex answered, then Claude replied (codex never saw this).
            bridge.memory.append_assistant("oc_a", "codex", "codex earlier")
            bridge.memory.append_assistant("oc_a", "claude", "claude chimed in")
            # Next human message to codex should include recap.
            prompt, _ = bridge.build_agent_prompt(
                "codex", self.event(), "new question", None, root
            )
            self.assertIn("---\n\n", prompt)
            self.assertIn("claude chimed in", prompt)

    def test_broadcast_responder_no_recap_when_no_peer_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.responders.set("oc_a", "codex")
            # Only codex turns — no peer gap, no recap needed.
            bridge.memory.append_assistant("oc_a", "codex", "codex answer")
            prompt, _ = bridge.build_agent_prompt(
                "codex", self.event(), "follow-up", None, root
            )
            self.assertNotIn("---\n\n", prompt)

    def test_no_reply_marks_agent_seen_so_next_turn_does_not_repeat_recap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "claude"))
            bridge.lark = FakeLark()
            fake_agents = NoReplyPromptAgents()
            bridge.agents = fake_agents
            bridge.responders.set("oc_a", "codex")
            bridge.memory.append_assistant("oc_a", "codex", "codex handled this solo")

            first = MessageEvent(
                event_id="evt_1",
                chat_id="oc_a",
                chat_type="group",
                content="@Claude welcome back",
                sender_id="ou_user",
                message_id="om_1",
                message_type="text",
            )
            second = MessageEvent(
                event_id="evt_2",
                chat_id="oc_a",
                chat_type="group",
                content="@Claude welcome again",
                sender_id="ou_user",
                message_id="om_2",
                message_type="text",
            )

            bridge.handle_event(first)
            bridge.handle_event(second)

            self.assertEqual(len(fake_agents.claude_prompts), 2)
            self.assertIn("codex handled this solo", fake_agents.claude_prompts[0])
            self.assertNotIn("codex handled this solo", fake_agents.claude_prompts[1])
            self.assertEqual(bridge.lark.markdowns, [])
            context = bridge.memory.context("oc_a")
            self.assertNotIn("[NO_REPLY]", context)
            self.assertNotIn("agent_seen", context)

    def test_sent_message_ids_are_recorded_as_non_discussion_when_direct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()

            bridge.send_bridge_markdown(
                "oc_a",
                "Codex answer",
                agent="codex",
                discussion_trigger=False,
            )

            match = bridge.outbox.match_message_id("oc_a", "om_fake_1")
            self.assertIsNotNone(match)
            self.assertEqual(match["agent"], "codex")
            self.assertFalse(match["discussion_trigger"])

    def test_recovers_completed_pending_run_and_queues_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "claude", max_turns=3))
            fake_lark = FakeLark()
            fake_agents = FakeAgents()
            fake_agents.recovered_reply = "Recovered Claude answer"
            bridge.lark = fake_lark
            bridge.agents = fake_agents
            run_id = "abc123"
            start_marker, end_marker = fake_agents.reply_markers("claude", run_id)
            bridge.pending_runs.start(
                run_id=run_id,
                chat_id="oc_a",
                agent="claude",
                route_text="run this",
                event_id="evt_a",
                message_id="om_user",
                sender_id="ou_user",
                message_type="text",
                chat_type="group",
                event_content="claude run this",
                source_agent=None,
                handoff_depth=0,
                start_marker=start_marker,
                end_marker=end_marker,
                session_name="pla-claude-oc_a",
                workspace=str(root),
                status_message_id=None,
                model_label="CLI default",
                effort_label="CLI default",
                timeout=900,
            )

            bridge.process_pending_runs("claude")

            self.assertEqual(fake_lark.markdowns, [("oc_a", "Recovered Claude answer")])
            self.assertEqual(bridge.handoffs.pending_for("codex"), [])
            self.assertEqual(bridge.pending_runs.pending_for("claude"), [])

    def test_still_running_agent_keeps_pending_run_for_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "claude"))
            fake_lark = FakeLark()
            bridge.lark = fake_lark
            bridge.agents = StillRunningAgents()
            event = self.event()

            reply = bridge.dispatch(Route("agent", text="run a long job", agent="claude"), event)

            self.assertEqual(reply, "")
            self.assertEqual(fake_lark.markdowns, [])
            pending = bridge.pending_runs.pending_for("claude")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].route_text, "run a long job")

    def test_recent_transcript_activity_prevents_pending_run_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "claude"))
            bridge.lark = FakeLark()
            transcript = root / "claude.jsonl"
            transcript.write_text('{"type":"response_item"}\n', encoding="utf-8")
            start = os.path.getmtime(transcript)
            future = start + 3
            os.utime(transcript, (future, future))
            bridge.agents = NoReplyAgents(NoReplyRuntime(transcript, offset=0))
            run_id = "abc123"
            start_marker, end_marker = bridge.agents.reply_markers("claude", run_id)
            bridge.pending_runs.start(
                run_id=run_id,
                chat_id="oc_a",
                agent="claude",
                route_text="run this",
                event_id="evt_a",
                message_id="om_user",
                sender_id="ou_user",
                message_type="text",
                chat_type="group",
                event_content="claude run this",
                source_agent=None,
                handoff_depth=0,
                start_marker=start_marker,
                end_marker=end_marker,
                session_name="pla-claude-oc_a",
                workspace=str(root),
                status_message_id=None,
                model_label="CLI default",
                effort_label="CLI default",
                timeout=1,
            )

            with patch("paper_lark_agents.app.time.time", return_value=future):
                bridge.process_pending_runs("claude")

            pending = bridge.pending_runs.pending_for("claude")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].run_id, run_id)

    def test_stale_message_event_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()
            fake_agents = FakeAgents()
            bridge.agents = fake_agents
            event = MessageEvent(
                event_id="evt_old",
                chat_id="oc_a",
                chat_type="group",
                content="old question",
                sender_id="ou_user",
                message_id="om_old",
                message_type="text",
                create_time="1",
            )

            bridge.handle_event(event)

            self.assertEqual(fake_agents.codex_prompts, [])
            self.assertEqual(bridge.lark.markdowns, [])

    def test_duplicate_message_agent_run_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bridge = PaperAgentBridge(self.settings(root, "codex"))
            bridge.lark = FakeLark()
            fake_agents = FakeAgents()
            bridge.agents = fake_agents
            event = self.event()

            bridge.handle_event(event)
            bridge.handle_event(event)

            self.assertEqual(len(fake_agents.codex_prompts), 1)

    def settings(
        self,
        root: Path,
        agent_mode: str,
        max_turns: int = 6,
    ):
        with patch.dict(
            os.environ,
            {
                "PLA_WORKSPACE": str(root),
                "PLA_WORKSPACE_ROOTS": str(root),
                "PLA_STATE_DIR": str(root / ".state"),
                "PLA_AGENT_MODE": agent_mode,
                "PLA_RESPOND_TO_ALL": "true",
                "PLA_ENABLE_AGENT_DISCUSSION": "true",
                "PLA_DIRECT_AGENT_HANDOFF": "true",
                "PLA_MAX_AGENT_DISCUSSION_TURNS": str(max_turns),
                "PLA_SEND_PROGRESS": "false",
            },
            clear=True,
        ):
            return load_settings(None)

    def event(self):
        return MessageEvent(
            event_id="evt_a",
            chat_id="oc_a",
            chat_type="group",
            content="question",
            sender_id="ou_user",
            message_id="om_user",
            message_type="text",
        )


if __name__ == "__main__":
    unittest.main()
