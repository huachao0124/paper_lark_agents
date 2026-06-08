import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
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


class AppHandoffTests(unittest.TestCase):
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

            self.assertIn("source: assistant:codex", fake_agents.claude_prompts[0])
            self.assertEqual(fake_lark.markdowns, [("oc_a", "Claude follow-up")])
            self.assertEqual(claude.handoffs.pending_for("claude"), [])
            self.assertEqual(claude.handoffs.pending_for("codex"), [])

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

            prompt = fake_agents.claude_prompts[0]
            # Claude now sees the human's original ask and codex's earlier turn,
            # even though it never processed those messages itself.
            self.assertIn("Is the ablation convincing?", prompt)
            self.assertIn("seed variance", prompt)
            self.assertIn("source: assistant:codex", prompt)

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
            self.assertNotIn("Recent conversation in this Feishu room", prompt)

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
            self.assertIn("Recent conversation in this Feishu room", prompt)
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
            self.assertIn("Recent conversation in this Feishu room", prompt)
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
            self.assertNotIn("Recent conversation in this Feishu room", prompt)

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

    def settings(self, root: Path, agent_mode: str, max_turns: int = 6):
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
