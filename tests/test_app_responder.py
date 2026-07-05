import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_lark_agents.app import PaperAgentBridge
from paper_lark_agents.config import load_settings
from paper_lark_agents.lark_cli import MessageEvent
from paper_lark_agents.router import Route


def make_event(content: str, chat_id: str = "oc_a") -> MessageEvent:
    return MessageEvent(
        event_id="evt_1",
        chat_id=chat_id,
        chat_type="group",
        content=content,
        sender_id="ou_user",
        message_id="om_user",
        message_type="text",
    )


class ResponderGateTests(unittest.TestCase):
    def settings(self, root: Path, **overrides):
        env = {
            "PLA_WORKSPACE": str(root),
            "PLA_STATE_DIR": str(root / ".state"),
            "PLA_AGENT_MODE": "claude",
            "PLA_RESPOND_TO_ALL": "true",
            "PLA_SEND_PROGRESS": "false",
            "PLA_ENABLE_MEMORY": "false",
        }
        env.update(overrides)
        with patch.dict(os.environ, env, clear=True):
            return load_settings(None)

    def bridge(self, root: Path, **overrides) -> PaperAgentBridge:
        return PaperAgentBridge(self.settings(root, **overrides))

    def test_command_sets_override_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))
            reply = bridge.handle_responder_command("oc_a", "codex")
            self.assertIn("Codex", reply)
            self.assertEqual(bridge.responders.current("oc_a"), "codex")
            self.assertTrue(bridge.responders.has_override("oc_a"))

    def test_command_sets_codebuddy_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))
            reply = bridge.handle_responder_command("oc_a", "codebuddy")
            self.assertIn("CodeBuddy", reply)
            self.assertEqual(bridge.responders.current("oc_a"), "codebuddy")

    def test_command_reset_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp), PLA_DEFAULT_RESPONDER="claude")
            bridge.handle_responder_command("oc_a", "codex")
            reset_reply = bridge.handle_responder_command("oc_a", "reset")
            self.assertIn("Claude", reset_reply)
            self.assertFalse(bridge.responders.has_override("oc_a"))
            bad = bridge.handle_responder_command("oc_a", "gemini")
            self.assertIn("Usage", bad)

    def test_apply_default_responder_gates_agent_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))
            bridge.responders.set("oc_a", "codex")
            claude_route = Route("agent", text="hi", agent="claude", broadcast=True)
            codex_route = Route("agent", text="hi", agent="codex", broadcast=True)
            self.assertIsNone(bridge.apply_default_responder(claude_route, "oc_a"))
            self.assertIs(bridge.apply_default_responder(codex_route, "oc_a"), codex_route)

    def test_apply_default_responder_filters_multi_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))
            bridge.responders.set("oc_a", "codex")
            route = Route(
                "multi_agent",
                text="hi",
                agent_texts={"codex": "hi", "claude": "hi"},
                broadcast=True,
            )
            gated = bridge.apply_default_responder(route, "oc_a")
            self.assertIsNotNone(gated)
            self.assertEqual(gated.agent_texts, {"codex": "hi"})

    def test_handle_event_skips_broadcast_for_non_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))  # this process is the Claude bot
            bridge.responders.set("oc_a", "codex")  # but Codex owns plain messages
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""
            bridge.handle_event(make_event("what should we read first?"))
            self.assertEqual(calls, [])

    def test_handle_event_runs_broadcast_for_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))  # this process is the Claude bot
            bridge.responders.set("oc_a", "claude")  # Claude owns plain messages
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""
            bridge.handle_event(make_event("what should we read first?"))
            self.assertEqual(len(calls), 1)

    def test_handle_event_runs_explicit_mention_regardless(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))  # Claude bot
            bridge.responders.set("oc_a", "codex")  # Codex is default responder
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""
            # Explicit @Claude must still reach Claude even though Codex is default.
            bridge.handle_event(make_event("@Claude critique the baseline"))
            self.assertEqual(len(calls), 1)

    def test_handle_event_runs_middle_mention_regardless(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(
                Path(tmp),
                PLA_STRICT_ALIAS_ROUTING="true",
                PLA_BOT_ALIASES="Claude",
            )
            bridge.responders.set("oc_a", "codex")
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""

            bridge.handle_event(make_event("这个问题 @Claude 看看"))

            self.assertEqual(bridge.responders.current("oc_a"), "claude")
            self.assertEqual(len(calls), 1)

    def test_handle_event_runs_later_mention_when_first_at_is_other_bot(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(
                Path(tmp),
                PLA_STRICT_ALIAS_ROUTING="true",
                PLA_BOT_ALIASES="Claude",
            )
            bridge.responders.set("oc_a", "codex")
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""

            bridge.handle_event(make_event("@Codex @Claude 都看看"))

            self.assertEqual(bridge.responders.current("oc_a"), "claude")
            self.assertEqual(len(calls), 1)

    def test_human_mention_switches_default_responder(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))  # Claude bot
            bridge.responders.set("oc_a", "codex")
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""

            bridge.handle_event(make_event("@Claude critique the baseline"))
            bridge.handle_event(make_event("继续说一下"))

            self.assertEqual(bridge.responders.current("oc_a"), "claude")
            self.assertEqual(len(calls), 2)

    def test_last_human_mention_wins_for_default_responder(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))  # Claude bot
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""

            bridge.handle_event(make_event("@Codex 先看 @Claude 再补一下"))

            self.assertEqual(bridge.responders.current("oc_a"), "claude")
            self.assertEqual(len(calls), 1)

    def test_human_mention_to_other_agent_makes_this_process_skip_plain_followup(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))  # Claude bot
            calls = []
            bridge.dispatch = lambda *a, **k: calls.append((a, k)) or ""

            bridge.handle_event(make_event("@Codex 你先看"))
            bridge.handle_event(make_event("继续说一下"))

            self.assertEqual(bridge.responders.current("oc_a"), "codex")
            self.assertEqual(calls, [])

    def test_mention_of_undeployed_agent_does_not_hijack_responder(self):
        # @CodeBuddy when codebuddy is NOT in the deployed roster must not point
        # the shared responder at a phantom — that would silence every real bot.
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp))  # deployed_agents defaults to codex,claude
            bridge.dispatch = lambda *a, **k: ""
            bridge.handle_event(make_event("@CodeBuddy 你看下"))
            # Responder unchanged (still the default), so codex/claude keep answering.
            self.assertFalse(bridge.responders.has_override("oc_a"))

    def test_mention_of_deployed_codebuddy_switches_responder(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = self.bridge(Path(tmp), PLA_DEPLOYED_AGENTS="codex,claude,codebuddy")
            bridge.dispatch = lambda *a, **k: ""
            bridge.handle_event(make_event("@CodeBuddy 你看下"))
            self.assertEqual(bridge.responders.current("oc_a"), "codebuddy")


if __name__ == "__main__":
    unittest.main()
