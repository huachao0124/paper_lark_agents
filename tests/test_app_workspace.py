import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_lark_agents.app import PaperAgentBridge
from paper_lark_agents.config import load_settings
from paper_lark_agents.lark_cli import LarkEvent, MessageEvent


class FakeAgents:
    def __init__(self, ready=True):
        self.warmups = []
        self.commands = []
        self.resets = []
        self.ready = ready

    def warmup_session(self, agent, chat_id, workspace=None, model=None, effort=None):
        self.warmups.append((agent, chat_id, workspace, model, effort))
        return f"pla-{agent}-{chat_id}"

    def send_session_command(
        self,
        agent,
        chat_id,
        command,
        workspace=None,
        model=None,
        effort=None,
    ):
        self.commands.append((agent, chat_id, command, workspace, model, effort))
        return True

    def wait_session_ready(self, agent, chat_id, timeout):
        return self.ready

    def detect_session_model(self, agent, chat_id):
        return None

    def detect_session_effort(self, agent, chat_id):
        return None

    def session_progress(self, agent, chat_id):
        return "Ready"

    def reset_session(self, agent, chat_id):
        self.resets.append((agent, chat_id))
        return []


class FakeLark:
    def __init__(self):
        self.cards = []
        self.updates = []
        self.markdowns = []
        self.pins = []

    def send_card(self, chat_id, card):
        self.cards.append((chat_id, card))
        return {"message_id": "card_1"}

    def update_card(self, message_id, card):
        self.updates.append((message_id, card))
        return {}

    def pin_message(self, message_id):
        self.pins.append(message_id)
        return {}

    def send_markdown(self, chat_id, markdown):
        self.markdowns.append((chat_id, markdown))
        return [{"message_id": f"om_fake_{len(self.markdowns)}"}]


class AppWorkspaceTests(unittest.TestCase):
    def test_bot_added_does_not_warm_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self.settings(root)
            bridge = PaperAgentBridge(settings)
            fake_agents = FakeAgents()
            bridge.agents = fake_agents

            bridge.handle_bot_added_event(LarkEvent("im.chat.member.bot.added_v1", "evt", "oc_a", {}))

            self.assertEqual(fake_agents.warmups, [])

    def test_workspace_set_warms_configured_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            settings = self.settings(root)
            bridge = PaperAgentBridge(settings)
            fake_agents = FakeAgents()
            bridge.agents = fake_agents
            fake_lark = FakeLark()
            bridge.lark = fake_lark

            reply = bridge.handle_workspace_command("oc_a", "project")

            self.assertIn("Workspace set for this group", reply)
            self.assertIn("Sessions started", reply)
            self.assertIn("Init completed", reply)
            self.assertEqual(
                [(agent, chat_id, workspace) for agent, chat_id, workspace, _, _ in fake_agents.warmups],
                [
                    ("codex", "oc_a", project.resolve()),
                    ("claude", "oc_a", project.resolve()),
                ],
            )
            self.assertEqual(
                [
                    (agent, chat_id, command, workspace)
                    for agent, chat_id, command, workspace, _, _ in fake_agents.commands
                ],
                [
                    ("codex", "oc_a", "/init", project.resolve()),
                    ("claude", "oc_a", "/init", project.resolve()),
                ],
            )
            self.assertEqual(len(fake_lark.cards), 1)
            self.assertEqual(fake_lark.pins, ["card_1"])
            self.assertGreaterEqual(len(fake_lark.updates), 1)
            final_card = fake_lark.updates[-1][1]
            self.assertEqual(final_card["header"]["title"]["content"], "AI Status")
            self.assertEqual(final_card["header"]["template"], "green")
            self.assertIn("Init completed", final_card["elements"][0]["text"]["content"])

    def test_clear_resets_group_state_and_preserves_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            settings = self.settings(root)
            bridge = PaperAgentBridge(settings)
            fake_agents = FakeAgents()
            bridge.agents = fake_agents
            bridge.workspaces.set("oc_a", "project")
            event = MessageEvent(
                event_id="evt_a",
                chat_id="oc_a",
                chat_type="group",
                content="question",
                sender_id="ou_user",
                message_id="om_user",
            )
            bridge.memory.append_user(event, "question")
            bridge.outbox.remember("oc_a", "reply", 3500, agent="codex")
            bridge.handoffs.enqueue(
                "oc_a",
                source_agent="codex",
                target_agent="claude",
                content="reply",
                origin_event_id="evt_a",
                origin_message_id="om_reply",
                sender_id="assistant:codex",
                depth=1,
            )

            reply = bridge.handle_clear_command("oc_a")

            self.assertIn("Cleared this group's bridge state", reply)
            self.assertEqual(
                fake_agents.resets,
                [
                    ("codex", "oc_a"),
                    ("claude", "oc_a"),
                ],
            )
            self.assertEqual(
                bridge.memory.context("oc_a"),
                "No previous discussion in this Feishu group yet.",
            )
            self.assertIsNone(bridge.outbox.match("oc_a", "reply"))
            self.assertEqual(bridge.handoffs.pending_for("claude"), [])
            self.assertEqual(bridge.chat_workspace("oc_a"), project.resolve())
            self.assertEqual(fake_agents.warmups, [])

    def test_clear_init_restarts_sessions_and_runs_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            settings = self.settings(root)
            bridge = PaperAgentBridge(settings)
            fake_agents = FakeAgents()
            bridge.agents = fake_agents
            fake_lark = FakeLark()
            bridge.lark = fake_lark
            bridge.workspaces.set("oc_a", "project")

            reply = bridge.handle_clear_command("oc_a", "init")

            self.assertIn("Cleared this group's bridge state", reply)
            self.assertIn("Sessions started", reply)
            self.assertIn("Init completed", reply)
            self.assertEqual(
                fake_agents.resets,
                [
                    ("codex", "oc_a"),
                    ("claude", "oc_a"),
                ],
            )
            self.assertEqual(
                [(agent, chat_id, workspace) for agent, chat_id, workspace, _, _ in fake_agents.warmups],
                [
                    ("codex", "oc_a", project.resolve()),
                    ("claude", "oc_a", project.resolve()),
                ],
            )
            self.assertEqual(
                [
                    (agent, chat_id, command, workspace)
                    for agent, chat_id, command, workspace, _, _ in fake_agents.commands
                ],
                [
                    ("codex", "oc_a", "/init", project.resolve()),
                    ("claude", "oc_a", "/init", project.resolve()),
                ],
            )
            self.assertEqual(len(fake_lark.cards), 1)
            self.assertEqual(fake_lark.pins, ["card_1"])
            self.assertGreaterEqual(len(fake_lark.updates), 1)
            self.assertEqual(len(fake_lark.markdowns), 0)
            final_card = fake_lark.updates[-1][1]
            self.assertEqual(final_card["header"]["template"], "green")
            self.assertIn("Init completed", final_card["elements"][0]["text"]["content"])

    def test_clear_init_pending_updates_status_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            settings = self.settings(root)
            bridge = PaperAgentBridge(settings)
            fake_agents = FakeAgents(ready=False)
            bridge.agents = fake_agents
            fake_lark = FakeLark()
            bridge.lark = fake_lark
            bridge.workspaces.set("oc_a", "project")

            reply = bridge.handle_clear_command("oc_a", "init")

            self.assertIn("Init still running", reply)
            self.assertNotIn("timed out", reply.lower())
            self.assertEqual(len(fake_lark.cards), 1)
            final_card = fake_lark.updates[-1][1]
            self.assertEqual(final_card["header"]["template"], "yellow")
            self.assertIn("Pending", card_text(final_card))

    def test_effort_command_sets_effort_and_confirms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self.settings(root)
            bridge = PaperAgentBridge(settings)
            fake_agents = FakeAgents()
            bridge.agents = fake_agents
            fake_lark = FakeLark()
            bridge.lark = fake_lark

            reply = bridge.handle_session_command("claude", "oc_a", "/effort ultracode")

            # User gets a text confirmation (not "" + a dead status card).
            self.assertIn("ultracode", reply)
            self.assertIn("effort", reply.lower())
            self.assertEqual(bridge.efforts.current("oc_a", "claude"), "ultracode")
            self.assertEqual(
                fake_agents.commands,
                [("claude", "oc_a", "/effort ultracode", root.resolve(), None, "ultracode")],
            )
            self.assertEqual(fake_agents.resets, [])
            # Session commands no longer create/patch the deprecated status card.
            self.assertEqual(fake_lark.cards, [])
            self.assertEqual(fake_lark.updates, [])

    def test_codex_effort_command_does_not_reset_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self.settings(root)
            bridge = PaperAgentBridge(settings)
            fake_agents = FakeAgents()
            bridge.agents = fake_agents
            fake_lark = FakeLark()
            bridge.lark = fake_lark

            reply = bridge.handle_session_command("codex", "oc_a", "/effort ultracode")

            self.assertIn("ultracode", reply)
            self.assertEqual(bridge.efforts.current("oc_a", "codex"), "ultracode")
            self.assertEqual(
                fake_agents.commands,
                [("codex", "oc_a", "/effort ultracode", root.resolve(), None, "ultracode")],
            )
            # Setting effort must not recreate the session (loses agent context).
            self.assertEqual(fake_agents.resets, [])

    def settings(self, root: Path):
        with patch.dict(
            os.environ,
            {
                "PLA_WORKSPACE": str(root),
                "PLA_WORKSPACE_ROOTS": str(root),
                "PLA_STATE_DIR": str(root / ".state"),
                "PLA_AGENT_MODE": "codex",
                "PLA_WORKSPACE_WARMUP_AGENTS": "codex,claude",
                "PLA_HANDLE_MANAGEMENT_COMMANDS": "true",
                "PLA_SESSION_COMMAND_WATCH_SECONDS": "1",
                "PLA_STATUS_UPDATE_SECONDS": "1",
            },
            clear=True,
        ):
            return load_settings(None)


def card_text(value):
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if key == "content" and isinstance(item, str):
                parts.append(item)
            else:
                parts.append(card_text(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, list):
        return "\n".join(card_text(item) for item in value)
    return ""


if __name__ == "__main__":
    unittest.main()
