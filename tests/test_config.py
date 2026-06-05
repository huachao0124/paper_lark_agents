import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_lark_agents.config import load_settings, proxy_command_prefix, proxy_env


class ConfigTests(unittest.TestCase):
    def test_event_keys_default_to_event_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PLA_WORKSPACE": tmp}, clear=True):
                settings = load_settings(None)

        self.assertEqual(settings.event_key, "im.message.receive_v1")
        self.assertEqual(settings.event_keys, ("im.message.receive_v1",))

    def test_event_keys_can_include_bot_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        f"PLA_WORKSPACE={tmp}",
                        (
                            "PLA_EVENT_KEYS=im.message.receive_v1,"
                            "im.chat.member.bot.added_v1,"
                            "im.chat.member.bot.deleted_v1,"
                            "im.chat.disbanded_v1"
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(str(env_path))

        self.assertEqual(
            settings.event_keys,
            (
                "im.message.receive_v1",
                "im.chat.member.bot.added_v1",
                "im.chat.member.bot.deleted_v1",
                "im.chat.disbanded_v1",
            ),
        )

    def test_export_style_env_lines_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        f"PLA_WORKSPACE={tmp}",
                        'export PLA_PROXY_URL="http://star-proxy.oa.com:3128"',
                        "export PLA_AGENT_PROXY_URL=http://127.0.0.1:7899",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                settings = load_settings(str(env_path))

        self.assertEqual(settings.proxy_url, "http://star-proxy.oa.com:3128")
        self.assertEqual(settings.agent_proxy_url, "http://127.0.0.1:7899")

    def test_proxy_env_sets_lowercase_and_uppercase_vars(self):
        env = proxy_env(
            "http://star-proxy.oa.com:3128",
            "localhost,127.0.0.1",
            base={},
        )

        self.assertEqual(env["http_proxy"], "http://star-proxy.oa.com:3128")
        self.assertEqual(env["https_proxy"], "http://star-proxy.oa.com:3128")
        self.assertEqual(env["HTTP_PROXY"], "http://star-proxy.oa.com:3128")
        self.assertEqual(env["HTTPS_PROXY"], "http://star-proxy.oa.com:3128")
        self.assertEqual(env["no_proxy"], "localhost,127.0.0.1")
        self.assertEqual(env["NO_PROXY"], "localhost,127.0.0.1")

    def test_proxy_command_prefix_sets_agent_proxy(self):
        prefix = proxy_command_prefix("http://127.0.0.1:7899", "localhost")

        self.assertEqual(prefix[0], "env")
        self.assertIn("http_proxy=http://127.0.0.1:7899", prefix)
        self.assertIn("https_proxy=http://127.0.0.1:7899", prefix)
        self.assertIn("no_proxy=localhost", prefix)

    def test_workspace_warmup_agents_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "PLA_WORKSPACE": tmp,
                    "PLA_WORKSPACE_WARMUP_AGENTS": "codex,claude,unknown,codex",
                },
                clear=True,
            ):
                settings = load_settings(None)

        self.assertEqual(settings.workspace_warmup_agents, ("codex", "claude"))

    def test_session_capture_and_history_defaults_cover_long_tui_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PLA_WORKSPACE": tmp}, clear=True):
                settings = load_settings(None)

        self.assertEqual(settings.session_capture_lines, 20000)
        self.assertEqual(settings.session_history_limit, 50000)
        self.assertEqual(settings.session_columns, 120)
        self.assertEqual(settings.session_rows, 80)
        self.assertEqual(settings.session_command_watch_seconds, 20)

    def test_direct_agent_handoff_defaults_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"PLA_WORKSPACE": tmp}, clear=True):
                settings = load_settings(None)

        self.assertTrue(settings.direct_agent_handoff)


if __name__ == "__main__":
    unittest.main()
