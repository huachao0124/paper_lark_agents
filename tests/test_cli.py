import unittest
from unittest.mock import patch

from paper_lark_agents import cli


class FakeProcess:
    def __init__(self, poll_values):
        self.poll_values = list(poll_values)
        self.terminated = False
        self.waited = False

    def poll(self):
        if len(self.poll_values) > 1:
            return self.poll_values.pop(0)
        return self.poll_values[0]

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


class ServeDuoTests(unittest.TestCase):
    def test_serve_duo_restarts_agent_child_instead_of_exiting(self):
        calls = []
        processes = []
        poll_sequences = [[1], [None], [0], [None]]

        def fake_popen(cmd):
            calls.append(cmd)
            proc = FakeProcess(poll_sequences[len(calls) - 1])
            processes.append(proc)
            return proc

        def fake_sleep(seconds):
            if seconds == 1:
                raise KeyboardInterrupt

        with patch("paper_lark_agents.cli.subprocess.Popen", side_effect=fake_popen):
            with patch("paper_lark_agents.cli.time.sleep", side_effect=fake_sleep):
                code = cli.serve_duo(".env.codex", ".env.claude")

        self.assertEqual(code, 130)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[0], calls[3])
        self.assertFalse(processes[0].terminated)
        self.assertTrue(processes[1].terminated)
        self.assertFalse(processes[2].terminated)
        self.assertTrue(processes[3].terminated)


class OomProtectionTests(unittest.TestCase):
    def test_sets_default_oom_score_adjustment(self):
        with patch.dict(cli.os.environ, {}, clear=True):
            with patch.object(cli.Path, "write_text") as write_text:
                cli._protect_bridge_process_from_oom()

        write_text.assert_called_once_with("-800\n", encoding="utf-8")

    def test_can_disable_oom_score_adjustment(self):
        with patch.dict(cli.os.environ, {"PLA_OOM_SCORE_ADJ": "off"}, clear=True):
            with patch.object(cli.Path, "write_text") as write_text:
                cli._protect_bridge_process_from_oom()

        write_text.assert_not_called()

    def test_clamps_oom_score_adjustment(self):
        with patch.dict(cli.os.environ, {"PLA_OOM_SCORE_ADJ": "-2000"}, clear=True):
            with patch.object(cli.Path, "write_text") as write_text:
                cli._protect_bridge_process_from_oom()

        write_text.assert_called_once_with("-1000\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
