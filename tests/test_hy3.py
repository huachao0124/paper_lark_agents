import os
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper_lark_agents.hy3 import (
    Hy3Client,
    Hy3Config,
    Hy3Error,
    load_internal_api_creds,
)


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout=stdout, stderr=stderr)


class LoadCredsTests(unittest.TestCase):
    def test_parses_export_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ".env").write_text(
                'export INTERNAL_API_USER="abc_user"\n'
                "export INTERNAL_API_TOKEN='tok123'\n"
                "# export INTERNAL_API_USER=\"commented\"\n",
                encoding="utf-8",
            )
            user, token = load_internal_api_creds(tmp)
            self.assertEqual(user, "abc_user")
            self.assertEqual(token, "tok123")

    def test_missing_file_returns_blanks(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_internal_api_creds(tmp), ("", ""))


class Hy3ClientTests(unittest.TestCase):
    def config(self, **overrides):
        base = dict(user="u", token="t", code_path="/code", model="hunyuan-3.0-preview-taiji")
        base.update(overrides)
        return Hy3Config(**base)

    def test_ask_returns_stripped_stdout(self):
        client = Hy3Client(self.config())
        with patch("paper_lark_agents.hy3.subprocess.run", return_value=_completed(stdout="claude\n")) as run:
            self.assertEqual(client.ask("who?"), "claude")
        # prompt is passed via stdin
        self.assertEqual(run.call_args.kwargs["input"], "who?")

    def test_ask_strips_proxy_and_sets_config_env(self):
        client = Hy3Client(self.config())
        with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:7899", "HTTPS_PROXY": "http://127.0.0.1:7899", "PATH": os.environ.get("PATH", "")}, clear=True):
            with patch("paper_lark_agents.hy3.subprocess.run", return_value=_completed(stdout="ok")) as run:
                client.ask("x")
        env = run.call_args.kwargs["env"]
        # proxy vars stripped (both cases)
        self.assertNotIn("http_proxy", env)
        self.assertNotIn("HTTPS_PROXY", env)
        # config injected
        self.assertEqual(env["INTERNAL_API_USER"], "u")
        self.assertEqual(env["INTERNAL_API_TOKEN"], "t")
        self.assertEqual(env["HY3_CODE_PATH"], "/code")
        self.assertEqual(env["HY3_MODEL"], "hunyuan-3.0-preview-taiji")
        self.assertEqual(env["EVAL_TASK_CREATOR"], "arimazhu")
        self.assertEqual(env["EVAL_TASK_NAME"], "debug")

    def test_missing_creds_raises_without_calling(self):
        client = Hy3Client(self.config(user="", token=""))
        with patch("paper_lark_agents.hy3.subprocess.run") as run:
            with self.assertRaises(Hy3Error):
                client.ask("x")
        run.assert_not_called()

    def test_nonzero_exit_raises(self):
        client = Hy3Client(self.config())
        with patch("paper_lark_agents.hy3.subprocess.run", return_value=_completed(stderr="boom", returncode=3)):
            with self.assertRaises(Hy3Error):
                client.ask("x")

    def test_empty_output_raises(self):
        client = Hy3Client(self.config())
        with patch("paper_lark_agents.hy3.subprocess.run", return_value=_completed(stdout="   ")):
            with self.assertRaises(Hy3Error):
                client.ask("x")

    def test_timeout_raises(self):
        client = Hy3Client(self.config())
        with patch("paper_lark_agents.hy3.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python", timeout=1)):
            with self.assertRaises(Hy3Error):
                client.ask("x")


if __name__ == "__main__":
    unittest.main()
