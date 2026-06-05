"""Thin client for the internal Hunyuan-3 (hy3) model.

Used by the supervisor router to make cheap "who should respond next" decisions.

The call is made in a SUBPROCESS, mirroring how the bridge already shells out to
lark-cli and the agent CLIs. This isolates three awkward properties of the
internal eval harness (``internal_api_eval.py``) from the long-lived daemon:

- it runs ``logging.basicConfig`` and resolves task metadata at import time;
- it is a ~275KB module pulled in only for one call path;
- the internal eval host is reachable by a DIRECT connection; the agent proxy
  the daemon inherits (http_proxy=127.0.0.1:7899) cannot reach it, so the
  child's env must have the proxy variables stripped.

So we strip proxy vars and set credentials + task metadata only in the child env.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
import sys


LOGGER = logging.getLogger(__name__)

DEFAULT_CODE_PATH = "/apdcephfs_sgfd/share_303735497/yixianliu/arimazhu/code"
DEFAULT_MODEL = "hunyuan-3.0-preview-taiji"

# Worker run as `python -c WORKER`. Reads the prompt from stdin, config from env,
# prints the extracted answer text to stdout. Kept dependency-free beyond what
# internal_api_eval already needs.
_WORKER = r"""
import os, sys, logging
logging.disable(logging.WARNING)
sys.path.insert(0, os.environ["HY3_CODE_PATH"])
from internal_api_eval import Api, HOST
prompt = sys.stdin.read()
api = Api(f"http://{HOST}", os.environ["INTERNAL_API_USER"], os.environ["INTERNAL_API_TOKEN"])
ret = api.call_data_eval(os.environ["HY3_MODEL"], prompt, []).json()
ans = None
if isinstance(ret, dict):
    if isinstance(ret.get("answer"), list) and ret["answer"]:
        ans = next((i.get("value") for i in ret["answer"] if i.get("type") == "text"), None)
    elif isinstance(ret.get("choices"), list) and ret["choices"]:
        msg = ret["choices"][0].get("message", {})
        ans = msg.get("content") or msg.get("reasoning_content")
if ans is None:
    sys.stderr.write("hy3: could not extract answer text from response\n")
    sys.exit(3)
sys.stdout.write(ans)
"""


class Hy3Error(RuntimeError):
    pass


@dataclass(frozen=True)
class Hy3Config:
    user: str
    token: str
    code_path: str = DEFAULT_CODE_PATH
    model: str = DEFAULT_MODEL
    task_creator: str = "arimazhu"
    task_name: str = "debug"
    timeout: int = 60
    python: str = sys.executable


def load_internal_api_creds(code_path: str) -> tuple[str, str]:
    """Read INTERNAL_API_USER / INTERNAL_API_TOKEN from <code_path>/.env.

    The internal eval harness hardcodes a different identity; we honour the
    .env-provided credentials instead.
    """
    user = token = ""
    env_path = Path(code_path) / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "INTERNAL_API_USER":
                user = value
            elif key == "INTERNAL_API_TOKEN":
                token = value
    return user, token


class Hy3Client:
    def __init__(self, config: Hy3Config):
        self.config = config

    def ask(self, prompt: str) -> str:
        """Send one prompt to hy3 and return its reply text. Raises Hy3Error."""
        cfg = self.config
        if not cfg.user or not cfg.token:
            raise Hy3Error("hy3 credentials missing (INTERNAL_API_USER / INTERNAL_API_TOKEN)")
        # Connect directly. The daemon inherits http_proxy=127.0.0.1:7899 (the
        # agent proxy), which cannot reach the internal eval host, so strip all
        # proxy vars from the child env.
        env = {
            key: value
            for key, value in os.environ.items()
            if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
        }
        env.update(
            {
                "HY3_CODE_PATH": cfg.code_path,
                "HY3_MODEL": cfg.model,
                "INTERNAL_API_USER": cfg.user,
                "INTERNAL_API_TOKEN": cfg.token,
                "EVAL_TASK_CREATOR": cfg.task_creator,
                "EVAL_TASK_NAME": cfg.task_name,
            }
        )
        try:
            proc = subprocess.run(
                [cfg.python, "-c", _WORKER],
                input=prompt,
                env=env,
                text=True,
                capture_output=True,
                timeout=cfg.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise Hy3Error(f"hy3 call timed out after {cfg.timeout}s") from exc
        except FileNotFoundError as exc:
            raise Hy3Error(f"python not found: {cfg.python}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise Hy3Error(f"hy3 call failed: {detail[-500:]}")
        text = (proc.stdout or "").strip()
        if not text:
            raise Hy3Error("hy3 returned empty output")
        return text
