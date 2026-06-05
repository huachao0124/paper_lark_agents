from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


@dataclass(frozen=True)
class DaemonPaths:
    run_dir: Path
    log_dir: Path
    pid_file: Path
    log_file: Path


def default_paths(root: Path) -> DaemonPaths:
    run_dir = root / ".run"
    log_dir = root / "logs"
    return DaemonPaths(
        run_dir=run_dir,
        log_dir=log_dir,
        pid_file=run_dir / "paper-lark-agents.pid",
        log_file=log_dir / "paper-lark-agents.log",
    )


def start_daemon(
    root: Path,
    codex_env: str = ".env.codex",
    claude_env: str = ".env.claude",
    verbose: bool = False,
) -> dict[str, object]:
    paths = default_paths(root)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    existing = read_pid(paths.pid_file)
    if existing and is_running(existing):
        return {
            "ok": True,
            "already_running": True,
            "pid": existing,
            "log_file": str(paths.log_file),
        }
    if existing:
        paths.pid_file.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "paper_lark_agents",
        "serve-duo",
        "--codex-env",
        codex_env,
        "--claude-env",
        claude_env,
    ]
    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_handle = paths.log_file.open("a", encoding="utf-8")
    log_handle.write(f"\n--- starting {' '.join(cmd)} ---\n")
    log_handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        text=True,
    )
    paths.pid_file.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1)
    running = is_running(proc.pid)
    return {
        "ok": running,
        "pid": proc.pid,
        "log_file": str(paths.log_file),
        "pid_file": str(paths.pid_file),
    }


def stop_daemon(root: Path, timeout: float = 10.0) -> dict[str, object]:
    paths = default_paths(root)
    pid = read_pid(paths.pid_file)
    if not pid:
        return {"ok": True, "running": False, "message": "No pid file."}
    if not is_running(pid):
        paths.pid_file.unlink(missing_ok=True)
        return {"ok": True, "running": False, "message": "Stale pid file removed."}

    os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_running(pid):
            paths.pid_file.unlink(missing_ok=True)
            return {"ok": True, "running": False, "pid": pid}
        time.sleep(0.2)

    os.killpg(pid, signal.SIGKILL)
    paths.pid_file.unlink(missing_ok=True)
    return {"ok": True, "running": False, "pid": pid, "killed": True}


def daemon_status(root: Path) -> dict[str, object]:
    paths = default_paths(root)
    pid = read_pid(paths.pid_file)
    running = bool(pid and is_running(pid))
    return {
        "running": running,
        "pid": pid,
        "pid_file": str(paths.pid_file),
        "log_file": str(paths.log_file),
    }


def read_log_tail(root: Path, lines: int = 80) -> str:
    paths = default_paths(root)
    if not paths.log_file.exists():
        return ""
    data = paths.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def read_pid(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

