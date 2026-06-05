from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable

from .config import Settings, proxy_env
from .tmux_runtime import TmuxReplyStillRunning, TmuxRuntimeError, TmuxSessionRuntime


@dataclass
class AgentResult:
    name: str
    text: str
    usage: dict | None = None
    transcript_path: str | None = None
    transcript_offset: int = 0


class AgentError(RuntimeError):
    pass


class AgentStillRunning(AgentError):
    pass


class AgentRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.codex_session = TmuxSessionRuntime(settings, "codex")
        self.claude_session = TmuxSessionRuntime(settings, "claude")

    def run_codex(
        self,
        prompt: str,
        chat_id: str | None = None,
        session_context: str | None = None,
        workspace: Path | None = None,
        model: str | None = None,
        effort: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        if self.settings.codex_runtime == "session":
            if not chat_id:
                raise AgentError("Codex session runtime requires chat_id.")
            try:
                reply = self.codex_session.run(
                    chat_id,
                    prompt,
                    self.settings.codex_timeout,
                    session_context=session_context,
                    workspace=workspace,
                    model=model,
                    effort=effort,
                    progress_callback=progress_callback,
                    progress_interval=self.settings.status_update_seconds,
                    run_id=run_id,
                )
            except TmuxReplyStillRunning as exc:
                raise AgentStillRunning(str(exc)) from exc
            except (TmuxRuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                raise AgentError(str(exc)) from exc
            return AgentResult("Codex", reply.text, reply.usage, reply.transcript_path, reply.transcript_offset)
        text = self._run(
            name="Codex",
            executable=self.settings.codex_cmd,
            args=self.settings.codex_args,
            prompt=prompt,
            timeout=self.settings.codex_timeout,
            workspace=workspace,
        )
        return AgentResult("Codex", text)

    def run_claude(
        self,
        prompt: str,
        chat_id: str | None = None,
        session_context: str | None = None,
        workspace: Path | None = None,
        model: str | None = None,
        effort: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        run_id: str | None = None,
    ) -> AgentResult:
        if self.settings.claude_runtime == "session":
            if not chat_id:
                raise AgentError("Claude session runtime requires chat_id.")
            try:
                reply = self.claude_session.run(
                    chat_id,
                    prompt,
                    self.settings.claude_timeout,
                    session_context=session_context,
                    workspace=workspace,
                    model=model,
                    effort=effort,
                    progress_callback=progress_callback,
                    progress_interval=self.settings.status_update_seconds,
                    run_id=run_id,
                )
            except TmuxReplyStillRunning as exc:
                raise AgentStillRunning(str(exc)) from exc
            except (TmuxRuntimeError, subprocess.CalledProcessError, FileNotFoundError) as exc:
                raise AgentError(str(exc)) from exc
            return AgentResult("Claude Code", reply.text, reply.usage, reply.transcript_path, reply.transcript_offset)
        text = self._run(
            name="Claude Code",
            executable=self.settings.claude_cmd,
            args=self.settings.claude_args,
            prompt=prompt,
            timeout=self.settings.claude_timeout,
            workspace=workspace,
        )
        return AgentResult("Claude Code", text)

    def send_session_command(
        self,
        agent: str,
        chat_id: str,
        command: str,
        workspace: Path | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> bool:
        if agent == "codex":
            if self.settings.codex_runtime != "session":
                return False
            runtime = self.codex_session
        elif agent == "claude":
            if self.settings.claude_runtime != "session":
                return False
            runtime = self.claude_session
        else:
            raise AgentError(f"Unknown agent: {agent}")
        try:
            return runtime.send_session_command(
                chat_id,
                command,
                workspace=workspace,
                model=model,
                effort=effort,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise AgentError(str(exc)) from exc

    def wait_session_ready(self, agent: str, chat_id: str, timeout: int) -> bool:
        if agent == "codex":
            if self.settings.codex_runtime != "session":
                return False
            runtime = self.codex_session
        elif agent == "claude":
            if self.settings.claude_runtime != "session":
                return False
            runtime = self.claude_session
        else:
            raise AgentError(f"Unknown agent: {agent}")
        try:
            return runtime.wait_for_command_ready(chat_id, timeout)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise AgentError(str(exc)) from exc

    def warmup_session(
        self,
        agent: str,
        chat_id: str,
        workspace: Path | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> str | None:
        if agent == "codex":
            if self.settings.codex_runtime != "session":
                return None
            runtime = self.codex_session
        elif agent == "claude":
            if self.settings.claude_runtime != "session":
                return None
            runtime = self.claude_session
        else:
            raise AgentError(f"Unknown agent: {agent}")
        workspace = (workspace or self.settings.workspace).expanduser().resolve()
        session_name = runtime.session_name(chat_id)
        try:
            runtime.ensure_session(session_name, chat_id, workspace, model, effort)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise AgentError(str(exc)) from exc
        return session_name

    def session_name(self, agent: str, chat_id: str) -> str:
        if agent == "codex":
            return self.codex_session.session_name(chat_id)
        if agent == "claude":
            return self.claude_session.session_name(chat_id)
        raise AgentError(f"Unknown agent: {agent}")

    def reply_markers(self, agent: str, run_id: str) -> tuple[str, str]:
        if agent == "codex":
            return self.codex_session.reply_markers(run_id)
        if agent == "claude":
            return self.claude_session.reply_markers(run_id)
        raise AgentError(f"Unknown agent: {agent}")

    def find_session_reply(
        self,
        agent: str,
        chat_id: str,
        start_marker: str,
        end_marker: str,
    ) -> str | None:
        if agent == "codex":
            if self.settings.codex_runtime != "session":
                return None
            runtime = self.codex_session
        elif agent == "claude":
            if self.settings.claude_runtime != "session":
                return None
            runtime = self.claude_session
        else:
            raise AgentError(f"Unknown agent: {agent}")
        try:
            return runtime.find_marked_reply(chat_id, start_marker, end_marker)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise AgentError(str(exc)) from exc

    def reset_session(self, agent: str, chat_id: str) -> list[Path]:
        if agent == "codex":
            if self.settings.codex_runtime == "session":
                return self.codex_session.reset_session(chat_id)
            return []
        if agent == "claude":
            if self.settings.claude_runtime == "session":
                return self.claude_session.reset_session(chat_id)
            return []
        raise AgentError(f"Unknown agent: {agent}")

    def detect_session_model(self, agent: str, chat_id: str) -> str | None:
        if agent == "codex" and self.settings.codex_runtime == "session":
            return self.codex_session.detect_model(chat_id)
        if agent == "claude" and self.settings.claude_runtime == "session":
            return self.claude_session.detect_model(chat_id)
        return None

    def detect_session_effort(self, agent: str, chat_id: str) -> str | None:
        if agent == "codex" and self.settings.codex_runtime == "session":
            return self.codex_session.detect_effort(chat_id)
        if agent == "claude" and self.settings.claude_runtime == "session":
            return self.claude_session.detect_effort(chat_id)
        return None

    def session_progress(self, agent: str, chat_id: str) -> str | None:
        if agent == "codex" and self.settings.codex_runtime == "session":
            return self.codex_session.session_progress(chat_id)
        if agent == "claude" and self.settings.claude_runtime == "session":
            return self.claude_session.session_progress(chat_id)
        return None

    def _run(
        self,
        name: str,
        executable: str,
        args: tuple[str, ...],
        prompt: str,
        timeout: int,
        workspace: Path | None = None,
    ) -> str:
        command = [executable, *args]
        if "-" in args:
            stdin = prompt
        else:
            command.append(prompt)
            stdin = None

        try:
            proc = subprocess.run(
                command,
                input=stdin,
                cwd=workspace or self.settings.workspace,
                env=proxy_env(self.settings.agent_proxy_url, self.settings.no_proxy),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AgentError(f"{name} command not found: {executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"{name} timed out after {timeout}s") from exc

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            detail = stderr or stdout or f"exit code {proc.returncode}"
            raise AgentError(f"{name} failed: {detail[-2000:]}")
        return stdout or stderr or "(no output)"
