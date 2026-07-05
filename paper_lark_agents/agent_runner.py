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
        suffix = settings.session_suffix
        codex_sid = f"codex-{suffix}" if suffix else None
        claude_sid = f"claude-{suffix}" if suffix else None
        codebuddy_sid = f"codebuddy-{suffix}" if suffix else None
        self.codex_session = TmuxSessionRuntime(settings, "codex", session_id=codex_sid)
        self.claude_session = TmuxSessionRuntime(settings, "claude", session_id=claude_sid)
        self.codebuddy_session = TmuxSessionRuntime(settings, "codebuddy", session_id=codebuddy_sid)
        # GPT-Pro as an isolated codex session (separate CODEX_HOME + internal
        # API key), so it has codex's tools without touching the subscription
        # codex session. Only built when PLA_GPT_PRO_RUNTIME=codex.
        self.gptpro_session: TmuxSessionRuntime | None = None
        if settings.gpt_pro_runtime == "codex":
            from .gpt_pro_agent import build_internal_api_key
            gptpro_sid = f"gptpro-{suffix}" if suffix else "gptpro"
            extra_env: dict[str, str] = {}
            if settings.gpt_pro_codex_home:
                extra_env["CODEX_HOME"] = settings.gpt_pro_codex_home
            if settings.gpt_pro_user and settings.gpt_pro_token:
                extra_env["INTERNAL_GPTPRO_KEY"] = build_internal_api_key(
                    settings.gpt_pro_user, settings.gpt_pro_token, settings.gpt_pro_model,
                    settings.gpt_pro_task_creator, settings.gpt_pro_task_name,
                )
            self.gptpro_session = TmuxSessionRuntime(
                settings, "codex", session_id=gptpro_sid, extra_env=extra_env,
            )

    def run_gpt_pro(
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
        """Run GPT-Pro as an isolated codex session (internal API)."""
        if not self.gptpro_session:
            raise AgentError("GPT-Pro codex runtime not initialized.")
        if not chat_id:
            raise AgentError("GPT-Pro session runtime requires chat_id.")
        try:
            reply = self.gptpro_session.run(
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
        return AgentResult("GPT-Pro", reply.text, reply.usage, reply.transcript_path, reply.transcript_offset)

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

    def run_codebuddy(
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
        if self.settings.codebuddy_runtime == "session":
            if not chat_id:
                raise AgentError("CodeBuddy session runtime requires chat_id.")
            try:
                reply = self.codebuddy_session.run(
                    chat_id,
                    prompt,
                    self.settings.codebuddy_timeout,
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
            return AgentResult("CodeBuddy", reply.text, reply.usage, reply.transcript_path, reply.transcript_offset)
        text = self._run(
            name="CodeBuddy",
            executable=self.settings.codebuddy_cmd,
            args=self.settings.codebuddy_args,
            prompt=prompt,
            timeout=self.settings.codebuddy_timeout,
            workspace=workspace,
        )
        return AgentResult("CodeBuddy", text)

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
        elif agent == "codebuddy":
            if self.settings.codebuddy_runtime != "session":
                return False
            runtime = self.codebuddy_session
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
        elif agent == "codebuddy":
            if self.settings.codebuddy_runtime != "session":
                return False
            runtime = self.codebuddy_session
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
    ) -> tuple[str | None, bool]:
        """Returns (session_name, needs_init). needs_init is True when the
        session was freshly created and has not received its context prompt."""
        if agent == "codex":
            if self.settings.codex_runtime != "session":
                return None, False
            runtime = self.codex_session
        elif agent == "claude":
            if self.settings.claude_runtime != "session":
                return None, False
            runtime = self.claude_session
        elif agent == "codebuddy":
            if self.settings.codebuddy_runtime != "session":
                return None, False
            runtime = self.codebuddy_session
        else:
            raise AgentError(f"Unknown agent: {agent}")
        workspace = (workspace or self.settings.workspace).expanduser().resolve()
        session_name = runtime.session_name(chat_id)
        try:
            needs_init = runtime.ensure_session(session_name, chat_id, workspace, model, effort)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise AgentError(str(exc)) from exc
        return session_name, needs_init

    def runtime_for(self, agent: str) -> "TmuxSessionRuntime":
        """Map an agent to its session runtime (pure mapping, no mode check).
        Used by recovery/interactive paths that already assume session mode."""
        if agent == "gpt-pro":
            return self.gptpro_session or self.codex_session
        if agent == "codebuddy":
            return self.codebuddy_session
        if agent == "claude":
            return self.claude_session
        return self.codex_session

    def _session_for(self, agent: str) -> "TmuxSessionRuntime | None":
        """Return the active session runtime for an agent, or None if that
        agent is not running in session mode."""
        if agent == "codex":
            return self.codex_session if self.settings.codex_runtime == "session" else None
        if agent == "claude":
            return self.claude_session if self.settings.claude_runtime == "session" else None
        if agent == "codebuddy":
            return self.codebuddy_session if self.settings.codebuddy_runtime == "session" else None
        if agent == "gpt-pro":
            return self.gptpro_session  # None unless PLA_GPT_PRO_RUNTIME=codex
        return None

    def session_name(self, agent: str, chat_id: str) -> str:
        if agent == "codex":
            return self.codex_session.session_name(chat_id)
        if agent == "claude":
            return self.claude_session.session_name(chat_id)
        if agent == "codebuddy":
            return self.codebuddy_session.session_name(chat_id)
        if agent == "gpt-pro" and self.gptpro_session:
            return self.gptpro_session.session_name(chat_id)
        raise AgentError(f"Unknown agent: {agent}")

    def reply_markers(self, agent: str, run_id: str) -> tuple[str, str]:
        if agent == "codex":
            return self.codex_session.reply_markers(run_id)
        if agent == "claude":
            return self.claude_session.reply_markers(run_id)
        if agent == "codebuddy":
            return self.codebuddy_session.reply_markers(run_id)
        if agent == "gpt-pro" and self.gptpro_session:
            return self.gptpro_session.reply_markers(run_id)
        raise AgentError(f"Unknown agent: {agent}")

    def find_session_reply(
        self,
        agent: str,
        chat_id: str,
        start_marker: str,
        end_marker: str,
    ) -> str | None:
        runtime = self._session_for(agent)
        if runtime is None:
            return None
        try:
            return runtime.find_marked_reply(chat_id, start_marker, end_marker)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise AgentError(str(exc)) from exc

    def reset_session(self, agent: str, chat_id: str) -> list[Path]:
        runtime = self._session_for(agent)
        if runtime is None:
            return []
        return runtime.reset_session(chat_id)

    def detect_session_model(self, agent: str, chat_id: str) -> str | None:
        runtime = self._session_for(agent)
        return runtime.detect_model(chat_id) if runtime else None

    def detect_session_effort(self, agent: str, chat_id: str) -> str | None:
        runtime = self._session_for(agent)
        return runtime.detect_effort(chat_id) if runtime else None

    def session_progress(self, agent: str, chat_id: str) -> str | None:
        runtime = self._session_for(agent)
        return runtime.session_progress(chat_id) if runtime else None

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
