from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable
import uuid

from .config import Settings, proxy_command_prefix
from .transcripts import (
    activity_detail,
    claude_activity,
    codex_activity,
    claude_followup_from_lines,
    claude_reply_from_lines,
    codex_reply_from_lines,
    encode_claude_project_dir,
    find_claude_recent_session_file,
    find_claude_session_file,
    find_codex_rollout,
    find_codex_rollout_by_id,
    read_new_jsonl,
    rollout_session_id,
)


# Bump when agent_session_context_prompt changes in a way existing long-lived
# sessions must learn (e.g. the @peer handoff contract). A mismatch re-injects
# the context block on the session's next turn without recreating the session.
LOGGER = logging.getLogger(__name__)

SESSION_PROMPT_VERSION = 10
EFFORT_TOKEN_PATTERN = r"[A-Za-z][A-Za-z0-9_-]*"


class TmuxRuntimeError(RuntimeError):
    pass


class TmuxReplyStillRunning(TmuxRuntimeError):
    pass


@dataclass(frozen=True)
class TmuxReply:
    text: str
    session_name: str
    run_id: str
    start_marker: str
    end_marker: str
    usage: dict | None = None
    # Cursor after the first reply — used to poll for follow-up replies.
    transcript_path: str | None = None
    transcript_offset: int = 0


class TmuxSessionRuntime:
    def __init__(
        self, settings: Settings, agent: str, session_id: str | None = None,
    ):
        self.settings = settings
        self.agent = agent
        self.session_id = session_id or agent
        self._chat_labels: dict[str, str] = {}
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_locks_guard = threading.Lock()

    def run(
        self,
        chat_id: str,
        prompt: str,
        timeout: int,
        session_context: str | None = None,
        workspace: Path | None = None,
        model: str | None = None,
        effort: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
        progress_interval: float = 30.0,
        run_id: str | None = None,
    ) -> TmuxReply:
        workspace = (workspace or self.settings.workspace).expanduser().resolve()
        session_name = self.session_name(chat_id)
        include_context = self.ensure_session(session_name, chat_id, workspace, model, effort)
        self.apply_model_if_needed(session_name, model)
        self.apply_effort_if_needed(session_name, effort)
        marker = run_id or uuid.uuid4().hex
        start_marker, end_marker = self.reply_markers(marker)
        # Snapshot the session transcript before submitting so we only read the
        # reply to *this* prompt; persist the cursor by run id for recovery.
        path_str, offset = self.transcript_cursor(session_name, chat_id, workspace)
        self.store_run_cursor(session_name, marker, path_str, offset)
        wrapped = self.wrap_prompt(
            prompt,
            include_context=include_context,
            session_context=session_context,
        )
        self.paste_and_submit(session_name, wrapped)
        reply_text, reply_usage, reply_cursor = self.wait_for_jsonl_reply(
            session_name,
            chat_id,
            workspace,
            path_str,
            offset,
            timeout,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
        if include_context:
            self.mark_session_initialized(session_name, chat_id, workspace)
        self.refresh_cli_session_files(session_name, chat_id)
        return TmuxReply(
            reply_text, session_name, marker, start_marker, end_marker, reply_usage,
            transcript_path=reply_cursor[0] if reply_cursor else None,
            transcript_offset=reply_cursor[1] if reply_cursor else 0,
        )

    def reply_markers(self, run_id: str) -> tuple[str, str]:
        return f"PLA_REPLY_START_{run_id}", f"PLA_REPLY_END_{run_id}"

    def session_name(self, chat_id: str) -> str:
        label = self._chat_labels.get(chat_id)
        sid = self.session_id
        # Short suffix from chat_id for uniqueness (handles duplicate group names).
        suffix = re.sub(r"[^A-Za-z0-9]+", "", chat_id)[-6:]
        if label:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")[:32]
            name = f"pla-{sid}-{safe_label}-{suffix}"
        else:
            safe_chat = re.sub(r"[^A-Za-z0-9_.-]+", "-", chat_id).strip("-")[:48]
            name = f"pla-{sid}-{safe_chat}"
        # Migrate: if a session exists under the old name (before session_id
        # was introduced, or before chat labels), rename it.
        if label:
            old_name = f"pla-{sid}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', chat_id).strip('-')[:48]}"
            if old_name != name and self.session_exists(old_name) and not self.session_exists(name):
                self._migrate_session_name(old_name, name)
        # Also migrate from the bare agent name when session_id differs.
        if sid != self.agent:
            old_agent_name = f"pla-{self.agent}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', chat_id).strip('-')[:48]}"
            if old_agent_name != name and self.session_exists(old_agent_name) and not self.session_exists(name):
                self._migrate_session_name(old_agent_name, name)
        return name

    def set_chat_label(self, chat_id: str, label: str) -> None:
        """Register a human-readable label (e.g. group name) for a chat_id."""
        if label and label != chat_id:
            self._chat_labels[chat_id] = label

    def _migrate_session_name(self, old_name: str, new_name: str) -> None:
        """Rename tmux session and metadata file from old to new name."""
        try:
            subprocess.run(
                ["tmux", "rename-session", "-t", old_name, new_name],
                text=True, capture_output=True, check=True,
            )
            LOGGER.info("renamed tmux session %s -> %s", old_name, new_name)
        except subprocess.CalledProcessError:
            return
        old_meta = self.settings.state_dir / "tmux" / f"{old_name}.json"
        new_meta = self.settings.state_dir / "tmux" / f"{new_name}.json"
        if old_meta.exists() and not new_meta.exists():
            data = json.loads(old_meta.read_text(encoding="utf-8"))
            data["session_name"] = new_name
            new_meta.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            old_meta.unlink()
        old_log = self.settings.state_dir / "tmux" / f"{old_name}.log"
        new_log = self.settings.state_dir / "tmux" / f"{new_name}.log"
        if old_log.exists() and not new_log.exists():
            old_log.rename(new_log)

    def find_marked_reply(self, chat_id: str, start_marker: str, end_marker: str) -> str | None:
        """Recover a finished reply from the session transcript (JSONL).

        Keeps the marker-based signature for the recovery plumbing, but the
        markers only carry the run id; the actual reply is read from the
        transcript cursor stored when the run was submitted.
        """
        run_id = run_id_from_marker(start_marker)
        session_name = self.session_name(chat_id)
        cursor = self.read_run_cursor(session_name, run_id)
        if not cursor:
            return None
        return self.read_jsonl_reply(cursor[0], cursor[1])

    def ensure_session(
        self,
        session_name: str,
        chat_id: str,
        workspace: Path,
        model: str | None = None,
        effort: str | None = None,
    ) -> bool:
        command = self.command(workspace, model=model, effort=effort)
        if self.session_exists(session_name):
            if self._cli_crashed(session_name):
                # The CLI process died but its conversation transcript is
                # intact on disk — keep the session id so the recreated
                # session resumes the same conversation. Guard against a
                # resume that itself crashes: only retry resume if the
                # previous recreation was not already a resume attempt.
                meta = self.read_metadata(session_name)
                resume_again = not meta.get("crash_resume_attempted") and bool(
                    str(meta.get("session_uuid") or "").strip()
                )
                LOGGER.warning(
                    "CLI crashed in %s, recreating session (resume=%s)",
                    session_name,
                    resume_again,
                )
                self.kill_session(session_name)
                self._clear_stale_transcript(session_name, keep_uuid=resume_again)
                self._update_metadata(session_name, crash_resume_attempted=resume_again)
            else:
                if self.read_metadata(session_name).get("crash_resume_attempted"):
                    # Session is alive and healthy — the post-crash resume
                    # worked, so allow resuming again on a future crash.
                    self._update_metadata(session_name, crash_resume_attempted=False)
                ws_match = self.session_workspace_matches(session_name, workspace)
                cmd_match = self.session_command_matches(session_name, command)
                if ws_match and cmd_match:
                    self.configure_window_size(session_name)
                    self.ensure_transcript_pipe(session_name)
                    self.refresh_detected_session_labels(session_name)
                    self._clear_stale_input(session_name)
                    return not self.session_initialized(session_name)
                # Only workspace changed (model/effort/args unchanged) — try to
                # preserve the agent's conversation context.
                only_workspace_changed = not ws_match and self._command_matches_ignoring_workspace(session_name, command)
                if only_workspace_changed:
                    if self._switch_workspace_in_place(session_name, chat_id, workspace):
                        return not self.session_initialized(session_name)
                # Fall through: kill and recreate (preserving context via resume).
                self.kill_session(session_name)
        self.configure_history_limit()
        # Resume the prior session when we recorded its id (survives container
        # restart — metadata lives on persistent .state). For codex, resume
        # works even across workspace changes (-C flag). For claude, resume is
        # only used when workspace is unchanged (it always /cd-switches first).
        prior_meta = self.read_metadata(session_name)
        prior_uuid = prior_meta.get("session_uuid")
        resume_uuid = None
        if isinstance(prior_uuid, str) and prior_uuid.strip():
            if self.agent == "codex":
                # codex resume always works — -C is stripped, tmux -c sets cwd.
                resume_uuid = prior_uuid.strip()
            elif str(prior_meta.get("workspace") or "") == str(workspace):
                resume_uuid = prior_uuid.strip()
        launch, keep_uuid = self.build_launch_command(command, resume_uuid)
        shell_command = " ".join(shlex.quote(part) for part in launch)
        columns = getattr(self.settings, "session_columns", 120)
        rows = getattr(self.settings, "session_rows", 80)
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-x",
                str(columns),
                "-y",
                str(rows),
                "-s",
                session_name,
                "-c",
                str(workspace),
                shell_command,
            ],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=True,
        )
        self.configure_window_size(session_name)
        self.write_metadata(session_name, chat_id, command, workspace)
        if keep_uuid:
            self.store_session_uuid(session_name, keep_uuid)
        self.ensure_transcript_pipe(session_name)
        self.wait_for_session_ready(session_name)
        self.refresh_detected_session_labels(session_name)
        self.apply_startup_commands(session_name)
        self.refresh_cli_session_files(session_name, chat_id)
        return True

    def command(
        self,
        workspace: Path,
        model: str | None = None,
        effort: str | None = None,
    ) -> list[str]:
        if self.agent == "codex":
            args = without_existing_cwd_args(self.settings.codex_session_args)
            args = with_model_arg(args, model or self.settings.codex_model, "-m")
            args = with_codex_effort_arg(args, effort or self.settings.codex_default_effort)
            args.extend(["-C", str(workspace)])
            command = [self.settings.codex_cmd, *args]
            return [
                *proxy_command_prefix(self.settings.agent_proxy_url, self.settings.no_proxy),
                *command,
            ]
        args = list(self.settings.claude_session_args)
        args = with_model_arg(args, self.settings.claude_model, "--model")
        command = [self.settings.claude_cmd, *args]
        return [
            *proxy_command_prefix(self.settings.agent_proxy_url, self.settings.no_proxy),
            *command,
        ]

    def build_launch_command(
        self, command: list[str], resume_uuid: str | None
    ) -> tuple[list[str], str | None]:
        """Build the actual launch command and the session id to persist.

        Returns (launch, keep_uuid). For codex fresh, keep_uuid is None (the id
        is captured later from the rollout); otherwise it is the id to store.
        """
        if self.agent == "claude":
            if resume_uuid:
                return ([*command, "--resume", resume_uuid], resume_uuid)
            new_uuid = str(uuid.uuid4())
            return ([*command, "--session-id", new_uuid], new_uuid)
        if resume_uuid:
            return (self._codex_resume_command(command, resume_uuid), resume_uuid)
        return (list(command), None)

    def _codex_resume_command(self, command: list[str], session_id: str) -> list[str]:
        # `codex resume <id> ...`: insert the resume subcommand right after the
        # codex executable, and drop `-C <cwd>` (the resumed session's cwd comes
        # from `tmux new-session -c`, and `resume` does not take -C the same way).
        out: list[str] = []
        inserted = False
        skip = False
        for token in command:
            if skip:
                skip = False
                continue
            if token == "-C":
                skip = True
                continue
            out.append(token)
            if not inserted and token == self.settings.codex_cmd:
                out.extend(["resume", session_id])
                inserted = True
        return out

    def _switch_workspace_in_place(
        self, session_name: str, chat_id: str, workspace: Path,
    ) -> bool:
        """Switch the workspace without killing the session.

        Claude: send /cd <path> into the running session.
        Codex: not supported in-place — return False to trigger kill+resume.
        """
        if self.agent == "claude":
            try:
                self.paste_and_submit(session_name, f"/cd {workspace}")
                time.sleep(1)
                self.update_metadata_workspace(session_name, chat_id, workspace)
                LOGGER.info("switched claude workspace in-place to %s", workspace)
                return True
            except (subprocess.CalledProcessError, OSError) as exc:
                LOGGER.warning("in-place workspace switch failed: %s", exc)
                return False
        # Codex has no /cd — fall through to kill+resume with new -C.
        return False

    def update_metadata_workspace(
        self, session_name: str, chat_id: str, workspace: Path,
    ) -> None:
        meta = self.read_metadata(session_name)
        meta["workspace"] = str(workspace)
        meta["command"] = self.command(workspace)
        path = self.metadata_path(session_name)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def session_workspace_matches(self, session_name: str, workspace: Path) -> bool:
        metadata = self.read_metadata(session_name)
        if not metadata:
            return workspace == self.settings.workspace
        recorded = metadata.get("workspace")
        if not recorded:
            return workspace == self.settings.workspace
        return Path(str(recorded)).expanduser().resolve() == workspace

    def session_command_matches(self, session_name: str, command: list[str]) -> bool:
        metadata = self.read_metadata(session_name)
        if not metadata:
            return True
        recorded = metadata.get("command")
        if not isinstance(recorded, list):
            return True
        return [str(part) for part in recorded] == command

    def _command_matches_ignoring_workspace(self, session_name: str, command: list[str]) -> bool:
        """Compare commands after stripping -C <dir> (codex workspace arg)."""
        metadata = self.read_metadata(session_name)
        recorded = metadata.get("command")
        if not isinstance(recorded, list):
            return True
        return without_existing_cwd_args(tuple(str(s) for s in recorded)) == without_existing_cwd_args(tuple(str(s) for s in command))

    def _cli_crashed(self, session_name: str) -> bool:
        try:
            screen = self.capture(session_name)
        except subprocess.CalledProcessError:
            return False
        lines = [l.strip() for l in screen.splitlines() if l.strip()]
        if not lines:
            return False
        tail = "\n".join(lines[-6:])
        if session_ready_for_input(tail, self.agent):
            return False
        if session_tail_busy(tail):
            return False
        has_bash = bool(re.search(r"\$\s*$|bash.*command not found", tail))
        has_cli = "❯" in tail or "›" in tail or "esc to interrupt" in tail
        return has_bash and not has_cli

    def _clear_stale_input(self, session_name: str) -> None:
        """Clear any leftover pasted content in the input line.

        If bridge was killed between paste and Enter, the prompt has
        stale text that would be submitted on the next paste_and_submit.
        """
        try:
            screen = self.capture(session_name)
            if "[Pasted Content" in screen or "Press up to edit queued" in screen:
                subprocess.run(
                    ["tmux", "send-keys", "-t", session_name, "C-u"],
                    text=True, capture_output=True, check=False,
                )
                subprocess.run(
                    ["tmux", "send-keys", "-t", session_name, "Escape"],
                    text=True, capture_output=True, check=False,
                )
                LOGGER.info("cleared stale input in %s", session_name)
        except (subprocess.CalledProcessError, AttributeError):
            pass

    def _clear_stale_transcript(self, session_name: str, keep_uuid: bool = False) -> None:
        data = self.read_metadata(session_name)
        if not data:
            return
        data.pop("transcript_path", None)
        if not keep_uuid:
            data.pop("session_uuid", None)
        data.pop("session_prompt_version", None)
        data.pop("run_cursors", None)
        path = self.metadata_path(session_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_matching_sessions(self) -> list[str]:
        prefix = f"pla-{self.session_id}-"
        proc = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            text=True, capture_output=True, check=False,
        )
        if proc.returncode != 0:
            return []
        return [
            name.strip() for name in proc.stdout.splitlines()
            if name.strip().startswith(prefix)
        ]

    def kill_session(self, session_name: str) -> None:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)

    def reset_session(self, chat_id: str) -> list[Path]:
        session_name = self.session_name(chat_id)
        self.refresh_cli_session_files(session_name, chat_id)
        metadata = self.read_metadata(session_name)
        self.kill_session(session_name)
        deleted = self.delete_cli_session_files(metadata, chat_id)
        self.metadata_path(session_name).unlink(missing_ok=True)
        self.transcript_path(session_name).unlink(missing_ok=True)
        return deleted

    def detect_model(self, chat_id: str) -> str | None:
        session_name = self.session_name(chat_id)
        # Prefer JSONL transcript — reliable, no tmux screen scraping needed.
        transcript = self._read_recent_transcript(session_name, chat_id)
        if transcript:
            from .transcripts import claude_model_from_lines, codex_model_from_lines
            model = (codex_model_from_lines if self.agent == "codex" else claude_model_from_lines)(transcript)
            if model:
                return model
        if not self.session_exists(session_name):
            return metadata_string(self.read_metadata(session_name), "detected_model", "model")
        try:
            detected = parse_session_model(self.capture_with_transcript(session_name), self.agent)
        except subprocess.CalledProcessError:
            detected = None
        if detected:
            return detected
        return metadata_string(self.read_metadata(session_name), "detected_model", "model")

    def _read_recent_transcript(self, session_name: str, chat_id: str, max_lines: int = 50) -> list[dict] | None:
        """Read the last N lines from the session's transcript JSONL."""
        meta = self.read_metadata(session_name)
        path_str = meta.get("transcript_path")
        if not path_str:
            return None
        path = Path(str(path_str))
        if not path.exists():
            return None
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        result = []
        for line in lines[-max_lines:]:
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result or None

    def detect_effort(self, chat_id: str) -> str | None:
        session_name = self.session_name(chat_id)
        # Prefer JSONL transcript — codex records reasoning effort in turn_context.
        transcript = self._read_recent_transcript(session_name, chat_id)
        if transcript:
            from .transcripts import claude_effort_from_lines, codex_effort_from_lines
            effort = (codex_effort_from_lines if self.agent == "codex" else claude_effort_from_lines)(transcript)
            if effort:
                return effort
        if not self.session_exists(session_name):
            return metadata_string(self.read_metadata(session_name), "detected_effort", "effort")
        try:
            detected = parse_session_effort(self.capture_with_transcript(session_name), self.agent)
        except subprocess.CalledProcessError:
            detected = None
        if detected:
            return detected
        return metadata_string(self.read_metadata(session_name), "detected_effort", "effort")

    def session_progress(self, chat_id: str) -> str | None:
        session_name = self.session_name(chat_id)
        if not self.session_exists(session_name):
            return None
        try:
            return summarize_session_progress(self.capture(session_name), self.agent)
        except subprocess.CalledProcessError:
            return None

    def poll_followup_reply(
        self, transcript_path: str, offset: int, timeout: int = 300,
    ) -> tuple[str | None, int]:
        """Poll for a follow-up reply after the first end_turn.

        Claude can produce multiple end_turn messages in one turn (e.g. agent
        spawns a background teammate, replies once, then replies again with the
        teammate's result). Returns (text, new_offset) where text is None if no
        follow-up appeared within *timeout* seconds. Use timeout=0 for a
        non-blocking single check.
        """
        if self.agent != "claude":
            return None, offset
        path = Path(transcript_path)
        if not path.exists():
            return None, offset
        deadline = time.time() + max(timeout, 0.1)
        cur = offset
        while time.time() < deadline:
            new, cur = read_new_jsonl(path, cur)
            if new:
                result = claude_followup_from_lines(new)
                if result is not None:
                    return result.text, cur
            time.sleep(2)
        return None, cur

    def configure_history_limit(self) -> None:
        limit = getattr(self.settings, "session_history_limit", 0)
        if not limit:
            return
        subprocess.run(
            [
                "tmux",
                "set-option",
                "-g",
                "history-limit",
                str(limit),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def configure_window_size(self, session_name: str) -> None:
        columns = getattr(self.settings, "session_columns", 120)
        rows = getattr(self.settings, "session_rows", 80)
        subprocess.run(
            [
                "tmux",
                "resize-pane",
                "-t",
                session_name,
                "-x",
                str(columns),
                "-y",
                str(rows),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def send_session_command(
        self,
        chat_id: str,
        command: str,
        workspace: Path | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> bool:
        workspace = (workspace or self.settings.workspace).expanduser().resolve()
        session_name = self.session_name(chat_id)
        if not self.session_exists(session_name):
            self.ensure_session(session_name, chat_id, workspace, model, effort)
        self.apply_model_if_needed(session_name, model)
        # Check if session is idle before sending slash commands — if busy,
        # the command gets queued as a message and won't execute.
        if command.strip().startswith("/"):
            try:
                screen = self.capture(session_name)
                if not session_ready_for_current_input(screen, self.agent):
                    return False
            except (subprocess.CalledProcessError, AttributeError):
                pass
        self.paste_and_submit(session_name, command)
        if command.strip().startswith(("/model", "/effort")):
            self._confirm_slash_dialog(session_name)
        if command.strip().startswith("/permissions"):
            self._select_codex_menu_option(session_name, "2")
        if effort:
            self.mark_session_effort(session_name, effort)
        self.refresh_cli_session_files(session_name, chat_id)
        return True

    def wait_for_command_ready(self, chat_id: str, timeout: int) -> bool:
        session_name = self.session_name(chat_id)
        if not self.session_exists(session_name):
            return False
        deadline = time.time() + max(1, timeout)
        time.sleep(1)
        while time.time() < deadline:
            try:
                captured = self.capture(session_name)
            except subprocess.CalledProcessError:
                time.sleep(1)
                continue
            if session_ready_for_current_input(captured, self.agent):
                return True
            time.sleep(1)
        return False

    def session_exists(self, session_name: str) -> bool:
        proc = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.returncode == 0

    def _session_lock(self, session_name: str) -> threading.Lock:
        with self._session_locks_guard:
            if session_name not in self._session_locks:
                self._session_locks[session_name] = threading.Lock()
            return self._session_locks[session_name]

    def paste_and_submit(self, session_name: str, text: str) -> None:
        with self._session_lock(session_name):
            self.dismiss_claude_feedback_prompt_if_visible(session_name)
            LOGGER.info("paste_and_submit to %s (%d chars)", session_name, len(text))
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
                handle.write(text)
                temp_path = Path(handle.name)
            try:
                subprocess.run(["tmux", "send-keys", "-t", session_name, "C-u"], check=True)
                subprocess.run(["tmux", "load-buffer", str(temp_path)], check=True)
                subprocess.run(["tmux", "paste-buffer", "-p", "-t", session_name], check=True)
                subprocess.run(["tmux", "send-keys", "-t", session_name, "Enter"], check=True)
            finally:
                temp_path.unlink(missing_ok=True)

    def dismiss_claude_feedback_prompt_if_visible(
        self,
        session_name: str,
        screen: str | None = None,
    ) -> bool:
        if self.agent != "claude":
            return False
        if screen is None:
            try:
                screen = self.capture(session_name)
            except subprocess.CalledProcessError:
                return False
        if not claude_feedback_prompt_visible(screen):
            return False
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "0"],
            text=True,
            capture_output=True,
            check=False,
        )
        return True

    def apply_startup_commands(self, session_name: str) -> None:
        if self.agent != "codex":
            return
        commands = getattr(self.settings, "codex_startup_commands", ())
        for command in commands:
            if not command.strip():
                continue
            self.paste_and_submit(session_name, command)
            time.sleep(1)
            if command.strip() == "/permissions":
                self._select_codex_menu_option(session_name, "2")

    def _select_codex_menu_option(self, session_name: str, option: str) -> None:
        time.sleep(1)
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, option],
            text=True, capture_output=True, check=False,
        )
        time.sleep(0.5)
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "Enter"],
            text=True, capture_output=True, check=False,
        )

    def _confirm_slash_dialog(self, session_name: str) -> None:
        """/model and /effort pop a confirmation that needs a second Enter to
        actually apply — paste_and_submit only presses Enter once (to run the
        command). Send the confirming Enter so no human has to. A stray Enter at
        an empty prompt is a harmless no-op when no dialog is shown."""
        time.sleep(1)  # let the confirmation dialog render
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, "Enter"],
            text=True, capture_output=True, check=False,
        )

    def apply_effort_if_needed(self, session_name: str, effort: str | None) -> None:
        if not effort or self.session_effort_matches(session_name, effort):
            return
        self.paste_and_submit(session_name, f"/effort {effort}")
        self._confirm_slash_dialog(session_name)
        self.mark_session_effort(session_name, effort)
        time.sleep(1)

    def apply_model_if_needed(self, session_name: str, model: str | None) -> None:
        if not model or self.session_model_matches(session_name, model):
            return
        self.paste_and_submit(session_name, f"/model {model}")
        self._confirm_slash_dialog(session_name)
        self.mark_session_model(session_name, model)
        time.sleep(1)

    def wait_for_reply(
        self,
        session_name: str,
        start_marker: str,
        end_marker: str,
        timeout: int,
        progress_callback: Callable[[str], None] | None = None,
        progress_interval: float = 30.0,
    ) -> str:
        deadline = time.time() + timeout
        last_capture = ""
        last_progress = 0.0
        feedback_dismissed = False
        while time.time() < deadline:
            screen = self.capture(session_name)
            last_capture = screen
            reply = extract_marked_reply(screen, start_marker, end_marker)
            if reply is None:
                reply = extract_mismatched_marked_reply_after_prompt(
                    screen,
                    start_marker,
                    end_marker,
                )
            if reply is None:
                transcript = normalize_terminal_text(self.read_transcript_tail(session_name))
                if transcript:
                    last_capture = transcript
                    reply = extract_marked_reply(transcript, start_marker, end_marker)
                    if reply is None:
                        reply = extract_mismatched_marked_reply_after_prompt(
                            transcript,
                            start_marker,
                            end_marker,
                        )
                if reply is None and transcript:
                    captured = normalize_terminal_text(f"{transcript}\n{screen}")
                    last_capture = captured
                    reply = extract_marked_reply(captured, start_marker, end_marker)
                    if reply is None:
                        reply = extract_mismatched_marked_reply_after_prompt(
                            captured,
                            start_marker,
                            end_marker,
                        )
            if reply is not None:
                return reply
            if not feedback_dismissed and self.dismiss_claude_feedback_prompt_if_visible(session_name, screen):
                feedback_dismissed = True
                time.sleep(0.5)
                continue
            if progress_callback and time.time() - last_progress >= max(1.0, progress_interval):
                display = "Codex" if self.agent == "codex" else "Claude"
                detail = summarize_session_progress(screen, self.agent)
                progress_callback(detail or f"Waiting for {display} reply.")
                last_progress = time.time()
            time.sleep(1)
        raise TmuxReplyStillRunning(
            f"{self.agent} session is still running; bridge watch window ended before "
            f"reply markers appeared in {session_name}. Last capture tail: {last_capture[-1200:]}"
        )

    def capture(self, session_name: str) -> str:
        proc = subprocess.run(
            [
                "tmux",
                "capture-pane",
                "-t",
                session_name,
                "-p",
                "-S",
                f"-{self.settings.session_capture_lines}",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        return normalize_terminal_text(proc.stdout)

    def capture_with_transcript(self, session_name: str) -> str:
        screen = self.capture(session_name)
        transcript = self.read_transcript_tail(session_name)
        if transcript:
            return normalize_terminal_text(f"{transcript}\n{screen}")
        return screen

    def transcript_path(self, session_name: str) -> Path:
        return self.settings.state_dir / "tmux" / f"{session_name}.log"

    def ensure_transcript_pipe(self, session_name: str) -> None:
        path = self.transcript_path(session_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        command = f"cat >> {shlex.quote(str(path))}"
        subprocess.run(
            ["tmux", "pipe-pane", "-o", "-t", session_name, command],
            text=True,
            capture_output=True,
            check=False,
        )

    def read_transcript_tail(self, session_name: str) -> str:
        path = self.transcript_path(session_name)
        if not path.exists():
            return ""
        max_bytes = max(262_144, getattr(self.settings, "session_capture_lines", 20000) * 200)
        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                data = handle.read()
        except OSError:
            return ""
        return data.decode("utf-8", errors="ignore")

    def wait_for_session_ready(self, session_name: str) -> None:
        deadline = time.time() + max(1.0, self.settings.session_startup_wait)
        last_capture = ""
        resume_dismissed = False
        while time.time() < deadline:
            try:
                last_capture = self.capture(session_name)
            except subprocess.CalledProcessError:
                time.sleep(0.5)
                continue
            if session_ready_for_input(last_capture, self.agent):
                return
            # Claude --resume on large sessions shows a dialog asking whether to
            # resume from summary (1) or full (2). Auto-select "2. Resume full
            # session as-is" so the bridge never stalls on the picker.
            if not resume_dismissed and "Resume full session as-is" in last_capture:
                subprocess.run(
                    ["tmux", "send-keys", "-t", session_name, "Down", "Enter"],
                    text=True, capture_output=True, check=False,
                )
                resume_dismissed = True
                LOGGER.info("auto-selected 'Resume full session' for %s", session_name)
            time.sleep(0.5)
        if last_capture:
            return
        time.sleep(0.5)

    def wrap_prompt(
        self,
        prompt: str,
        include_context: bool = False,
        session_context: str | None = None,
    ) -> str:
        # No reply markers: the reply is read from the CLI's native session
        # transcript (JSONL), so the model only needs the message itself. Reply
        # conventions ([NO_REPLY], artifacts as Markdown links) live in the
        # one-time session context.
        context_block = ""
        if include_context:
            context_block = f"""{session_context or self.default_session_context()}

"""
        return f"""{context_block}Message:
{prompt}
"""

    def default_session_context(self) -> str:
        display_name = "Codex" if self.agent == "codex" else "Claude Code"
        return f"""Session setup for {display_name}.

This is a long-lived Feishu research-room session for exactly one group.
Treat the CLI conversation history as this group's continuing memory.
Decide whether each Feishu message needs a reply. If not, use {self.settings.no_reply_token}.
Keep replies concise, research-focused, and suitable for Feishu.
Do not edit local files or run long experiments unless a human explicitly asks."""

    def write_metadata(
        self,
        session_name: str,
        chat_id: str,
        command: list[str],
        workspace: Path,
    ) -> None:
        meta_dir = self.settings.state_dir / "tmux"
        meta_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "session_name": session_name,
            "agent": self.agent,
            "chat_id": chat_id,
            "workspace": str(workspace),
            "command": command,
            "created_at": time.time(),
            "session_prompt_version": 0,
        }
        (meta_dir / f"{session_name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def metadata_path(self, session_name: str) -> Path:
        return self.settings.state_dir / "tmux" / f"{session_name}.json"

    def session_initialized(self, session_name: str) -> bool:
        data = self.read_metadata(session_name)
        if not data:
            return False
        return data.get("session_prompt_version") == SESSION_PROMPT_VERSION

    def session_effort_matches(self, session_name: str, effort: str) -> bool:
        data = self.read_metadata(session_name)
        return data.get("effort") == effort

    def session_model_matches(self, session_name: str, model: str) -> bool:
        data = self.read_metadata(session_name)
        return data.get("model") == model

    def read_metadata(self, session_name: str) -> dict[str, object]:
        path = self.metadata_path(session_name)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def mark_session_initialized(
        self,
        session_name: str,
        chat_id: str,
        workspace: Path,
    ) -> None:
        path = self.metadata_path(session_name)
        data: dict[str, object] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        data["session_name"] = session_name
        data["agent"] = self.agent
        data["chat_id"] = chat_id
        data["workspace"] = str(workspace)
        data["session_prompt_version"] = SESSION_PROMPT_VERSION
        data["session_prompt_initialized_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_session_effort(self, session_name: str, effort: str) -> None:
        path = self.metadata_path(session_name)
        data = self.read_metadata(session_name)
        data["session_name"] = session_name
        data["agent"] = self.agent
        data["effort"] = effort
        data["effort_updated_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_session_model(self, session_name: str, model: str) -> None:
        path = self.metadata_path(session_name)
        data = self.read_metadata(session_name)
        data["session_name"] = session_name
        data["agent"] = self.agent
        data["model"] = model
        data["model_updated_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def refresh_detected_session_labels(self, session_name: str) -> None:
        try:
            captured = self.capture_with_transcript(session_name)
        except subprocess.CalledProcessError:
            return
        detected_model = parse_session_model(captured, self.agent)
        detected_effort = parse_session_effort(captured, self.agent)
        if not detected_model and not detected_effort:
            return
        data = self.read_metadata(session_name)
        if detected_model:
            data["detected_model"] = detected_model
            data["detected_model_updated_at"] = time.time()
        if detected_effort:
            data["detected_effort"] = detected_effort
            data["detected_effort_updated_at"] = time.time()
        data["session_name"] = session_name
        data["agent"] = self.agent
        path = self.metadata_path(session_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def refresh_cli_session_files(self, session_name: str, chat_id: str) -> None:
        data = self.read_metadata(session_name)
        if not data:
            return
        workspace = metadata_path_value(data, "workspace")
        created_at = metadata_float(data, "created_at")
        existing = {
            str(path)
            for path in data.get("cli_session_files", [])
            if isinstance(path, str) and path.strip()
        }
        discovered = discover_cli_session_files(
            self.agent,
            chat_id,
            workspace,
            created_at,
            self.codex_state_dir(),
            self.claude_state_dir(),
        )
        paths = sorted(existing | {str(path) for path in discovered})
        if not paths:
            return
        data["cli_session_files"] = paths
        data["cli_session_files_updated_at"] = time.time()
        path = self.metadata_path(session_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete_cli_session_files(self, metadata: dict[str, object], chat_id: str) -> list[Path]:
        workspace = metadata_path_value(metadata, "workspace")
        created_at = metadata_float(metadata, "created_at")
        paths = {
            Path(str(path)).expanduser().resolve()
            for path in metadata.get("cli_session_files", [])
            if isinstance(path, str) and path.strip()
        }
        paths.update(
            discover_cli_session_files(
                self.agent,
                chat_id,
                workspace,
                created_at,
                self.codex_state_dir(),
                self.claude_state_dir(),
            )
        )
        session_ids = session_ids_for_files(paths, self.agent)
        if self.agent == "claude":
            paths.update(discover_claude_process_state_files(self.claude_state_dir(), session_ids))
            paths.update(discover_claude_subagent_paths(paths))
        deleted: list[Path] = []
        for path in sorted(paths):
            if not safe_cli_session_path(path, self.agent, self.codex_state_dir(), self.claude_state_dir()):
                continue
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(path)
        if self.agent == "codex":
            prune_codex_history(self.codex_state_dir(), chat_id, session_ids)
        if self.agent == "claude":
            prune_jsonl_lines(self.claude_state_dir() / "history.jsonl", chat_id, session_ids)
        return deleted

    def codex_state_dir(self) -> Path:
        value = getattr(self.settings, "codex_state_dir", None)
        if isinstance(value, Path):
            return value.expanduser().resolve()
        return Path("~/.codex").expanduser().resolve()

    def claude_state_dir(self) -> Path:
        value = getattr(self.settings, "claude_state_dir", None)
        if isinstance(value, Path):
            return value.expanduser().resolve()
        return Path("~/.claude").expanduser().resolve()

    # ---- native transcript (JSONL) reply path ----

    def claude_projects_root(self) -> Path:
        return self.claude_state_dir() / "projects"

    def codex_sessions_root(self) -> Path:
        return self.codex_state_dir() / "sessions"

    def resolve_transcript_path(
        self, session_name: str, chat_id: str, workspace: Path
    ) -> Path | None:
        meta = self.read_metadata(session_name)
        pinned = meta.get("transcript_path")
        if isinstance(pinned, str) and pinned and Path(pinned).exists():
            return Path(pinned)
        path: Path | None = None
        if self.agent == "codex":
            created = metadata_float(meta, "created_at") or 0.0
            existing = meta.get("session_uuid")
            # Prefer id-based lookup: the rollout's session_meta.cwd is frozen at
            # creation, so cwd matching breaks after a `/workspace` switch or a
            # cross-machine migration. Matching the known session id is stable.
            if isinstance(existing, str) and existing.strip():
                path = find_codex_rollout_by_id(
                    self.codex_sessions_root(), existing.strip()
                )
            if path is None:
                path = find_codex_rollout(
                    self.codex_sessions_root(), workspace, min_mtime=max(0.0, created - 5)
                )
            # Capture codex's session id from its rollout so it can be resumed
            # later (codex picks a fresh id per launch; we don't control it).
            if path is not None and not (isinstance(existing, str) and existing.strip()):
                sid = rollout_session_id(path)
                if sid:
                    self.store_session_uuid(session_name, sid)
        else:
            sid = meta.get("session_uuid")
            if isinstance(sid, str) and sid:
                path = find_claude_session_file(self.claude_projects_root(), sid)
                if path is None:
                    path = (
                        self.claude_projects_root()
                        / encode_claude_project_dir(workspace)
                        / f"{sid}.jsonl"
                    )
            if path is None or not path.exists():
                recent = find_claude_recent_session_file(self.claude_projects_root(), workspace)
                if recent is not None:
                    path = recent
        if path is not None and path.exists():
            self.pin_transcript(session_name, path)
        return path

    def transcript_cursor(
        self, session_name: str, chat_id: str, workspace: Path
    ) -> tuple[str, int]:
        path = self.resolve_transcript_path(session_name, chat_id, workspace)
        if path is not None and path.exists():
            try:
                return (str(path), path.stat().st_size)
            except OSError:
                return (str(path), 0)
        return (str(path) if path else "", 0)

    def parse_transcript_reply(self, lines: list[dict]):
        return (
            codex_reply_from_lines(lines)
            if self.agent == "codex"
            else claude_reply_from_lines(lines)
        )

    def read_jsonl_reply(self, path_str: str, offset: int) -> str | None:
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            return None
        lines, _ = read_new_jsonl(path, offset)
        result = self.parse_transcript_reply(lines)
        return result.text if result is not None else None

    def wait_for_jsonl_reply(
        self,
        session_name: str,
        chat_id: str,
        workspace: Path,
        path_str: str,
        offset: int,
        timeout: int,
        progress_callback: Callable[[str], None] | None = None,
        progress_interval: float = 30.0,
    ) -> tuple[str, dict | None, tuple[str, int] | None]:
        deadline = time.time() + timeout
        path = Path(path_str) if path_str else None
        cur = offset
        acc: list[dict] = []
        last_progress = 0.0
        last_action: str | None = None
        feedback_dismissed = False
        while time.time() < deadline:
            if path is None or not path.exists():
                resolved = self.resolve_transcript_path(session_name, chat_id, workspace)
                if resolved is None or not resolved.exists():
                    time.sleep(1)
                    continue
                if str(resolved) != path_str:
                    cur = 0  # a freshly created session's transcript just appeared
                path = resolved
            new, cur = read_new_jsonl(path, cur)
            if new:
                acc.extend(new)
                result = self.parse_transcript_reply(acc)
                if result is not None:
                    self.pin_transcript(session_name, path)
                    return result.text, result.usage, (str(path), cur)
            # Update the card when the activity changes OR at least every 10s,
            # throttled to `progress_interval` minimum between updates.
            if progress_callback:
                act = codex_activity(acc) if self.agent == "codex" else claude_activity(acc)
                elapsed = time.time() - last_progress
                action_changed = act["action"] != last_action
                if elapsed >= max(1.0, progress_interval) and (action_changed or elapsed >= 10):
                    progress_callback(activity_detail(acc, self.agent))
                    last_action = act["action"]
                    last_progress = time.time()
            # Peek at the screen to clear blocking interactive prompts (the
            # feedback survey can appear multiple times or take a moment to
            # dismiss, so keep retrying — not just once).
            try:
                screen = self.capture(session_name)
            except subprocess.CalledProcessError:
                screen = ""
            if screen and self.dismiss_claude_feedback_prompt_if_visible(session_name, screen):
                time.sleep(0.5)
                continue
            time.sleep(1)
        raise TmuxReplyStillRunning(
            f"{self.agent} session {session_name} did not finish within {timeout}s "
            f"(transcript {path_str or 'unresolved'})."
        )

    # ---- per-run cursor + session-id persisted in metadata ----

    def _update_metadata(self, session_name: str, **fields: object) -> None:
        path = self.metadata_path(session_name)
        data = self.read_metadata(session_name)
        data.update(fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def store_session_uuid(self, session_name: str, session_uuid: str) -> None:
        self._update_metadata(session_name, session_uuid=session_uuid)

    def pin_transcript(self, session_name: str, path: Path) -> None:
        data = self.read_metadata(session_name)
        if data.get("transcript_path") == str(path):
            return
        self._update_metadata(session_name, transcript_path=str(path))

    def store_run_cursor(self, session_name: str, run_id: str, path_str: str, offset: int) -> None:
        data = self.read_metadata(session_name)
        cursors = data.get("run_cursors")
        if not isinstance(cursors, dict):
            cursors = {}
        cursors[run_id] = [path_str, int(offset)]
        if len(cursors) > 50:
            for key in list(cursors.keys())[:-50]:
                cursors.pop(key, None)
        self._update_metadata(session_name, run_cursors=cursors)

    def read_run_cursor(self, session_name: str, run_id: str) -> tuple[str, int] | None:
        data = self.read_metadata(session_name)
        cursors = data.get("run_cursors")
        if isinstance(cursors, dict):
            value = cursors.get(run_id)
            if isinstance(value, list) and len(value) == 2:
                return (str(value[0]), int(value[1]))
        return None


def run_id_from_marker(marker: str) -> str:
    for prefix in ("PLA_REPLY_START_", "PLA_REPLY_END_"):
        if marker.startswith(prefix):
            return marker[len(prefix):]
    return marker


def metadata_path_value(data: dict[str, object], key: str) -> Path | None:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def metadata_float(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def discover_cli_session_files(
    agent: str,
    chat_id: str,
    workspace: Path | None,
    created_at: float | None,
    codex_state_dir: Path,
    claude_state_dir: Path,
) -> set[Path]:
    if agent == "codex":
        return discover_codex_session_files(codex_state_dir, chat_id, workspace, created_at)
    if agent == "claude":
        return discover_claude_session_files(claude_state_dir, chat_id, workspace, created_at)
    return set()


def discover_codex_session_files(
    state_dir: Path,
    chat_id: str,
    workspace: Path | None,
    created_at: float | None,
) -> set[Path]:
    roots = [state_dir / "sessions", state_dir / "archived_sessions"]
    return {
        path.resolve()
        for root in roots
        for path in iter_files(root, "*.jsonl")
        if codex_file_matches(path, chat_id, workspace, created_at)
    }


def discover_claude_session_files(
    state_dir: Path,
    chat_id: str,
    workspace: Path | None,
    created_at: float | None,
) -> set[Path]:
    root = state_dir / "projects"
    return {
        path.resolve()
        for path in iter_files(root, "*.jsonl")
        if "/subagents/" not in str(path)
        and claude_file_matches(path, chat_id, workspace, created_at)
    }


def iter_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    try:
        return [path for path in root.rglob(pattern) if path.is_file()]
    except OSError:
        return []


def codex_file_matches(
    path: Path,
    chat_id: str,
    workspace: Path | None,
    created_at: float | None,
) -> bool:
    text = read_text_sample(path, max_bytes=4_000_000)
    if valid_chat_id(chat_id) and chat_id in text:
        return True
    meta = codex_session_meta(path, text)
    if not workspace or not meta:
        return False
    cwd = meta.get("cwd")
    if not path_matches_workspace(cwd, workspace):
        return False
    timestamp = iso_to_epoch(str(meta.get("timestamp") or ""))
    return timestamp_close(timestamp, created_at)


def claude_file_matches(
    path: Path,
    chat_id: str,
    workspace: Path | None,
    created_at: float | None,
) -> bool:
    text = read_text_sample(path, max_bytes=4_000_000)
    if valid_chat_id(chat_id) and chat_id in text:
        return True
    if not workspace or not path_matches_workspace(first_jsonl_cwd(text), workspace):
        return False
    timestamp = first_jsonl_timestamp(text)
    return timestamp_close(timestamp, created_at) or timestamp_close(path_mtime(path), created_at)


def read_text_sample(path: Path, max_bytes: int = 1_000_000) -> str:
    try:
        with path.open("rb") as handle:
            return handle.read(max_bytes).decode("utf-8", errors="ignore")
    except OSError:
        return ""


def codex_session_meta(path: Path, text: str | None = None) -> dict[str, object]:
    raw = text if text is not None else read_text_sample(path, max_bytes=64_000)
    first = raw.splitlines()[0] if raw.splitlines() else ""
    try:
        data = json.loads(first)
    except json.JSONDecodeError:
        return {}
    if data.get("type") != "session_meta":
        return {}
    payload = data.get("payload")
    return payload if isinstance(payload, dict) else {}


def first_jsonl_cwd(text: str) -> str | None:
    for line in text.splitlines()[:80]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        cwd = data.get("cwd")
        if isinstance(cwd, str) and cwd.strip():
            return cwd
    return None


def first_jsonl_timestamp(text: str) -> float | None:
    for line in text.splitlines()[:80]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = data.get("timestamp")
        if isinstance(value, str):
            parsed = iso_to_epoch(value)
            if parsed is not None:
                return parsed
    return None


def path_matches_workspace(value: object, workspace: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).expanduser().resolve() == workspace.expanduser().resolve()
    except OSError:
        return False


def iso_to_epoch(value: str) -> float | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def path_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def timestamp_close(value: float | None, created_at: float | None, window: float = 900.0) -> bool:
    if value is None or created_at is None:
        return False
    return created_at - 120.0 <= value <= created_at + window


def session_ids_for_files(paths: set[Path], agent: str) -> set[str]:
    session_ids: set[str] = set()
    for path in paths:
        if agent == "codex":
            meta = codex_session_meta(path)
            session_id = meta.get("id")
            if isinstance(session_id, str) and session_id:
                session_ids.add(session_id)
                continue
            match = re.search(r"(019e[0-9a-f-]+)\.jsonl$", path.name)
            if match:
                session_ids.add(match.group(1))
        elif agent == "claude":
            session_id = claude_session_id(path)
            if session_id:
                session_ids.add(session_id)
    return session_ids


def claude_session_id(path: Path) -> str | None:
    if path.suffix == ".jsonl":
        text = read_text_sample(path, max_bytes=64_000)
        for line in text.splitlines()[:20]:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = data.get("sessionId")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return path.stem if looks_like_uuid(path.stem) else None
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None
        value = data.get("sessionId") if isinstance(data, dict) else None
        return value.strip() if isinstance(value, str) and value.strip() else None
    return None


def looks_like_uuid(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value))


def valid_chat_id(value: str) -> bool:
    return bool(re.fullmatch(r"oc_[A-Za-z0-9]{16,}", value or ""))


def discover_claude_process_state_files(state_dir: Path, session_ids: set[str]) -> set[Path]:
    if not session_ids:
        return set()
    root = state_dir / "sessions"
    result: set[Path] = set()
    for path in iter_files(root, "*.json"):
        session_id = claude_session_id(path)
        if session_id in session_ids:
            result.add(path.resolve())
    return result


def discover_claude_subagent_paths(paths: set[Path]) -> set[Path]:
    result: set[Path] = set()
    for path in paths:
        if path.suffix != ".jsonl" or not looks_like_uuid(path.stem):
            continue
        subdir = path.with_suffix("")
        if subdir.is_dir():
            result.add(subdir.resolve())
    return result


def safe_cli_session_path(
    path: Path,
    agent: str,
    codex_state_dir: Path,
    claude_state_dir: Path,
) -> bool:
    path = path.expanduser().resolve()
    if agent == "codex":
        return (
            is_relative_to(path, codex_state_dir / "sessions")
            or is_relative_to(path, codex_state_dir / "archived_sessions")
        ) and path.suffix == ".jsonl"
    if agent == "claude":
        if is_relative_to(path, claude_state_dir / "projects"):
            return path.suffix == ".jsonl" or path.is_dir()
        return is_relative_to(path, claude_state_dir / "sessions") and path.suffix == ".json"
    return False


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.expanduser().resolve())
        return True
    except ValueError:
        return False


def prune_codex_history(state_dir: Path, chat_id: str, session_ids: set[str]) -> None:
    prune_jsonl_lines(state_dir / "history.jsonl", chat_id, session_ids)
    prune_jsonl_lines(state_dir / "session_index.jsonl", chat_id, session_ids)


def prune_jsonl_lines(path: Path, chat_id: str, session_ids: set[str]) -> None:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    tokens = {*session_ids}
    if valid_chat_id(chat_id):
        tokens.add(chat_id)
    tokens.discard("")
    if not tokens:
        return
    kept = [line for line in lines if not any(token in line for token in tokens)]
    if len(kept) == len(lines):
        return
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    temp_path.replace(path)


def extract_marked_reply(text: str, start_marker: str, end_marker: str) -> str | None:
    start_re = marker_line_pattern(start_marker)
    end_re = marker_line_pattern(end_marker)
    start_indices: list[int] = []
    replies: list[str] = []
    last_end_index = -1
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if start_re.match(line):
            start_indices.append(index)
            continue
        if not end_re.match(line):
            continue
        preceding_starts = [start for start in start_indices if last_end_index < start < index]
        last_end_index = index
        if not preceding_starts:
            continue
        start = preceding_starts[-1]
        replies.append("\n".join(lines[start + 1 : index]).strip())
    if not replies:
        return None
    return replies[-1]


def extract_mismatched_marked_reply_after_prompt(
    text: str,
    expected_start_marker: str,
    expected_end_marker: str,
) -> str | None:
    lines = text.splitlines()
    prompt_index = latest_prompt_instruction_index(lines, expected_start_marker)
    if prompt_index is None:
        return None
    return extract_latest_generic_marked_reply(
        "\n".join(lines[prompt_index + 1 :]),
        excluded_markers={expected_start_marker, expected_end_marker},
    )


def latest_prompt_instruction_index(lines: list[str], start_marker: str) -> int | None:
    start_re = marker_line_pattern(start_marker)
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if start_marker not in line:
            continue
        if start_re.match(line):
            continue
        return index
    return None


def extract_latest_generic_marked_reply(
    text: str,
    excluded_markers: set[str] | None = None,
) -> str | None:
    excluded_markers = excluded_markers or set()
    marker_prefix = r"[^\S\r\n]*(?:[●•]\s*)?"
    start_re = re.compile(rf"^{marker_prefix}PLA_REPLY_START_([A-Za-z0-9_-]+)[^\S\r\n]*$")
    end_re = re.compile(rf"^{marker_prefix}PLA_REPLY_END_([A-Za-z0-9_-]+)[^\S\r\n]*$")
    starts: list[tuple[str, int]] = []
    replies: list[str] = []
    last_end_index = -1
    lines = text.splitlines()
    for index, line in enumerate(lines):
        start_match = start_re.match(line)
        if start_match:
            marker = f"PLA_REPLY_START_{start_match.group(1)}"
            if marker not in excluded_markers:
                starts.append((start_match.group(1), index))
            continue
        end_match = end_re.match(line)
        if not end_match:
            continue
        marker = f"PLA_REPLY_END_{end_match.group(1)}"
        if marker in excluded_markers:
            continue
        preceding_starts = [
            start_index
            for marker_id, start_index in starts
            if marker_id == end_match.group(1) and last_end_index < start_index < index
        ]
        last_end_index = index
        if not preceding_starts:
            continue
        start = preceding_starts[-1]
        reply = "\n".join(lines[start + 1 : index]).strip()
        if reply:
            replies.append(reply)
    if not replies:
        return None
    return replies[-1]


def marker_line_pattern(marker: str) -> re.Pattern[str]:
    marker_prefix = r"[^\S\r\n]*(?:[●•]\s*)?"
    return re.compile(rf"^{marker_prefix}{re.escape(marker)}[^\S\r\n]*$")


def normalize_terminal_text(text: str) -> str:
    text = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    while "\b" in text:
        updated = re.sub(r".\b", "", text)
        if updated == text:
            return text.replace("\b", "")
        text = updated
    return text


def session_ready_for_input(text: str, agent: str) -> bool:
    if agent == "claude":
        return "❯" in text or "Try \"" in text or "don't ask on" in text
    if agent == "codex":
        return "›" in text or "context left" in text
    return bool(text.strip())


def session_ready_for_current_input(text: str, agent: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-12:])
    if session_tail_busy(tail):
        return False
    return session_ready_for_input(tail, agent)


def claude_feedback_prompt_visible(text: str) -> bool:
    lines = [line for line in normalize_terminal_text(text).splitlines() if line.strip()]
    tail = "\n".join(lines[-14:])
    return "How is Claude doing this session?" in tail and "0: Dismiss" in tail


def session_tail_busy(text: str) -> bool:
    return bool(
        re.search(
            r"Compacting conversation|esc to interrupt|\bWorking\b|^\s*(?:\*|✽|✢|✶)\s+",
            text,
            re.MULTILINE,
        )
    )


def session_has_background_work(text: str) -> bool:
    return bool(
        re.search(
            r"Waiting for \d+ background agent"
            r"|esc to interrupt"
            r"|Bootstrapping"
            r"|↓ to manage",
            text,
        )
    )


def parse_session_model(text: str, agent: str) -> str | None:
    if agent == "codex":
        footer = re.compile(
            r"(?m)^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s+"
            rf"(?:{EFFORT_TOKEN_PATTERN})\s+·\s+/"
        )
        matches = footer.findall(text)
        if matches:
            return matches[-1]
    if agent == "claude":
        tui_line = re.compile(
            r"(?m)^[^\n]*?([A-Z][A-Za-z0-9 .()/_-]*?)\s+with\s+"
            rf"(?:{EFFORT_TOKEN_PATTERN})\s+effort\s+·\s+Claude"
        )
        matches = tui_line.findall(text)
        if matches:
            return matches[-1].strip()
    explicit = re.compile(r"(?im)^\s*model\s*[:=]\s*`?([A-Za-z0-9][A-Za-z0-9._/-]*)`?")
    matches = explicit.findall(text)
    if matches:
        return matches[-1].strip().rstrip("`")
    return None


def parse_session_effort(text: str, agent: str) -> str | None:
    command_result = re.compile(
        rf"(?im)\bSet\s+effort\s+level\s+to\s+({EFFORT_TOKEN_PATTERN})\b"
    )
    command_matches = command_result.findall(text)
    if command_matches:
        return command_matches[-1].lower()
    if agent == "codex":
        footer = re.compile(
            r"(?m)^\s*[A-Za-z0-9][A-Za-z0-9._-]*\s+"
            rf"({EFFORT_TOKEN_PATTERN})\s+·\s+/"
        )
        matches = footer.findall(text)
        if matches:
            return matches[-1].lower()
    if agent == "claude":
        tui_line = re.compile(
            rf"(?im)\bwith\s+({EFFORT_TOKEN_PATTERN})\s+effort\b"
        )
        matches = tui_line.findall(text)
        if matches:
            return matches[-1].lower()
    explicit = re.compile(rf"(?im)\beffort\s*[:=]\s*`?({EFFORT_TOKEN_PATTERN})`?")
    matches = explicit.findall(text)
    if matches:
        return matches[-1].lower()
    return None


def metadata_string(data: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def summarize_session_progress(text: str, agent: str, limit: int = 260) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        cleaned = clean_progress_line(line)
        if not cleaned or skip_progress_line(cleaned, agent):
            continue
        if cleaned.startswith(("• ", "- ", "└ ", "✻ ", "● ", "⎿ ")):
            return truncate_progress(cleaned, limit)
    for line in reversed(lines):
        cleaned = clean_progress_line(line)
        if cleaned and not skip_progress_line(cleaned, agent):
            return truncate_progress(cleaned, limit)
    return None


def clean_progress_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" │")


def skip_progress_line(line: str, agent: str) -> bool:
    if not line:
        return True
    if "PLA_REPLY_START_" in line or "PLA_REPLY_END_" in line:
        return True
    if line.startswith(
        (
            "- Always put the literal start marker",
            "- Then write the reply body",
            "- If no reply is useful",
            "- Include only content intended for the Feishu group",
            "- note with observable actions/results is OK",
            "- If you created or want to share local artifacts",
            "- explicit Markdown link/image",
            "- Then put the literal end marker",
            "- Do not put either marker",
        )
    ):
        return True
    if line.startswith(("Output protocol", "Message:", "content:", "source:", "chat_id:")):
        return True
    if line.startswith(("›", "❯", "⏵", "─", "━", "⚠ MCP startup interrupted")):
        return True
    if line.startswith("✻ Baked"):
        return True
    if re.search(r"\b(?:minimal|low|medium|high|xhigh|max)\s+·\s+/", line):
        return True
    if agent == "codex" and line in {"OpenAI Codex", "Working"}:
        return True
    return False


def truncate_progress(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def without_existing_cwd_args(args: tuple[str, ...]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-C", "--cd"}:
            skip_next = True
            continue
        if arg.startswith("--cd="):
            continue
        cleaned.append(arg)
    return cleaned


def with_model_arg(args: list[str], model: str | None, flag: str) -> list[str]:
    if not model or has_model_arg(args):
        return args
    return [*args, flag, model]


def has_model_arg(args: list[str]) -> bool:
    for index, arg in enumerate(args):
        if arg in {"-m", "--model"}:
            return index + 1 < len(args)
        if arg.startswith("--model="):
            return True
    return False


def with_codex_effort_arg(args: list[str], effort: str | None) -> list[str]:
    if not effort or has_codex_effort_arg(args):
        return args
    return [*args, "-c", f'model_reasoning_effort="{effort}"']


def has_codex_effort_arg(args: list[str]) -> bool:
    for index, arg in enumerate(args):
        if arg == "-c" and index + 1 < len(args):
            if str(args[index + 1]).startswith("model_reasoning_effort="):
                return True
        if arg.startswith("model_reasoning_effort="):
            return True
    return False
