from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
from typing import Literal

from .effort import EffortError, normalize_effort
from .responders import ResponderError, normalize_responder


AgentMode = Literal["codex", "claude", "both", "tasks"]


def load_env_file(path: str | os.PathLike[str] | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: bool = False) -> bool:
    value = _env(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = _env(name, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    value = _env(name, str(default)).strip()
    try:
        return float(value)
    except ValueError:
        return default


def _args(name: str, default: str) -> tuple[str, ...]:
    return tuple(shlex.split(_env(name, default)))


def _optional(name: str) -> str | None:
    value = _env(name, "").strip()
    return value or None


def _effort(name: str) -> str | None:
    value = _optional(name)
    if not value:
        return None
    try:
        return normalize_effort(value)
    except EffortError:
        return None


def _responder(name: str, default: str = "both") -> str:
    try:
        return normalize_responder(_env(name, default))
    except ResponderError:
        return "both"


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    value = _env(name, default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _paths(name: str, default: tuple[Path, ...]) -> tuple[Path, ...]:
    values = _csv(name)
    if not values:
        return default
    return tuple(Path(value).expanduser().resolve() for value in values)


def proxy_env(
    proxy_url: str | None,
    no_proxy: str | None,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    if proxy_url:
        env["http_proxy"] = proxy_url
        env["https_proxy"] = proxy_url
        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
    if no_proxy:
        env["no_proxy"] = no_proxy
        env["NO_PROXY"] = no_proxy
    return env


def proxy_command_prefix(proxy_url: str | None, no_proxy: str | None) -> list[str]:
    if not proxy_url and not no_proxy:
        return []
    assignments: list[str] = []
    if proxy_url:
        assignments.extend(
            [
                f"http_proxy={proxy_url}",
                f"https_proxy={proxy_url}",
                f"HTTP_PROXY={proxy_url}",
                f"HTTPS_PROXY={proxy_url}",
            ]
        )
    if no_proxy:
        assignments.extend([f"no_proxy={no_proxy}", f"NO_PROXY={no_proxy}"])
    return ["env", *assignments]


@dataclass(frozen=True)
class Settings:
    lark_cli: str
    lark_profile: str | None
    dashboard_lark_profile: str | None
    event_key: str
    event_keys: tuple[str, ...]
    proxy_url: str | None
    agent_proxy_url: str | None
    no_proxy: str | None
    chat_id: str | None
    chat_id_exclude: tuple[str, ...]
    bot_open_id: str | None
    agent_mode: AgentMode
    bot_aliases: tuple[str, ...]
    enable_tasks: bool
    enable_debate: bool
    workspace: Path
    workspace_roots: tuple[Path, ...]
    handle_management_commands: bool
    workspace_warmup_agents: tuple[str, ...]
    codex_cmd: str
    codex_args: tuple[str, ...]
    codex_runtime: str
    codex_session_args: tuple[str, ...]
    codex_startup_commands: tuple[str, ...]
    codex_state_dir: Path
    codex_model: str | None
    codex_default_effort: str | None
    codex_timeout: int
    claude_cmd: str
    claude_args: tuple[str, ...]
    claude_runtime: str
    claude_session_args: tuple[str, ...]
    claude_state_dir: Path
    claude_model: str | None
    claude_default_effort: str | None
    claude_timeout: int
    task_as: str
    tasklist_id: str | None
    respond_to_all: bool
    default_responder: str
    send_progress: bool
    dashboard_enabled: bool
    dashboard_host: str
    dashboard_port: int
    dashboard_public_url: str | None
    dashboard_tab_enabled: bool
    dashboard_tab_name: str
    max_message_chars: int
    enable_artifacts: bool
    max_artifacts: int
    enable_inbound_files: bool
    max_inbound_files: int
    max_event_age_seconds: int
    status_update_seconds: int
    enable_memory: bool
    state_dir: Path
    memory_turns: int
    memory_chars: int
    outbox_ttl_seconds: int
    enable_agent_discussion: bool
    direct_agent_handoff: bool
    max_agent_discussion_turns: int
    agent_discussion_window_seconds: int
    no_reply_token: str
    session_startup_wait: float
    session_capture_lines: int
    session_history_limit: int
    session_columns: int
    session_rows: int
    session_command_watch_seconds: int


def load_settings(env_file: str | None = ".env") -> Settings:
    load_env_file(env_file)
    workspace = Path(_env("PLA_WORKSPACE", os.getcwd())).expanduser().resolve()
    event_key = _env("PLA_EVENT_KEY", "im.message.receive_v1")
    agent_mode = _env("PLA_AGENT_MODE", "both").strip().lower()
    if agent_mode not in {"codex", "claude", "both", "tasks"}:
        agent_mode = "both"
    bot_aliases = _csv("PLA_BOT_ALIASES")
    if not bot_aliases and agent_mode == "codex":
        bot_aliases = ("Codex",)
    if not bot_aliases and agent_mode == "claude":
        bot_aliases = ("Claude Code", "Claude")
    return Settings(
        lark_cli=_env("PLA_LARK_CLI", "lark-cli"),
        lark_profile=_optional("PLA_LARK_PROFILE"),
        dashboard_lark_profile=_optional("PLA_DASHBOARD_LARK_PROFILE"),
        event_key=event_key,
        event_keys=_csv("PLA_EVENT_KEYS", event_key),
        proxy_url=_optional("PLA_PROXY_URL"),
        agent_proxy_url=_optional("PLA_AGENT_PROXY_URL"),
        no_proxy=_env("PLA_NO_PROXY", "localhost,127.0.0.1,::1"),
        chat_id=_optional("PLA_CHAT_ID"),
        chat_id_exclude=tuple(s.strip() for s in os.environ.get("PLA_CHAT_ID_EXCLUDE", "").split(",") if s.strip()),
        bot_open_id=_optional("PLA_BOT_OPEN_ID"),
        agent_mode=agent_mode,  # type: ignore[arg-type]
        bot_aliases=bot_aliases,
        enable_tasks=_bool("PLA_ENABLE_TASKS", agent_mode in {"both", "tasks"}),
        enable_debate=_bool("PLA_ENABLE_DEBATE", agent_mode == "both"),
        workspace=workspace,
        workspace_roots=_paths("PLA_WORKSPACE_ROOTS", (workspace,)),
        handle_management_commands=_bool("PLA_HANDLE_MANAGEMENT_COMMANDS", True),
        workspace_warmup_agents=valid_agents(_csv("PLA_WORKSPACE_WARMUP_AGENTS")),
        codex_cmd=_env("PLA_CODEX_CMD", "codex"),
        codex_args=_args(
            "PLA_CODEX_ARGS",
            "exec --skip-git-repo-check -",
        ),
        codex_runtime=_env("PLA_CODEX_RUNTIME", "session").strip().lower(),
        codex_session_args=_args(
            "PLA_CODEX_SESSION_ARGS",
            "--no-alt-screen",
        ),
        codex_startup_commands=_csv(
            "PLA_CODEX_STARTUP_COMMANDS",
            "/permissions auto-review",
        ),
        codex_state_dir=Path(
            _env("PLA_CODEX_STATE_DIR", os.environ.get("CODEX_HOME", "~/.codex"))
        ).expanduser().resolve(),
        codex_model=_optional("PLA_CODEX_MODEL"),
        codex_default_effort=_effort("PLA_CODEX_DEFAULT_EFFORT"),
        codex_timeout=_int("PLA_CODEX_TIMEOUT", 900),
        claude_cmd=_env("PLA_CLAUDE_CMD", "claude"),
        claude_args=_args(
            "PLA_CLAUDE_ARGS",
            "-p --output-format text",
        ),
        claude_runtime=_env("PLA_CLAUDE_RUNTIME", "session").strip().lower(),
        claude_session_args=_args(
            "PLA_CLAUDE_SESSION_ARGS",
            "--permission-mode auto --teammate-mode in-process",
        ),
        claude_state_dir=Path(
            _env("PLA_CLAUDE_STATE_DIR", os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude"))
        ).expanduser().resolve(),
        claude_model=_optional("PLA_CLAUDE_MODEL"),
        claude_default_effort=_effort("PLA_CLAUDE_DEFAULT_EFFORT"),
        claude_timeout=_int("PLA_CLAUDE_TIMEOUT", 900),
        task_as=_env("PLA_TASK_AS", "user"),
        tasklist_id=_optional("PLA_TASKLIST_ID"),
        respond_to_all=_bool("PLA_RESPOND_TO_ALL", False),
        default_responder=_responder("PLA_DEFAULT_RESPONDER"),
        send_progress=_bool("PLA_SEND_PROGRESS", True),
        dashboard_enabled=_bool("PLA_DASHBOARD_ENABLED", False),
        dashboard_host=_env("PLA_DASHBOARD_HOST", "127.0.0.1"),
        dashboard_port=max(1, _int("PLA_DASHBOARD_PORT", 8765)),
        dashboard_public_url=_optional("PLA_DASHBOARD_PUBLIC_URL"),
        dashboard_tab_enabled=_bool("PLA_DASHBOARD_TAB_ENABLED", False),
        dashboard_tab_name=_env("PLA_DASHBOARD_TAB_NAME", "AI 状态"),
        max_message_chars=max(800, _int("PLA_MAX_MESSAGE_CHARS", 3500)),
        enable_artifacts=_bool("PLA_ENABLE_ARTIFACTS", True),
        max_artifacts=_int("PLA_MAX_ARTIFACTS", 8),
        enable_inbound_files=_bool("PLA_ENABLE_INBOUND_FILES", True),
        max_inbound_files=max(1, _int("PLA_MAX_INBOUND_FILES", 8)),
        max_event_age_seconds=max(0, _int("PLA_MAX_EVENT_AGE_SECONDS", 3600)),
        status_update_seconds=_int("PLA_STATUS_UPDATE_SECONDS", 30),
        enable_memory=_bool("PLA_ENABLE_MEMORY", True),
        state_dir=Path(_env("PLA_STATE_DIR", ".state")).expanduser().resolve(),
        memory_turns=_int("PLA_MEMORY_TURNS", 24),
        memory_chars=_int("PLA_MEMORY_CHARS", 9000),
        outbox_ttl_seconds=_int("PLA_OUTBOX_TTL_SECONDS", 86400),
        enable_agent_discussion=_bool("PLA_ENABLE_AGENT_DISCUSSION", True),
        direct_agent_handoff=_bool("PLA_DIRECT_AGENT_HANDOFF", True),
        max_agent_discussion_turns=_int("PLA_MAX_AGENT_DISCUSSION_TURNS", 6),
        agent_discussion_window_seconds=_int("PLA_AGENT_DISCUSSION_WINDOW_SECONDS", 600),
        no_reply_token=_env("PLA_NO_REPLY_TOKEN", "[NO_REPLY]"),
        session_startup_wait=_float("PLA_SESSION_STARTUP_WAIT", 4.0),
        session_capture_lines=_int("PLA_SESSION_CAPTURE_LINES", 20000),
        session_history_limit=_int("PLA_SESSION_HISTORY_LIMIT", 50000),
        session_columns=max(80, _int("PLA_SESSION_COLUMNS", 120)),
        session_rows=max(24, _int("PLA_SESSION_ROWS", 80)),
        session_command_watch_seconds=_int("PLA_SESSION_COMMAND_WATCH_SECONDS", 20),
    )


def valid_agents(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        agent = value.strip().lower()
        if agent in {"codex", "claude"} and agent not in result:
            result.append(agent)
    return tuple(result)
