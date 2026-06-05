"""Read agent replies from the CLIs' native session transcripts (JSONL).

This replaces the terminal marker-scraping path. The prompt is still pasted into
the tmux session (subscription quota), but the reply is read from the structured
transcript the CLI writes to disk, so no PLA_REPLY_START/END markers are needed.

Two formats, confirmed against live files:

Claude (~/.claude/projects/<encoded-cwd>/<session-id>.jsonl):
    one JSON object per line; assistant turns are
    {"type":"assistant","message":{"role","content":[{type,text|...}],
     "stop_reason","usage",...}}
    A turn is finished when the latest assistant message's stop_reason is a
    terminal reason (end_turn / stop_sequence / max_tokens); "tool_use" means
    it is mid-turn. Launch with `claude --session-id <uuid>` so the file path is
    deterministic.

Codex (~/.codex/sessions/Y/M/D/rollout-<ts>-<uuid>.jsonl):
    first line {"type":"session_meta","payload":{"cwd",...}}; the turn finishes
    with an event {"type":"event_msg","payload":{"type":"task_complete",
    "last_agent_message": <final text>, ...}}. We match the rollout file by
    session_meta.cwd + recency.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path


CLAUDE_TERMINAL_STOP_REASONS = {"end_turn", "stop_sequence", "max_tokens"}


@dataclass(frozen=True)
class TurnResult:
    text: str
    usage: dict | None = None


# ---------------------------------------------------------------- parsing

def _claude_message_text(message: dict) -> str:
    blocks = message.get("content")
    if not isinstance(blocks, list):
        return ""
    parts = [
        b["text"]
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    return "\n".join(parts).strip()


def claude_reply_from_lines(lines: list[dict]) -> TurnResult | None:
    """Return the finished reply from new Claude jsonl lines, or None if the
    turn has not completed yet."""
    messages = [
        o["message"]
        for o in lines
        if o.get("type") == "assistant" and isinstance(o.get("message"), dict)
    ]
    if not messages:
        return None
    last = messages[-1]
    if last.get("stop_reason") not in CLAUDE_TERMINAL_STOP_REASONS:
        return None  # still generating / awaiting a tool result
    text = _claude_message_text(last)
    if not text:
        # Final message carried no text (answer was in an earlier block before a
        # tool call) — fall back to all assistant text in this turn.
        text = "\n".join(t for m in messages if (t := _claude_message_text(m))).strip()
    usage = last.get("usage") if isinstance(last.get("usage"), dict) else None
    return TurnResult(text=text, usage=usage)


def claude_followup_from_lines(lines: list[dict]) -> TurnResult | None:
    """Check for a follow-up reply in lines read AFTER the first end_turn.

    Claude can produce multiple end_turn messages in one turn (e.g. when a
    background teammate finishes and Claude posts the result). This function
    looks for a *new* terminal assistant message in the continuation lines.
    Returns None if the agent is still working or no new reply appeared.

    Note: we do NOT stop on user messages because new prompts (handoffs) can
    be injected while a background agent is still running — Claude interleaves
    the subagent result with the new turn in the same transcript.
    """
    messages = [
        o["message"]
        for o in lines
        if o.get("type") == "assistant" and isinstance(o.get("message"), dict)
    ]
    if not messages:
        return None
    last = messages[-1]
    if last.get("stop_reason") not in CLAUDE_TERMINAL_STOP_REASONS:
        return None
    text = _claude_message_text(last)
    if not text:
        return None
    usage = last.get("usage") if isinstance(last.get("usage"), dict) else None
    return TurnResult(text=text, usage=usage)


_CLAUDE_TOOL_LABEL = {
    "Bash": "⚙️ bash", "Read": "📖 读", "Write": "✍️ 写", "Edit": "✏️ 改",
    "Grep": "🔎 grep", "Glob": "🔎 glob", "WebSearch": "🔍 搜网", "WebFetch": "🌐 抓取",
    "Task": "🤝 子代理", "TodoWrite": "📝 待办", "NotebookEdit": "✏️ notebook",
}
_CLAUDE_TARGET_KEYS = ("file_path", "command", "pattern", "query", "url", "path", "description")


def _short(value: object, limit: int = 40) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def fmt_tokens(n: int | None) -> str:
    if not n:
        return ""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def claude_activity(lines: list[dict]) -> dict:
    msgs = [
        o["message"] for o in lines
        if o.get("type") == "assistant" and isinstance(o.get("message"), dict)
    ]
    steps = len(msgs)
    tokens = sum(int((m.get("usage") or {}).get("output_tokens") or 0) for m in msgs)
    action = ""
    for m in reversed(msgs):
        blocks = m.get("content") or []
        tools = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        if tools:
            tool = tools[-1]
            inp = tool.get("input") or {}
            target = next((inp[k] for k in _CLAUDE_TARGET_KEYS if k in inp), "")
            if isinstance(target, str) and "/" in target and " " not in target:
                target = target.rsplit("/", 1)[-1]
            label = _CLAUDE_TOOL_LABEL.get(tool.get("name", ""), f"🔧 {tool.get('name', '')}")
            action = f"{label} {_short(target)}".strip()
            break
        if any(isinstance(b, dict) and b.get("type") == "text" and (b.get("text") or "").strip() for b in blocks):
            action = "✍️ 整理回复"
            break
    return {"action": action or "✻ 思考中", "steps": steps, "tokens": tokens, "message": ""}


def _codex_call_action(payload: dict) -> str:
    name = payload.get("name") or ""
    raw = payload.get("arguments")
    if raw is None:
        raw = payload.get("input")
    arg = ""
    if isinstance(raw, str) and raw.lstrip().startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        for key in ("cmd", "command", "file_path", "path", "query", "pattern"):
            if data.get(key):
                arg = str(data[key])
                break
    elif isinstance(raw, dict):
        for key in ("cmd", "command", "file_path", "path", "query"):
            if raw.get(key):
                arg = str(raw[key])
                break
    label = "⚙️ exec" if name in ("exec_command", "shell", "bash") else (f"🔧 {name}" if name else "🔧 tool")
    return f"{label} {_short(arg)}".strip() if arg else label


def codex_activity(lines: list[dict]) -> dict:
    steps = 0
    action = ""
    tokens = 0
    last_message = ""
    for o in lines:
        kind_outer = o.get("type")
        if kind_outer == "response_item":
            payload = o.get("payload") or {}
            ptype = payload.get("type")
            if ptype in ("function_call", "custom_tool_call"):
                steps += 1
                action = _codex_call_action(payload)
            elif ptype == "web_search_call":
                action = "🔍 搜网"
        elif kind_outer == "event_msg":
            payload = o.get("payload") or {}
            kind = payload.get("type")
            if kind == "web_search_end":
                action = f"🔍 搜索 {_short(payload.get('query', ''))}"
            elif kind == "patch_apply_end":
                action = "✏️ 改文件"
            elif kind == "agent_message":
                action = "✍️ 回复中"
                msg = (payload.get("message") or "").strip()
                if msg:
                    last_message = msg
            elif kind == "task_started":
                action = "✻ 开始"
            elif kind == "token_count":
                # total_token_usage is cumulative for the whole session and
                # re-counts the growing context each request, so it balloons.
                # last_token_usage is per-request; summing the generated
                # tokens (output + reasoning) gives this turn's real output.
                last = (payload.get("info") or {}).get("last_token_usage") or {}
                tokens += int(last.get("output_tokens") or 0) + int(last.get("reasoning_output_tokens") or 0)
    return {"action": action or "✻ 思考中", "steps": steps, "tokens": tokens, "message": last_message}


def activity_detail(lines: list[dict], agent: str) -> str:
    """Live-activity summary for the status card: agent message (persistent) +
    current action + step count + tokens."""
    act = codex_activity(lines) if agent == "codex" else claude_activity(lines)
    status_parts = [act["action"]]
    if act["steps"]:
        status_parts.append(f"{act['steps']} 步")
    tok = fmt_tokens(act["tokens"])
    if tok:
        status_parts.append(f"{tok} tok")
    status_line = " · ".join(status_parts)
    message = act.get("message") or ""
    if message:
        return f"{message}\n\n{status_line}"
    return status_line


def codex_reply_from_lines(lines: list[dict]) -> TurnResult | None:
    """Return the finished reply from new Codex rollout lines, or None if no
    task_complete event is present yet."""
    text: str | None = None
    usage: dict | None = None
    for o in lines:
        if o.get("type") != "event_msg":
            continue
        payload = o.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if kind == "task_complete":
            value = payload.get("last_agent_message")
            if isinstance(value, str):
                text = value
        elif kind == "token_count":
            info = payload.get("info")
            usage = info if isinstance(info, dict) else payload
    if text is None:
        return None
    return TurnResult(text=text.strip(), usage=usage)


# ---------------------------------------------------------------- locating

def claude_model_from_lines(lines: list[dict]) -> str | None:
    """Extract the model name from Claude JSONL lines."""
    for o in reversed(lines):
        model = (o.get("message") or {}).get("model")
        if model and isinstance(model, str):
            return model
    return None


def codex_model_from_lines(lines: list[dict]) -> str | None:
    """Extract the model name from Codex rollout lines."""
    for o in lines:
        if o.get("type") == "turn_context":
            model = (o.get("payload") or {}).get("model") or o.get("model")
            if model and isinstance(model, str):
                return model
    return None


def encode_claude_project_dir(workspace: Path) -> str:
    """Claude encodes the cwd into the project dir name by replacing every run
    of non-alphanumeric chars with a single dash (leading dash kept)."""
    raw = str(workspace)
    return "".join(c if c.isalnum() else "-" for c in raw)


def find_claude_session_file(projects_root: Path, session_id: str) -> Path | None:
    """Locate <session_id>.jsonl under the projects root. session_id is the
    uuid we launched the session with, so the match is exact."""
    if not session_id:
        return None
    matches = glob.glob(str(projects_root / "*" / f"{session_id}.jsonl"))
    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


def find_claude_recent_session_file(projects_root: Path, workspace: Path) -> Path | None:
    """Fallback for an existing session with no known session-id: the most
    recently modified jsonl in the workspace's project dir (the one being
    written). Reliable only when a workspace has a single active session — when
    several sessions share a workspace, launch with --session-id instead. Pin
    the resolved path in session metadata so this heuristic runs at most once.
    """
    proj = projects_root / encode_claude_project_dir(workspace)
    files = glob.glob(str(proj / "*.jsonl"))
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


def rollout_cwd(path: Path) -> str | None:
    """Read the session_meta cwd from the first line of a Codex rollout."""
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return None
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return None
    if obj.get("type") != "session_meta":
        return None
    payload = obj.get("payload")
    if isinstance(payload, dict):
        cwd = payload.get("cwd")
        return str(cwd) if cwd else None
    return None


def rollout_session_id(path: Path) -> str | None:
    """The codex session id from a rollout's session_meta first line (used to
    `codex resume <id>`)."""
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return None
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        return None
    if obj.get("type") != "session_meta":
        return None
    payload = obj.get("payload")
    if isinstance(payload, dict):
        sid = payload.get("id")
        return str(sid) if sid else None
    return None


def find_codex_rollout(sessions_root: Path, workspace: Path, min_mtime: float = 0.0) -> Path | None:
    """Most recent Codex rollout whose session_meta.cwd matches the workspace
    and that was modified at/after min_mtime (when the session was started)."""
    target = str(workspace)
    candidates = glob.glob(str(sessions_root / "*" / "*" / "*" / "rollout-*.jsonl"))
    best: tuple[float, Path] | None = None
    for raw in candidates:
        path = Path(raw)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime + 1e-3 < min_mtime:
            continue
        if rollout_cwd(path) != target:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    return best[1] if best else None


def find_codex_rollout_by_id(sessions_root: Path, session_id: str) -> Path | None:
    """Locate a Codex rollout by its session_meta id, ignoring cwd.

    `find_codex_rollout` matches on the rollout's frozen session_meta.cwd, which
    breaks whenever the live workspace differs from where the session was first
    created — after a `/workspace` switch or a cross-machine migration. When the
    session id is known we match on it directly, which is stable across both.
    """
    if not session_id:
        return None
    # Fast path: the id is embedded in the rollout filename.
    direct = glob.glob(str(sessions_root / "*" / "*" / "*" / f"rollout-*-{session_id}.jsonl"))
    best: tuple[float, Path] | None = None
    for raw in direct:
        path = Path(raw)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    if best is not None:
        return best[1]
    # Slow path: filename convention may differ — read each session_meta id.
    for raw in glob.glob(str(sessions_root / "*" / "*" / "*" / "rollout-*.jsonl")):
        path = Path(raw)
        if rollout_session_id(path) == session_id:
            return path
    return None


# ---------------------------------------------------------------- incremental read

def read_new_jsonl(path: Path, offset: int) -> tuple[list[dict], int]:
    """Read JSON objects appended to `path` after byte `offset`.

    Returns (objects, new_offset). A trailing partial line (write in progress)
    is left unconsumed so it is re-read whole next time. Unparseable complete
    lines are skipped.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return [], offset

    if not data:
        return [], offset

    # Keep a trailing partial line (no newline yet) for next read.
    if data.endswith(b"\n"):
        consumed = len(data)
        chunk = data
    else:
        nl = data.rfind(b"\n")
        if nl == -1:
            return [], offset  # nothing complete yet
        consumed = nl + 1
        chunk = data[:consumed]

    objects: list[dict] = []
    for line in chunk.decode("utf-8", "ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)
    return objects, offset + consumed
