from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal


RouteKind = Literal[
    "ignore",
    "help",
    "agent",
    "multi_agent",
    "debate",
    "task",
    "workspace",
    "clear",
    "responder",
    "import_memory",
    "session_command",
    "shell_command",
]
AgentName = Literal["codex", "claude"]


@dataclass(frozen=True)
class TaskRequest:
    summary: str
    description: str = ""
    assignee: str | None = None
    due: str | None = None
    tasklist_id: str | None = None


@dataclass(frozen=True)
class Route:
    kind: RouteKind
    text: str = ""
    agent: AgentName | None = None
    agent_texts: dict[AgentName, str] | None = None
    task: TaskRequest | None = None
    # True when the agent(s) were chosen by the respond-to-all fallthrough rather
    # than explicit addressing (@mention, /codex, etc.). Only broadcast routes are
    # subject to the per-chat default-responder gate.
    broadcast: bool = False


HELP_TEXT = """Available commands:

/codex <question>
/claude <question>
/both <question> — send to both agents at once
/debate <paper, claim, or question>
/import <source_chat_id> — import room memory from another group
/workspace [path|reset]
/responder [codex|claude|both|reset]
/clear [init]
@Codex /<session command>
@Claude /<session command>
/help

Ordinary group messages are forwarded in full to the active assistant session(s).
Each assistant decides whether to reply or return [NO_REPLY].
"""


def normalize_content(content: str) -> str:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", content).strip()


def route_message(
    content: str,
    respond_to_all: bool = False,
    enabled_agents: Iterable[AgentName] = ("codex", "claude"),
    bot_aliases: Iterable[str] = (),
    default_agent: AgentName | None = None,
    strict_alias: bool = False,
) -> Route:
    text = normalize_content(content)
    if not text:
        return Route("ignore")

    enabled_order = tuple(agent for agent in enabled_agents if agent in {"codex", "claude"})
    enabled = set(enabled_agents)
    lowered = text.lower()

    alias_remainder = None
    command_text = text
    command_lowered = lowered
    if default_agent in enabled:
        alias_remainder = _strip_alias(text, bot_aliases, exact=strict_alias)
        if alias_remainder is not None:
            command_text = alias_remainder
            command_lowered = command_text.lower()

    for prefix in ("/clear", "!clear", "clear"):
        remainder = _strip_prefix(command_text, command_lowered, prefix)
        if remainder is not None:
            return Route("clear", text=remainder)

    if alias_remainder is not None and default_agent in enabled and is_raw_session_command(alias_remainder):
        return Route("session_command", text=alias_remainder, agent=default_agent)

    # "! command" — raw shell passthrough: send the line (with the "!")
    # directly into the agent's input without any wrapping. The agent's own
    # shell mode will execute it.
    if alias_remainder is not None and default_agent in enabled and is_shell_command(alias_remainder):
        return Route("shell_command", text=alias_remainder, agent=default_agent)

    if command_lowered in {"/help", "!help", "help"}:
        return Route("help", text=HELP_TEXT)

    for prefix in ("/workspace", "!workspace", "/cwd", "!cwd", "workspace:", "cwd:"):
        remainder = _strip_prefix(command_text, command_lowered, prefix)
        if remainder is not None:
            return Route("workspace", text=remainder)

    for prefix in ("/responder", "!responder", "responder:"):
        remainder = _strip_prefix(command_text, command_lowered, prefix)
        if remainder is not None:
            return Route("responder", text=remainder)

    for prefix in ("/import", "!import"):
        remainder = _strip_prefix(command_text, command_lowered, prefix)
        if remainder is not None:
            return Route("import_memory", text=remainder)

    # In strict alias mode, the bot's own alias is the ONLY trigger for
    # explicit agent routing. Skip agent commands, multi-agent directives,
    # and /both — but still allow /debate, /task, and respond_to_all (each
    # process uses its own default_agent so there is no cross-instance conflict).
    if strict_alias and alias_remainder is None:
        for prefix in ("/both", "!both", "@both"):
            remainder = _strip_prefix(text, lowered, prefix)
            if remainder is not None and remainder:
                return Route(
                    "multi_agent",
                    text=remainder,
                    agent_texts={agent: remainder for agent in enabled_order},
                )
        for prefix in ("/debate", "!debate", "debate:"):
            remainder = _strip_prefix(command_text, command_lowered, prefix)
            if remainder is not None:
                if respond_to_all and default_agent in enabled:
                    return Route(
                        "agent",
                        text=_format_debate_broadcast_text(remainder),
                        agent=default_agent,
                    )
                return Route("debate", text=remainder)
        for prefix in ("/task", "!task", "/todo", "!todo", "task:", "todo:"):
            remainder = _strip_prefix(command_text, command_lowered, prefix)
            if remainder is not None:
                task = parse_task_request(remainder)
                return Route("task", text=remainder, task=task)
        if respond_to_all:
            # Don't broadcast messages explicitly addressed to a known agent
            # name — let the target bot's process handle them (which returns
            # ignore when the agent is not enabled in that process).
            explicit = _route_explicit_agent_command(text, lowered, enabled)
            if explicit is not None:
                return Route("ignore")
            if default_agent in enabled:
                return Route("agent", text=text, agent=default_agent, broadcast=True)
        return Route("ignore")

    # When strict alias mode is on AND the alias was matched, route directly to
    # the default agent without checking hardcoded @codex/@claude patterns (which
    # might steal messages addressed to another instance of the same agent type).
    if strict_alias and alias_remainder is not None and default_agent in enabled:
        if alias_remainder:
            if is_shell_command(alias_remainder):
                return Route("shell_command", text=alias_remainder, agent=default_agent)
            return Route("agent", text=alias_remainder, agent=default_agent)
        return Route("ignore")

    multi_agent_commands = split_multi_agent_directives(text)
    if multi_agent_commands and all(
        is_raw_session_command(agent_text) for agent_text in multi_agent_commands.values()
    ):
        enabled_texts = {
            agent: multi_agent_commands[agent]
            for agent in enabled_order
            if agent in multi_agent_commands
        }
        if enabled_texts:
            if len(enabled_texts) == 1:
                agent, agent_text = next(iter(enabled_texts.items()))
                return Route("session_command", text=agent_text, agent=agent)
            return Route("multi_agent", text=text, agent_texts=enabled_texts)

    agent_command_candidates = [(text, lowered)]
    acknowledged = _strip_leading_acknowledgement(text)
    if acknowledged != text:
        agent_command_candidates.append((acknowledged, acknowledged.lower()))

    for command_source, command_source_lowered in agent_command_candidates:
        explicit_route = _route_explicit_agent_command(command_source, command_source_lowered, enabled)
        if explicit_route is not None:
            return explicit_route

    for prefix in ("/both", "!both", "@both"):
        remainder = _strip_prefix(text, lowered, prefix)
        if remainder is not None and remainder:
            return Route(
                "multi_agent",
                text=remainder,
                agent_texts={agent: remainder for agent in enabled_order},
            )

    for prefix in ("/debate", "!debate", "debate:"):
        remainder = _strip_prefix(text, lowered, prefix)
        if remainder is not None:
            if respond_to_all and default_agent in enabled:
                return Route(
                    "agent",
                    text=_format_debate_broadcast_text(remainder),
                    agent=default_agent,
                )
            return Route("debate", text=remainder)

    for prefix in ("/task", "!task", "/todo", "!todo", "task:", "todo:"):
        remainder = _strip_prefix(text, lowered, prefix)
        if remainder is not None:
            task = parse_task_request(remainder)
            return Route("task", text=remainder, task=task)

    if respond_to_all:
        if default_agent in enabled:
            return Route("agent", text=text, agent=default_agent, broadcast=True)
        if enabled_order:
            return Route(
                "multi_agent",
                text=text,
                agent_texts={agent: text for agent in enabled_order},
                broadcast=True,
            )

    return Route("ignore")


def _route_explicit_agent_command(
    text: str,
    lowered: str,
    enabled: set[AgentName],
) -> Route | None:
    for prefix, agent in (
        ("@claude code", "claude"),
        ("claude code", "claude"),
        ("/codex", "codex"),
        ("!codex", "codex"),
        ("@codex", "codex"),
        ("codex:", "codex"),
        ("codex", "codex"),
        ("/claude", "claude"),
        ("!claude", "claude"),
        ("@claude", "claude"),
        ("claude:", "claude"),
        ("claude", "claude"),
    ):
        remainder = _strip_prefix(text, lowered, prefix)
        if remainder is None:
            continue
        if _is_bare_agent_prefix(prefix) and _starts_with_agent_connector(remainder):
            continue
        if agent not in enabled:
            if _is_bare_agent_prefix(prefix) and not is_raw_session_command(remainder):
                continue
            return Route("ignore")
        if is_raw_session_command(remainder):
            return Route("session_command", text=remainder, agent=agent)  # type: ignore[arg-type]
        if is_shell_command(remainder):
            return Route("shell_command", text=remainder, agent=agent)  # type: ignore[arg-type]
        if _is_bare_agent_prefix(prefix):
            continue
        return Route("agent", text=remainder, agent=agent)  # type: ignore[arg-type]
    return None


def split_multi_agent_directives(content: str) -> dict[AgentName, str] | None:
    text = _strip_leading_acknowledgement(content)
    mentions = list(_iter_agent_mentions(text))
    if len({agent for _, _, agent in mentions}) < 2:
        return None

    shared_context = _clean_agent_directive_segment(text[: mentions[0][0]])
    segments: dict[AgentName, list[str]] = {}
    for index, (start, end, agent) in enumerate(mentions):
        next_start = mentions[index + 1][0] if index + 1 < len(mentions) else len(text)
        segment = _clean_agent_directive_segment(text[end:next_start])
        if not segment or _starts_with_agent_connector(segment):
            continue
        segments.setdefault(agent, []).append(segment)
    if len(segments) < 2:
        return None

    directives: dict[AgentName, str] = {}
    for agent, agent_segments in segments.items():
        directive = "\n".join(agent_segments)
        if shared_context and not is_raw_session_command(directive):
            directive = _prepend_shared_context(shared_context, agent, directive)
        directives[agent] = directive
    if len(directives) < 2:
        return None
    return directives


def _iter_agent_mentions(text: str):
    # An explicit "@name" may carry a bot display-name suffix ("@Codex-Mac");
    # bare names keep their old, suffix-free matching to avoid false positives.
    pattern = re.compile(
        r"@(?:claude\s+code|claude|codex)(?:-[A-Za-z0-9]+)?|claude\s+code|claude|codex",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        start, end = match.span()
        if start > 0 and not _is_left_mention_boundary(text[start - 1]):
            continue
        if end < len(text) and not _is_prefix_boundary(text[end]):
            continue
        raw = match.group(0).lower()
        agent: AgentName = "codex" if "codex" in raw else "claude"
        yield start, end, agent


_ADDRESSED_AGENT_RE = re.compile(r"@\s*(claude\s+code|claude|codex)", re.IGNORECASE)


def addressed_agents(text: str) -> set[AgentName]:
    """Agents explicitly @-addressed in a reply.

    Used to detect when one assistant hands the thread to the other by writing
    e.g. "@Codex" / "@Claude" in its reply. The leading "@" is required so that a
    passing mention of the name (e.g. "like Codex said") does not trigger a
    handoff. Case-insensitive, so "@codex" and "@Codex" both count.
    """
    found: set[AgentName] = set()
    for match in _ADDRESSED_AGENT_RE.finditer(text or ""):
        raw = match.group(1).lower()
        found.add("codex" if "codex" in raw else "claude")
    return found


def _clean_agent_directive_segment(text: str) -> str:
    text = re.sub(r"^[\s,，、:：;；]+", "", text)
    text = re.sub(r"[\s,，、:：;；]+$", "", text)
    return text.strip()


def _format_debate_broadcast_text(text: str) -> str:
    return (
        "Feishu command: /debate\n"
        "Please participate in a focused two-assistant discussion. Be concrete, "
        "skeptical, and useful; include evidence, risks, and a next step.\n\n"
        f"Prompt:\n{text}"
    )


def _prepend_shared_context(
    shared_context: str,
    agent: AgentName,
    directive: str,
) -> str:
    return f"{shared_context}\n\n给 {_agent_display_name(agent)} 的任务：{directive}"


def _agent_display_name(agent: AgentName) -> str:
    return "Codex" if agent == "codex" else "Claude"


def _is_bare_agent_prefix(prefix: str) -> bool:
    return prefix in {"codex", "claude", "claude code"}


def _starts_with_agent_connector(text: str) -> bool:
    return bool(re.match(r"^(?:和|跟|与|and\b)", text.strip(), flags=re.IGNORECASE))


def _is_left_mention_boundary(char: str) -> bool:
    return not (char.isascii() and (char.isalnum() or char in {"_", "-"}))


def is_raw_session_command(text: str) -> bool:
    return text.lstrip().startswith("/")


def is_shell_command(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("!") and len(stripped) > 1 and not stripped[1:].lstrip().startswith("!")


def _strip_prefix(
    text: str, lowered: str, prefix: str, allow_suffix: bool = True,
) -> str | None:
    if not lowered.startswith(prefix):
        return None
    consumed = len(prefix)
    if not prefix.endswith(":") and len(lowered) > consumed:
        next_char = lowered[consumed]
        # Allow a bot display-name suffix on an explicitly-marked mention: Feishu
        # renders an @ of the "Codex-Mac" bot as "@Codex-Mac", so accept
        # "@codex-mac" / "/claude-mac" as addressing codex/claude. Bare names
        # ("codex-cli is great") are left alone to avoid false positives.
        if next_char == "-" and allow_suffix and prefix[:1] in {"@", "/", "!"}:
            suffix = re.match(r"-[A-Za-z0-9]+", lowered[consumed:])
            after = consumed + suffix.end() if suffix else consumed
            if suffix and (after >= len(lowered) or _is_prefix_boundary(lowered[after])):
                consumed = after
            else:
                return None
        elif not _is_prefix_boundary(next_char):
            return None
    remainder = text[consumed:].lstrip()
    if remainder[:1] in {":", "：", ",", "，", "、", ";", "；"}:
        remainder = remainder[1:].strip()
    return remainder


def _is_prefix_boundary(char: str) -> bool:
    if char in {" ", "\n", "\t", ":", "：", ",", "，", "、", ";", "；", "/", "?"}:
        return True
    return not (char.isascii() and (char.isalnum() or char in {"_", "-"}))


def _strip_leading_acknowledgement(text: str) -> str:
    return re.sub(
        r"^(?:嗯|好|好的|行|可以|对|是的|ok|okay)[\s,，、:：;；]+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _strip_alias(
    text: str, aliases: Iterable[str], exact: bool = False,
) -> str | None:
    lowered = text.lower()
    for raw_alias in aliases:
        alias = raw_alias.strip()
        if not alias:
            continue
        forms = {alias, f"@{alias}"}
        if alias.startswith("@"):
            forms.add(alias[1:])
        for form in sorted(forms, key=len, reverse=True):
            remainder = _strip_prefix(
                text, lowered, form.lower(), allow_suffix=not exact,
            )
            if remainder is not None:
                return remainder
    return None


def parse_task_request(text: str) -> TaskRequest:
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if not parts:
        raise ValueError("Task summary is required.")

    summary = parts[0]
    data: dict[str, str] = {}
    for part in parts[1:]:
        if ":" not in part:
            data.setdefault("description", "")
            data["description"] = (data["description"] + "\n" + part).strip()
            continue
        key, value = part.split(":", 1)
        data[key.strip().lower()] = value.strip()

    description = data.get("desc") or data.get("description") or ""
    return TaskRequest(
        summary=summary,
        description=description,
        assignee=data.get("assignee"),
        due=data.get("due"),
        tasklist_id=data.get("tasklist"),
    )
