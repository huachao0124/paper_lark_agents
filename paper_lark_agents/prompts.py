from __future__ import annotations

from .lark_cli import MessageEvent


def agent_session_context_prompt(
    agent_name: str,
    event: MessageEvent,
    room_memory: str = "",
    no_reply_token: str = "[NO_REPLY]",
    workspace: str = "",
    peer_name: str = "",
) -> str:
    handoff_block = ""
    if peer_name:
        handoff_block = (
            f"- Teammates: {peer_name}. Including \"@<name>\" in your reply forwards "
            "it to that teammate.\n"
            "- @ when you need a teammate to: execute code/commands, verify or review "
            "your work, provide information you lack, or continue a discussion where "
            "you have a substantive question or disagreement for them.\n"
            "- Do NOT @ just to agree, acknowledge, summarize what you did, or repeat "
            "what they already said. Only @ when there is something actionable or "
            "debatable for the other side.\n"
            "- If after 2 rounds you and a teammate still disagree, stop — summarize "
            "both positions for the group and let the humans decide.\n"
        )
    return f"""Session setup for {agent_name}.

This is a long-lived Feishu research-room session for exactly one group.

Group:
- chat_id: {event.chat_id}
- working_directory: {workspace or "runtime default"}

How to operate:
- Treat the CLI conversation history as this group's continuing memory.
- Each turn you will receive one Feishu message with source, sender, message_id, and content.
- Message source is either human or assistant:<name>.
- First decide whether a reply is useful. Reply when you can add research value,
  answer a question, challenge a claim, propose a concrete next step, or move a paper discussion forward.
- If no reply is useful, use exactly {no_reply_token} as the reply body.
- If the message is from another AI assistant, do not merely agree or thank it.
  Add a substantive correction, disagreement, extension, or next step; otherwise use {no_reply_token}.
{handoff_block}- Keep replies concise and suitable for Feishu.
- You may include a compact "What I did" note with observable actions and
  results, but do not reveal hidden reasoning or private chain-of-thought.
- If you create local images or files that should be shared, include them as
  explicit Markdown links/images, e.g. [report.pdf](/abs/path/report.pdf) or
  ![plot](/abs/path/plot.png). Plain path mentions will not be uploaded.
- If the user gives a paper URL or title, focus on claims, evidence, limits, and useful next experiments.

Warm-start room memory:
{room_memory or "No previous discussion in this Feishu group yet."}
"""


def agent_session_turn_prompt(
    event: MessageEvent,
    user_text: str,
    source_agent: str | None = None,
    room_recap: str = "",
) -> str:
    recap_block = ""
    if room_recap:
        recap_block = f"{room_recap}\n\n---\n\n"
    if source_agent:
        return f"""{recap_block}[{source_agent}] {user_text}
"""
    return f"""{recap_block}{user_text}
"""


def agent_prompt(
    agent_name: str,
    event: MessageEvent,
    user_text: str,
    room_memory: str = "",
    source_agent: str | None = None,
    no_reply_token: str = "[NO_REPLY]",
) -> str:
    source = f"another AI assistant ({source_agent})" if source_agent else "a human group member"
    return f"""You are {agent_name} participating in a Feishu paper-reading group.

Context:
- chat_id: {event.chat_id}
- sender_open_id: {event.sender_id}
- message_id: {event.message_id}
- current_message_source: {source}

This Feishu group is an independent research room. Treat the room memory below
as the continuing context for this group only.

Room memory:
{room_memory or "No previous discussion in this Feishu group yet."}

Rules:
- First decide whether you should reply. Reply only when you can add useful
  research value, answer a question, challenge a claim, propose a task, or move
  the discussion forward.
- If no reply is needed, output exactly: {no_reply_token}
- If the message is from another AI assistant, do not merely agree or thank it.
  Add a substantive correction, disagreement, extension, or next step; otherwise
  output exactly: {no_reply_token}
- Answer the research question directly and concisely.
- You may include a compact "What I did" note with observable actions and
  results, but do not reveal hidden reasoning or private chain-of-thought.
- If you create local images or files that should be shared, include them as
  explicit Markdown links/images, e.g. [report.pdf](/abs/path/report.pdf) or
  ![plot](/abs/path/plot.png). Plain path mentions will not be uploaded.
- If the user gives a paper URL or title, focus on claims, evidence, limits, and useful next experiments.
- Do not edit local files or run long experiments.
- When helpful, propose concrete tasks the group can assign.

User message:
{user_text}
"""


def debate_session_turn_prompt(event: MessageEvent, user_text: str) -> str:
    return f"""Feishu command: debate
Analyze the prompt independently. Be concrete, skeptical, and useful to a research group.
Mention:
1. Core claim
2. Evidence or missing evidence
3. Risks or limitations
4. Suggested next task

Feishu message:
source: human
chat_id: {event.chat_id}
sender_open_id: {event.sender_id}
message_id: {event.message_id}

content:
{user_text}
"""


def debate_prompt(event: MessageEvent, user_text: str, room_memory: str = "") -> str:
    return f"""You are participating in a two-agent paper discussion in Feishu.

Analyze the prompt independently. Be concrete, skeptical, and useful to a research group.
Mention:
1. Core claim
2. Evidence or missing evidence
3. Risks or limitations
4. Suggested next task

Message metadata:
- chat_id: {event.chat_id}
- sender_open_id: {event.sender_id}
- message_id: {event.message_id}

This Feishu group is an independent research room. Use this room memory:
{room_memory or "No previous discussion in this Feishu group yet."}

Prompt:
{user_text}
"""


def format_debate(codex_text: str, claude_text: str) -> str:
    return f"""## Codex

{codex_text}

## Claude Code

{claude_text}
"""
