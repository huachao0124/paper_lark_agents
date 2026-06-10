"""Detect and forward interactive prompts from agent tmux sessions."""
from __future__ import annotations

import re
import logging

LOGGER = logging.getLogger(__name__)

# Patterns that indicate an interactive prompt waiting for user input.
_INTERACTIVE_MARKERS = [
    "Enter to select",
    "Press enter to confirm or esc to cancel",
    "Press enter to confirm or esc to go back",
    "1. Yes, proceed (y)",
    "1. Yes, switch to",
    "No, go back",
    "› 1.",
    "❯ 1.",
]


def detect_interactive_prompt(screen: str) -> dict | None:
    """Parse an interactive prompt from a tmux screen capture.

    Returns a dict with 'title', 'options' (list of str), and 'raw' (the
    matched screen section), or None if no prompt is detected.
    """
    lines = [l for l in screen.splitlines() if l.strip()]
    if not lines:
        return None
    tail = "\n".join(lines[-20:])
    if not any(marker in tail for marker in _INTERACTIVE_MARKERS):
        return None
    # Parse options from the tail.
    options: list[str] = []
    title = ""
    for line in lines[-20:]:
        stripped = line.strip()
        # Title line: bold text or a question ending with ?
        if stripped.endswith("?") and not title:
            title = stripped
        # Option lines: "› 1. ...", "  2. ...", "1. Yes, proceed (y)", etc.
        m = re.match(r"^[›❯>\s]*(\d+)\.\s+(.+)", stripped)
        if m:
            idx = m.group(1)
            text = m.group(2).strip()
            options.append(f"{idx}. {text}")
    # A real selection menu always has at least two options; a single match
    # is usually an ordinary numbered list in agent output.
    if len(options) < 2:
        return None
    return {
        "title": title or "请选择：",
        "options": options,
        "raw": tail,
    }


def format_prompt_message(prompt: dict) -> str:
    """Format an interactive prompt as a Feishu message."""
    lines = [f"**{prompt['title']}**", ""]
    for opt in prompt["options"]:
        lines.append(f"- {opt}")
    lines.append("")
    lines.append("_回复数字选择，或回复 esc 取消_")
    return "\n".join(lines)
