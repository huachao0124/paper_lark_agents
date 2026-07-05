from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time


@dataclass(frozen=True)
class StatusCard:
    agent_name: str
    state: str
    detail: str
    workspace: Path
    model: str
    effort: str
    started_at: float

    def to_card(self) -> dict[str, object]:
        elapsed = max(0, int(time.time() - self.started_at))
        return {
            "config": {
                "wide_screen_mode": True,
                "update_multi": True,
            },
            "header": {
                "template": template_for_state(self.state),
                "title": {
                    "tag": "plain_text",
                    "content": f"{self.agent_name} status",
                },
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": "\n".join(
                            [
                                f"**State:** {label_for_state(self.state)}",
                                f"**Model:** {self.model}",
                                f"**Effort:** {self.effort}",
                                f"**Workspace:** {self.workspace}",
                                f"**Elapsed:** {elapsed}s",
                                f"**Detail:** {self.detail}",
                            ]
                        ),
                    },
                }
            ],
        }


def turn_reply_card(
    agent_name: str,
    state: str,
    body: str,
    *,
    model: str = "",
    effort: str = "",
    started_at: float = 0.0,
) -> dict[str, object]:
    """A per-turn card that shows live activity while running and becomes the
    final reply when done. Header carries the agent + state; the body is the
    activity line (running) or the answer (done)."""
    elements: list[dict[str, object]] = [
        {"tag": "markdown", "content": body or "…"}
    ]
    footer_bits = []
    if model:
        footer_bits.append(model)
    if effort:
        footer_bits.append(effort)
    if started_at and state == "running":
        footer_bits.append(f"{max(0, int(time.time() - started_at))}s")
    if footer_bits:
        elements.append(
            {"tag": "markdown", "content": "_" + " · ".join(footer_bits) + "_"}
        )
    return {
        "schema": "2.0",
        "header": {
            "template": template_for_state(state),
            "title": {"tag": "plain_text", "content": f"{agent_name} · {label_for_state(state)}"},
        },
        "body": {"elements": elements},
    }


def template_for_state(state: str) -> str:
    if state == "done":
        return "green"
    if state == "pending":
        return "yellow"
    if state == "failed":
        return "red"
    if state == "skipped":
        return "grey"
    return "blue"


def label_for_state(state: str) -> str:
    labels = {
        "running": "Running",
        "done": "Done",
        "pending": "Pending",
        "failed": "Failed",
        "skipped": "Skipped",
    }
    return labels.get(state, state.title())
