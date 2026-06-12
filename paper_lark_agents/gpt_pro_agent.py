"""GPT-Pro agent: calls internal API platform (OpenAI-compatible) directly.

No tmux session — just an API call with chat history as context.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GptProConfig:
    host: str
    user: str
    token: str
    model: str = "gpt-5.5-pro"
    task_creator: str = "arimazhu"
    task_name: str = "debug"
    timeout: int = 600
    max_context_chars: int = 80000


def call_gpt_pro(config: GptProConfig, messages: list[dict[str, str]]) -> str:
    """Call the internal API platform with OpenAI Chat Completions format.

    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns the assistant reply text.
    """
    from openai import OpenAI
    import httpx

    extension = {
        "task_creator": config.task_creator,
        "task_id": "",
        "task_name": config.task_name,
        "task_source": "9",
        "caller_token": "",
    }
    extra_encoded = urllib.parse.quote(json.dumps(extension, ensure_ascii=False))
    api_key = (
        f"{config.user}:{config.token}"
        f"?provider=openai&timeout={config.timeout}"
        f"&model={config.model}&usage=1"
        f"&extra={extra_encoded}"
    )
    base_url = f"http://{config.host}/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(connect=60.0, read=config.timeout, write=60.0, pool=10.0),
    )

    LOGGER.info("calling GPT-Pro (%s) with %d messages", config.model, len(messages))
    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
    )
    reply = response.choices[0].message.content or ""
    model_used = response.model or config.model
    LOGGER.info("GPT-Pro replied: %d chars, model=%s", len(reply), model_used)
    return reply, model_used
