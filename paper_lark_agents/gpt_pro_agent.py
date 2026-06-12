"""GPT-Pro agent: calls internal API platform (OpenAI Responses API) directly.

No tmux session — just an API call with chat history as context.
gpt-5.5-pro is a reasoning model served via the Responses API (/v1/responses),
NOT chat completions.
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
    effort: str = "xhigh"
    task_creator: str = "arimazhu"
    task_name: str = "debug"
    timeout: int = 600
    max_context_chars: int = 80000


def call_gpt_pro(config: GptProConfig, messages: list[dict[str, str]]) -> tuple[str, str]:
    """Call the internal API platform via the OpenAI Responses API.

    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns (reply_text, model_used).
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
    # The API key encodes provider + model (real upstream name, NOT the
    # -passthrough alias) + usage flags.
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

    LOGGER.info("calling GPT-Pro (%s, effort=%s) with %d messages", config.model, config.effort, len(messages))
    response = client.responses.create(
        model=config.model,
        input=messages,
        reasoning={"effort": config.effort},
    )

    # Check for upstream error embedded in the response body.
    err = getattr(response, "error", None)
    if err:
        msg = getattr(err, "message", str(err))
        raise RuntimeError(f"GPT-Pro upstream error: {msg}")

    # Responses API: prefer output_text convenience field, fall back to walking output.
    reply = getattr(response, "output_text", None) or ""
    if not reply and getattr(response, "output", None):
        parts = []
        for item in response.output:
            for content in getattr(item, "content", None) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
        reply = "\n".join(parts)

    model_used = getattr(response, "model", None) or config.model
    LOGGER.info("GPT-Pro replied: %d chars, model=%s", len(reply), model_used)
    return reply, model_used
