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
    # Keep total input under ~272K tokens (~200K Chinese chars) to avoid the
    # 2x input / 1.5x output surcharge gpt-5.5-pro applies above that threshold.
    max_context_chars: int = 200000
    # When the folded background exceeds this, summarize it instead of sending
    # raw — keeps cost down on long-running groups.
    summarize_threshold_chars: int = 40000
    instructions: str = (
        "你是 GPT-Pro，一个加入飞书研究群的助手，通过内部 API 平台运行 gpt-5.5-pro。"
        "群里还有 Codex 和 Claude 两个助手以及用户。只有用户 @ 你时你才会被调用。"
        "请用中文简洁、专业地回答，给出有依据的判断而不是泛泛而谈。"
    )


def _make_client(config: GptProConfig):
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
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(connect=60.0, read=config.timeout, write=60.0, pool=10.0),
    )


def summarize_background(config: GptProConfig, background: str) -> str:
    """Compress a long conversation background into a concise summary using a
    cheaper low-effort call. Used so long-running groups don't re-send the full
    raw history on every GPT-Pro turn."""
    client = _make_client(config)
    LOGGER.info("summarizing GPT-Pro background (%d chars)", len(background))
    prompt = (
        "把下面这个飞书群的历史对话压缩成一段简洁的背景摘要，保留关键结论、决定、"
        "待办和重要事实，省略寒暄和重复。用中文，控制在 800 字以内：\n\n" + background
    )
    response = client.responses.create(
        model=config.model,
        input=[{"role": "user", "content": prompt}],
        reasoning={"effort": "low"},
    )
    err = getattr(response, "error", None)
    if err:
        raise RuntimeError(f"GPT-Pro summarize error: {getattr(err, 'message', str(err))}")
    summary = getattr(response, "output_text", None) or ""
    LOGGER.info("background summarized: %d -> %d chars", len(background), len(summary))
    return summary


def call_gpt_pro(config: GptProConfig, messages: list[dict[str, str]]) -> tuple[str, str]:
    """Non-streaming call. Returns (reply_text, model_used)."""
    client = _make_client(config)
    LOGGER.info("calling GPT-Pro (%s, effort=%s) with %d messages", config.model, config.effort, len(messages))
    response = client.responses.create(
        model=config.model,
        input=messages,
        instructions=config.instructions,
        reasoning={"effort": config.effort},
    )
    err = getattr(response, "error", None)
    if err:
        raise RuntimeError(f"GPT-Pro upstream error: {getattr(err, 'message', str(err))}")
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


def stream_gpt_pro(config: GptProConfig, messages: list[dict[str, str]], on_update):
    """Streaming call via the Responses API. Calls on_update(kind, accumulated)
    as tokens arrive — kind is "reasoning" (thinking summary) or "answer"
    (final text). Returns (final_reply, model_used).

    on_update is throttled by the caller; here we call it on every meaningful
    delta. The caller decides how often to push to Feishu.
    """
    client = _make_client(config)
    LOGGER.info("streaming GPT-Pro (%s, effort=%s) with %d messages", config.model, config.effort, len(messages))

    reasoning_buf: list[str] = []
    answer_buf: list[str] = []
    model_used = config.model

    with client.responses.stream(
        model=config.model,
        input=messages,
        instructions=config.instructions,
        reasoning={"effort": config.effort, "summary": "auto"},
    ) as stream:
        for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.reasoning_summary_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    reasoning_buf.append(delta)
                    on_update("reasoning", "".join(reasoning_buf))
            elif etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    answer_buf.append(delta)
                    on_update("answer", "".join(answer_buf))
            elif etype == "response.completed":
                resp = getattr(event, "response", None)
                if resp is not None:
                    model_used = getattr(resp, "model", None) or model_used
            elif etype == "error":
                msg = getattr(event, "message", "") or "stream error"
                raise RuntimeError(f"GPT-Pro stream error: {msg}")

        final = stream.get_final_response()
        model_used = getattr(final, "model", None) or model_used
        reply = getattr(final, "output_text", None) or "".join(answer_buf)

    LOGGER.info("GPT-Pro stream done: %d chars, model=%s", len(reply), model_used)
    return reply, model_used
