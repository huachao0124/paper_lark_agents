"""LarkSDK — drop-in replacement for LarkCLI using lark-oapi Python SDK.

Same public interface as LarkCLI so app.py needs only a factory swap.
Falls back to LarkCLI when PLA_LARK_APP_ID is not set.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Callable

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateFileRequest,
    CreateFileRequestBody,
    CreateImageRequest,
    CreateImageRequestBody,
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetMessageResourceRequest,
    ListChatRequest,
    PatchMessageRequest,
    PatchMessageRequestBody,
    CreateChatRequest,
    CreateChatRequestBody,
    CreatePinRequest,
    CreatePinRequestBody,
    CreateChatTabRequest,
    CreateChatTabRequestBody,
    ChatTab,
    ChatTabContent,
    UpdateTabsChatTabRequest,
    UpdateTabsChatTabRequestBody,
    ListTabsChatTabRequest,
)

from .config import Settings
from .lark_cli import (
    LarkCLIError,
    LarkEvent,
    MessageEvent,
    _chunks,
    _idempotency_key,
)
from .router import TaskRequest


LOGGER = logging.getLogger(__name__)

_FILE_TYPE_MAP = {
    ".opus": "opus", ".mp3": "mp3", ".mp4": "mp4",
    ".pdf": "pdf", ".doc": "doc", ".docx": "doc",
    ".xls": "xls", ".xlsx": "xls",
    ".ppt": "ppt", ".pptx": "ppt",
}


class LarkSDK:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.proxy_url:
            os.environ.setdefault("HTTP_PROXY", settings.proxy_url)
            os.environ.setdefault("HTTPS_PROXY", settings.proxy_url)
        if settings.no_proxy:
            os.environ.setdefault("NO_PROXY", settings.no_proxy)
        self._client = (
            lark.Client.builder()
            .app_id(settings.lark_app_id)
            .app_secret(settings.lark_app_secret)
            .domain(lark.FEISHU_DOMAIN)
            .build()
        )

    # ---- messaging ----

    def send_markdown(self, chat_id: str, markdown: str) -> list[dict[str, Any]]:
        results = []
        for index, chunk in enumerate(_chunks(markdown, self.settings.max_message_chars), start=1):
            key = _idempotency_key(chat_id, chunk, index)
            content = json.dumps({"text": chunk}, ensure_ascii=False)
            body = (
                CreateMessageRequestBody.builder()
                .receive_id(chat_id)
                .msg_type("text")
                .content(content)
                .uuid(key)
                .build()
            )
            request = (
                CreateMessageRequest.builder()
                .receive_id_type("chat_id")
                .request_body(body)
                .build()
            )
            response = self._request_with_retry(
                lambda r=request: self._client.im.v1.message.create(r)
            )
            results.append(self._msg_response_dict(response))
        return results

    def send_image(self, chat_id: str, image_path: str) -> dict[str, Any]:
        with open(image_path, "rb") as f:
            upload_body = (
                CreateImageRequestBody.builder()
                .image_type("message")
                .image(f)
                .build()
            )
            upload_req = CreateImageRequest.builder().request_body(upload_body).build()
            upload_resp = self._request_with_retry(
                lambda r=upload_req: self._client.im.v1.image.create(r)
            )
        image_key = upload_resp.data.image_key
        content = json.dumps({"image_key": image_key})
        body = (
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("image")
            .content(content)
            .uuid(_idempotency_key(chat_id, "image", image_path))
            .build()
        )
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(body)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.message.create(r)
        )
        return self._msg_response_dict(response)

    def send_file(self, chat_id: str, file_path: str) -> dict[str, Any]:
        suffix = Path(file_path).suffix.lower()
        file_type = _FILE_TYPE_MAP.get(suffix, "stream")
        with open(file_path, "rb") as f:
            upload_body = (
                CreateFileRequestBody.builder()
                .file_type(file_type)
                .file_name(Path(file_path).name)
                .file(f)
                .build()
            )
            upload_req = CreateFileRequest.builder().request_body(upload_body).build()
            upload_resp = self._request_with_retry(
                lambda r=upload_req: self._client.im.v1.file.create(r)
            )
        file_key = upload_resp.data.file_key
        content = json.dumps({"file_key": file_key, "file_name": Path(file_path).name})
        body = (
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("file")
            .content(content)
            .uuid(_idempotency_key(chat_id, "file", file_path))
            .build()
        )
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(body)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.message.create(r)
        )
        return self._msg_response_dict(response)

    def send_card(self, chat_id: str, card: dict[str, Any]) -> dict[str, Any]:
        content = json.dumps(card, ensure_ascii=False, separators=(",", ":"))
        body = (
            CreateMessageRequestBody.builder()
            .receive_id(chat_id)
            .msg_type("interactive")
            .content(content)
            .uuid(_idempotency_key(chat_id, "card", content))
            .build()
        )
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(body)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.message.create(r)
        )
        return self._msg_response_dict(response)

    def update_card(self, message_id: str, card: dict[str, Any]) -> dict[str, Any]:
        content = json.dumps(card, ensure_ascii=False, separators=(",", ":"))
        body = PatchMessageRequestBody.builder().content(content).build()
        request = (
            PatchMessageRequest.builder()
            .message_id(message_id)
            .request_body(body)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.message.patch(r)
        )
        return self._normalize(response)

    def pin_message(self, message_id: str) -> dict[str, Any]:
        body = CreatePinRequestBody.builder().message_id(message_id).build()
        request = CreatePinRequest.builder().request_body(body).build()
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.message_pin.create(r)
        )
        return self._normalize(response)

    # ---- resources ----

    def download_message_resource(
        self,
        message_id: str,
        file_key: str,
        resource_type: str,
        output_relative_path: str,
    ) -> Path:
        request = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(file_key)
            .type(resource_type)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.message_resource.get(r)
        )
        output_path = self.settings.workspace / output_relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(response, "file") and response.file:
            output_path.write_bytes(response.file.read())
        elif hasattr(response, "raw") and hasattr(response.raw, "content"):
            output_path.write_bytes(response.raw.content)
        return output_path.resolve()

    # ---- chats ----

    def list_chats(self) -> list[dict[str, Any]]:
        all_chats: list[dict[str, Any]] = []
        page_token = ""
        while True:
            builder = ListChatRequest.builder().page_size(100)
            if page_token:
                builder = builder.page_token(page_token)
            request = builder.build()
            response = self._request_with_retry(
                lambda r=request: self._client.im.v1.chat.list(r)
            )
            for item in response.data.items or []:
                all_chats.append({
                    "chat_id": getattr(item, "chat_id", ""),
                    "name": getattr(item, "name", ""),
                })
            if not response.data.has_more:
                break
            page_token = response.data.page_token or ""
        return all_chats

    def create_chat(
        self,
        name: str,
        users: str | None = None,
        bots: str | None = None,
        description: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        body_builder = (
            CreateChatRequestBody.builder()
            .name(name)
            .chat_mode("group")
            .chat_type("private")
        )
        if users:
            body_builder = body_builder.user_id_list(
                [u.strip() for u in users.split(",") if u.strip()]
            )
        if bots:
            body_builder = body_builder.bot_id_list(
                [b.strip() for b in bots.split(",") if b.strip()]
            )
        if description:
            body_builder = body_builder.description(description)
        body = body_builder.build()
        request = (
            CreateChatRequest.builder()
            .set_bot_manager(True)
            .request_body(body)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.chat.create(r)
        )
        return self._normalize(response)

    # ---- chat tabs ----

    def list_chat_tabs(self, chat_id: str) -> dict[str, Any]:
        request = ListTabsChatTabRequest.builder().chat_id(chat_id).build()
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.chat_tab.list_tabs(r)
        )
        return self._normalize(response)

    def create_chat_tab(self, chat_id: str, tab_name: str, doc_url: str) -> dict[str, Any]:
        tab = (
            ChatTab.builder()
            .tab_name(tab_name)
            .tab_type("doc")
            .tab_content(ChatTabContent.builder().doc(doc_url).build())
            .build()
        )
        body = CreateChatTabRequestBody.builder().chat_tabs([tab]).build()
        request = (
            CreateChatTabRequest.builder()
            .chat_id(chat_id)
            .request_body(body)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.chat_tab.create(r)
        )
        return self._normalize(response)

    def update_chat_tab(self, chat_id: str, tab_id: str, tab_name: str, doc_url: str) -> dict[str, Any]:
        tab = (
            ChatTab.builder()
            .tab_id(tab_id)
            .tab_name(tab_name)
            .tab_type("doc")
            .tab_content(ChatTabContent.builder().doc(doc_url).build())
            .build()
        )
        body = UpdateTabsChatTabRequestBody.builder().chat_tabs([tab]).build()
        request = (
            UpdateTabsChatTabRequest.builder()
            .chat_id(chat_id)
            .request_body(body)
            .build()
        )
        response = self._request_with_retry(
            lambda r=request: self._client.im.v1.chat_tab.update_tabs(r)
        )
        return self._normalize(response)

    # ---- docs (raw API — legacy v2 XML format) ----

    def create_doc(self, content: str) -> dict[str, Any]:
        request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.POST)
            .uri("/open-apis/doc/v2/create")
            .token_types({lark.AccessTokenType.TENANT})
            .body({"content": content, "folder_token": ""})
            .build()
        )
        response = self._client.request(request)
        return self._base_response_dict(response)

    def update_doc(self, doc: str, content: str) -> dict[str, Any]:
        request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.POST)
            .uri(f"/open-apis/doc/v2/{doc}/content")
            .token_types({lark.AccessTokenType.TENANT})
            .body({"content": content})
            .build()
        )
        response = self._client.request(request)
        return self._base_response_dict(response)

    # ---- tasks ----

    def create_task(self, task: TaskRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"summary": task.summary}
        if task.description:
            body["description"] = task.description
        if task.due:
            body["due"] = {"timestamp": task.due}
        request = (
            lark.BaseRequest.builder()
            .http_method(lark.HttpMethod.POST)
            .uri("/open-apis/task/v2/tasks")
            .token_types({lark.AccessTokenType.TENANT})
            .body(body)
            .build()
        )
        response = self._client.request(request)
        return self._base_response_dict(response)

    # ---- events (WebSocket) ----

    def consume_events(self, on_event: Callable[[LarkEvent], None]) -> None:
        from lark_oapi import ws as lark_ws

        handler = (
            lark.EventDispatcherHandler.builder(
                self.settings.lark_encrypt_key or "", ""
            )
            .register_p2_im_message_receive_v1(
                lambda data: self._dispatch_im_event(data, "im.message.receive_v1", on_event)
            )
            .register_p2_im_chat_member_bot_added_v1(
                lambda data: self._dispatch_lifecycle_event(data, "im.chat.member.bot.added_v1", on_event)
            )
            .register_p2_im_chat_member_bot_deleted_v1(
                lambda data: self._dispatch_lifecycle_event(data, "im.chat.member.bot.deleted_v1", on_event)
            )
            .register_p2_im_chat_disbanded_v1(
                lambda data: self._dispatch_lifecycle_event(data, "im.chat.disbanded_v1", on_event)
            )
            .build()
        )

        ws_client = lark_ws.Client(
            self.settings.lark_app_id,
            self.settings.lark_app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )

        LOGGER.info("starting WebSocket event consumer (SDK mode)")
        ws_client.start()

    def _dispatch_im_event(self, data: Any, event_type: str, on_event: Callable[[LarkEvent], None]) -> None:
        raw = self._p2_message_to_raw(data, event_type)
        on_event(LarkEvent.from_json(raw))

    def _dispatch_lifecycle_event(self, data: Any, event_type: str, on_event: Callable[[LarkEvent], None]) -> None:
        chat_id = ""
        if hasattr(data, "event") and data.event:
            chat_id = getattr(data.event, "chat_id", "") or ""
        raw: dict[str, Any] = {
            "header": {
                "event_type": event_type,
                "event_id": getattr(data.header, "event_id", "") if data.header else "",
            },
            "event": {"chat_id": chat_id},
        }
        on_event(LarkEvent.from_json(raw))

    def _p2_message_to_raw(self, data: Any, event_type: str) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": "2.0"}
        if data.header:
            result["header"] = {
                "event_id": data.header.event_id or "",
                "event_type": event_type,
            }
        event_dict: dict[str, Any] = {}
        if hasattr(data, "event") and data.event:
            ev = data.event
            if hasattr(ev, "message") and ev.message:
                msg = ev.message
                content = msg.content or ""
                if msg.message_type == "text":
                    try:
                        content = json.loads(content).get("text", content)
                    except (json.JSONDecodeError, AttributeError):
                        pass
                mentions = []
                for m in msg.mentions or []:
                    mentions.append({"key": m.key or "", "name": m.name or ""})
                event_dict["message"] = {
                    "message_id": msg.message_id or "",
                    "chat_id": msg.chat_id or "",
                    "chat_type": msg.chat_type or "",
                    "content": content,
                    "message_type": msg.message_type or "",
                    "create_time": str(msg.create_time or ""),
                    "mentions": mentions,
                }
            if hasattr(ev, "sender") and ev.sender:
                sender = ev.sender
                open_id = ""
                if hasattr(sender, "sender_id") and sender.sender_id:
                    open_id = getattr(sender.sender_id, "open_id", "") or ""
                event_dict["sender"] = {"sender_id": {"open_id": open_id}}
        result["event"] = event_dict
        return result

    # ---- internals ----

    def _request_with_retry(self, call: Callable, retries: int = 3) -> Any:
        import time as _time
        last_err = ""
        for attempt in range(retries):
            response = call()
            if response.success():
                return response
            code = response.code or 0
            msg = response.msg or ""
            last_err = f"API error {code}: {msg}"
            if attempt < retries - 1 and _retryable_sdk_error(code, msg):
                _time.sleep(1 + attempt)
                continue
            raise LarkCLIError(last_err)
        raise LarkCLIError(last_err or "lark-oapi request failed")

    def _normalize(self, response: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if hasattr(response, "data") and response.data:
            data = response.data
            result["data"] = _obj_to_dict(data)
        return result

    def _msg_response_dict(self, response: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if hasattr(response, "data") and response.data:
            msg = response.data
            result["data"] = {
                "message_id": getattr(msg, "message_id", ""),
            }
            result["message_id"] = getattr(msg, "message_id", "")
        return result

    def _base_response_dict(self, response: Any) -> dict[str, Any]:
        if isinstance(response, lark.BaseResponse):
            if response.code != 0:
                raise LarkCLIError(f"API error {response.code}: {response.msg}")
            raw = response.raw
            if hasattr(raw, "json"):
                try:
                    return raw.json()
                except Exception:
                    pass
            if hasattr(raw, "content"):
                try:
                    return json.loads(raw.content)
                except Exception:
                    pass
        return {}


def _obj_to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_obj_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _obj_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {
            k: _obj_to_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith("_") and v is not None
        }
    return str(obj)


def _retryable_sdk_error(code: int, msg: str) -> bool:
    if code in {230049, 99991400, 99991401}:
        return True
    lowered = msg.lower()
    return any(
        token in lowered
        for token in ("timeout", "temporary", "network", "connection")
    )
