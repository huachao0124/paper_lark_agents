from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


RESOURCE_MESSAGE_TYPES = {"image", "file", "audio", "video", "post"}


@dataclass(frozen=True)
class InboundResource:
    key: str
    resource_type: str
    name: str
    message_type: str


def extract_inbound_resources(message_type: str, content: str) -> list[InboundResource]:
    message_type = message_type.strip().lower()
    if message_type not in RESOURCE_MESSAGE_TYPES:
        return []
    payload = parse_content(content)
    resources = dedupe_resources(extract_payload_resources(message_type, payload))
    if message_type == "image":
        return resources or [
            InboundResource(key, "image", key, message_type)
            for key in all_regex(content, r"img_(?!key\b)[A-Za-z0-9_=-]+")
        ]

    if message_type in {"file", "audio", "video"}:
        if resources:
            return resources
        return [
            InboundResource(key, "file", name_for_key(content, key) or key, message_type)
            for key in all_regex(content, r"file_(?!key\b)[A-Za-z0-9_=-]+")
        ]

    if message_type == "post":
        resources.extend(
            InboundResource(key, "image", key, message_type)
            for key in all_regex(content, r"img_(?!key\b)[A-Za-z0-9_=-]+")
        )
        resources.extend(
            InboundResource(key, "file", name_for_key(content, key) or key, message_type)
            for key in all_regex(content, r"file_(?!key\b)[A-Za-z0-9_=-]+")
        )
        return dedupe_resources(resources)

    return []


def post_text_without_resources(content: str) -> str:
    payload = parse_content(content)
    if payload:
        text = "\n".join(extract_text_values(payload))
    else:
        text = content
    text = re.sub(r"\[Image:\s*img_(?!key\b)[A-Za-z0-9_=-]+\]", " ", text)
    text = re.sub(r"<(?:file|audio|video|img|image)\b[^>]*>", " ", text)
    text = re.sub(r"\bimg_(?!key\b)[A-Za-z0-9_=-]+\b", " ", text)
    text = re.sub(r"\bfile_(?!key\b)[A-Za-z0-9_=-]+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text_values(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "text" and isinstance(child, str) and child.strip():
                texts.append(child.strip())
            else:
                texts.extend(extract_text_values(child))
    elif isinstance(value, list):
        for child in value:
            texts.extend(extract_text_values(child))
    return texts


def extract_payload_resources(message_type: str, value: Any) -> list[InboundResource]:
    resources: list[InboundResource] = []
    if isinstance(value, dict):
        image_key = value.get("image_key") or value.get("imageKey")
        if isinstance(image_key, str) and image_key.strip():
            resources.append(InboundResource(image_key.strip(), "image", image_key.strip(), message_type))
        file_key = value.get("file_key") or value.get("fileKey")
        if isinstance(file_key, str) and file_key.strip():
            name = (
                value.get("file_name")
                or value.get("fileName")
                or value.get("name")
                or value.get("title")
                or file_key
            )
            resources.append(InboundResource(file_key.strip(), "file", str(name).strip(), message_type))
        for child in value.values():
            resources.extend(extract_payload_resources(message_type, child))
    elif isinstance(value, list):
        for child in value:
            resources.extend(extract_payload_resources(message_type, child))
    return resources


def inbound_output_relative_path(
    base_cwd: Path,
    chat_workspace: Path,
    chat_id: str,
    message_id: str,
    resource: InboundResource,
) -> str:
    base_cwd = base_cwd.expanduser().resolve()
    chat_workspace = chat_workspace.expanduser().resolve()
    if is_relative_to(chat_workspace, base_cwd):
        root = chat_workspace
    else:
        root = base_cwd
    directory = root / ".lark_uploads" / safe_name(chat_id) / safe_name(message_id or resource.key)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / safe_filename(resource.name or resource.key)
    return path.relative_to(base_cwd).as_posix()


def parse_content(content: str) -> Any:
    text = content.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def all_regex(text: str, pattern: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in re.finditer(pattern, text)))


def name_for_key(text: str, key: str) -> str | None:
    tag_match = re.search(rf"<[^>]*{re.escape(key)}[^>]*>", text)
    if not tag_match:
        return None
    tag = tag_match.group(0)
    for attr in ("name", "file_name", "filename", "title"):
        attr_match = re.search(rf"""{attr}=["']([^"']+)["']""", tag)
        if attr_match:
            return attr_match.group(1)
    return None


def dedupe_resources(resources: list[InboundResource]) -> list[InboundResource]:
    deduped: list[InboundResource] = []
    seen: set[tuple[str, str]] = set()
    for resource in resources:
        identity = (resource.resource_type, resource.key)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(resource)
    return deduped


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe[:80] or "unknown"


def safe_filename(value: str) -> str:
    name = Path(value).name.strip()
    name = re.sub(r"[\x00-\x1f/\\:]+", "-", name).strip(". ")
    if not name:
        name = "attachment"
    return name[:160]


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
