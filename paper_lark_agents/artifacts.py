from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shlex
import shutil
from urllib.parse import unquote, urlparse


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
FILE_EXTENSIONS = {
    ".csv",
    ".docx",
    ".gz",
    ".htm",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".pdf",
    ".pptx",
    ".tar",
    ".tgz",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".zip",
}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | FILE_EXTENSIONS
SUPPORTED_EXTENSION_PATTERN = "|".join(sorted(suffix.lstrip(".") for suffix in SUPPORTED_EXTENSIONS))
GENERATED_ARTIFACT_CUES = (
    "artifact",
    "created",
    "generated",
    "output",
    "saved",
    "wrote",
    "written",
    "产出",
    "保存",
    "写好",
    "周报",
    "报告",
    "新建",
    "生成",
    "草稿",
    "做成",
    "整理为",
    "输出",
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    upload_path: str
    kind: str

    @property
    def name(self) -> str:
        return self.path.name


class ArtifactRelay:
    def __init__(
        self,
        base_cwd: Path,
        state_dir: Path,
        allowed_roots: tuple[Path, ...],
        max_artifacts: int = 8,
        include_tmp: bool = True,
    ):
        self.base_cwd = base_cwd.expanduser().resolve()
        self.max_artifacts = max(0, max_artifacts)
        state_dir = state_dir.expanduser().resolve()
        if is_relative_to(state_dir, self.base_cwd):
            self.stage_dir = state_dir / "artifacts"
        else:
            self.stage_dir = self.base_cwd / ".state" / "artifacts"
        roots = [self.base_cwd, *allowed_roots]
        if include_tmp:
            roots.append(Path("/tmp"))
        self.allowed_roots = tuple(dedupe_paths(path.expanduser().resolve() for path in roots))

    def collect(self, text: str, workspace: Path) -> list[Artifact]:
        workspace = workspace.expanduser().resolve()
        artifacts: list[Artifact] = []
        seen: set[Path] = set()
        for candidate in extract_path_candidates(text):
            path = self.resolve_candidate(candidate, workspace)
            if path is None or path in seen:
                continue
            seen.add(path)
            upload_path = self.upload_argument(path)
            kind = "image" if path.suffix.lower() in IMAGE_EXTENSIONS else "file"
            artifacts.append(Artifact(path=path, upload_path=upload_path, kind=kind))
            if len(artifacts) >= self.max_artifacts:
                break
        return artifacts

    def resolve_candidate(self, candidate: str, workspace: Path) -> Path | None:
        candidate = clean_candidate(candidate)
        if not candidate or is_remote_or_key(candidate):
            return None
        parsed = urlparse(candidate)
        if parsed.scheme == "file":
            candidate = unquote(parsed.path)

        try:
            raw = Path(candidate).expanduser()
        except RuntimeError:
            return None
        choices = [raw] if raw.is_absolute() else [workspace / raw, self.base_cwd / raw]
        for choice in choices:
            path = choice.resolve(strict=False)
            if (
                path.exists()
                and path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and self.is_allowed(path)
            ):
                return path
        return None

    def is_allowed(self, path: Path) -> bool:
        return any(is_relative_to(path, root) for root in self.allowed_roots)

    def upload_argument(self, path: Path) -> str:
        if is_relative_to(path, self.base_cwd):
            return path.relative_to(self.base_cwd).as_posix()
        staged = self.stage(path)
        return staged.relative_to(self.base_cwd).as_posix()

    def stage(self, path: Path) -> Path:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        name = safe_filename(path.name)
        staged = self.stage_dir / f"{digest}-{name}"
        if not staged.exists() or staged.stat().st_mtime < path.stat().st_mtime:
            shutil.copy2(path, staged)
        return staged


def extract_path_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        candidates.append(first_markdown_target(match.group(1)))

    for match in re.finditer(r"""(?i)<(?:img|a)\b[^>]*(?:src|href)=["']([^"']+)["']""", text):
        candidates.append(match.group(1))

    candidates.extend(extract_plain_path_candidates(text))

    return [candidate for candidate in candidates if looks_like_path(clean_candidate(candidate))]


def extract_plain_path_candidates(text: str) -> list[str]:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_./~-])"
        rf"(?:[~./]?[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+|"
        rf"[A-Za-z0-9_.@+-]+\.({SUPPORTED_EXTENSION_PATTERN}))"
        rf"(?![A-Za-z0-9_./~-])",
        re.IGNORECASE,
    )
    candidates: list[str] = []
    for match in pattern.finditer(text):
        if generated_cue_precedes_path(text, match.start()):
            candidates.append(match.group(0))
    return candidates


def generated_cue_precedes_path(text: str, path_start: int) -> bool:
    prefix = text[max(0, path_start - 160) : path_start].lower()
    return any(cue in prefix for cue in GENERATED_ARTIFACT_CUES)


def first_markdown_target(value: str) -> str:
    value = value.strip()
    try:
        parts = shlex.split(value)
    except ValueError:
        parts = value.split()
    return parts[0] if parts else ""


def clean_candidate(value: str) -> str:
    value = value.strip().strip("<>")
    value = value.strip("\"'")
    return value.rstrip(".,;:")


def looks_like_path(value: str) -> bool:
    if not value or is_remote_or_key(value):
        return False
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return False
    if parsed.scheme == "file":
        value = parsed.path
    path = Path(value)
    return (
        value.startswith(("/", "./", "../", "~"))
        or "/" in value
        or path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def is_remote_or_key(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered.startswith(("http://", "https://", "data:"))
        or value.startswith(("img_", "file_"))
    )


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def dedupe_paths(paths) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "artifact"
