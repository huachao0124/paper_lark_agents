# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local Python bridge that turns local **Codex** and **Claude Code** CLIs into two members of a Feishu (Lark) research group. It listens to group messages, routes prompts to the local agent CLIs, lets the two bots discuss with each other, creates Feishu tasks, and sends replies back — all through the external `lark-cli` tool. Codex and Claude are *not* Feishu users; each is driven inside a long-lived tmux session.

The package has **no third-party dependencies** (standard library only, `requires-python >=3.10`). All Feishu I/O is shelled out to `lark-cli`; agent execution is shelled out to `codex` / `claude` running under `tmux`.

## Commands

```bash
# Run the full test suite (no network / Feishu / tmux needed — pure logic + parsing)
python -m pytest -q
python -m unittest discover -s tests        # equivalent

# Run a single test module / case
python -m pytest tests/test_router.py -q
python -m pytest tests/test_router.py::RouterTests::test_codex_route -q

# Inspect routing without touching Feishu (great for debugging router.py changes)
python -m paper_lark_agents --env .env.codex route "@Codex Is the ablation convincing?"

# Ask a local agent once, bypassing Feishu
python -m paper_lark_agents ask --agent claude "Two sentence paper-review checklist."

# Run one bot bridge (one process per Feishu app/bot)
python -m paper_lark_agents --env .env.codex serve

# Run both bots + the status dashboard together
python -m paper_lark_agents serve-duo --codex-env .env.codex --claude-env .env.claude

# Background daemon (writes .run/*.pid and logs/*.log)
python -m paper_lark_agents daemon-start   # also daemon-status / daemon-logs / daemon-stop
```

Runtime prerequisites (must be on PATH): `lark-cli`, `codex`, `claude`, `tmux`. The entry point is `paper_lark_agents.cli:main` (also exposed as the `paper-lark-agents` script).

## The two-bot model

Codex and Claude run as **separate bridge processes**, each bound to its own Feishu app/bot via a `lark-cli` profile and its own env file (`.env.codex`, `.env.claude`). `PLA_AGENT_MODE` (`codex` | `claude` | `both` | `tasks`) decides which agent a process represents. Because both processes receive the *same* group events, group-wide actions are owned by exactly one bot to avoid duplicates:

- `PLA_ENABLE_TASKS` / `PLA_ENABLE_DEBATE` — only one bot should own `/task` and `/debate`.
- `PLA_HANDLE_MANAGEMENT_COMMANDS` — only one bot replies to `/workspace` and `/clear`; the other reads the shared state on its next turn.

The example envs default Codex to owning tasks/debate/management and Claude to off.

## Configuration

`config.py:load_settings(env_file)` reads `PLA_*` variables into a frozen `Settings` dataclass. Important behaviors:

- `load_env_file` **does not override variables already in `os.environ`** — the inherited shell environment wins over the `.env` file.
- Proxy policy is explicit and split: `PLA_PROXY_URL` is applied to bridge-side tools (`lark-cli`), while `PLA_AGENT_PROXY_URL` is applied to the agent CLIs launched in tmux. Never rely on inherited `http_proxy`. See `proxy_env` / `proxy_command_prefix`.
- Most defaults live in `load_settings`, not in the `.env.example` files — read `config.py` to know a flag's real default.

## Architecture

Message flow, end to end:

1. **`cli.py`** parses args, loads `Settings`, and dispatches to a subcommand. `serve` constructs a single `PaperAgentBridge`.
2. **`PaperAgentBridge.serve()`** (`app.py`) starts two background daemon threads — the **pending-run worker** and the **handoff worker** — then blocks in `lark.consume_events`.
3. **`lark_cli.py:LarkCLI.consume_events`** spawns one `lark-cli event consume <EventKey>` subprocess per key in `PLA_EVENT_KEYS`, keeps their stdin open (lark-cli exits on EOF), and parses NDJSON lines into `LarkEvent`s. All Feishu reads/writes (`send_markdown`, `create_task`, `download_message_resource`, `create_chat`, dashboard cards) go through this class with `--as bot`/`--as user` and idempotency keys.
4. **`app.py:handle_event`** filters (wrong chat, self-sent, non-group, stale, unsupported type), handles standalone uploads, downloads inbound attachments, then calls the router.
5. **`router.py:route_message`** is **pure, side-effect-free, and the most heavily tested module.** It turns raw message text into a `Route(kind=...)` where kind ∈ `ignore | help | agent | multi_agent | debate | task | workspace | clear | session_command`. It understands `@Codex`/`@Claude Code`/`/codex`/bare names/aliases, multi-agent directives in one message, raw session commands (anything starting with `/`), Chinese punctuation, and leading-acknowledgement stripping. Change routing behavior here and add cases to `tests/test_router.py`.
6. **`PaperAgentBridge.dispatch`** executes the route: creates a Feishu task, runs `/workspace`/`/clear`, forwards a raw session command, or runs the agent(s).

### Agent execution = persistent tmux sessions (`tmux_runtime.py`)

This is the most intricate part. When `PLA_*_RUNTIME=session` (the default), each (agent, chat) pair gets a long-lived tmux session named `pla-<agent>-<chat_id>`:

- The agent CLI is started **once** per session (`ensure_session`), in the chat's workspace. The session is recreated if its workspace, model, effort, or launch command changes.
- Each turn, the prompt is **pasted and submitted** into the session, wrapped with unique markers `PLA_REPLY_START_<run_id>` / `PLA_REPLY_END_<run_id>`. The reply is recovered by scraping the tmux **screen capture** plus a **transcript pipe** and extracting text between the markers (`find_marked_reply`, `extract_marked_reply`). This marker scraping is sensitive to terminal rendering — that is why Codex runs with `--no-alt-screen`.
- Per-session metadata (workspace, model, effort, initialized-flag, detected labels) is persisted next to the session and used to decide whether to inject the one-time session context prompt.
- A legacy one-shot path (`AgentRunner._run` calling `codex exec` / `claude -p` over stdin) still exists for `PLA_*_RUNTIME` values other than `session`.

### Resilience: pending runs

Agent turns routinely outlast a single dispatch. `pending_runs.py:PendingRunStore` persists every in-flight run to `.state/pending_runs.jsonl`. If the reply marker isn't present yet, `dispatch` raises/returns early and the **`pending_run_loop` thread** keeps polling the tmux session for the end marker, then sends the final reply when ready. This survives bridge restarts. `AgentStillRunning` signals "not done yet."

### Agent-to-agent discussion

When `PLA_RESPOND_TO_ALL=true`, both bots see ordinary messages and each decides whether to reply (emitting the `[NO_REPLY]` token to stay silent). To let the bots talk to each other without infinite loops:

- `outbox.py:AssistantOutbox` (`.state/assistant_outbox.jsonl`) records each bot's own outgoing replies. A bot ignores its **own** echo (`source_agent_for_event`) and only the *other* bot may continue.
- Discussion is bounded by `PLA_MAX_AGENT_DISCUSSION_TURNS` within `PLA_AGENT_DISCUSSION_WINDOW_SECONDS`.
- `handoff.py:AgentHandoffQueue` (`.state/agent_handoffs.jsonl`) supports direct handoffs, processed by the **`handoff_loop` thread**.

### Per-chat state under `.state/` (shared by both bridge processes)

Each Feishu group is an independent "research room." Because two processes share `.state`, several stores use `.lock` files and `*_claims/` directories for cross-process coordination:

- `chats/<chat_id>/history.jsonl` — room memory (`memory.py:ChatMemory`, bounded by `PLA_MEMORY_TURNS` / `PLA_MEMORY_CHARS`).
- `chat_workspaces.json` — per-chat `/workspace` overrides (`workspace.py`); the requested path **must be inside `PLA_WORKSPACE_ROOTS`**, and changing it recreates the tmux session.
- `chat_efforts.json` + model store — per-chat effort/model set via `@Codex /effort` etc. (`effort.py`, `models.py`).
- `chat_responders.json` — per-chat default responder (`responders.py`); decides which bot answers an *unaddressed* `respond_to_all` message. The router tags those routes `broadcast=True`; `PaperAgentBridge.apply_default_responder` gates them. `@`-mentions, `/codex`/`/claude`, and bot-to-bot discussion are never gated. Switch from Feishu with `/responder`, env default `PLA_DEFAULT_RESPONDER`.
- `pending_runs.jsonl`, `agent_handoffs.jsonl`, `assistant_outbox.jsonl`, `status_dashboards.json`.

### Prompts, status, files

- `prompts.py` builds the one-time session-context prompt, per-turn prompts, and debate prompts. The session-context prompt is what tells the agent how to behave as a group member (when to reply, the `[NO_REPLY]` contract, how to surface artifacts as Markdown links).
- `status_dashboard.py` / `status_card.py` render the interactive Feishu status card updated during long turns (`PLA_SEND_PROGRESS`). Cards are `interactive` messages so they don't re-trigger the other bot. `web_dashboard.py` (`dashboard-server`) serves an HTTP/JSON view over the shared `status_dashboards.json`.
- `artifacts.py:ArtifactRelay` scans final replies for local paths and uploads images/files to Feishu — but only files under `PLA_WORKSPACE_ROOTS` / `PLA_WORKSPACE` / `/tmp`.
- `inbound_files.py` + `LarkCLI.download_message_resource` pull user-uploaded image/file/audio/video/`post` resources into `.lark_uploads/<chat_id>/<message_id>/` before routing.

## Conventions

- Pure-logic modules (`router`, `memory`, `outbox`, `workspace`, `effort`, `models`, `config`, `inbound_files`, `artifacts`, plus the parsing helpers in `tmux_runtime`/`status_dashboard`) are designed to be unit-tested without Feishu, tmux, or network. Keep new logic testable the same way and mirror the existing `tests/test_*.py` layout.
- Outgoing replies are chunked to `PLA_MAX_MESSAGE_CHARS` and sent with idempotency keys to avoid duplicate/oversized Feishu messages.
- `from __future__ import annotations` is used throughout; frozen dataclasses for config/value types.
