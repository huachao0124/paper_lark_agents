# paper-lark-agents

Local bridge for a Feishu research group:

- Listen to group messages through `lark-cli event consume`.
- Let visible Feishu app bots respond as `@Codex`, `@Claude Code`, and optionally `@CodeBuddy`.
- Route `/codex`, `/claude`, `/codebuddy`, and `/debate` prompts to local assistant CLIs.
- Create Feishu tasks from `/task ...` messages.
- Send replies back to the same Feishu group.

Codex and Claude Code are local CLIs, not Feishu users. To make them feel like
two real group members, create two Feishu apps/bots and run one bridge process
for each app profile:

- Feishu bot display name: `Codex`; local backend: `codex exec`
- Feishu bot display name: `Claude Code`; local backend: `claude -p`
- Optional Feishu bot display name: `CodeBuddy`; local backend: `codebuddy`

## Setup

1. Make sure these commands work:

   ```bash
   lark-cli --help
   codex --help
   claude --help
   codebuddy --help
   ```

2. Configure two `lark-cli` profiles. Each profile should point at a different
   Feishu app/bot with the matching display name.

   ```bash
   lark-cli config init --new --name codex
   lark-cli config init --new --name claude
   lark-cli profile list
   ```

3. Enable the required scopes/events in both Feishu developer apps.

   Minimum bot-side scopes:

   - `im:message.p2p_msg:readonly` for direct-message `im.message.receive_v1`
   - `im:message.group_msg` if you want the bot to receive normal group
     messages without `@` mentions
   - `im:message.group_at_msg:readonly` if you only want the bot to receive
     group messages that mention it
   - `im:chat.members:bot_access` for `im.chat.member.bot.added_v1`, which
     lets the bridge warm a session when the bot is added to a group
   - `im:message` for sending messages
   - `im:chat:create` if the bot creates groups
   - `im:chat.members:write_only` if the bot adds members

   Enable these app events for both bots:

   - `im.message.receive_v1`
   - `im.chat.member.bot.added_v1`

   For Feishu Tasks:

   - `task:task:write`
   - `task:tasklist:write` if you create tasklists

4. Copy and edit the two environment files:

   ```bash
   cd /apdcephfs_sgfd/share_303735497/yixianliu/arimazhu/paper-lark-agents
   cp .env.codex.example .env.codex
   cp .env.claude.example .env.claude
   cp .env.codebuddy.example .env.codebuddy
   ```

   Keep `PLA_LARK_PROFILE=codex` in `.env.codex`,
   `PLA_LARK_PROFILE=claude` in `.env.claude`, and
   `PLA_LARK_PROFILE=codebuddy` in `.env.codebuddy`.

   Network proxy policy is explicit and does not rely on the shell's inherited
   `http_proxy` values:

   ```dotenv
   # Bridge-side tools such as lark-cli.
   PLA_PROXY_URL=http://star-proxy.oa.com:3128

   # Local agent CLIs launched in tmux sessions.
   PLA_AGENT_PROXY_URL=http://127.0.0.1:7899
   PLA_NO_PROXY=localhost,127.0.0.1,::1
   ```

   `PLA_CHAT_ID` is optional. Leave it empty if you want to create/invite groups
   directly in Feishu: once both bots are added to a group, they will respond
   there when mentioned. Fill `oc_xxx` only if you want to lock the bridge to
   one specific group.

5. Create or choose a Feishu group, then invite both app bots.

   The simplest workflow is entirely inside Feishu:

   - Create a group.
   - Add `Codex` and `Claude Code`.
   - Send a normal message, or address one assistant by name when you want only
     that assistant to answer.

   No `chat_id` copy/paste is needed when `PLA_CHAT_ID` is empty. When a bot is
   added to the group, the bridge receives `im.chat.member.bot.added_v1` and
   prestarts that group's long-lived CLI session in the background, so the first
   real message does not have to pay the cold-start cost.

Each Feishu group is treated as an independent research room. The bridge writes
that group's local memory under `.state/chats/<chat_id>/history.jsonl`, and only
that group's recent discussion is sent back to Codex or Claude Code on the next
mention. New groups start with empty memory automatically.

By default `PLA_RESPOND_TO_ALL=true`, so both assistants see normal group
messages without requiring `@`. Each model is asked to decide whether a reply is
useful; it can output `[NO_REPLY]` to stay silent.

The assistants can also discuss with each other. The bridge records recent
assistant replies in `.state/assistant_outbox.jsonl`, ignores each assistant's
own echo, and lets the other assistant decide whether to continue. Discussion is
bounded by `PLA_MAX_AGENT_DISCUSSION_TURNS` within
`PLA_AGENT_DISCUSSION_WINDOW_SECONDS` so the bots cannot loop forever.

Both assistants run as long-lived per-group sessions by default:

- `PLA_CODEX_RUNTIME=session`
- `PLA_CLAUDE_RUNTIME=session`
- `PLA_CODEBUDDY_RUNTIME=session`

By default the bridge does not force a model or effort at startup. Codex keeps
`--no-alt-screen` so tmux can capture the reply markers reliably; Claude starts
the same way as the project `teams.sh`.

```dotenv
PLA_CODEX_SESSION_ARGS=--no-alt-screen
PLA_CODEX_STARTUP_COMMANDS=/permissions auto-review
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
PLA_CLAUDE_SESSION_ARGS=--permission-mode auto --teammate-mode in-process
PLA_CODEBUDDY_SESSION_ARGS=--permission-mode acceptEdits
PLA_CODEX_MODEL=
PLA_CODEX_DEFAULT_EFFORT=
PLA_CLAUDE_MODEL=
PLA_CLAUDE_DEFAULT_EFFORT=
PLA_CODEBUDDY_MODEL=
PLA_CODEBUDDY_DEFAULT_EFFORT=xhigh
```

You can set model or effort from Feishu when needed:

```text
@Codex /effort xhigh
@Claude /effort max
@Codex /model gpt-5.5
@Claude /model opus
@CodeBuddy /effort xhigh
```

Targeted slash commands are pasted into the named assistant's tmux session
exactly as written, creating that session first if needed. For example,
`@Codex /help` or `@Claude /model opus` goes to that session directly instead of
being wrapped as a normal Feishu research prompt. Codex effort is the exception:
the bridge persists it and recreates the group's Codex session, because Codex
CLI applies reasoning effort at startup.

Bridge-level commands are intentionally unmentioned. Use plain `/workspace` to
manage the bridge's per-group working directory. Use `@Codex /workspace ...` if
you really want Codex itself to receive a `/workspace` slash command.

For each Feishu `chat_id`, the bridge lazily creates tmux sessions named like
`pla-codex-<chat_id>` and `pla-claude-<chat_id>`. Messages are pasted into those
sessions, and replies are captured with per-turn markers. This avoids the old
`claude -p` path and makes each Feishu group behave like a pair of persistent
local CLI conversations.

When `PLA_SEND_PROGRESS=true`, the bridge sends an interactive status card
before long assistant turns and updates that same card while waiting for reply
markers. The card shows state, model, effort, workspace, elapsed time, and a
short detail line. Update cadence is controlled by `PLA_STATUS_UPDATE_SECONDS`.
Status cards are `interactive` messages, so they do not trigger the other
assistant to reply.

When `PLA_ENABLE_ARTIFACTS=true`, final assistant replies are scanned for local
artifact paths. Markdown images/links, backticked paths, and plain paths like
`results/plot.png` or `/tmp/report.pdf` are uploaded automatically if the file
exists and is under `PLA_WORKSPACE_ROOTS`, `PLA_WORKSPACE`, or `/tmp`. Images
are sent as Feishu image messages; supported non-image outputs such as PDF, CSV,
Markdown, JSON, logs, Office files, and zip/tar archives are sent as file
messages. Files outside the bridge's allowed roots are ignored.

When `PLA_ENABLE_INBOUND_FILES=true`, user-uploaded Feishu `image`, `file`,
`audio`, and `video` messages, plus `post` rich-text messages with embedded
`image_key` / `file_key` resources, are downloaded through
`im +messages-resources-download` before routing. Downloads are stored under the
current group workspace in `.lark_uploads/<chat_id>/<message_id>/`, and the
bridge replies with the absolute local path(s). Standalone attachment messages
are not forwarded to Codex or Claude; the path is recorded in room memory so a
later text message can refer to it. If a `post` contains normal text plus
embedded resources, the assistant receives the original post text plus the local
paths. Artifact uploads created by the bridge are recorded by `message_id` so
they do not get reprocessed as new user uploads.

Each group can override the working directory from Feishu with `/workspace`.
The override is stored in `.state/chat_workspaces.json` and is shared by both
assistants. To keep group chat members from pointing the CLIs at arbitrary
server paths, the requested directory must be inside `PLA_WORKSPACE_ROOTS`
(comma-separated; defaults to `PLA_WORKSPACE`). Changing the workspace causes
the next assistant message in that group to start a fresh tmux session in the
new directory.

   Optional: create a group through one bot:

   ```bash
   python -m paper_lark_agents --env .env.codex create-chat \
     --name "Paper Lab Agents" \
     --users ou_xxx,ou_yyy \
     --bots cli_codex_app,cli_claude_app
   ```

   Or create the group manually in Feishu, invite both bots, then find the
   group id:

   ```bash
   lark-cli --profile codex im +chat-search --as bot --query "Paper Lab Agents"
   ```

6. Start both bot bridges:

   ```bash
   python -m paper_lark_agents serve-duo --codex-env .env.codex --claude-env .env.claude
   ```

   Run CodeBuddy as a third bridge process when you have created its Feishu
   app/profile:

   ```bash
   python -m paper_lark_agents --env .env.codebuddy serve
   ```

   Or run them as a background daemon:

   ```bash
   python -m paper_lark_agents daemon-start
   python -m paper_lark_agents daemon-status
   python -m paper_lark_agents daemon-logs
   python -m paper_lark_agents daemon-stop
   ```

## Group commands

```text
/help
What should we read first for this paper?
@Codex What is the main limitation of this paper?
@Claude Code Please critique the experiment design.
@CodeBuddy Please implement the repro harness.
/codex What is the main limitation of this paper?
/claude Please critique the experiment design.
/codebuddy Please implement the repro harness.
@Codex /effort xhigh
@Claude /effort max
@CodeBuddy /effort xhigh
/debate Compare this method with DINO and SAM.
/task Read the related work | assignee:ou_xxx | due:+2d | desc:focus on missing baselines
/workspace
/workspace papers/project-a
/workspace reset
/responder
/responder claude
/responder codebuddy
/responder reset
```

When `PLA_RESPOND_TO_ALL=true`, both bots otherwise answer every message. Use
`/responder` to pick who owns ordinary, unaddressed messages in a group:

- `/responder codex` / `/responder claude` / `/responder codebuddy` — only that bot answers plain
  messages; the others stay silent unless `@`-mentioned.
- `/responder both` — both answer (the default).
- `/responder` — show the current setting; `/responder reset` falls back to the
  `PLA_DEFAULT_RESPONDER` env default.

The choice is stored per group in `.state/chat_responders.json` and shared by
both bridge processes, so they always agree on who responds. It never affects
`@`-mentions, `/codex`/`/claude` commands, or bot-to-bot discussion — those
always reach the named assistant. Like `/workspace`, the confirmation is owned
by whichever bot has `PLA_HANDLE_MANAGEMENT_COMMANDS=true`.

In the two-bot setup, `.env.codex.example` and `.env.claude.example` default to
`PLA_ENABLE_TASKS=false` and `PLA_ENABLE_DEBATE=false`. This prevents duplicate
tasks or duplicate debate replies when both bots receive the same group event.
Turn one of those flags on for exactly one bot if you want that bot to own
group-wide `/task` or `/debate` commands.

The same duplicate-handling pattern applies to management commands such as
`/workspace`: `.env.codex.example` sets `PLA_HANDLE_MANAGEMENT_COMMANDS=true`
and `.env.claude.example` sets it to `false`, so Codex owns the confirmation
message while Claude reads the shared workspace setting on the next turn.

Task syntax:

- The first segment is the task summary.
- Optional segments use `key:value`.
- Supported keys: `assignee`, `due`, `desc`, `description`, `tasklist`.

## Local smoke checks

```bash
python -m unittest discover -s tests
python -m paper_lark_agents --env .env.codex route "@Codex Is the ablation convincing?"
python -m paper_lark_agents --env .env.claude route "@Claude Code Is the ablation convincing?"
python -m paper_lark_agents ask --agent claude "Give me a two sentence paper review checklist."
```

## Notes

- The event consumer keeps stdin open, because `lark-cli event consume` exits
  when stdin reaches EOF.
- Replies are chunked before sending to avoid oversized Feishu messages.
- Codex runs in read-only sandbox by default. Claude is started with tools
  disabled by default. Adjust `.env` if you want stronger local capabilities.
