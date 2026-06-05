# Migrate paper-lark-agents to macbook

This guide migrates the bridge + agent sessions from a Linux server to macbook.
Designed to be read and executed by Codex/Claude Code on the macbook.

## Prerequisites

```bash
# Install tools (if not already present)
brew install tmux
# Install and login: codex, claude code, lark-cli
# Use the SAME accounts as the server
```

## 1. Clone and configure

```bash
git clone <REPO_URL> ~/paper-lark-agents
cd ~/paper-lark-agents
```

Create `.env.codex` and `.env.claude` by copying from `.env.codex.example` / `.env.claude.example`,
then update these fields for macbook paths:

```
PLA_WORKSPACE=~/projects          # or wherever your working dirs are
PLA_WORKSPACE_ROOTS=~/projects
PLA_STATE_DIR=.state
```

All other fields (PLA_LARK_PROFILE, PLA_BOT_ALIASES, event keys, proxy, etc.)
stay the same as the server. The Feishu bot credentials are tied to the app,
not the machine.

### lark-cli profiles

```bash
lark-cli config init              # follow prompts for codex app
lark-cli profile create claude    # follow prompts for claude app
```

Use the same `app_id` / `app_secret` as the server. Verify:
```bash
lark-cli im +chat-list --as bot --profile codex
lark-cli im +chat-list --as bot --profile claude
```

## 2. Migrate session data (optional, per-group)

To preserve agent conversation context for a specific group, copy the
session transcript files from the server.

### Claude session

```bash
# On server: find the session file
cat .state/tmux/pla-claude-<GroupName>-<suffix>.json
# Look for "session_uuid" and "transcript_path"

# Copy transcript to macbook:
mkdir -p ~/.claude/projects/_migration/
scp server:<transcript_path> ~/.claude/projects/_migration/

# Verify resume works:
claude --resume <session_uuid>
```

### Codex session

```bash
# On server: find the rollout file
cat .state/tmux/pla-codex-<GroupName>-<suffix>.json
# Look for "session_uuid" and "transcript_path"

# Copy rollout to macbook (preserve directory structure):
mkdir -p ~/.codex/sessions/2026/06/05/   # adjust date
scp server:<transcript_path> ~/.codex/sessions/2026/06/05/

# Verify resume works:
codex resume <session_uuid>
```

### Bridge state (room memory)

```bash
# Copy the group's chat history for room memory context:
mkdir -p ~/paper-lark-agents/.state/chats/
scp -r server:paper-lark-agents/.state/chats/<chat_id_dir> \
    ~/paper-lark-agents/.state/chats/

# Copy tmux metadata (session names, cursors):
mkdir -p ~/paper-lark-agents/.state/tmux/
scp server:paper-lark-agents/.state/tmux/pla-*-<GroupName>*.json \
    ~/paper-lark-agents/.state/tmux/

# Copy shared state files:
scp server:paper-lark-agents/.state/chat_responders.json \
    ~/paper-lark-agents/.state/
scp server:paper-lark-agents/.state/chat_efforts.json \
    ~/paper-lark-agents/.state/ 2>/dev/null
scp server:paper-lark-agents/.state/chat_models.json \
    ~/paper-lark-agents/.state/ 2>/dev/null
```

**Important**: Update `workspace` paths in the copied `.state/tmux/*.json`
files to point to the macbook workspace directory.

## 3. Current session IDs (as of 2026-06-05)

| Group | chat_id | Claude session | Codex session |
|-------|---------|----------------|---------------|
| VisualTrust | oc_ba89bb5f4c5c811e46774c5f9094376a | ffe3e379-cadb-4539-b29e-9679bf0875f4 | 019e8749-9133-7263-9acb-a996783462a4 |
| VideoAgent | oc_3349d64e2d90567f1c1949335e566a1e | c9e0a5b4-4aee-4d83-9d5c-3015b5acdead | (check server metadata) |
| sts2_self_improve_harness | oc_fe8260ac005b56c6629f4711b6afe44e | e6d7caac-7c3c-4c1b-9afe-765c43578c1f | 019e958a-5b51-7230-9fbc-fe1529d8bd5a |

## 4. Switch operation

**CRITICAL: Never run both server and macbook bridge simultaneously.
Both consume the same Feishu events and will duplicate all replies.**

```bash
# === Stop server ===
# On server:
cd paper-lark-agents
python -m paper_lark_agents daemon-stop

# === Start macbook ===
# On macbook:
cd ~/paper-lark-agents
python -m paper_lark_agents daemon-start

# Verify:
python -m paper_lark_agents daemon-status
tmux ls | grep pla-
```

To switch back:
```bash
# macbook: stop
python -m paper_lark_agents daemon-stop
# server: start
python -m paper_lark_agents daemon-start
```

## 5. Verify after migration

```bash
# Check all groups are accessible:
python -m paper_lark_agents daemon-status

# Check tmux sessions created with human-readable names:
tmux ls | grep pla-
# Expected: pla-claude-VisualTrust-94376a, pla-codex-sts2-6afe44e, etc.

# Send a test message in any Feishu group and verify:
# 1. Progress card appears
# 2. Agent replies
# 3. Room memory updates in .state/chats/
```

## Notes

- Agent sessions start fresh on macbook (new tmux). If you copied transcript
  files, the bridge will `--resume` / `codex resume` to restore context.
- Room memory (history.jsonl) provides context even without session resume.
- Proxy settings may differ — macbook likely doesn't need `PLA_PROXY_URL`
  or `PLA_AGENT_PROXY_URL`. Set them to empty or remove.
- `PLA_NO_PROXY` can be left empty on macbook.
