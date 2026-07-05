from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import Settings
from .status_dashboard import DashboardSnapshot, StatusDashboardStore


LOGGER = logging.getLogger(__name__)


def serve_dashboard(settings: Settings) -> None:
    store = StatusDashboardStore(settings.state_dir)
    server = ThreadingHTTPServer(
        (settings.dashboard_host, settings.dashboard_port),
        lambda *args, **kwargs: DashboardHandler(store, *args, **kwargs),
    )
    LOGGER.info(
        "dashboard listening on http://%s:%s",
        settings.dashboard_host,
        settings.dashboard_port,
    )
    server.serve_forever()


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, store: StatusDashboardStore, *args: object, **kwargs: object):
        self.store = store
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self.write_json({"ok": True})
            return
        if parsed.path == "/api/status":
            self.write_json({"chats": self.all_chats()})
            return
        if parsed.path.startswith("/api/status/"):
            chat_id = unquote(parsed.path.removeprefix("/api/status/"))
            self.write_json(snapshot_to_dict(self.store.snapshot(chat_id)))
            return
        if parsed.path.startswith("/dashboard/"):
            chat_id = unquote(parsed.path.removeprefix("/dashboard/"))
            self.write_html(dashboard_html(chat_id))
            return
        self.write_html(index_html(self.all_chats()))

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.debug("dashboard %s", format % args)

    def all_chats(self) -> dict[str, Any]:
        try:
            data = json.loads(self.store.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def write_json(self, data: object) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def write_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def snapshot_to_dict(snapshot: DashboardSnapshot) -> dict[str, Any]:
    return {
        "chat_id": snapshot.chat_id,
        "message_id": snapshot.message_id,
        "pinned": snapshot.pinned,
        "doc_url": snapshot.doc_url,
        "doc_token": snapshot.doc_token,
        "tab_id": snapshot.tab_id,
        "statuses": snapshot.statuses,
        "bridge": snapshot.bridge,
    }


def index_html(chats: dict[str, Any]) -> str:
    links = "\n".join(
        f'<li><a href="/dashboard/{escape(chat_id)}">{escape(chat_id)}</a></li>'
        for chat_id in sorted(chats)
    )
    if not links:
        links = "<li>No dashboard state yet.</li>"
    return base_html(
        "AI Dashboard",
        f"""
        <main>
          <h1>AI Dashboard</h1>
          <ul>{links}</ul>
        </main>
        """,
    )


def dashboard_html(chat_id: str) -> str:
    safe_chat_id = json.dumps(chat_id)
    return base_html(
        "AI Dashboard",
        f"""
        <main>
          <header class="top">
            <div>
              <h1>AI Dashboard</h1>
              <p id="chat" class="muted"></p>
            </div>
            <div id="stamp" class="stamp">Loading</div>
          </header>
          <section id="bridge" class="band hidden"></section>
          <section class="agents">
            <article id="codex" class="agent"></article>
            <article id="claude" class="agent"></article>
            <article id="codebuddy" class="agent"></article>
          </section>
        </main>
        <script>
        const chatId = {safe_chat_id};
        document.getElementById('chat').textContent = chatId;

        function label(state) {{
          const labels = {{
            running: 'Running',
            done: 'Done',
            pending: 'Pending',
            failed: 'Failed',
            skipped: 'No reply needed',
            idle: 'Idle'
          }};
          return labels[state] || (state || 'idle');
        }}

        function age(startedAt) {{
          const value = Number(startedAt);
          if (!value) return 'n/a';
          const seconds = Math.max(0, Math.floor(Date.now() / 1000 - value));
          if (seconds < 60) return seconds + 's';
          const minutes = Math.floor(seconds / 60);
          if (minutes < 60) return minutes + 'm ' + (seconds % 60) + 's';
          const hours = Math.floor(minutes / 60);
          return hours + 'h ' + (minutes % 60) + 'm';
        }}

        function esc(value) {{
          return String(value ?? '').replace(/[&<>"']/g, ch => ({{
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
          }}[ch]));
        }}

        function renderAgent(id, data) {{
          const names = {{ codex: 'Codex', claude: 'Claude', codebuddy: 'CodeBuddy' }};
          const name = names[id] || id;
          const target = document.getElementById(id);
          const state = data?.state || 'idle';
          target.className = 'agent ' + state;
          if (!data) {{
            target.innerHTML = `
              <div class="agent-head"><h2>${{name}}</h2><span class="pill idle">Idle</span></div>
              <p class="muted">No session activity yet.</p>`;
            return;
          }}
          target.innerHTML = `
            <div class="agent-head">
              <h2>${{name}}</h2>
              <span class="pill ${{esc(state)}}">${{esc(label(state))}}</span>
            </div>
            <dl>
              <dt>Model</dt><dd>${{esc(data.model || 'unknown')}}</dd>
              <dt>Effort</dt><dd>${{esc(data.effort || 'unknown')}}</dd>
              <dt>Workspace</dt><dd><code>${{esc(data.workspace || 'unknown')}}</code></dd>
              <dt>Elapsed</dt><dd>${{esc(age(data.started_at))}}</dd>
              <dt>Detail</dt><dd>${{esc(data.detail || 'No detail.')}}</dd>
            </dl>`;
        }}

        function renderBridge(data) {{
          const target = document.getElementById('bridge');
          if (!data) {{
            target.classList.add('hidden');
            return;
          }}
          target.classList.remove('hidden');
          target.innerHTML = `<strong>Bridge</strong> · ${{esc(label(data.state))}} · ${{esc(data.detail || '')}}`;
        }}

        async function refresh() {{
          try {{
            const response = await fetch('/api/status/' + encodeURIComponent(chatId), {{ cache: 'no-store' }});
            const data = await response.json();
            renderBridge(data.bridge);
            renderAgent('codex', data.statuses?.codex);
            renderAgent('claude', data.statuses?.claude);
            renderAgent('codebuddy', data.statuses?.codebuddy);
            document.getElementById('stamp').textContent = 'Updated ' + new Date().toLocaleTimeString();
          }} catch (error) {{
            document.getElementById('stamp').textContent = 'Disconnected';
          }}
        }}

        refresh();
        setInterval(refresh, 1000);
        </script>
        """,
    )


def base_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --text: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #ffffff;
      --blue: #1d5fd3;
      --green: #16855f;
      --yellow: #a15c00;
      --red: #b42318;
      --grey: #59606c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 18px;
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 22px; font-weight: 700; }}
    h2 {{ font-size: 17px; font-weight: 700; }}
    .top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .muted {{ color: var(--muted); }}
    .stamp {{
      white-space: nowrap;
      color: var(--muted);
      font-size: 12px;
      padding-top: 5px;
    }}
    .band {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 12px;
    }}
    .hidden {{ display: none; }}
    .agents {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .agent {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }}
    .agent-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .pill {{
      border-radius: 999px;
      padding: 2px 8px;
      color: #fff;
      background: var(--grey);
      font-size: 12px;
      white-space: nowrap;
    }}
    .pill.running {{ background: var(--blue); }}
    .pill.done {{ background: var(--green); }}
    .pill.pending {{ background: var(--yellow); }}
    .pill.failed {{ background: var(--red); }}
    .pill.skipped, .pill.idle {{ background: var(--grey); }}
    dl {{
      display: grid;
      grid-template-columns: 78px minmax(0, 1fr);
      gap: 7px 10px;
      margin: 0;
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
    }}
    dd {{
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
    ul {{ padding-left: 18px; }}
    a {{ color: var(--blue); }}
    @media (max-width: 720px) {{
      .agents {{ grid-template-columns: 1fr; }}
      .top {{ flex-direction: column; }}
      .stamp {{ padding-top: 0; }}
    }}
  </style>
</head>
<body>{body}</body>
</html>"""


def escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
