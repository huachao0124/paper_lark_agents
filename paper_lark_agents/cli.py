from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from .agent_runner import AgentRunner
from .app import PaperAgentBridge
from .config import load_settings
from .daemon import daemon_status, read_log_tail, start_daemon, stop_daemon
from .lark_cli import LarkCLI
from .router import parse_task_request, route_message
from .web_dashboard import serve_dashboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-lark-agents")
    parser.add_argument("--env", default=".env", help="Path to env file.")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        subparser.add_argument("--env", default=argparse.SUPPRESS, help="Path to env file.")
        subparser.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS)
        return subparser

    add_common(sub.add_parser("serve", help="Listen to Feishu group events."))
    add_common(sub.add_parser("dashboard-server", help="Serve the AI status dashboard."))

    duo_cmd = add_common(sub.add_parser("serve-duo", help="Run Codex and Claude bot bridges."))
    duo_cmd.add_argument("--codex-env", default=".env.codex")
    duo_cmd.add_argument("--claude-env", default=".env.claude")

    daemon_start = add_common(sub.add_parser("daemon-start", help="Start bridges in the background."))
    daemon_start.add_argument("--codex-env", default=".env.codex")
    daemon_start.add_argument("--claude-env", default=".env.claude")

    add_common(sub.add_parser("daemon-stop", help="Stop the background bridge."))
    add_common(sub.add_parser("daemon-status", help="Show background bridge status."))
    daemon_logs = add_common(sub.add_parser("daemon-logs", help="Print bridge log tail."))
    daemon_logs.add_argument("--lines", type=int, default=80)

    route_cmd = add_common(sub.add_parser("route", help="Show how a message would route."))
    route_cmd.add_argument("message")

    ask_cmd = add_common(sub.add_parser("ask", help="Ask a local agent once."))
    ask_cmd.add_argument("--agent", choices=["codex", "claude", "codebuddy", "both"], default="both")
    ask_cmd.add_argument("prompt")

    send_cmd = add_common(sub.add_parser("send", help="Send markdown to a Feishu chat."))
    send_cmd.add_argument("--chat-id", required=True)
    send_cmd.add_argument("markdown")

    task_cmd = add_common(sub.add_parser("task", help="Create a Feishu task from task syntax."))
    task_cmd.add_argument("task_text")

    chat_cmd = add_common(sub.add_parser("create-chat", help="Create a Feishu group chat."))
    chat_cmd.add_argument("--name", required=True)
    chat_cmd.add_argument("--users", default="")
    chat_cmd.add_argument("--bots", default="")
    chat_cmd.add_argument("--description", default="")
    chat_cmd.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _protect_bridge_process_from_oom()
    settings = load_settings(args.env)

    try:
        if args.command == "serve":
            PaperAgentBridge(settings).serve()
            return 0
        if args.command == "dashboard-server":
            if not settings.dashboard_enabled:
                print("dashboard disabled")
                return 0
            serve_dashboard(settings)
            return 0
        if args.command == "serve-duo":
            return serve_duo(args.codex_env, args.claude_env, args.verbose)
        if args.command == "daemon-start":
            result = start_daemon(
                Path.cwd(),
                codex_env=args.codex_env,
                claude_env=args.claude_env,
                verbose=args.verbose,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("ok") else 1
        if args.command == "daemon-stop":
            result = stop_daemon(Path.cwd())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("ok") else 1
        if args.command == "daemon-status":
            print(json.dumps(daemon_status(Path.cwd()), ensure_ascii=False, indent=2))
            return 0
        if args.command == "daemon-logs":
            print(read_log_tail(Path.cwd(), lines=args.lines))
            return 0
        if args.command == "route":
            enabled_agents, default_agent = route_agent_settings(settings.agent_mode)
            route = route_message(
                args.message,
                settings.respond_to_all,
                enabled_agents=enabled_agents,
                bot_aliases=settings.bot_aliases,
                default_agent=default_agent,
            )
            print(route)
            return 0
        if args.command == "ask":
            runner = AgentRunner(settings)
            if args.agent in {"codex", "both"}:
                print("## Codex")
                print(runner.run_codex(args.prompt, "manual").text)
            if args.agent in {"claude", "both"}:
                print("## Claude Code")
                print(runner.run_claude(args.prompt, "manual").text)
            if args.agent == "codebuddy":
                print("## CodeBuddy")
                print(runner.run_codebuddy(args.prompt, "manual").text)
            return 0
        if args.command == "send":
            result = LarkCLI(settings).send_markdown(args.chat_id, args.markdown)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "task":
            task = parse_task_request(args.task_text)
            result = LarkCLI(settings).create_task(task)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "create-chat":
            result = LarkCLI(settings).create_chat(
                name=args.name,
                users=args.users,
                bots=args.bots,
                description=args.description,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2


def _protect_bridge_process_from_oom() -> None:
    raw = os.environ.get("PLA_OOM_SCORE_ADJ", "-800").strip()
    if raw.lower() in {"", "off", "false", "none", "disable", "disabled"}:
        return
    try:
        value = int(raw)
    except ValueError:
        logging.warning("invalid PLA_OOM_SCORE_ADJ=%r; skipping OOM protection", raw)
        return
    value = max(-1000, min(1000, value))
    try:
        Path("/proc/self/oom_score_adj").write_text(f"{value}\n", encoding="utf-8")
    except OSError as exc:
        logging.debug("failed to set oom_score_adj=%s: %s", value, exc)


def route_agent_settings(agent_mode: str) -> tuple[tuple[str, ...], str | None]:
    if agent_mode == "codex":
        return ("codex",), "codex"
    if agent_mode == "claude":
        return ("claude",), "claude"
    if agent_mode == "codebuddy":
        return ("codebuddy",), "codebuddy"
    if agent_mode == "tasks":
        return (), None
    return ("codex", "claude"), None


def serve_duo(codex_env: str, claude_env: str, verbose: bool = False) -> int:
    specs = [
        {"name": "codex", "cmd": _serve_cmd(codex_env, verbose), "restart": True},
        {"name": "claude", "cmd": _serve_cmd(claude_env, verbose), "restart": True},
        {"name": "dashboard", "cmd": _dashboard_cmd(codex_env, verbose), "restart": False},
    ]
    children = [
        {**spec, "process": subprocess.Popen(spec["cmd"])}
        for spec in specs
    ]
    try:
        while children:
            for child_info in list(children):
                child = child_info["process"]
                code = child.poll()
                if code is None:
                    continue
                name = str(child_info["name"])
                cmd = list(child_info["cmd"])
                if not child_info["restart"]:
                    children.remove(child_info)
                    if code != 0:
                        logging.warning("%s process exited with code %s", name, code)
                    continue
                logging.warning("%s bridge exited with code %s; restarting", name, code)
                time.sleep(2)
                child_info["process"] = subprocess.Popen(cmd)
            time.sleep(1)
    except KeyboardInterrupt:
        for child_info in children:
            child_info["process"].terminate()
        for child_info in children:
            child_info["process"].wait(timeout=10)
        return 130
    return 0


def _serve_cmd(env_file: str, verbose: bool) -> list[str]:
    cmd = [sys.executable, "-m", "paper_lark_agents", "--env", env_file]
    if verbose:
        cmd.append("--verbose")
    cmd.append("serve")
    return cmd


def _dashboard_cmd(env_file: str, verbose: bool) -> list[str]:
    cmd = [sys.executable, "-m", "paper_lark_agents", "--env", env_file]
    if verbose:
        cmd.append("--verbose")
    cmd.append("dashboard-server")
    return cmd


if __name__ == "__main__":
    raise SystemExit(main())
