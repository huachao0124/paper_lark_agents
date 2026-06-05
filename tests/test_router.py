import unittest

from paper_lark_agents.router import addressed_agents, parse_task_request, route_message


class AddressedAgentsTests(unittest.TestCase):
    def test_at_codex_detected(self):
        self.assertEqual(addressed_agents("好的,@Codex 你看下 pipeline 顺序"), {"codex"})

    def test_case_insensitive(self):
        self.assertEqual(addressed_agents("@codex 看看"), {"codex"})
        self.assertEqual(addressed_agents("@CLAUDE thoughts?"), {"claude"})

    def test_at_claude_code_form(self):
        self.assertEqual(addressed_agents("@Claude Code 接一下"), {"claude"})

    def test_passing_mention_without_at_is_ignored(self):
        self.assertEqual(addressed_agents("这点和 Codex 之前说的一样"), set())

    def test_both_addressed(self):
        self.assertEqual(addressed_agents("@Codex @Claude 你俩都看下"), {"codex", "claude"})

    def test_empty(self):
        self.assertEqual(addressed_agents(""), set())


class RouterTests(unittest.TestCase):
    def test_codex_route(self):
        route = route_message("/codex critique this paper")
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "critique this paper")
        self.assertFalse(route.broadcast)

    def test_bot_display_name_suffix_is_addressed(self):
        # Feishu renders an @ of a "Codex-Mac" bot as "@Codex-Mac"; it must be
        # recognized as addressing codex, not treated as an unaddressed broadcast.
        route = route_message("@Codex-Mac use deepseek")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "use deepseek")
        self.assertFalse(route.broadcast)
        route = route_message("@Claude-Mac thoughts?")
        self.assertEqual(route.agent, "claude")
        self.assertFalse(route.broadcast)

    def test_hyphenated_word_is_not_addressing(self):
        # A bare hyphenated word that merely contains an agent name stays a
        # broadcast (no false positive without an explicit "@").
        route = route_message(
            "codex-cli is a great tool",
            respond_to_all=True,
            default_agent="codex",
        )
        self.assertTrue(route.broadcast)

    def test_responder_command_routes(self):
        for text in ("/responder claude", "!responder claude", "responder: claude"):
            route = route_message(text)
            self.assertEqual(route.kind, "responder", text)
            self.assertEqual(route.text, "claude", text)

    def test_responder_command_without_arg(self):
        route = route_message("/responder")
        self.assertEqual(route.kind, "responder")
        self.assertEqual(route.text, "")

    def test_respond_to_all_plain_message_is_broadcast(self):
        route = route_message(
            "what should we read first?",
            respond_to_all=True,
            enabled_agents=("claude",),
            default_agent="claude",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "claude")
        self.assertTrue(route.broadcast)

    def test_respond_to_all_multi_agent_is_broadcast(self):
        route = route_message(
            "what should we read first?",
            respond_to_all=True,
            enabled_agents=("codex", "claude"),
            default_agent=None,
        )
        self.assertEqual(route.kind, "multi_agent")
        self.assertTrue(route.broadcast)

    def test_explicit_mention_is_not_broadcast(self):
        route = route_message(
            "@Codex critique this",
            respond_to_all=True,
            enabled_agents=("codex", "claude"),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "critique this")
        self.assertFalse(route.broadcast)

    def test_claude_route(self):
        route = route_message("claude: summarize limitations")
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "claude")
        self.assertEqual(route.text, "summarize limitations")

    def test_bare_claude_name_broadcasts_full_text_to_default_agent(self):
        route = route_message(
            "claude你先出清单吧",
            respond_to_all=True,
            enabled_agents=("claude",),
            default_agent="claude",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "claude")
        self.assertEqual(route.text, "claude你先出清单吧")

    def test_acknowledged_bare_claude_name_broadcasts_full_text_to_default_agent(self):
        route = route_message(
            "嗯，claude你先出清单吧",
            respond_to_all=True,
            enabled_agents=("claude",),
            default_agent="claude",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "claude")
        self.assertEqual(route.text, "嗯，claude你先出清单吧")

    def test_bare_other_agent_name_broadcasts_to_single_agent_mode(self):
        route = route_message(
            "claude你先出清单吧",
            respond_to_all=True,
            enabled_agents=("codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "claude你先出清单吧")

    def test_acknowledged_other_agent_name_broadcasts_to_single_agent_mode(self):
        route = route_message(
            "嗯，claude你先出清单吧",
            respond_to_all=True,
            enabled_agents=("codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "嗯，claude你先出清单吧")

    def test_bare_agent_name_with_fullwidth_punctuation_broadcasts(self):
        route = route_message("Codex，继续校验", respond_to_all=True)
        self.assertEqual(route.kind, "multi_agent")
        self.assertEqual(
            route.agent_texts,
            {
                "codex": "Codex，继续校验",
                "claude": "Codex，继续校验",
            },
        )

    def test_bare_agent_session_command(self):
        route = route_message("Claude /effort ultracode", respond_to_all=True)
        self.assertEqual(route.kind, "session_command")
        self.assertEqual(route.agent, "claude")
        self.assertEqual(route.text, "/effort ultracode")

    def test_plain_multi_agent_directives_broadcast_when_both_enabled(self):
        text = "claude继续写代码，codex用28.45.33.95和28.45.32.245去跑通视频生成模型"
        route = route_message(
            text,
            respond_to_all=True,
        )
        self.assertEqual(route.kind, "multi_agent")
        self.assertEqual(
            route.agent_texts,
            {
                "claude": text,
                "codex": text,
            },
        )

    def test_plain_multi_agent_directives_preserve_full_message(self):
        text = (
            "新建好了，在git@example.com:owner/repo.git，access token在../.env里面，"
            "email是user@example.com，user是exampleuser，不要把你们作为coauthor上传上去，"
            "codex去上传，做完之后claude去check"
        )
        route = route_message(
            text,
            respond_to_all=True,
        )
        self.assertEqual(route.kind, "multi_agent")
        self.assertEqual(route.agent_texts, {"codex": text, "claude": text})

    def test_multi_agent_session_commands_do_not_keep_shared_context(self):
        route = route_message(
            "麻烦 codex /effort xhigh，claude /effort max",
            respond_to_all=True,
        )
        self.assertEqual(route.kind, "multi_agent")
        self.assertEqual(
            route.agent_texts,
            {
                "codex": "/effort xhigh",
                "claude": "/effort max",
            },
        )

    def test_plain_multi_agent_directives_broadcast_to_enabled_agent(self):
        text = "claude继续写代码，codex用28.45.33.95和28.45.32.245去跑通视频生成模型"
        route = route_message(
            text,
            respond_to_all=True,
            enabled_agents=("codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, text)

    def test_plain_multi_agent_directives_broadcast_to_claude(self):
        text = "claude继续写代码，codex用28.45.33.95和28.45.32.245去跑通视频生成模型"
        route = route_message(
            text,
            respond_to_all=True,
            enabled_agents=("claude",),
            default_agent="claude",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "claude")
        self.assertEqual(route.text, text)

    def test_agent_conjunction_broadcasts_not_first_agent(self):
        route = route_message("codex和claude讨论一下", respond_to_all=True)
        self.assertEqual(route.kind, "multi_agent")
        self.assertEqual(
            route.agent_texts,
            {
                "codex": "codex和claude讨论一下",
                "claude": "codex和claude讨论一下",
            },
        )

    def test_both_sends_to_all_agents(self):
        route = route_message("/both 看看这篇论文")
        self.assertEqual(route.kind, "multi_agent")
        self.assertIn("codex", route.agent_texts)
        self.assertIn("claude", route.agent_texts)
        self.assertEqual(route.agent_texts["codex"], "看看这篇论文")
        self.assertEqual(route.agent_texts["claude"], "看看这篇论文")
        self.assertFalse(route.broadcast)

    def test_both_ignores_responder_gate(self):
        # /both is explicit, not broadcast — responder gate should not block.
        route = route_message(
            "/both check this",
            respond_to_all=True,
            enabled_agents=("codex", "claude"),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "multi_agent")
        self.assertIn("claude", route.agent_texts)

    def test_both_empty_is_ignored(self):
        route = route_message("/both")
        self.assertNotEqual(route.kind, "multi_agent")

    def test_debate_route(self):
        route = route_message("/debate is the ablation enough?")
        self.assertEqual(route.kind, "debate")
        self.assertEqual(route.text, "is the ablation enough?")

    def test_debate_broadcasts_to_default_agent(self):
        route = route_message(
            "/debate is the ablation enough?",
            respond_to_all=True,
            enabled_agents=("codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "codex")
        self.assertIn("Feishu command: /debate", route.text)
        self.assertIn("is the ablation enough?", route.text)

    def test_task_route(self):
        route = route_message(
            "/task Read SAM paper | assignee:ou_abc | due:+2d | desc:focus on data"
        )
        self.assertEqual(route.kind, "task")
        self.assertIsNotNone(route.task)
        self.assertEqual(route.task.summary, "Read SAM paper")
        self.assertEqual(route.task.assignee, "ou_abc")
        self.assertEqual(route.task.due, "+2d")
        self.assertEqual(route.task.description, "focus on data")

    def test_workspace_route_wins_over_respond_to_all(self):
        route = route_message("/workspace papers/project-a", respond_to_all=True)
        self.assertEqual(route.kind, "workspace")
        self.assertEqual(route.text, "papers/project-a")

    def test_clear_route_wins_over_respond_to_all(self):
        route = route_message("/clear", respond_to_all=True)
        self.assertEqual(route.kind, "clear")
        self.assertEqual(route.text, "")

    def test_clear_route_after_bot_alias_wins_over_session_command(self):
        route = route_message(
            "@Codex /clear init",
            respond_to_all=True,
            enabled_agents=("codex",),
            bot_aliases=("Codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "clear")
        self.assertEqual(route.text, "init")

    def test_workspace_route_after_bot_alias_is_session_command(self):
        route = route_message(
            "@Codex /workspace",
            respond_to_all=True,
            enabled_agents=("codex",),
            bot_aliases=("Codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "session_command")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "/workspace")

    def test_task_parser_with_extra_description(self):
        task = parse_task_request("Run ablation | include seed variance")
        self.assertEqual(task.summary, "Run ablation")
        self.assertEqual(task.description, "include seed variance")

    def test_ignore_by_default(self):
        self.assertEqual(route_message("hello").kind, "ignore")

    def test_command_prefix_needs_boundary(self):
        self.assertEqual(route_message("/codexology").kind, "ignore")

    def test_respond_to_all(self):
        route = route_message("hello", respond_to_all=True)
        self.assertEqual(route.kind, "multi_agent")
        self.assertEqual(route.agent_texts, {"codex": "hello", "claude": "hello"})

    def test_respond_to_all_uses_default_agent(self):
        route = route_message(
            "直接聊论文",
            respond_to_all=True,
            enabled_agents=("codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "直接聊论文")

    def test_bot_alias_routes_to_default_agent(self):
        route = route_message(
            "@Claude Code critique this",
            enabled_agents=("claude",),
            bot_aliases=("Claude Code", "Claude"),
            default_agent="claude",
        )
        self.assertEqual(route.kind, "agent")
        self.assertEqual(route.agent, "claude")
        self.assertEqual(route.text, "critique this")

    def test_bot_alias_slash_routes_to_session_command(self):
        route = route_message(
            "@Codex /effort xhigh",
            enabled_agents=("codex",),
            bot_aliases=("Codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "session_command")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "/effort xhigh")

    def test_bot_alias_help_routes_to_session_command(self):
        route = route_message(
            "@Codex /help",
            enabled_agents=("codex",),
            bot_aliases=("Codex",),
            default_agent="codex",
        )
        self.assertEqual(route.kind, "session_command")
        self.assertEqual(route.agent, "codex")
        self.assertEqual(route.text, "/help")

    def test_explicit_other_agent_command_is_ignored(self):
        route = route_message(
            "@Codex /effort xhigh",
            respond_to_all=True,
            enabled_agents=("claude",),
            bot_aliases=("Claude",),
            default_agent="claude",
        )
        self.assertEqual(route.kind, "ignore")

    def test_disabled_agent_command_is_ignored(self):
        route = route_message("/claude critique this", enabled_agents=("codex",))
        self.assertEqual(route.kind, "ignore")


if __name__ == "__main__":
    unittest.main()
