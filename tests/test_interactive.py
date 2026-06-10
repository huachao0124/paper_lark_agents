import unittest

from paper_lark_agents.interactive import detect_interactive_prompt, format_prompt_message


class DetectInteractivePromptTests(unittest.TestCase):
    def test_detects_permission_menu(self):
        screen = (
            "Allow Claude to edit files in this workspace?\n"
            "❯ 1. Yes, proceed (y)\n"
            "  2. No, go back (esc)\n"
            "Enter to select\n"
        )
        prompt = detect_interactive_prompt(screen)
        self.assertIsNotNone(prompt)
        self.assertEqual(len(prompt["options"]), 2)
        self.assertIn("Yes, proceed", prompt["options"][0])

    def test_detects_model_switch_menu(self):
        screen = (
            "Switch model for this session?\n"
            "❯ 1. Yes, switch to claude-fable-5\n"
            "  2. No, go back\n"
        )
        prompt = detect_interactive_prompt(screen)
        self.assertIsNotNone(prompt)
        self.assertEqual(len(prompt["options"]), 2)

    def test_rejects_single_numbered_line(self):
        # An ordinary numbered list in agent output must not be mistaken
        # for a selection menu — menus always have at least two options.
        screen = (
            "Here is my plan:\n"
            "1. Yes, switch to the new tokenizer first\n"
            "Press enter to confirm or esc to cancel later steps.\n"
        )
        self.assertIsNone(detect_interactive_prompt(screen))

    def test_rejects_plain_output(self):
        screen = "All tests passed.\nDone in 1.4s\n"
        self.assertIsNone(detect_interactive_prompt(screen))

    def test_format_prompt_message(self):
        prompt = {"title": "Switch model?", "options": ["1. Yes", "2. No"], "raw": ""}
        text = format_prompt_message(prompt)
        self.assertIn("Switch model?", text)
        self.assertIn("- 1. Yes", text)
        self.assertIn("esc", text)


if __name__ == "__main__":
    unittest.main()
