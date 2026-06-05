import unittest

from paper_lark_agents.lark_cli import LarkEvent, MessageEvent


class LarkEventTests(unittest.TestCase):
    def test_message_event_extracts_flat_shape(self):
        event = LarkEvent.from_json(
            {
                "type": "im.message.receive_v1",
                "event_id": "evt_a",
                "chat_id": "oc_a",
                "content": "hello",
                "sender_id": "ou_a",
                "message_id": "om_a",
            }
        )

        self.assertEqual(event.event_type, "im.message.receive_v1")
        self.assertEqual(event.event_id, "evt_a")
        self.assertEqual(event.chat_id, "oc_a")

        message = MessageEvent.from_json(event.raw)
        self.assertEqual(message.content, "hello")
        self.assertEqual(message.message_id, "om_a")

    def test_bot_added_event_extracts_v2_shape(self):
        event = LarkEvent.from_json(
            {
                "schema": "2.0",
                "header": {
                    "event_id": "evt_b",
                    "event_type": "im.chat.member.bot.added_v1",
                },
                "event": {
                    "chat_id": "oc_b",
                    "name": "Research Room",
                },
            }
        )

        self.assertEqual(event.event_type, "im.chat.member.bot.added_v1")
        self.assertEqual(event.event_id, "evt_b")
        self.assertEqual(event.chat_id, "oc_b")

    def test_message_event_extracts_v2_message_shape(self):
        event = LarkEvent.from_json(
            {
                "schema": "2.0",
                "header": {
                    "event_id": "evt_nested",
                    "event_type": "im.message.receive_v1",
                },
                "event": {
                    "sender": {
                        "sender_id": {
                            "open_id": "ou_nested",
                        },
                    },
                    "message": {
                        "chat_id": "oc_nested",
                        "chat_type": "group",
                        "content": "nested hello",
                        "message_id": "om_nested",
                        "message_type": "text",
                        "create_time": "1780434068000",
                    },
                },
            }
        )

        message = MessageEvent.from_json(event.raw)
        self.assertEqual(message.event_id, "evt_nested")
        self.assertEqual(message.chat_id, "oc_nested")
        self.assertEqual(message.sender_id, "ou_nested")
        self.assertEqual(message.message_id, "om_nested")
        self.assertEqual(message.create_time, "1780434068000")

    def test_chat_disbanded_event_extracts_v2_shape(self):
        event = LarkEvent.from_json(
            {
                "schema": "2.0",
                "header": {
                    "event_id": "evt_c",
                    "event_type": "im.chat.disbanded_v1",
                },
                "event": {
                    "chat_id": "oc_c",
                    "name": "Research Room",
                },
            }
        )

        self.assertEqual(event.event_type, "im.chat.disbanded_v1")
        self.assertEqual(event.event_id, "evt_c")
        self.assertEqual(event.chat_id, "oc_c")


if __name__ == "__main__":
    unittest.main()
