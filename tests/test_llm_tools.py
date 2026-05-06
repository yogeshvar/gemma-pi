"""Tests for chat_with_tools (llama-server client mocked)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from config import Settings
from core.llm import chat_with_tools


class ChatWithToolsTests(unittest.TestCase):
    def test_runs_tool_then_returns_answer(self) -> None:
        settings = Settings(
            web_search_max_tool_rounds=3,
            llamacpp_model="test-model",
        )
        r1_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "weather"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        r2_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "It will be sunny.",
                    }
                }
            ]
        }
        mock_r1 = MagicMock(ok=True)
        mock_r1.json.return_value = r1_body
        mock_r2 = MagicMock(ok=True)
        mock_r2.json.return_value = r2_body

        with patch("core.llm._post_chat_completions", side_effect=[mock_r1, mock_r2]) as post:
            with patch("core.llm.search_web", return_value="search results here") as sw:
                out = chat_with_tools(
                    settings,
                    [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                )

        self.assertEqual(out, "It will be sunny.")
        sw.assert_called_once()
        self.assertEqual(post.call_count, 2)
        second_kwargs = post.call_args_list[1].kwargs
        msgs = second_kwargs["messages"]
        self.assertTrue(any(m.get("role") == "tool" for m in msgs))

    def test_unknown_tool_name(self) -> None:
        settings = Settings(web_search_max_tool_rounds=3, llamacpp_model="m")
        r1_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "nope", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }
        r2_body = {
            "choices": [{"message": {"role": "assistant", "content": "Done."}}]
        }
        mock_r1 = MagicMock(ok=True)
        mock_r1.json.return_value = r1_body
        mock_r2 = MagicMock(ok=True)
        mock_r2.json.return_value = r2_body
        with patch("core.llm._post_chat_completions", side_effect=[mock_r1, mock_r2]):
            with patch("core.llm.search_web") as sw:
                out = chat_with_tools(settings, [{"role": "user", "content": "x"}])
        sw.assert_not_called()
        self.assertEqual(out, "Done.")

    def test_falls_back_to_plain_chat_when_model_rejects_tools(self) -> None:
        settings = Settings(web_search_max_tool_rounds=3, llamacpp_model="gemma3-1b-it")

        mock_err = MagicMock()
        mock_err.ok = False
        mock_err.status_code = 400
        mock_err.text = "this model does not support tools (status code: 400)"

        with patch("core.llm._post_chat_completions", return_value=mock_err):
            with patch("core.llm.chat", return_value="plain reply") as plain:
                out = chat_with_tools(
                    settings,
                    [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
                )

        self.assertEqual(out, "plain reply")
        plain.assert_called_once()
        call_msgs = plain.call_args[0][1]
        self.assertEqual(
            call_msgs,
            [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        )
        self.assertTrue(plain.call_args[1].get("stream", False))


if __name__ == "__main__":
    unittest.main()
