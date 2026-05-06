"""Tests for chat_with_tools (Ollama client mocked)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from config import Settings
from core.llm import chat_with_tools
from ollama._types import ChatResponse, Message


class ChatWithToolsTests(unittest.TestCase):
    def test_runs_tool_then_returns_answer(self) -> None:
        settings = Settings(
            web_search_max_tool_rounds=3,
            ollama_model="test-model",
        )
        tc = Message.ToolCall(
            function=Message.ToolCall.Function(name="web_search", arguments={"query": "weather"})
        )
        msg_tool = Message(role="assistant", content="", tool_calls=[tc])
        msg_final = Message(role="assistant", content="It will be sunny.")
        r1 = ChatResponse(model="test-model", message=msg_tool)
        r2 = ChatResponse(model="test-model", message=msg_final)

        mock_client = MagicMock()
        mock_client.chat.side_effect = [r1, r2]

        with patch("core.llm._client", return_value=mock_client):
            with patch("core.llm.search_web", return_value="search results here") as sw:
                out = chat_with_tools(
                    settings,
                    [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                )

        self.assertEqual(out, "It will be sunny.")
        sw.assert_called_once()
        self.assertEqual(mock_client.chat.call_count, 2)
        second_call_kwargs = mock_client.chat.call_args_list[1].kwargs
        self.assertIn("messages", second_call_kwargs)
        msgs = second_call_kwargs["messages"]
        self.assertTrue(any(m.get("role") == "tool" for m in msgs))

    def test_unknown_tool_name(self) -> None:
        settings = Settings(web_search_max_tool_rounds=3, ollama_model="m")
        tc = Message.ToolCall(
            function=Message.ToolCall.Function(name="nope", arguments={})
        )
        msg_tool = Message(role="assistant", tool_calls=[tc])
        msg_final = Message(role="assistant", content="Done.")
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ChatResponse(model="m", message=msg_tool),
            ChatResponse(model="m", message=msg_final),
        ]
        with patch("core.llm._client", return_value=mock_client):
            with patch("core.llm.search_web") as sw:
                out = chat_with_tools(settings, [{"role": "user", "content": "x"}])
        sw.assert_not_called()
        self.assertEqual(out, "Done.")


if __name__ == "__main__":
    unittest.main()
