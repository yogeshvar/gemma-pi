"""Ollama think= wiring."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from config import Settings
from core.llm import _chat_with_think_fallback, _think_kw


class ThinkKwTests(unittest.TestCase):
    def test_false_sends_think_false(self) -> None:
        self.assertEqual(_think_kw(Settings(ollama_think=False)), {"think": False})

    def test_true_sends_think_true(self) -> None:
        self.assertEqual(_think_kw(Settings(ollama_think=True)), {"think": True})

    def test_none_omits(self) -> None:
        self.assertEqual(_think_kw(Settings(ollama_think=None)), {})

    def test_fallback_strips_think_on_typeerror(self) -> None:
        settings = Settings(ollama_think=False, ollama_model="m")

        def fake_chat(**kwargs: object) -> str:
            if "think" in kwargs:
                raise TypeError("unexpected keyword argument 'think'")
            return "ok"

        client = MagicMock()
        client.chat = fake_chat
        out = _chat_with_think_fallback(
            client, settings, model="m", messages=[], stream=False
        )
        self.assertEqual(out, "ok")


if __name__ == "__main__":
    unittest.main()
