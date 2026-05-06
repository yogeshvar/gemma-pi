"""Tests for sync_ollama_model_from_server."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from config import Settings
from core.llm import sync_ollama_model_from_server


def _list_resp(tags: list[str]) -> MagicMock:
    models = [MagicMock(model=t) for t in tags]
    r = MagicMock()
    r.models = models
    return r


class OllamaSyncTests(unittest.TestCase):
    def test_uses_configured_when_installed(self) -> None:
        settings = Settings(ollama_model="beta:latest")
        mock = MagicMock()
        mock.list.return_value = _list_resp(["alpha:1", "beta:latest", "gamma"])
        with patch("core.llm._client", return_value=mock):
            sync_ollama_model_from_server(settings)
        self.assertEqual(settings.ollama_model, "beta:latest")

    def test_empty_config_picks_first_sorted(self) -> None:
        settings = Settings(ollama_model="")
        mock = MagicMock()
        mock.list.return_value = _list_resp(["zoo", "alpha", "mid"])
        with patch("core.llm._client", return_value=mock):
            sync_ollama_model_from_server(settings)
        self.assertEqual(settings.ollama_model, "alpha")

    def test_missing_config_falls_back_to_first_sorted(self) -> None:
        settings = Settings(ollama_model="not-there:7b")
        mock = MagicMock()
        mock.list.return_value = _list_resp(["z:1", "a:2"])
        with patch("core.llm._client", return_value=mock):
            sync_ollama_model_from_server(settings)
        self.assertEqual(settings.ollama_model, "a:2")

    def test_list_failure_leaves_model(self) -> None:
        settings = Settings(ollama_model="unchanged")
        mock = MagicMock()
        mock.list.side_effect = ConnectionError("refused")
        with patch("core.llm._client", return_value=mock):
            sync_ollama_model_from_server(settings)
        self.assertEqual(settings.ollama_model, "unchanged")


if __name__ == "__main__":
    unittest.main()
