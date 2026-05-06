"""Tests for core.web_search (HTTP mocked)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from config import Settings
from core.web_search import search_web


class BraveSearchTests(unittest.TestCase):
    def test_brave_formats_results(self) -> None:
        settings = Settings(
            web_search_provider="brave",
            brave_api_key="test-token",
            web_search_max_results=2,
            web_search_max_chars=5000,
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "T1",
                        "description": "D1",
                        "url": "https://a.example",
                    },
                    {
                        "title": "T2",
                        "description": "D2",
                        "url": "https://b.example",
                    },
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("core.web_search.requests.get", return_value=mock_resp) as g:
            out = search_web(settings, "hello world")
        g.assert_called_once()
        self.assertIn("T1", out)
        self.assertIn("https://a.example", out)
        self.assertIn("2.", out)

    def test_brave_missing_key(self) -> None:
        settings = Settings(web_search_provider="brave", brave_api_key="")
        out = search_web(settings, "q")
        self.assertIn("BRAVE_API_KEY", out)


class TavilySearchTests(unittest.TestCase):
    def test_tavily_formats_results(self) -> None:
        settings = Settings(
            web_search_provider="tavily",
            tavily_api_key="tv-key",
            web_search_max_results=1,
            web_search_max_chars=5000,
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"title": "News", "content": "Body text", "url": "https://news.test/x"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("core.web_search.requests.post", return_value=mock_resp) as p:
            out = search_web(settings, "query")
        p.assert_called_once()
        self.assertIn("News", out)
        self.assertIn("Body text", out)

    def test_empty_query(self) -> None:
        settings = Settings()
        self.assertIn("empty", search_web(settings, "  ").lower())


class DDGSSearchTests(unittest.TestCase):
    def test_ddgs_uses_package(self) -> None:
        settings = Settings(
            web_search_provider="ddgs",
            web_search_max_results=1,
            web_search_max_chars=2000,
        )
        fake_hit = {"title": "H", "body": "B", "href": "https://z"}
        inner = MagicMock()
        inner.text.return_value = [fake_hit]
        factory = MagicMock(return_value=inner)
        inner.__enter__ = MagicMock(return_value=inner)
        inner.__exit__ = MagicMock(return_value=False)

        with patch("duckduckgo_search.DDGS", factory):
            out = search_web(settings, "anything")
        self.assertIn("H", out)
        self.assertIn("https://z", out)


if __name__ == "__main__":
    unittest.main()
