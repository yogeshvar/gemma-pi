"""Tests for prompt merging including prompts/web/."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from config import Settings
from core.prompts import build_system_message, load_web_prompts


class WebPromptMergeTests(unittest.TestCase):
    def test_web_prompts_only_when_enabled(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "01_base.md").write_text("BASEMARKER", encoding="utf-8")
            web = root / "web"
            web.mkdir()
            (web / "01_tool.md").write_text("WEBRISK_UNIQUE_STRING", encoding="utf-8")

            off = Settings(prompt_dir=root, web_search_enabled=False)
            self.assertEqual(load_web_prompts(off), "")
            msg_off = build_system_message(off)
            self.assertIn("BASEMARKER", msg_off)
            self.assertNotIn("WEBRISK_UNIQUE_STRING", msg_off)

            on = Settings(prompt_dir=root, web_search_enabled=True)
            msg_on = build_system_message(on)
            self.assertIn("BASEMARKER", msg_on)
            self.assertIn("WEBRISK_UNIQUE_STRING", msg_on)

    def test_system_prompt_suffix_is_last(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "01_base.md").write_text("BASE", encoding="utf-8")
            web = root / "web"
            web.mkdir()
            (web / "01_w.md").write_text("WEB", encoding="utf-8")
            s = Settings(
                prompt_dir=root,
                web_search_enabled=True,
                system_prompt="SUFFIXLAST",
            )
            msg = build_system_message(s)
            self.assertLess(msg.index("WEB"), msg.index("SUFFIXLAST"))


if __name__ == "__main__":
    unittest.main()
