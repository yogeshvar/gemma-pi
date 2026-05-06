"""Tests for PipeWire-friendly audio resolution and startup check."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from config import Settings
from core.audio_io import (
    resolve_input_device,
    resolve_output_device,
    verify_audio_devices,
)


class ResolveInputTests(unittest.TestCase):
    def test_explicit_string_passthrough(self) -> None:
        s = Settings(input_device="pipewire")
        self.assertEqual(resolve_input_device(s), "pipewire")

    def test_explicit_int_passthrough(self) -> None:
        s = Settings(input_device=2)
        self.assertEqual(resolve_input_device(s), 2)

    @patch("core.audio_io.sd.default")
    def test_none_uses_default_when_valid(self, mock_default: MagicMock) -> None:
        mock_default.device = (3, 3)
        s = Settings(input_device=None)
        self.assertEqual(resolve_input_device(s), 3)

    @patch("core.audio_io.sd.default")
    def test_none_raises_when_default_minus_one(self, mock_default: MagicMock) -> None:
        mock_default.device = (-1, 2)
        s = Settings(input_device=None)
        with self.assertRaises(RuntimeError) as ctx:
            resolve_input_device(s)
        self.assertIn("PI_ASSISTANT_INPUT_DEVICE", str(ctx.exception))


class ResolveOutputTests(unittest.TestCase):
    @patch("core.audio_io.sd.default")
    def test_none_uses_output_default(self, mock_default: MagicMock) -> None:
        mock_default.device = (2, 0)
        s = Settings(output_device=None)
        self.assertEqual(resolve_output_device(s), 0)

    @patch("core.audio_io.sd.default")
    def test_none_raises_when_output_invalid(self, mock_default: MagicMock) -> None:
        mock_default.device = (0, -1)
        s = Settings(output_device=None)
        with self.assertRaises(RuntimeError) as ctx:
            resolve_output_device(s)
        self.assertIn("OUTPUT_DEVICE", str(ctx.exception))


class VerifyAudioTests(unittest.TestCase):
    @patch("core.audio_io.sd.query_devices")
    @patch("core.audio_io.sd.default")
    def test_verify_queries_both(self, mock_default: MagicMock, mock_query: MagicMock) -> None:
        mock_default.device = (1, 2)
        mock_query.side_effect = [
            {"name": "Mic", "default_samplerate": 48000},
            {"name": "Speaker", "default_samplerate": 48000},
        ]
        s = Settings()
        verify_audio_devices(s)
        self.assertEqual(mock_query.call_count, 2)

    @patch("core.audio_io.sd.query_devices")
    def test_verify_wraps_query_failure(self, mock_query: MagicMock) -> None:
        mock_query.side_effect = OSError("boom")
        s = Settings(input_device="pipewire", output_device="pipewire")
        with self.assertRaises(RuntimeError) as ctx:
            verify_audio_devices(s)
        self.assertIn("pipewire", str(ctx.exception))


class SettingsCoercionTests(unittest.TestCase):
    def test_env_style_numeric_string_becomes_int(self) -> None:
        s = Settings.model_validate({"input_device": "0"})
        self.assertEqual(s.input_device, 0)

    def test_empty_string_becomes_none(self) -> None:
        s = Settings.model_validate({"output_device": ""})
        self.assertIsNone(s.output_device)


if __name__ == "__main__":
    unittest.main()
