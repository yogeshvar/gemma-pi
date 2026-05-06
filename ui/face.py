"""BMO-style face: mint field, black dot eyes, capsule mouth with teeth (user ref)."""

from __future__ import annotations

from typing import Sequence

import pygame

from core.assistant import AssistantState, UIFrame

from .animations import BlinkController, breathing_scale, lerp
from .colors import palette_for


class FaceView:
    def __init__(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self._blink = BlinkController()
        pygame.font.init()
        self._font_title = self._load_font(40)
        self._font_sub = self._load_font(20)

    @staticmethod
    def _load_font(px: int) -> pygame.font.Font:
        for name in ("arial rounded mt bold", "nunito", "arial", "helvetica"):
            try:
                return pygame.font.SysFont(name, px, bold=True)
            except Exception:
                continue
        return pygame.font.Font(None, px)

    def draw(self, surface: pygame.Surface, frame: UIFrame, now: float) -> None:
        pal = palette_for(frame.state)
        surface.fill(pal.face_bg)

        breath = breathing_scale(now)
        open_scale = self._blink.update(now)

        cx = self.w // 2
        eye_y = int(self.h * 0.30)
        # Wide-set black dots (reference proportions)
        spread = int(self.w * 0.19)
        base_r = int(min(self.w, self.h) * 0.055 * breath)
        ry = max(2, int(base_r * open_scale))
        self._draw_eye_dots(surface, cx - spread, eye_y, base_r, ry)
        self._draw_eye_dots(surface, cx + spread, eye_y, base_r, ry)

        mouth_amp = self._mouth_amplitude(frame)
        self._draw_bmo_mouth(surface, cx, int(self.h * 0.54), mouth_amp, pal)

        self._draw_waveform(surface, frame.waveform, pal.accent)
        self._draw_status(surface, frame, pal.text)

    def _draw_eye_dots(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        rx: int,
        ry: int,
    ) -> None:
        if ry < rx * 0.35:
            # Blink: thin line
            pygame.draw.line(
                surface,
                (12, 18, 16),
                (cx - rx, cy),
                (cx + rx, cy),
                max(3, rx // 4),
            )
        else:
            pygame.draw.ellipse(surface, (12, 18, 16), pygame.Rect(cx - rx, cy - ry, rx * 2, ry * 2))

    def _draw_bmo_mouth(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        amp: float,
        pal,
    ) -> None:
        """Capsule mouth: thick outline, white teeth bar, dark green interior."""
        amp = max(0.0, min(1.0, amp))
        w = int(lerp(100, 200, amp))
        h = int(lerp(10, 52, amp))
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, max(8, h))
        r = max(4, h // 2)

        # Interior (inset so outline stays visible)
        inset = 5
        inner = rect.inflate(-inset * 2, -inset * 2)
        if inner.height >= 10:
            pygame.draw.rect(surface, pal.mouth_dark, inner, border_radius=r - 3)
            teeth_h = max(4, int(inner.height * 0.28))
            teeth_rect = pygame.Rect(inner.left, inner.top, inner.width, teeth_h)
            pygame.draw.rect(surface, pal.teeth, teeth_rect, border_radius=max(2, teeth_h // 3))
        else:
            # Nearly closed: thin dark slot
            pygame.draw.rect(surface, pal.mouth_outline, inner, border_radius=r - 2)

        pygame.draw.rect(surface, pal.mouth_outline, rect, width=5, border_radius=r)

    def _mouth_amplitude(self, frame: UIFrame) -> float:
        if frame.state != AssistantState.SPEAKING or not frame.envelope:
            if frame.state == AssistantState.SPEAKING:
                return 0.12
            if frame.state == AssistantState.LISTENING:
                return 0.06
            if frame.state == AssistantState.ERROR:
                return 0.03
            return 0.04
        idx = int(frame.speak_progress * (len(frame.envelope) - 1))
        return float(frame.envelope[idx])

    def _draw_waveform(
        self, surface: pygame.Surface, samples: Sequence[float], color: tuple[int, int, int]
    ) -> None:
        if not samples:
            return
        base_y = int(self.h * 0.72)
        bar_w = max(3, self.w // (len(samples) * 2))
        x0 = (self.w - (len(samples) * bar_w * 2)) // 2
        for i, v in enumerate(samples):
            h = int(6 + v * 40)
            x = x0 + i * bar_w * 2
            pygame.draw.rect(surface, color, pygame.Rect(x, base_y - h, bar_w, h), border_radius=2)

    def _draw_status(
        self,
        surface: pygame.Surface,
        frame: UIFrame,
        text_rgb: tuple[int, int, int],
    ) -> None:
        margin = 20
        title = self._font_title.render(frame.caption, True, text_rgb)
        tw = max(1, min(title.get_width(), self.w - margin * 2))
        surf_title = pygame.transform.smoothscale(title, (tw, title.get_height()))
        surface.blit(surf_title, (margin, self.h - int(self.h * 0.20)))

        sub_text = frame.subtitle
        if sub_text:
            sub = self._font_sub.render(sub_text, True, text_rgb)
            sw = max(1, min(sub.get_width(), self.w - margin * 2))
            surf_sub = pygame.transform.smoothscale(sub, (sw, sub.get_height()))
            surface.blit(surf_sub, (margin, self.h - int(self.h * 0.11)))
