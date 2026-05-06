"""BMO-style face: mint field, black dot eyes, capsule mouth with teeth (user ref)."""

from __future__ import annotations

import math
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
        # Large, readable status on a small touchscreen (e.g. 800×480).
        hero = max(40, min(62, int(min(width, height) * 0.072)))
        sub = max(28, min(46, int(min(width, height) * 0.052)))
        idle_title = max(30, min(48, int(min(width, height) * 0.055)))
        self._font_hero = self._load_font(hero)
        self._font_sub = self._load_font(sub)
        self._font_title = self._load_font(idle_title)

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

        mouth_amp = self._mouth_amplitude(frame, now)
        self._draw_bmo_mouth(surface, cx, int(self.h * 0.54), mouth_amp, pal)

        self._draw_waveform(surface, frame.waveform, pal.accent)
        self._draw_status(surface, frame, pal, now)

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

    def _mouth_amplitude(self, frame: UIFrame, now: float) -> float:
        if frame.state == AssistantState.THINKING:
            # Gentle “processing” mouth motion
            return 0.05 + 0.04 * (0.5 + 0.5 * math.sin(now * 3.2))
        if frame.state != AssistantState.SPEAKING or not frame.envelope:
            if frame.state == AssistantState.SPEAKING:
                return 0.12
            if frame.state == AssistantState.LISTENING:
                return 0.06 + 0.02 * (0.5 + 0.5 * math.sin(now * 5.0))
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

    def _animated_caption(self, frame: UIFrame, now: float) -> str:
        cap = frame.caption
        if frame.state not in (AssistantState.LISTENING, AssistantState.THINKING):
            return cap
        dots = (".", "..", "…", "")
        i = int(now * 1.7) % len(dots)
        return cap.rstrip(".") + dots[i]

    def _wrap_text(self, text: str, font: pygame.font.Font, max_w: int) -> list[str]:
        """Word-wrap without horizontal squish — long tokens are split."""
        words = text.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            test = f"{cur} {w}".strip()
            if font.size(test)[0] <= max_w:
                cur = test
                continue
            if cur:
                lines.append(cur)
            if font.size(w)[0] <= max_w:
                cur = w
                continue
            chunk = ""
            for ch in w:
                t2 = chunk + ch
                if font.size(t2)[0] <= max_w:
                    chunk = t2
                else:
                    if chunk:
                        lines.append(chunk)
                    chunk = ch
            cur = chunk
        if cur:
            lines.append(cur)
        return lines

    def _subtitle_for_frame(self, frame: UIFrame, now: float) -> str:
        if frame.subtitle:
            return frame.subtitle
        if frame.state == AssistantState.THINKING:
            hints = (
                "Transcribing what you said…",
                "Talking to the language model…",
                "Still here — almost ready…",
            )
            return hints[int(now * 0.5) % len(hints)]
        return ""

    def _draw_status(
        self,
        surface: pygame.Surface,
        frame: UIFrame,
        pal,
        now: float,
    ) -> None:
        margin = 16
        panel_w = self.w - margin * 2
        max_w = panel_w - 56
        text_rgb = pal.text
        title_font = (
            self._font_hero
            if frame.state
            in (
                AssistantState.LISTENING,
                AssistantState.THINKING,
                AssistantState.SPEAKING,
                AssistantState.ERROR,
            )
            else self._font_title
        )

        caption = self._animated_caption(frame, now)
        title_surf = title_font.render(caption, True, text_rgb)
        tw = max(1, min(title_surf.get_width(), max_w))
        title_draw = pygame.transform.smoothscale(title_surf, (tw, title_surf.get_height()))

        sub_text = self._subtitle_for_frame(frame, now)
        sub_lines: list[str] = []
        if sub_text:
            wrap_w = panel_w - 28
            sub_lines = self._wrap_text(sub_text, self._font_sub, wrap_w)
            max_lines = 6 if frame.state == AssistantState.ERROR else 4
            sub_lines = sub_lines[:max_lines]

        line_gap = 8
        sub_line_h = self._font_sub.get_linesize()
        desired_h = 22 + title_draw.get_height()
        if sub_lines:
            desired_h += 10 + len(sub_lines) * (sub_line_h + line_gap)
        panel_h = int(min(self.h * 0.52, max(desired_h, 100)))
        panel_top = self.h - panel_h - 10

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((248, 255, 252, 242))
        border_c = (*pal.accent, 85)
        pygame.draw.rect(panel, border_c, panel.get_rect(), width=2, border_radius=14)
        surface.blit(panel, (margin, panel_top))

        tx = margin + 48
        if frame.state == AssistantState.LISTENING:
            pulse = 0.5 + 0.5 * math.sin(now * 6.5)
            r = int(8 + pulse * 6)
            pygame.draw.circle(
                surface,
                (218, 72, 72),
                (margin + 26, panel_top + 18 + title_draw.get_height() // 2),
                r,
            )
        surface.blit(title_draw, (tx, panel_top + 14))

        if sub_lines:
            y = panel_top + 20 + title_draw.get_height()
            for line in sub_lines:
                surf = self._font_sub.render(line, True, text_rgb)
                surface.blit(surf, (margin + 14, y))
                y += sub_line_h + line_gap
