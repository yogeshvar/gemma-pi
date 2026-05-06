#!/usr/bin/env python3
"""Pi Assistant — entry point (Pygame UI or headless CLI loop)."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from rich.logging import RichHandler

from config import get_settings
from core.audio_io import verify_audio_devices


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def run_cli() -> None:
    from core.assistant import AssistantController, AssistantState

    settings = get_settings()
    ctrl = AssistantController(settings)
    logging.getLogger(__name__).info(
        "CLI mode — type Enter to talk, f=forget last exchange, q=quit (Ctrl+C exits)."
    )
    try:
        while True:
            line = input("Press Enter to talk | f | q: ").strip().lower()
            if line == "q":
                break
            if line == "f":
                if ctrl.state == AssistantState.IDLE:
                    ctrl.forget_last_exchange()
                    print("Forgot last exchange.")
                continue
            ctrl.on_enter_from_idle()
            while ctrl.state != AssistantState.IDLE:
                snap = ctrl.ui_snapshot()
                tail = (snap.subtitle or "")[:72]
                print(f"\r{snap.caption}  {tail:<72}", end="", flush=True)
                time.sleep(0.08)
            print()
    finally:
        ctrl.close()


def run_ui() -> None:
    import pygame

    from core.assistant import AssistantController, AssistantState
    from ui.face import FaceView

    settings = get_settings()
    pygame.init()
    flags = pygame.FULLSCREEN if settings.fullscreen else 0
    screen = pygame.display.set_mode(
        (settings.screen_width, settings.screen_height),
        flags,
    )
    pygame.display.set_caption("Pi Assistant")
    clock = pygame.time.Clock()
    face = FaceView(settings.screen_width, settings.screen_height)
    ctrl = AssistantController(settings)

    running = True
    try:
        while running:
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        ctrl.on_enter_from_idle()
                    elif event.key == pygame.K_f and ctrl.state == AssistantState.IDLE:
                        ctrl.forget_last_exchange()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    ctrl.on_tap()

            frame = ctrl.ui_snapshot(now)
            face.draw(screen, frame, now)
            pygame.display.flip()
            clock.tick(60)
    finally:
        ctrl.close()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi Assistant")
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Terminal loop (no Pygame) for SSH / laptop testing",
    )
    args = parser.parse_args()
    settings = get_settings()
    _setup_logging(settings.log_level)
    verify_audio_devices(settings)
    if args.cli:
        run_cli()
    else:
        run_ui()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
