"""BMO-style palette — mint face, state shifts kept subtle."""

from __future__ import annotations

from dataclasses import dataclass

from core.assistant import AssistantState


@dataclass(frozen=True)
class Palette:
    """Face background + UI chrome that still reads on mint."""

    face_bg: tuple[int, int, int]
    accent: tuple[int, int, int]
    accent2: tuple[int, int, int]
    text: tuple[int, int, int]
    mouth_outline: tuple[int, int, int]
    mouth_dark: tuple[int, int, int]
    teeth: tuple[int, int, int]


def palette_for(state: AssistantState) -> Palette:
    # Reference: light seafoam / mint screen; states nudge hue slightly.
    if state == AssistantState.LISTENING:
        face = (168, 232, 218)
        acc = (20, 100, 88)
    elif state == AssistantState.THINKING:
        face = (188, 228, 198)
        acc = (70, 90, 55)
    elif state == AssistantState.SPEAKING:
        face = (198, 224, 210)
        acc = (90, 55, 75)
    else:
        face = (178, 236, 214)
        acc = (28, 78, 72)
    return Palette(
        face_bg=face,
        accent=acc,
        accent2=acc,
        text=(22, 52, 48),
        mouth_outline=(28, 52, 48),
        mouth_dark=(55, 92, 78),
        teeth=(252, 252, 250),
    )


# Legacy export — face no longer uses dark navy base
BACKGROUND_BASE = (178, 236, 214)
