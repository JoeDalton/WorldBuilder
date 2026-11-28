"""
The ``render`` section of :class:`~world_builder.settings.WorldSettings`:
how many pixels the view is drawn at and how it reaches the screen (render
scale, MSAA, FXAA, the upscaler; :mod:`world_builder.post`).

Choices, not quality levels: no presets of its own (as the display's), every
knob kept when a preset is picked. The default draws the view at the
output's size with no anti-aliasing, as before the section existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from world_builder.quality import LIVE, REBUILD, QualitySettings, knob

#: ``RenderSettings.upscaler``: a bilinear stretch (or a plain copy at scale 1).
BILINEAR = "bilinear"
#: ``RenderSettings.upscaler``: AMD FidelityFX Super Resolution 1 (EASU, then
#: RCAS's sharpening).
FSR1 = "fsr1"

#: AMD's FSR 1 quality modes: their render scale per axis (1.3x, 1.5x, 1.7x
#: and 2x upscales).
FSR_MODES: dict[str, float] = {
    "Ultra quality": 0.77,
    "Quality": 0.67,
    "Balanced": 0.59,
    "Performance": 0.5,
}

#: Ultra performance: a 3x upscale per axis (FSR 2 and later's ratio). Past
#: FSR 1's design limit (2x), for hardware that cannot draw more.
ULTRA_PERFORMANCE = 0.33

#: The render scales a menu offers, best first: native, AMD's FSR 1 modes,
#: then ultra performance. Labels as a menu shows them.
SCALE_PRESETS: dict[str, float] = {
    "Native": 1.0,
    **FSR_MODES,
    "Ultra performance": ULTRA_PERFORMANCE,
}

#: The least render scale: ultra performance's (FSR 1 is made for upscales up
#: to 2x per axis, 0.5; below it the image softens further).
MIN_SCALE = ULTRA_PERFORMANCE


@dataclass(frozen=True)
class RenderSettings(QualitySettings):
    """How the view is drawn and reaches the screen (see the module).

    :ivar scale: the view's size over its output's, per axis
    :ivar msaa: the view's multisamples; 0 = the window's own (none by
        default)
    :ivar fxaa: FXAA over the view, at its size (before an upscale)
    :ivar upscaler: :data:`BILINEAR` or :data:`FSR1`
    :ivar sharpness: FSR 1's sharpening (RCAS), in stops: 0 the most, each
        +1 halves it
    """

    scale: float = knob(
        1.0,
        "Render scale",
        REBUILD,
        range=(MIN_SCALE, 1.0),
        kept=True,
        help="The view's resolution over the window's, per axis: fewer pixels "
        "drawn (the clouds' march follows it), upscaled to the window. "
        "The HUD stays at the window's. FSR 1 is made for 0.5 and up.",
    )
    msaa: int = knob(
        0,
        "MSAA",
        REBUILD,
        choices={"Off": 0, "2x": 2, "4x": 4, "8x": 8, "16x": 16},
        kept=True,
        help="Multisampled edges: costs memory and fill with the samples. "
        "The driver may grant fewer.",
    )
    fxaa: bool = knob(
        False,
        "FXAA",
        REBUILD,
        kept=True,
        help="A cheap edge blur over the view. Forced under FSR 1 without "
        "MSAA: FSR 1 wants anti-aliased input.",
    )
    upscaler: str = knob(
        BILINEAR,
        "Upscaler",
        REBUILD,
        choices={"Bilinear": BILINEAR, "FSR 1": FSR1},
        kept=True,
        help="How the view is stretched to the window: bilinear (soft), or "
        "AMD's FSR 1 (edge-adaptive, then sharpened).",
    )
    sharpness: float = knob(
        0.1,
        "FSR sharpness",
        LIVE,
        unit="stops",
        range=(0.0, 2.0),
        kept=True,
        help="FSR 1's sharpening: 0 the strongest, each +1 halves it.",
    )

    LABEL: ClassVar = "Render"
    PRESETS: ClassVar = {}  # choices, the same at every level
    LEVELLED: ClassVar = False

    @property
    def fsr(self) -> bool:
        """Whether FSR 1 upscales the view."""
        return self.upscaler == FSR1

    @property
    def fxaa_forced(self) -> bool:
        """Whether FXAA runs although :attr:`fxaa` is off: FSR 1 wants an
        anti-aliased input, and FXAA is its floor without MSAA."""
        return self.fsr and self.msaa == 0

    @property
    def fxaa_active(self) -> bool:
        """Whether the FXAA pass runs."""
        return self.fxaa or self.fxaa_forced

    @property
    def needs_chain(self) -> bool:
        """Whether the view needs a post chain: anything but a plain draw at
        the output's size."""
        return self.scale < 1.0 or self.msaa > 0 or self.fxaa or self.fsr

    def render_size(self, output: tuple[int, int]) -> tuple[int, int]:
        """:returns: the view's size for an *output* of (width, height)
        pixels: :attr:`scale` of it, at least a pixel"""
        return (
            max(1, round(output[0] * self.scale)),
            max(1, round(output[1] * self.scale)),
        )
