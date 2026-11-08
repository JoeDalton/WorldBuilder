"""
How frames reach the screen: a frame-rate cap (:class:`FrameCap`), the
``display`` section of :class:`~world_builder.settings.WorldSettings`.

Why a cap: a desktop compositor shows a window's frames on its display's
refresh (vsync or not). A frame that takes a little more than one refresh
(6.9 ms against a 180 Hz display's 5.5) waits for the next: most frames show
for one refresh, every third or fourth for two, and turning the view
stutters although the mean frame rate is fine (performance.md §27). Capped
at a divisor of the refresh the frame rate can hold (90 fps on 180 Hz),
every frame shows for the same time. 0 turns the cap off (benchmarks, a
variable-refresh display, a fullscreen window the compositor leaves alone).

The cap is Panda3D's global clock in its limited mode (``ClockObject``'s
``M_limited``): the frame loop waits out the rest of each frame. It is
process-wide; the last :meth:`FrameCap.set_settings` wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from panda3d.core import ClockObject

from world_builder.quality import QualitySettings, knob


@dataclass(frozen=True)
class DisplaySettings(QualitySettings):
    """How frames reach the screen (see the module).

    :ivar frame_cap: the most frames a second; 0 = no cap
    """

    frame_cap: int = knob(
        90,
        "Frame cap",
        unit="fps",
        range=(0, 480),
        kept=True,
        help="The most frames a second, 0 for none. A divisor of the "
        "display's refresh the game can hold (90 on 180 Hz, 72 on 144 Hz) "
        "shows every frame for the same time: smooth turns. Off, a frame "
        "that misses a refresh is held for two: stutter.",
    )

    LABEL: ClassVar = "Display"
    PRESETS: ClassVar = {}  # a choice, the same at every level
    LEVELLED: ClassVar = False


class FrameCap:
    """Applies :class:`DisplaySettings` to Panda3D's global clock (see the
    module): a game builds one (``World`` does) and hands it to
    ``WorldSettings.apply``."""

    SETTINGS_TYPE: ClassVar = DisplaySettings

    def __init__(self, settings: DisplaySettings | None = None, clock: Any = None):
        """
        :param settings: the cap; None = the default (90 fps)
        :param clock: the clock to cap; None = Panda3D's global clock
        """
        self._clock = clock or ClockObject.get_global_clock()
        self.settings = settings or DisplaySettings()
        self.set_settings(self.settings)

    def set_settings(self, settings: DisplaySettings) -> None:
        """Cap the frame rate (live), or not (``frame_cap`` 0)."""
        self.settings = settings
        if settings.frame_cap > 0:
            self._clock.set_mode(ClockObject.M_limited)
            self._clock.set_frame_rate(float(settings.frame_cap))
        else:
            self._clock.set_mode(ClockObject.M_normal)
