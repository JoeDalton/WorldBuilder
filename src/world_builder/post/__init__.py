"""
From a drawn view to the screen: render scale, MSAA, FXAA and AMD's FSR 1
(:class:`PostChain`), set by the ``render`` section of
:class:`~world_builder.settings.WorldSettings` (:class:`RenderSettings`).

FSR 1 is spatial: it needs the view's colour only (no motion vectors or
jitter, unlike FSR 2 and later). EASU stretches the view to the output along
its edges; RCAS then sharpens it. Shaders ported from AMD's ``ffx_fsr1.h``
(MIT; its notice is in each).
"""

from world_builder.post.chain import PostChain, target_texture
from world_builder.post.settings import (
    BILINEAR,
    FSR1,
    FSR_MODES,
    MIN_SCALE,
    RenderSettings,
)

__all__ = [
    "BILINEAR",
    "FSR1",
    "FSR_MODES",
    "MIN_SCALE",
    "PostChain",
    "RenderSettings",
    "target_texture",
]
