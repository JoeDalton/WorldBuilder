"""
cloud.py — The cloud types and what they cost (CPU only).

There is no "cloud" object: a cloud type is one density field over a slab of sky
(noise.py), clouds are its features, and a billboard is a marker saying "there is
cloud here, come and sample it". A planet's billboards are placed per
cube-sphere tile by rejection-sampling the very field the shader draws
(sphere.py), so the field alone bounds each cloud and its crisp edge is the
field's edge.

Here: the per-type presets (:data:`PRESETS`), what the clouds cost
(:class:`CloudSettings` and its presets; :func:`with_settings`) and the
billboards' sprite atlas.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from panda3d.core import Filename, PNMImage, Texture, TexturePool

from world_builder.clouds.noise import (
    MAX_LAYERS,
    MIE_ASYMMETRY_ICE,
    MIE_ASYMMETRY_WATER,
    CloudOptics,
    DensityField,
    delta_eddington,
    field_offset,
)
from world_builder.quality import REBUILD, QualityLevel, QualitySettings, knob

#: Most billboard shells a planet's clouds take: one cull bin each between
#: the opaque and transparent bins (``sphere._SHELL_BIN_SORTS``), which the
#: sky and the deck share. Their layers fit ``MAX_LAYERS``.
MAX_SHELLS = 6
assert MAX_SHELLS < MAX_LAYERS


class CloudType(Enum):
    """The preset cloud types: the keys of :data:`PRESETS`."""

    CUMULUS = "cumulus"
    STRATUS = "stratus"
    CIRRUS = "cirrus"
    CUMULONIMBUS = "cumulonimbus"
    #: Cumulus that grow with the weather's convective depth through
    #: congestus to cumulonimbus (:mod:`genera`)
    CONVECTIVE = "convective"


@dataclass(frozen=True)
class CloudSpec:
    """Everything that defines one cloud type.

    field            its :class:`DensityField` — shape; drives placement, so a
                     change needs a rebuild
    optics           its :class:`CloudOptics` — shading only, changeable live
    radius           (min, max) billboard radius, metres; match the finest octave
    aspect           billboard long:short ratio. Area-preserving (sqrt(a) by
                     1/sqrt(a)), so extinction and count are untouched; the long
                     axis lies level
    deep_radius      a convective field's billboards' radius at depth 1, over
                     their radius at depth 0 (it grows linearly between): a
                     cumulonimbus is drawn by as many billboards per kilometre
                     of its features, not per cubic metre (the overlap holds,
                     so the extinction does)
    """

    field: DensityField
    optics: CloudOptics = CloudOptics()
    radius: tuple = (70.0, 140.0)
    aspect: float = 1.0
    deep_radius: float = 1.0


PRESETS = {
    CloudType.CUMULUS: CloudSpec(
        # 2 km isotropic features over a 1 km slab; scattered fair-weather puffs,
        # well below the percolation point.
        field=DensityField(
            slab=(1000.0, 1000.0),
            noise_scale=(0.002, 0.002, 0.002),
            coverage=0.3,
            edge_softness=1.3,
            base_ramp=6.0,
            offset=field_offset(0),
        ),
        optics=CloudOptics(
            shadow_strength=0.3,
            # 8x stays clear of the tonemap's shoulder; the reference's 24x
            # washes a dusk deck into one flat colour.
            forward_gain=8.0,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_WATER),
        ),
    ),
    CloudType.CONVECTIVE: CloudSpec(
        # The cumulus field, its clouds' top rising with the depth from 1 km
        # to the cap, 8.3 km above the base: congestus at 4-6 km, the
        # cumulonimbus's anvil spreading under the cap, a smooth lens reaching
        # past the tower. A cell octave (8 km features) gathers the deep ones
        # into towers.
        field=DensityField(
            slab=(2200.0, 8300.0),
            noise_scale=(0.002, 0.002, 0.002),
            coverage=0.3,
            edge_softness=1.3,
            base_ramp=6.0,
            offset=field_offset(0),
            min_top=1000.0 / 8300.0,
            anvil=0.2,
            cell_weight=0.5,
        ),
        optics=CloudOptics(
            shadow_strength=0.3,
            forward_gain=8.0,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_WATER),
        ),
        deep_radius=2.5,
    ),
    CloudType.STRATUS: CloudSpec(
        # Overcast: a sheet with holes, soft-edged, flat underneath. Its top
        # undulates under the slab's (a strong erosion): cut flat there, each
        # billboard shows the cut as a straight edge, and the deck seen from
        # above turns to blocks.
        field=DensityField(
            slab=(600.0, 500.0),
            density=0.05,
            noise_scale=(0.0015, 0.0015, 0.006),
            coverage=0.25,
            edge_softness=2,
            base_ramp=8.0,
            top_erosion=0.35,
            top_exponent=2.0,
            offset=field_offset(1),
        ),
        # Optically thickest: phase washed out, no silver lining to speak of.
        optics=CloudOptics(
            sun_steps=3,
            sun_step_length=90.0,
            multiple_scattering=0.10,
            powder_length=150.0,
            max_optical_depth=1.8,
            forward_gain=2.2,
            backward_gain=1.3,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_WATER),
        ),
        radius=(90.0, 180.0),
    ),
    CloudType.CIRRUS: CloudSpec(
        # Anisotropy IS the fibrous look: 10 km along X, 1.6 km along Y, 130 m in Z.
        field=DensityField(
            slab=(8000.0, 400.0),
            density=0.008,
            noise_scale=(0.0004, 0.0025, 0.03),
            # A veil over most of the sky that hides almost none of it.
            coverage=0.81,
            edge_softness=0.76,
            base_ramp=3.0,
            top_erosion=0.10,
            top_exponent=2.0,
            offset=field_offset(2),
        ),
        # Thin ice cloud: near single scattering, so it keeps most of its raw
        # anisotropy and lights up strongly around the sun.
        optics=CloudOptics(
            sun_steps=2,
            sun_step_length=120.0,
            shadow_strength=0.6,
            multiple_scattering=0.45,
            powder_strength=0.25,
            powder_length=300.0,
            max_optical_depth=0.8,
            forward_gain=12.0,
            backward_gain=1.1,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_ICE, applicability=0.15),
        ),
        radius=(90.0, 200.0),
        # Gentler than the field's 6.25:1, or the anisotropy is double-counted.
        aspect=3.5,
    ),
    CloudType.CUMULONIMBUS: CloudSpec(
        # Few isolated towers; low erosion over a deep slab flattens into an anvil.
        field=DensityField(
            slab=(800.0, 2400.0),
            density=0.04,
            noise_scale=(0.0012, 0.0012, 0.0006),
            coverage=0.15,
            edge_softness=0.38,
            base_ramp=6.0,
            top_erosion=0.08,
            top_exponent=4.0,
            offset=field_offset(3),
        ),
        # A long march to reach the near-black base of a storm tower.
        optics=CloudOptics(
            sun_steps=6,
            sun_step_length=320.0,
            multiple_scattering=0.05,
            powder_length=250.0,
            max_optical_depth=2.0,
            forward_gain=2.6,
            backward_gain=1.4,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_WATER),
        ),
        radius=(90.0, 180.0),
    ),
}


# ── Quality ────────────────────────────────────────────────────────────────────


#: A planet's noise octaves per shell (the last repeats, and goes to the
#: deck), cheapest first: :attr:`CloudSettings.shell_octaves`' choices. The
#: finest octave (~125 m features for cumulus) is still ~6 px at shell 1's far
#: edge (27 km) at 1600x900, ~3 px at shell 2's (54 km), so dropping it is
#: only invisible from shell 3 on.
_OCTAVE_SCHEDULES = ((4, 3, 3, 2), (4, 4, 3, 3), (4, 4, 4, 3), (4,))


@dataclass(frozen=True)
class CloudSettings(QualitySettings):
    """What the clouds cost (:mod:`world_builder.quality`), for a planet's
    (:class:`sphere.CloudVolume`, and :class:`cirrus.CloudSheet` above it).

    :ivar renderer: how the low clouds are drawn: ``"billboards"``
        (:class:`sphere.CloudVolume`) or ``"march"``, a raymarch at a
        fraction of the view's resolution (:class:`march.CloudMarch`, which
        needs the split cameras). The knobs of the other renderer are
        ignored. Rebuild
    :ivar march_steps: most steps of the raymarch per pixel (it stops
        sooner once the cloud is opaque). Live
    :ivar march_resolution: the raymarch's resolution over the view's, per
        axis. Live
    :ivar interior_growth: billboards deep inside a cloud grow, fewer of
        them (:func:`sphere.interior_levels`). Rebuild
    :ivar count_scale: billboards relative to the types as authored: their
        sizes go as count^(-1/3), so coverage and opacity hold
        (:func:`scale_count`). Rebuild
    :ivar march_scale: sun-march steps relative to the types', over the same
        reach (:func:`scale_march`). Live
    :ivar overlap_near: billboard overlap in the nearest shell
        (volume per unit of cloud volume; the cost, not the opacity). Rebuild
    :ivar overlap_far: in the shells past it. Rebuild
    :ivar shell_octaves: noise octaves per shell, the last
        repeating (and going to the deck). Rebuild
    :ivar shells: billboard shells before the deck takes
        over, each twice as far as the last. Rebuild
    :ivar reflected: drawn in the ocean's reflection. Live
    :ivar half_resolution: drawn at half the view's resolution and laid over
        it (``clouds.half_res``; World, given the split cameras, and the
        demo draw that pass). Live
    :ivar uploads_per_frame: tiles uploaded per frame. Live
    :ivar resorts_per_frame: tiles re-sorted per frame. Live
    :ivar tile_cache: tiles kept beyond those wanted. Live
    """

    renderer: str = knob(
        "billboards",
        "Renderer",
        REBUILD,
        choices={"Billboards": "billboards", "Raymarch": "march"},
        help="Billboards, or a raymarch at a fraction of the resolution "
        "(experimental).",
        kept=True,
    )
    march_steps: int = knob(
        64,
        "Raymarch steps",
        unit="steps",
        range=(8, 256),
        help="Most samples per pixel of the raymarch: its smoothness along "
        "the view (raymarch only).",
    )
    march_resolution: float = knob(
        0.5,
        "Raymarch resolution",
        choices={"Quarter": 0.25, "Half": 0.5, "Full": 1.0},
        help="The raymarch's resolution, per axis, over the view's (raymarch only).",
    )
    interior_growth: bool = knob(
        False,
        "Larger billboards inside",
        REBUILD,
        advanced=True,
        help="Draw the inside of big clouds with fewer, larger billboards: "
        "some 15 % less work near them, softer detail.",
    )
    count_scale: float = knob(
        1.0,
        "Billboards",
        REBUILD,
        unit="x",
        range=(0.1, 4.0),
        help="How many billboards draw the clouds (smaller as there are more): "
        "the clouds' main cost, and their smoothness.",
    )
    march_scale: float = knob(
        1.0,
        "Lighting",
        unit="x",
        range=(0.25, 2.0),
        help="Steps of each billboard's march toward the sun: the clouds' "
        "self-shadowing.",
    )
    overlap_near: float = knob(
        2.5,
        "Overlap, near",
        REBUILD,
        range=(0.5, 4.0),
        advanced=True,
        help="How much the nearest billboards overlap: smoothness up close.",
    )
    overlap_far: float = knob(
        2.5,
        "Overlap, far",
        REBUILD,
        range=(0.5, 4.0),
        advanced=True,
        help="The same, farther out.",
    )
    shell_octaves: tuple[int, ...] = knob(
        (4, 4, 4, 3),
        "Detail",
        REBUILD,
        choices={"-".join(map(str, v)): v for v in _OCTAVE_SCHEDULES},
        help="Noise octaves near to far: the clouds' fine detail in the distance.",
    )
    shells: int = knob(
        4,
        "Billboard distance",
        REBUILD,
        unit="shells",
        range=(1, MAX_SHELLS),
        advanced=True,
        help="Billboard shells, each twice as far as the last, before the "
        "deck takes over.",
    )
    reflected: bool = knob(
        True,
        "In reflections",
        help="Draw the clouds in the sea's reflection (about 1-1.7 ms).",
    )
    half_resolution: bool = knob(
        False,
        "Half-resolution clouds",
        help="Draw the clouds at half the view's resolution: about half their "
        "cost low over the planet, softer cloud edges against the ground.",
    )
    uploads_per_frame: int = knob(
        3, "Tile uploads per frame", range=(1, 32), advanced=True
    )
    resorts_per_frame: int = knob(
        4, "Tile re-sorts per frame", range=(1, 32), advanced=True
    )
    tile_cache: int = knob(
        160, "Tiles kept", unit="tiles", range=(0, 2000), advanced=True
    )

    LABEL: ClassVar = "Clouds"
    # The count is where the frame time is and costs no opacity; the march
    # carries the clouds' form, so it is stepped far more gently. Overlap
    # (a volume fraction) does not change opacity either (the extinction is
    # divided by it), only the cost; the near shell keeps the most, where
    # overlap shows as smoothness.
    PRESETS: ClassVar = {
        QualityLevel.LOW: {
            "count_scale": 0.25,
            "march_scale": 4.0 / 9.0,
            "overlap_near": 1.5,
            "overlap_far": 1.0,
            "shell_octaves": (4, 3, 3, 2),
            "reflected": False,
            "march_steps": 24,
            "march_resolution": 0.25,
            "interior_growth": True,
            "half_resolution": True,
        },
        QualityLevel.MID: {
            "count_scale": 0.5,
            "march_scale": 2.0 / 3.0,
            "overlap_near": 2.0,
            "overlap_far": 1.5,
            "shell_octaves": (4, 4, 3, 3),
            "march_steps": 40,
            "interior_growth": True,
            "half_resolution": True,
        },
        QualityLevel.ULTRA: {
            "count_scale": 2.0,
            "march_scale": 1.5,
            "shell_octaves": (4,),
            "march_steps": 96,
        },
    }


def scale_count(spec: CloudSpec, count: float) -> CloudSpec:
    """A cloud type with *count* times its billboards, as large in all: their
    radii go as count^(-1/3), not the volume fraction (at a fixed fraction
    the count falls as radius⁻³, the trade each LOD shell makes; cutting the
    fraction instead strips billboards from the fringes and loses
    silhouette: measured -18% vs -8.8% at LOW).

    :returns: a scaled copy (*spec* itself at 1)
    """
    if count == 1.0:
        return spec
    return replace(spec, radius=tuple(r * count ** (-1.0 / 3.0) for r in spec.radius))


def scale_march(spec: CloudSpec, march: float) -> CloudSpec:
    """A cloud type with *march* times its sun-march steps over the same
    REACH (steps x length): a coarser sampling of the same path approximates
    Beer-Lambert well; a shorter path does not.

    :returns: a scaled copy (*spec* itself at 1)
    """
    if march == 1.0:
        return spec
    optics = spec.optics
    # int(x + 0.5), not round(): half-to-even would make a 3-step type lose a step
    # going UP to ULTRA.
    steps = max(1, int(optics.sun_steps * march + 0.5))
    reach = optics.sun_steps * optics.sun_step_length
    return replace(
        spec, optics=replace(optics, sun_steps=steps, sun_step_length=reach / steps)
    )


def with_settings(spec: CloudSpec, settings: CloudSettings) -> CloudSpec:
    """A cloud type at the settings' cost (:func:`scale_count`,
    :func:`scale_march`)."""
    return scale_march(scale_count(spec, settings.count_scale), settings.march_scale)


# ── Sprite atlas ──────────────────────────────────────────────────────────────

_ASSET_DIR = Path(__file__).parent
ATLAS_PNG = _ASSET_DIR / "cloud_atlas.png"
ATLAS_JSON = _ASSET_DIR / "cloud_atlas.json"


def load_cloud_atlas(game: Any) -> tuple[Texture, list[tuple]]:
    """Load the cloud sprite atlas, linearly filtered and clamped at its edges
    (so a sprite neither wraps nor bleeds into its atlas neighbours).

    Through ``game.app.asset_manager`` when the game has one, so a game can
    preload the atlas with its other assets; straight from disk otherwise.

    The sprites are a per-quad THICKNESS profile, not a silhouette: their alpha
    scales how much volume a fragment stands for, while the density field decides
    where the cloud ends.

    :param game: the game object, or None to load from disk
    :returns: (Texture, rects) with rects a list of (u, v, du, dv)
    """
    asset_manager = getattr(getattr(game, "app", None), "asset_manager", None)
    if asset_manager is not None:
        tex = asset_manager.get_asset(
            asset_type="texture", path=ATLAS_PNG
        ).get_texture()
    else:
        tex = TexturePool.load_texture(Filename.from_os_specific(str(ATLAS_PNG)))
    tex.set_magfilter(Texture.FT_linear)
    tex.set_minfilter(Texture.FT_linear)
    tex.set_wrap_u(Texture.WM_clamp)
    tex.set_wrap_v(Texture.WM_clamp)
    return tex, atlas_rects()


def atlas_rects() -> list[tuple]:
    """The atlas rects, read from the JSON descriptor with no GL: one dict per
    sprite with u_min, v_min, u_size, v_size, in 0-1 UV space already flipped
    for OpenGL's bottom-left origin.

    :returns: a list of (u, v, du, dv) tuples, one per sprite
    """
    with open(ATLAS_JSON) as f:
        return [
            (r["u_min"], r["v_min"], r["u_size"], r["v_size"]) for r in json.load(f)
        ]


@functools.cache
def atlas_mean_alpha() -> float:
    """Mean sprite alpha over the atlas: the fraction of a billboard's disc that
    carries thickness, which the extinction must be divided by or every cloud comes
    out proportionally see-through. Read from the PNG, so it works headless.

    :raises OSError: if the atlas PNG cannot be read
    """
    image = PNMImage()
    if not image.read(Filename.from_os_specific(str(ATLAS_PNG))):
        raise OSError(f"could not read the cloud atlas at {ATLAS_PNG}")
    if not image.has_alpha():
        return 1.0
    width, height = image.get_x_size(), image.get_y_size()
    # The alpha as an array (a loop of get_alpha calls took ~80 ms): a texture
    # holds it bottom-up, PNMImage rows are top-down.
    texture = Texture()
    texture.load(image)
    alpha = np.frombuffer(texture.get_ram_image_as("A").get_data(), dtype=np.uint8)
    alpha = alpha.reshape(height, width)[::-1] / float(image.get_maxval())
    totals = []
    for u_min, v_min, u_size, v_size in atlas_rects():
        # UVs are bottom-left origin; PNMImage rows are top-down.
        x0 = int(round(u_min * width))
        x1 = max(x0 + 1, int(round((u_min + u_size) * width)))
        y0 = int(round((1.0 - v_min - v_size) * height))
        y1 = max(y0 + 1, int(round((1.0 - v_min) * height)))
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x1 > x0 and y1 > y0:
            totals.append(float(alpha[y0:y1, x0:x1].mean()))
    return float(np.mean(totals)) if totals else 1.0
