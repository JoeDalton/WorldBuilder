"""
cloud.py — The cloud types and what they cost (CPU only).

There is no "cloud" object: a cloud type is one density field over a slab of sky
(noise.py), and clouds are its features. The raymarch (march.py) samples that
field along each pixel's ray, so the field alone bounds each cloud and its crisp
edge is the field's edge.

Here: the per-type presets (:data:`PRESETS`) and what the clouds cost
(:class:`CloudSettings` and its presets; :func:`with_settings`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar

from world_builder.clouds.noise import (
    DETAIL_LEVELS,
    MIE_ASYMMETRY_ICE,
    MIE_ASYMMETRY_WATER,
    CloudOptics,
    DensityField,
    delta_eddington,
    field_offset,
)
from world_builder.quality import (
    REBUILD,
    RESTART,
    QualityLevel,
    QualitySettings,
    knob,
)

#: Most distance bands of the march's detail (each a detail level of a
#: genus, plus its far level): they fit ``DETAIL_LEVELS``.
MAX_OCTAVE_BANDS = 6
assert MAX_OCTAVE_BANDS < DETAIL_LEVELS


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

    field            its :class:`DensityField` — shape; a change needs a
                     rebuild
    optics           its :class:`CloudOptics` — shading only, changeable live
    """

    field: DensityField
    optics: CloudOptics = CloudOptics()


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
            # Deep clouds tower: the cell octave gathers a congestus nearly
            # as it does a cumulonimbus (as the square root of the depth),
            # and the threshold stays low up the column and rises late
            # (tests/test_clouds_congestus.py: at depth 1/2 the tops stopped
            # 1.5-2 km under their local top, and what rose was scattered).
            cell_weight=1.0,
            cell_power=0.5,
            deep_exponent=10.0,
            deep_erosion=0.25,
        ),
        optics=CloudOptics(
            shadow_strength=0.3,
            forward_gain=8.0,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_WATER),
        ),
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
        # Its sunward depth scaled as the cumulus's is (CONVECTIVE's 0.3,
        # their stand-in for multiple scattering): at 1 (physical, single
        # scattering) the sheets self-shadowed three times the cumulus's,
        # grey tops at noon and navy undersides.
        optics=CloudOptics(
            sun_steps=3,
            sun_step_length=90.0,
            shadow_strength=0.35,
            multiple_scattering=0.10,
            powder_length=150.0,
            max_optical_depth=1.8,
            forward_gain=2.2,
            backward_gain=1.3,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_WATER),
        ),
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
        # Gentler than the field's 6.25:1, or the anisotropy is double-counted.
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
    (:class:`march.CloudMarch`, a raymarch at a fraction of the view's
    resolution which needs the split cameras, its high layer the cirrus).
    A config saved with the billboards' knobs (renderer,
    billboard count, overlap, interior growth, half resolution, tile
    uploads, re-sorts and cache) loads with those skipped.

    :ivar march_steps: most steps of the raymarch per pixel (it stops
        sooner once the cloud is opaque). Live
    :ivar march_block: the view's pixels per raymarched pixel, (across,
        up): the march traces one pixel of each block a frame, in turn
        (``march.block_order``). Live
    :ivar march_refresh_moving: while the camera moves, the march traces a
        smaller block (``march.MOVING_BLOCKS``). Live
    :ivar march_scale: sun-march steps relative to the types', over the same
        reach (:func:`scale_march`). Live
    :ivar shell_octaves: noise octaves per distance band of the march, the
        last repeating (and going to the deck). Rebuild
    :ivar octave_bands: the march's distance bands before the deck's far field,
        each twice as far as the last. Rebuild
    :ivar reflected: drawn in the ocean's reflection. Live
    :ivar light_refresh_growth: the march's light (the sun's colour through
        the air, twilight, eclipses: cloudSunAndSky) is taken anew at a cloud
        sample once the ray is this many times farther than where it was
        last taken (and at least :attr:`light_refresh_min` farther): a far
        cloud behind a near one gets its own light, not the near one's. 0:
        once a pixel, at its first cloud. Live
    :ivar light_refresh_min: the least distance between two takes, metres on
        a 600 km planet (scaled by the planet's radius). Live
    :ivar weather_nimbostratus: the weather's nimbostratus drawn (its
        stratus thickening into it, genera.GeneraSpec.weather_nimbostratus):
        off under the high preset, where a front's rain clouds cost 2-6 ms.
        Restart
    """

    march_steps: int = knob(
        64,
        "Raymarch steps",
        unit="steps",
        range=(8, 256),
        help="Most samples per pixel of the raymarch: its smoothness along "
        "the view (raymarch only).",
    )
    march_block: tuple[int, int] = knob(
        (4, 2),
        "Raymarch resolution",
        choices={
            "Quarter": (4, 4),
            "Ninth (3 x 3)": (3, 3),
            "Eighth (4 x 2)": (4, 2),
            "Half": (2, 2),
            "Full": (1, 1),
        },
        help="The share of the view's pixels the raymarch traces each frame (it "
        "gathers the others over the frames): a quarter of the resolution per axis "
        "traces one pixel in 16, a ninth one in 9, an eighth one in 8, a half one "
        "in 4 (raymarch only).",
    )
    march_refresh_moving: bool = knob(
        False,
        "Raymarch: faster refresh in motion",
        advanced=True,
        help="While the camera moves, the raymarch traces twice as many "
        "pixels a frame (an eighth to a quarter, a quarter to an eighth), "
        "back to its own share once still: at twice the raymarch's cost in "
        "motion (raymarch only).",
    )
    march_scale: float = knob(
        1.0,
        "Lighting",
        unit="x",
        range=(0.25, 2.0),
        help="Steps of each sample's march toward the sun: the clouds' self-shadowing.",
    )
    shell_octaves: tuple[int, ...] = knob(
        (4, 4, 4, 3),
        "Detail",
        REBUILD,
        choices={"-".join(map(str, v)): v for v in _OCTAVE_SCHEDULES},
        help="Noise octaves near to far: the clouds' fine detail in the distance.",
    )
    octave_bands: int = knob(
        4,
        "Detail distance",
        REBUILD,
        unit="bands",
        range=(1, MAX_OCTAVE_BANDS),
        advanced=True,
        help="Distance bands of the raymarch's detail, each twice as far as "
        "the last, before the deck's far field takes over.",
    )
    reflected: bool = knob(
        True,
        "In reflections",
        help="Draw the clouds in the sea's reflection (about 1-1.7 ms).",
    )
    light_refresh_growth: float = knob(
        1.4,
        "Light along the view",
        choices={
            "Once a pixel": 0.0,
            "Coarse (2x)": 2.0,
            "1.6x": 1.6,
            "1.4x": 1.4,
            "Fine (1.25x)": 1.25,
        },
        help="How often the clouds' light (the sun's colour, twilight) is taken "
        "again along a view ray: each time the ray is this many times farther. "
        "Far clouds behind near ones get their own sunset colour; finer costs "
        "a little more where a ray crosses many clouds.",
    )
    light_refresh_min: float = knob(
        2000.0,
        "Light refresh distance",
        unit="m",
        range=(100.0, 50000.0),
        advanced=True,
        help="The least distance between two takes of the clouds' light along a "
        "ray, on a 600 km planet (scaled by the planet's size).",
    )
    weather_nimbostratus: bool = knob(
        True,
        "Rain clouds",
        RESTART,
        help="The weather's nimbostratus: its stratus thickening into a dark "
        "rain cloud up to 2 km over fronts and lows (2-6 ms where they are in "
        "view). Applies when the clouds are next built.",
    )

    LABEL: ClassVar = "Clouds"
    # Saved by label: "Quarter", "Half" and "Full" keep their blocks.
    RENAMED: ClassVar = {"march_resolution": "march_block", "shells": "octave_bands"}
    # The sun march carries the clouds' form, so it is stepped gently.
    PRESETS: ClassVar = {
        QualityLevel.LOW: {
            "march_scale": 4.0 / 9.0,
            "shell_octaves": (4, 3, 3, 2),
            "reflected": False,
            "march_steps": 24,
            "march_block": (4, 4),
            "weather_nimbostratus": False,
            "light_refresh_growth": 2.0,
        },
        QualityLevel.MID: {
            "march_scale": 2.0 / 3.0,
            "shell_octaves": (4, 4, 3, 3),
            "march_steps": 40,
            "march_block": (4, 4),
            "weather_nimbostratus": False,
            "light_refresh_growth": 1.6,
        },
        QualityLevel.ULTRA: {
            "march_scale": 1.5,
            "shell_octaves": (4,),
            "march_steps": 96,
            "march_block": (2, 2),
            "light_refresh_growth": 1.25,
        },
    }


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
    """A cloud type at the settings' cost (:func:`scale_march`)."""
    return scale_march(spec, settings.march_scale)
