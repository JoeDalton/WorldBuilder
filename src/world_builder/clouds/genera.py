"""
A planet's low and middle clouds by genus: stratus, stratocumulus,
nimbostratus, altostratus, altocumulus, and the convective
continuum from fair-weather cumulus through congestus to cumulonimbus,
drawn from the weather (:func:`weather.synoptic_cube`), with optional
hand-placed features over it.

The genera (:data:`GENUS_PRESETS`, :attr:`GeneraSpec.genera`), each one
density field over its own slab, which the raymarch marches as one union
(:class:`march.CloudMarch`; slabs may overlap):

* **stratus**, **stratocumulus**, **nimbostratus**, **altostratus** and
  **altocumulus**: recipe genera (:attr:`GenusPreset.recipe`; clouds.md,
  "The recipe genera"), a flat grey sheet low down, a layer of lumpy closed
  cells, a thick dark rain-cloud layer, a thin sheet the sun shows through
  and a thin layer of small rounded patches, each its coverage a map of its
  own (:func:`genus_coverage`);
* **convective** (:attr:`cloud.CloudType.CONVECTIVE`): the cumulus field,
  whose clouds grow with the convective **depth** (0 to 1, a third map):
  their top rises from the cumulus's 1 km to the slab's cap
  (:attr:`noise.DensityField.min_top`), a coarse cell octave gathers them
  into towers kilometres wide (``cell_weight``), the threshold falls under
  the cap of the deepest into an anvil (``anvil``). The anvil spreads past
  its tower as a smooth lens, over a fourth map (its spread:
  :func:`noise.shelf_lift`), streaming downwind on the upper wind
  (:func:`anvil_spread`).

The maps (:func:`genus_planes`, cube maps as the weather's) are the
weather's own: its convective coverage, depth and anvil shelf, and the
stratus's coverage its stratus and stratocumulus (:data:`WEATHER_GENERA`,
:func:`genus_coverage`; the other genera are their features' alone for
now); plus, around :attr:`GeneraSpec.focus`, the features of
:attr:`GeneraSpec.features`.

**Coverage to the field.** The share of columns holding cloud depends on
how tall they may grow, so the convective field's fbm shift is tabulated by
depth as well as coverage (:func:`depth_shift_tables`, one
:func:`weather.shift_table` per depth of :data:`DEPTHS`), and the weather
cube carries the depth beside the coverage and the shift
(:func:`genus_cube`), and the anvil's spread (its shift is the field's:
:func:`noise.with_shelf`).

Design notes: docs/source/clouds.md, "Genera".
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from world_builder.clouds.cloud import CloudType
from world_builder.clouds.noise import (
    MIE_ASYMMETRY_ICE,
    DensityField,
    _smoothstep,
    delta_eddington,
    sphere_column_peaks,
)
from world_builder.clouds.weather import (
    SYNOPTIC_CHANNELS,
    WEATHER_SIZE,
    WeatherSpec,
    cirrus_cover_map,
    coverage_map,
    coverage_to_shift,
    sample_cube,
    shift_table,
    synoptic_cube,
)
from world_builder.clouds.weather import (
    upper_stream as weather_upper_stream,
)
from world_builder.sky.starfield import cube_directions
from world_builder.terrain.chart import TangentFrame, latlon

#: The genus maps' cube size: twice the weather's (1.85 km texels on a
#: 600 km planet), for features a few kilometres across.
GENUS_SIZE = 2 * WEATHER_SIZE
#: The convective depths the fbm shift is tabulated at (and interpolated
#: between).
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: The coverage around a cumulonimbus's core: its anvil's overhang, then
#: the congestus ring.
_CB_RING_COVERAGE = 0.45
#: A cumulonimbus's anvil shelf, in radii of it: its spread 1 out to the
#: first, 0 by the second (its underside rising, a lens), with no wind.
_ANVIL_REACH = (1.0, 3.0)
#: The anvil on the upper wind (:func:`anvil_spread`), at the jets' fastest
#: (1): its reach downwind this many times longer, upwind and across this
#: many times shorter (it streams downwind, its upwind edge sharp).
ANVIL_DOWNWIND = 2.5
ANVIL_UPWIND = 0.5
ANVIL_ACROSS = 0.25
#: The primary genus's weather (:func:`genus_cube`).
CONVECTIVE = "convective"


@dataclass(frozen=True)
class Feature:
    """One feature of the placeholder maps: a disc of a genus.

    :ivar east: its centre east of the focus, metres
    :ivar north: north of it, metres
    :ivar radius: its radius, metres; it fades out over its outer third
    :ivar genus: a cumulus species (:data:`CONVECTIVE_SPECIES`):
        ``"cumulonimbus"`` (depth 1 to 1.4 radii: its core, then its anvil's
        overhang; a ring of congestus to 2.2 radii; its anvil's shelf to 3
        radii), ``"congestus"`` (depth 0.5) and the rest of
        :data:`GENUS_DEPTHS`; or another genus of :data:`GENUS_PRESETS`
        (``"stratus"``, ``"stratocumulus"``, ``"nimbostratus"``,
        ``"altostratus"``, ``"altocumulus"``), which fills its own coverage map
        (:func:`genus_coverage`); or a high genus (:data:`HIGH_CHANNELS`),
        its channel of coverage cube B
    :ivar coverage: the share of its columns holding cloud at its core (a
        cumulonimbus's near 1: a solid tower, the noise carving its edges;
        less, and a deep column holds scattered blobs at every height)
    """

    east: float
    north: float
    radius: float
    genus: str
    coverage: float = 0.6


@dataclass(frozen=True)
class GenusPreset:
    """A genus of the union march (:class:`noise.GenusField`'s source):
    its cloud type (the preset's field and optics), where its coverage comes
    from, its far shadow, and its slab when not the type's.

    :ivar slab: (base, thickness) above sea level, metres; None = the type's
        (the stratus's base is :attr:`GeneraSpec.stratus_altitude`)
    :ivar recipe: a recipe genus's knobs (:class:`noise.GenusField`'s
        ``noise_kind``, ``cell_size``, ``cell_fill``, ``stretch``,
        ``vertical_stretch``, ``profile``, ``erosion``, ``erosion_base``):
        the genus evaluates the recipe (``genusRecipeDensity``); None = its
        type's field as it is (the convective clouds)
    :ivar field: overrides of its type's :class:`noise.DensityField`
        (``edge_softness``, ``noise_scale``, ``density``...), its threshold
        then resolved anew; None = none
    :ivar optics: overrides of its type's :class:`noise.CloudOptics`
        (``multiple_scattering``, ``forward_gain``...): genera of one type
        that are lit apart (the lighting carries a sheet's identity:
        cloud_genera_research.md, section 5); None = none
    """

    cloud_type: CloudType
    coverage: tuple = ("weather", 0)
    far_shadow: float = 0.0
    slab: tuple | None = None
    recipe: dict | None = None
    field: dict | None = None
    optics: dict | None = None


#: The genera by name: the primary one (the convective clouds) reads the
#: weather cube, the others a channel of a coverage cube. The stratiform
#: ones are recipe genera (clouds.md, "The recipe genera", has the why of
#: each knob; cloud_genera_research.md, section 3, the recipes).
GENUS_PRESETS = {
    "convective": GenusPreset(CloudType.CONVECTIVE),
    # A low, grey, uniform sheet (WMO: "a flat, uniform base, sometimes
    # fractus"; outline sharp): value fbm of 10 km features (the research's
    # 10-30 km), three times longer along the focus's east, a wide window
    # (soft, even density inside), the top soft, the base flat, its outline
    # ragged (wispy erosion) only low down.
    "stratus": GenusPreset(
        CloudType.STRATUS,
        ("A", 0),
        far_shadow=1.0,
        recipe=dict(
            noise_kind="value",
            stretch=3.0,
            profile=(0.03, 0.3, False),
            erosion=(-1.5, 1.0),
            erosion_base=0.4,
        ),
        field=dict(
            noise_scale=(0.0004, 0.0004, 0.0004),
            octave_weights=(0.6, 0.3, 0.08, 0.02),
            edge_softness=1.2,
            top_erosion=0.1,
            top_exponent=2.0,
        ),
    ),
    # Lumpy, tessellated grey-white cells with dark parts (WMO: "always some
    # dark parts", non-fibrous, flat-based, lumpy-topped): closed Worley
    # cells 1.5 km wide (F2 - F1, twice as tall as wide: columns) mixed
    # with a little fbm, the threshold rising toward the top (domes:
    # a cell's centre reaches higher than its rim), the base flat, billowy
    # erosion. Density 0.045 a metre, about the stratus's 0.05 (its water,
    # 0.45 against 0.25-0.3 g/m3, argues for more; at 0.07 its undersides
    # went navy).
    "stratocumulus": GenusPreset(
        CloudType.STRATUS,
        ("A", 1),
        far_shadow=1.0,
        slab=(800.0, 1000.0),
        recipe=dict(
            noise_kind="cells",
            cell_size=1500.0,
            cell_fill=0.75,
            vertical_stretch=2.0,
            profile=(0.08, 0.1, False),
            erosion=(3.0, 1.0),
        ),
        field=dict(
            noise_scale=(0.002, 0.002, 0.002),
            edge_softness=0.8,
            top_erosion=0.4,
            top_exponent=1.5,
            density=0.045,
        ),
    ),
    # A thick, dark, featureless rain-cloud layer spanning the low and
    # middle etages (WMO: "grey, often dark ... thick enough throughout to
    # blot out the sun"): value fbm of 20 km features (the research's 20-50
    # km) three times longer along east, its fine octaves faint
    # (featureless), a narrow window (solid inside: uniform), a top-hat over
    # 0.6-5 km, its base diffuse (soft over its lowest 12 %, wispy erosion
    # of 500 m cells over its lowest 15 %: ragged where it thins), the
    # threshold rising a little toward the top (shoulders, not a brick, where
    # the coverage fades). Denser than the stratus and lit darker: its
    # sunward depth scaled less than the stratus's (0.4 to 0.35), a weak
    # forward lobe; its sun steps longer (4.4 km of it).
    "nimbostratus": GenusPreset(
        CloudType.STRATUS,
        ("A", 2),
        far_shadow=1.0,
        slab=(600.0, 4400.0),
        recipe=dict(
            noise_kind="value",
            stretch=3.0,
            cell_size=3000.0,
            profile=(0.12, 0.15, False),
            erosion=(-2.0, 1.0),
            erosion_base=0.15,
        ),
        field=dict(
            noise_scale=(0.0002, 0.0002, 0.0002),
            octave_weights=(0.6, 0.3, 0.08, 0.02),
            edge_softness=1.0,
            top_erosion=0.15,
            top_exponent=2.0,
            density=0.08,
        ),
        optics=dict(
            sun_steps=3,
            sun_step_length=250.0,
            shadow_strength=0.4,
            multiple_scattering=0.35,
            forward_gain=1.6,
            backward_gain=1.1,
        ),
    ),
    # A grey or bluish sheet, uniform or striated, through whose thinner
    # parts the sun shows as through ground glass (WMO; never a halo):
    # value fbm of 16 km features stretched five times along east, its last
    # octave dropped (the march's far levels drop it: in a window this wide
    # the cut drew a contour across the sheet), a wide window (its density
    # varies: thin and thicker parts), soft at both ends, no erosion (at
    # -0.5 its 500 m cells, squeezed along east, sawed its rim). Thin (a
    # thirtieth of the stratus's density: the sun's disc shows through) and
    # lit as ice: more sky fill (grey-bluish), an ice cloud's forward lobe.
    # Over 4.7-6 km: clear of the altocumulus's 3.5-3.8 km (a sample inside
    # two genera's slabs costs both).
    "altostratus": GenusPreset(
        CloudType.STRATUS,
        ("A", 3),
        far_shadow=1.0,
        slab=(4700.0, 1300.0),
        recipe=dict(
            noise_kind="value",
            stretch=5.0,
            cell_size=3000.0,
            profile=(0.3, 0.3, False),
        ),
        field=dict(
            noise_scale=(0.00025, 0.00025, 0.00025),
            octave_weights=(0.6, 0.3, 0.1, 0.0),
            edge_softness=2.0,
            top_erosion=0.05,
            top_exponent=2.0,
            density=0.0015,
        ),
        optics=dict(
            # Thin ice: single scattering, its sunward depth physical (the
            # stratus's 0.35 let three quarters of the sun through a full sheet)
            shadow_strength=1.0,
            multiple_scattering=0.2,
            forward_gain=3.0,
            forward_anisotropy=delta_eddington(MIE_ASYMMETRY_ICE, applicability=0.5),
        ),
    ),
    # Rounded white or grey patches in a thin layer, with shading, small
    # clear gaps between them, each 1-5 degrees across seen from the ground
    # (WMO): closed Worley cells (F2 - F1: their walls stay clear however
    # high the coverage; 1 - F1's blobs merged into chains) 350 m apart (an
    # element is about 1.2 cells: 4-6 degrees overhead at 3.5 km; at 800 m,
    # 10 degrees), most of the base value, the rest fbm of 3 km features
    # (the cells' sizes jitter), half again taller than wide (one element
    # per column), a lens (4h(1-h): rounded tops and bases) over a slab
    # 300 m thick (flatter than wide: at 1 km the elements were tall
    # pebbles seen from the side), billowy erosion at a sixth of the cells
    # (at 2 their rims were polygonal). Dense enough for grey cores (at
    # 0.012 all white). Unstretched: the erosion's cells are read in the
    # recipe's coordinates, which a stretch squeezes (the altostratus's
    # rim sawn into teeth). Its last octave dropped (the far levels' cut).
    # Over 3.5-3.8 km: inside the nimbostratus's slab, under the
    # altostratus's base.
    "altocumulus": GenusPreset(
        CloudType.STRATUS,
        ("B", 0),
        far_shadow=1.0,
        slab=(3500.0, 300.0),
        recipe=dict(
            noise_kind="cells",
            cell_size=350.0,
            cell_fill=0.8,
            vertical_stretch=1.5,
            profile=(0.0, 0.0, True),
            erosion=(3.0, 1.0),
        ),
        field=dict(
            noise_scale=(0.0013, 0.0013, 0.0013),
            octave_weights=(0.6, 0.3, 0.1, 0.0),
            edge_softness=0.8,
            density=0.05,
        ),
    ),
}


#: The high layer's slab (base, thickness) above sea level, metres: its own
#: sky band above the union of the genera (the convective clouds' top, 10.5
#: km; a gap of 500 m, so :func:`march.sky_bands` never merges them). Real
#: cirrus spans about 6-12 km at mid-latitudes; here it must clear the
#: anvils. The march crosses it in one or two samples (clouds.md, "The high
#: layer"), not in steps.
HIGH_SLAB = (11000.0, 1500.0)
#: The high genera's channels of coverage cube B (its channel 0 is the
#: altocumulus's): every one is read at once where a ray crosses the high
#: slab, each channel its own recipe.
HIGH_CHANNELS = {"cirrus": 1, "cirrostratus": 2, "cirrocumulus": 3}


@dataclass(frozen=True)
class HighOptics:
    """A high genus's optics (the march's ``uHighExtinction``, ``uHighCap``,
    ``uHighShape``): its own, where the genera of the march share their
    type's (:class:`noise.CloudOptics`); the light is the thin slab's for
    all (``thinSlabRadiance``, the cirrus's ice phase).

    :ivar depth: the optical depth (exp2 units) of a full veil (density 1)
        straight through the slab
    :ivar cap: what a crossing's optical depth saturates toward (a grazing
        one's; ``march.thin_optical_depth``)
    :ivar halo: the 22 degree halo's strength in its phase, 0 = none
        (``cloud_march.glsl``'s ``haloPhase``)
    """

    depth: float
    cap: float
    halo: float = 0.0


#: The high genera's optics. The cirrus: a streak's density about 1 at a
#: tuft, 0.3-0.6 along its fibres, so 0.1-1 optical depth straight up (the
#: research's 0.03-3, SIRTA's typical 0.1-1): thin, white, the sky plain
#: between; a cap of 2.5; a faint 22 degree halo, only where a streak is
#: (narrow: "only fragments of a halo"). The cirrostratus: a veil of 0.2
#: (the research's 0.03-0.3 for the thin layers), the sun plain through it,
#: milky toward the horizon (its cap: a grazing crossing at most about
#: 0.6), and its 22 degree halo. The cirrocumulus: a grain's core 1.2,
#: between the grains none (its mean over the grains about a fifth of it,
#: the far patch's).
HIGH_OPTICS = {
    "cirrus": HighOptics(depth=1.5, cap=2.5, halo=0.4),
    "cirrostratus": HighOptics(depth=0.2, cap=1.2, halo=1.0),
    "cirrocumulus": HighOptics(depth=1.2, cap=2.5),
}

#: The convective depth of each cumulus species a feature can be (a
#: cumulonimbus is depth 1 and its anvil, :func:`genus_planes`): its clouds'
#: top rises with it (``DensityField.min_top``), a congestus's to 6.85 km.
GENUS_DEPTHS = {"humilis": 0.0, "mediocris": 0.2, "congestus": 0.5}
#: The features of the convective genus (the weather cube's coverage and
#: depth); every other feature's genus has a coverage map of its own.
CONVECTIVE_SPECIES = ("cumulonimbus", *GENUS_DEPTHS)

#: The showcase's northern row (:func:`showcase_features`): (genus,
#: bearing east of north in degrees, distance in metres, radius in metres,
#: coverage), west to east: the cumulus species. A cumulonimbus last, east
#: of the rest: its anvil streams to 3 radii and farther downwind.
SHOWCASE_ROW = (
    ("humilis", -42.0, 35e3, 8e3, 0.35),
    ("mediocris", -12.0, 35e3, 8e3, 0.45),
    ("congestus", 16.0, 35e3, 7e3, 0.65),
    ("cumulonimbus", 60.0, 45e3, 5e3, 1.0),
)
#: The southern row, as :data:`SHOWCASE_ROW`, east to west: the stratiform
#: and mid-level genera, sheets 14 km wide, a slot each 30 degrees apart
#: (19.7 km between centres). A slot whose genus is not a preset is
#: left out (:func:`showcase_features`). The low
#: genera east of south, away from the sun (azimuth 210 at the start), so
#: their undersides are not shot against it; the altostratus at the sun
#: ("ground glass"), the altocumulus beyond it (shot from under it: its
#: pattern of patches is its identity).
SHOWCASE_SOUTH = (
    ("stratocumulus", 120.0, 38e3, 7e3, 0.88),
    ("stratus", 150.0, 38e3, 7e3, 0.9),
    ("nimbostratus", 180.0, 38e3, 7e3, 0.95),
    ("altostratus", 210.0, 38e3, 7e3, 0.9),
    ("altocumulus", 240.0, 38e3, 7e3, 0.85),
)


#: The high genera's slots, as :data:`SHOWCASE_ROW`'s, overhead, apart
#: from each other: the high layer (11-12.5 km) is over the rows, so each
#: is shot from the start looking up at :data:`SHOWCASE_HIGH_PITCH`, the
#: rows' clouds under the view's bottom edge or off its sides. The
#: cirrostratus a veil 64 km wide to the west (a gap between the
#: altocumulus, 240, and the humilis, 318); its pose looks at the sun
#: through it (the sun set at azimuth 280, elevation 35, for it). The
#: cirrocumulus a patch 12 km wide east-south-east, over the low sheets
#: (the stratocumulus, 120), away from the cumulonimbus's anvil (60).
#: The cirrus a field 80 km wide far to the north, beyond the cumulus row
#: (35-45 km; the cirrostratus's veil reaches over the start, and the
#: cirrocumulus and the cumulonimbus's anvil fill the east): its pose
#: stands under its near edge (:data:`SHOWCASE_HIGH_EYE`), the row behind.
SHOWCASE_HIGH = (
    ("cirrostratus", 280.0, 25e3, 32e3, 0.9),
    ("cirrocumulus", 120.0, 16e3, 6e3, 0.9),
    ("cirrus", 0.0, 90e3, 40e3, 0.55),
)
#: Each high genus's pose's pitch, degrees: the cirrostratus's at the sun's
#: elevation, the cirrocumulus's at the slab's middle over its centre, the
#: cirrus's at the slab's middle 14 km on (inside its core).
SHOWCASE_HIGH_PITCH = {"cirrostratus": 35.0, "cirrocumulus": 34.0, "cirrus": 40.0}
#: Where each high genus's pose stands, metres along its bearing from the
#: start (the start itself where not listed).
SHOWCASE_HIGH_EYE = {"cirrus": 55e3}


def showcase_slots() -> tuple[tuple, ...]:
    """The showcase's rows (:data:`SHOWCASE_ROW`, :data:`SHOWCASE_SOUTH`,
    :data:`SHOWCASE_HIGH`) less the slots whose genus is neither a cumulus
    species, a preset nor a high genus."""
    return tuple(
        slot
        for slot in SHOWCASE_ROW + SHOWCASE_SOUTH + SHOWCASE_HIGH
        if slot[0] in CONVECTIVE_SPECIES
        or slot[0] in GENUS_PRESETS
        or slot[0] in HIGH_CHANNELS
    )


def showcase_features() -> tuple[Feature, ...]:
    """The separated-genera showcase: one feature per genus of the march (and
    per cumulus species), the cumulus in a row along the northern horizon
    from the focus, the stratiform and mid-level genera along the southern,
    apart (:func:`showcase_slots`). With ``GeneraSpec(background=False)``
    there is no other cloud: each genus is seen alone."""
    return tuple(
        Feature(
            distance * math.sin(math.radians(bearing)),
            distance * math.cos(math.radians(bearing)),
            radius,
            genus,
            coverage,
        )
        for genus, bearing, distance, radius, coverage in showcase_slots()
    )


#: The demo's: within view of the demo's start, a cumulonimbus 40 km north
#: ringed by congestus, another 70 km north-east, a line of congestus to the
#: west, and a stratus deck to the south.
DEFAULT_FEATURES = (
    Feature(0.0, 40e3, 6e3, "cumulonimbus", 1.0),
    Feature(45e3, 55e3, 5e3, "cumulonimbus", 1.0),
    Feature(-30e3, 20e3, 12e3, "congestus", 0.65),
    Feature(-30e3, 40e3, 10e3, "congestus", 0.65),
    Feature(0.0, -30e3, 25e3, "stratus", 0.9),
)


@dataclass(frozen=True)
class GeneraSpec:
    """A planet's low clouds by genus (see the module); frozen and picklable.

    :ivar weather: the weather whose coverage the fair-weather cumulus keep
    :ivar focus: (lat, lon) degrees the features are placed around
    :ivar features: the features
    :ivar stratus_altitude: the stratiform slab's base, metres (it may
        overlap the convective clouds' slab: the march takes them as one)
    :ivar background: the weather's fair-weather cumulus between the
        features; False = none (the showcase: each genus alone)
    :ivar wind: the upper wind blowing the anvils, (east, north), the jets'
        fastest 1, the same for all; None = the weather's at each
        cumulonimbus (:func:`upper_wind_at`)
    :ivar high: the high genera (:data:`HIGH_CHANNELS`), drawn by the march's
        thin-layer path where a feature covers them (:func:`high_genera`)
    :ivar high_slab: their slab, (base, thickness) above sea level, metres
    :ivar cirrus_weather: the cirrus covers the weather's cirrus cover too
        (:func:`weather.cirrus_cover_map`), not only its features: the march
        draws the planet's cirrus (``PlanetClouds(cirrus=True)`` sets it, the
        demo's ``--cirrus``); False = its features alone (the showcase)
    """

    weather: WeatherSpec
    focus: tuple[float, float] = (0.0, 0.0)
    features: tuple[Feature, ...] = field(default=DEFAULT_FEATURES)
    stratus_altitude: float = 900.0
    wind: tuple[float, float] | None = None
    genera: tuple[str, ...] = (
        "convective",
        "stratus",
        "stratocumulus",
        "nimbostratus",
        "altostratus",
        "altocumulus",
    )
    background: bool = True
    high: tuple[str, ...] = tuple(HIGH_CHANNELS)
    high_slab: tuple[float, float] = HIGH_SLAB
    cirrus_weather: bool = False


def weather_cirrus(weather: WeatherSpec) -> GeneraSpec:
    """The high layer of a march without genera (one cloud type): the
    weather's cirrus alone (:attr:`GeneraSpec.cirrus_weather`), no genus,
    no feature. Only its high layer is read (:func:`high_genera`,
    :func:`coverage_cube`'s cube B, :func:`high_wind_cube`).

    :param weather: the weather whose cirrus cover and upper wind it follows
    """
    return GeneraSpec(
        weather,
        features=(),
        genera=(),
        background=False,
        high=("cirrus",),
        cirrus_weather=True,
    )


def high_genera(spec: GeneraSpec, covered: bool = False) -> list[tuple[str, int]]:
    """The high genera of *spec* (:attr:`GeneraSpec.high`) and their channels
    of coverage cube B.

    :param covered: only those a feature covers, as the march takes them
        (:func:`genus_specs`' *covered*): with none it compiles no high
        layer at all (``march.march_fragment``)
    :returns: (name, channel) each, in :attr:`GeneraSpec.high`'s order
    """
    return [
        (name, HIGH_CHANNELS[name])
        for name in spec.high
        if not covered
        or any(f.genus == name for f in spec.features)
        or (name == "cirrus" and spec.cirrus_weather)
    ]


def high_jet(spec: GeneraSpec) -> np.ndarray:
    """The cirrostratus's and the cirrocumulus's stretch axis (the march's
    ``uHighJet``): the upper wind at the focus (:attr:`GeneraSpec.wind`,
    else :func:`upper_wind_at`), a unit vector in the planet frame, tangent
    there. One for the whole layer, as the recipe genera's shared axis (the
    cirrus follows the wind's own form on the GPU, :func:`high_wind_cube`).

    :returns: (3,) float64; the focus's east where the wind is calm
    """
    frame = TangentFrame.at(*spec.focus)
    wind = (
        np.asarray(spec.wind, dtype=np.float64)
        if spec.wind is not None
        else upper_wind_at(spec.weather, np.asarray(frame.up, dtype=np.float64))
    )
    axis = wind[0] * np.asarray(frame.east) + wind[1] * np.asarray(frame.north)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.asarray(frame.east, dtype=np.float64)
    return axis / norm


#: The recipe genera the weather covers, and the channels of
#: :func:`weather.synoptic_cube` each takes (the larger): the stratus sheet
#: takes the weather's stratus and stratocumulus. The others are their
#: features' alone for now: each genus the march draws costs it as a full
#: one wherever it is (+0.65 ms at 2560 x 1440, :func:`genus_specs`).
WEATHER_GENERA = {"stratus": ("stratus", "stratocumulus")}
#: A weather genus channel under this everywhere holds no cloud
#: (:func:`has_coverage`): the march skips the genus.
WEATHER_COVER_FLOOR = 0.01


def has_coverage(spec: GeneraSpec, name: str) -> bool:
    """Whether genus *name* of *spec* holds cloud anywhere: the primary one
    (the weather's) always; another where the weather's channels of it
    (:data:`WEATHER_GENERA`, with :attr:`GeneraSpec.background`) or a
    feature is its (:func:`genus_coverage`)."""
    if GENUS_PRESETS[name].coverage[0] == "weather":
        return True
    if any(f.genus == name for f in spec.features):
        return True
    if not spec.background or name not in WEATHER_GENERA:
        return False
    picks = [SYNOPTIC_CHANNELS.index(n) for n in WEATHER_GENERA[name]]
    return bool(synoptic_cube(spec.weather)[..., picks].max() > WEATHER_COVER_FLOOR)


def genus_specs(
    spec: GeneraSpec, altitude: float | None = None, covered: bool = False
) -> list[tuple]:
    """Each genus of *spec* as the march builds it (:data:`GENUS_PRESETS`):
    its name, cloud type, slab and coverage source.

    :param altitude: the convective clouds' base, metres; None = the type's
    :param covered: only the genera holding cloud somewhere
        (:func:`has_coverage`), as the march takes them: an empty genus
        would cost it all the same (in a band it shares, the generic genus
        loop instead of the band's dedicated one: +0.65 ms at 2560 x 1440
        for an empty stratocumulus over the stratus's band)
    :returns: (name, cloud type, base or None, thickness or None, coverage)
        per genus, in :attr:`GeneraSpec.genera`'s order (None: the type's)
    """
    out = []
    for name in spec.genera:
        if covered and not has_coverage(spec, name):
            continue
        preset = GENUS_PRESETS[name]
        base, thickness = preset.slab if preset.slab is not None else (None, None)
        if preset.slab is None and preset.cloud_type is CloudType.CONVECTIVE:
            base = altitude
        elif preset.slab is None and preset.cloud_type is CloudType.STRATUS:
            base = spec.stratus_altitude
        out.append((name, preset.cloud_type, base, thickness, preset.coverage))
    return out


def _feature_dirs(spec: GeneraSpec) -> list[tuple[Feature, np.ndarray, float]]:
    """:returns: each feature, its centre's direction and angular radius"""
    frame = TangentFrame.at(*spec.focus)
    radius = spec.weather.planet.radius
    out = []
    for f in spec.features:
        offset = math.hypot(f.east, f.north) / radius
        heading = (f.east * frame.east + f.north * frame.north) / max(
            math.hypot(f.east, f.north), 1e-9
        )
        centre = math.cos(offset) * frame.up + math.sin(offset) * heading
        out.append((f, centre, f.radius / radius))
    return out


#: The step the upper wind's stream function is differenced over, metres.
_WIND_STEP = 1000.0


def _frame(centre: np.ndarray) -> TangentFrame:
    lat, lon = latlon(np.asarray(centre)[None])
    return TangentFrame.at(float(lat[0]), float(lon[0]))


def upper_wind_at(spec: WeatherSpec, centre: np.ndarray) -> np.ndarray:
    """The upper wind at a direction: the cirrus's (the meandering jets,
    swirled by the weather systems: :func:`weather.upper_stream`), which
    blows along its stream function's contours, ``up x grad psi``.

    :param centre: (3,) unit direction
    :returns: (east, north), the jets' fastest 1
    """
    frame = _frame(centre)
    step = _WIND_STEP / spec.planet.radius
    probes = np.array(
        [
            centre + step * frame.east,
            centre - step * frame.east,
            centre + step * frame.north,
            centre - step * frame.north,
        ]
    )
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    psi = weather_upper_stream(spec, probes)
    grad_east = (psi[0] - psi[1]) / (2.0 * _WIND_STEP)
    grad_north = (psi[2] - psi[3]) / (2.0 * _WIND_STEP)
    return np.array([-grad_north, grad_east])


#: The upper wind's cube map (:func:`high_wind_cube`): texels per face edge
#: (7.4 km on a 600 km planet; the wind turns over hundreds of kilometres).
HIGH_WIND_SIZE = 128


def _tangent_east_north(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north unit vectors at unit *dirs* (any at the poles)."""
    east = np.cross([0.0, 0.0, 1.0], dirs)
    norm = np.linalg.norm(east, axis=-1, keepdims=True)
    east = np.where(norm > 1e-9, east / np.maximum(norm, 1e-12), [1.0, 0.0, 0.0])
    return east, np.cross(dirs, east)


def upper_stream(spec: GeneraSpec, dirs: ArrayLike) -> np.ndarray:
    """The upper wind's stream function at unit *dirs*, metres: the
    weather's (:func:`weather.upper_stream`), or with
    :attr:`GeneraSpec.wind` set, a uniform wind's, that wind at the focus
    (``psi = speed * R * dirs . (axis x up)``), the jets' fastest 1. The
    wind blows along its contours, ``up x grad psi`` (:func:`upper_wind_at`).

    :returns: (N,) float64
    """
    dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    if spec.wind is None:
        return weather_upper_stream(spec.weather, dirs)
    frame = TangentFrame.at(*spec.focus)
    up = np.asarray(frame.up, dtype=np.float64)
    wind = np.asarray(spec.wind, dtype=np.float64)
    axis = wind[0] * np.asarray(frame.east) + wind[1] * np.asarray(frame.north)
    return spec.weather.planet.radius * (dirs @ np.cross(axis, up))


def upper_wind_field(
    spec: GeneraSpec, dirs: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """The upper wind at unit *dirs*, in the planet frame, and its stream
    function (:func:`upper_stream`): ``up x grad psi`` by central
    differences over :data:`_WIND_STEP` (:func:`upper_wind_at`'s, for many
    directions at once).

    :returns: (wind (N, 3), tangent, the jets' fastest 1; psi (N,), metres)
    """
    dirs = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    east, north = _tangent_east_north(dirs)
    step = _WIND_STEP / spec.weather.planet.radius

    def psi(offset):
        probe = dirs + step * offset
        return upper_stream(spec, probe / np.linalg.norm(probe, axis=1, keepdims=True))

    grad_east = (psi(east) - psi(-east)) / (2.0 * _WIND_STEP)
    grad_north = (psi(north) - psi(-north)) / (2.0 * _WIND_STEP)
    wind = -grad_north[:, None] * east + grad_east[:, None] * north
    return wind, upper_stream(spec, dirs)


@functools.cache
def high_wind_cube(spec: GeneraSpec, size: int = HIGH_WIND_SIZE) -> np.ndarray:
    """The upper wind on the GPU (the march's ``highWind``; cached per
    process): the cirrus's flow (``cloud_march.glsl``'s ``cirrusStreaks``)
    turns with it over the planet, around the weather's lows, where one jet
    axis (:func:`high_jet`) drifts off it far from the focus.

    :returns: (6, size, size, 4) float32: the wind (planet frame, tangent,
        the jets' fastest 1) and its stream function (metres: its contours
        are the streamlines, the cirrus's across-the-wind coordinate), the
        layout of :func:`sky.starfield.cube_directions`
    """
    dirs = cube_directions(size).reshape(-1, 3)
    wind, psi = upper_wind_field(spec, dirs)
    cube = np.concatenate([wind, psi[:, None]], axis=1)
    return cube.reshape(6, size, size, 4).astype(np.float32)


def anvil_spread(
    dirs: np.ndarray, centre: np.ndarray, angle: float, wind: np.ndarray
) -> np.ndarray:
    """A cumulonimbus's anvil spread at *dirs*: 1 to :data:`_ANVIL_REACH`'s
    first radius, 0 by its second, the reach stretched downwind and
    shortened upwind and across on the upper wind (:data:`ANVIL_DOWNWIND`,
    ``ANVIL_UPWIND``, ``ANVIL_ACROSS`` at the jets' fastest): the anvil
    streams downwind, as a plume.

    :param dirs: (N, 3) unit directions
    :param centre: (3,) the tower's centre
    :param angle: its radius, radians
    :param wind: (east, north) at it, the jets' fastest 1
    """
    frame = _frame(centre)
    speed = float(np.hypot(*wind))
    along = frame.east
    if speed > 1e-9:
        along = (wind[0] * frame.east + wind[1] * frame.north) / speed
    across = np.cross(centre, along)
    # Polar coordinates around the tower: its distance (radians, in its
    # radii), along the great circle's heading.
    cosine = np.clip(dirs @ centre, -1.0, 1.0)
    tangent = dirs - cosine[:, None] * centre
    norm = np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)
    distance = np.arccos(cosine) / angle
    x = distance * (tangent @ along) / norm
    y = distance * (tangent @ across) / norm
    speed = min(speed, 1.0)
    x = np.where(
        x > 0.0, x / (1.0 + ANVIL_DOWNWIND * speed), x / (1.0 - ANVIL_UPWIND * speed)
    )
    y = y / (1.0 - ANVIL_ACROSS * speed)
    return 1.0 - _smoothstep(*_ANVIL_REACH, np.hypot(x, y))


def _weather_channels(spec: GeneraSpec, names: tuple, size: int) -> np.ndarray:
    """The weather's genus channels *names* (:data:`weather.SYNOPTIC_CHANNELS`)
    at the cube's directions, smooth, upsampled from its own map; zeros
    without :attr:`GeneraSpec.background`.

    :returns: (6 * size * size, len(names)) float64
    """
    if not spec.background:
        return np.zeros((6 * size * size, len(names)))
    dirs = cube_directions(size).reshape(-1, 3)
    picks = [SYNOPTIC_CHANNELS.index(n) for n in names]
    return sample_cube(synoptic_cube(spec.weather)[..., picks], dirs).astype(np.float64)


@functools.cache
def genus_coverage(spec: GeneraSpec, name: str, size: int = GENUS_SIZE) -> np.ndarray:
    """A genus's own coverage map (a genus other than the convective one):
    the weather's (its channels of :func:`weather.synoptic_cube`,
    :data:`WEATHER_GENERA`, with :attr:`GeneraSpec.background`), and over
    it its features' (:attr:`Feature.genus` *name*), each its coverage over
    its core, fading out over its outer third (cached per process).

    :returns: (6, size, size) float32, the layout of
        :func:`sky.starfield.cube_directions`
    """
    dirs = cube_directions(size).reshape(-1, 3)
    if name in WEATHER_GENERA:
        cover = _weather_channels(spec, WEATHER_GENERA[name], size).max(axis=1)
    else:
        cover = np.zeros(len(dirs))
    for f, centre, angle in _feature_dirs(spec):
        if f.genus != name:
            continue
        distance = np.arccos(np.clip(dirs @ centre, -1.0, 1.0)) / angle
        core = 1.0 - _smoothstep(2.0 / 3.0, 1.0, distance)
        cover = np.maximum(cover, f.coverage * core)
    return cover.reshape(6, size, size).astype(np.float32)


@functools.cache
def genus_planes(spec: GeneraSpec, size: int = GENUS_SIZE) -> np.ndarray:
    """The convective genus's maps as cube maps (cached per process): the
    weather's (:func:`weather.coverage_map` and
    :func:`weather.synoptic_cube`, smooth, upsampled from their own maps),
    the features over them.

    :returns: (6, size, size, 4) float32: the convective coverage, the
        convective depth, the stratus's coverage (:func:`genus_coverage`) and
        the anvil shelf's spread, the layout of
        :func:`sky.starfield.cube_directions`
    """
    dirs = cube_directions(size).reshape(-1, 3)
    convective = sample_cube(coverage_map(spec.weather), dirs).astype(np.float64)
    if not spec.background:
        convective = np.zeros(len(dirs))
    weather = _weather_channels(spec, ("depth", "shelf"), size)
    depth, shelf = weather[:, 0], weather[:, 1]
    stratiform = genus_coverage(spec, "stratus", size).reshape(-1)
    for f, centre, angle in _feature_dirs(spec):
        if f.genus not in CONVECTIVE_SPECIES:
            continue  # a map of its own (genus_coverage)
        distance = np.arccos(np.clip(dirs @ centre, -1.0, 1.0)) / angle
        core = 1.0 - _smoothstep(2.0 / 3.0, 1.0, distance)
        if f.genus == "cumulonimbus":
            # Depth 1 over the core and around it, where the sparser columns
            # hold cloud mostly in the anvil's band (an overhang); congestus
            # beyond, gone by 2.2 radii.
            deep = 1.0 - _smoothstep(1.2, 1.6, distance)
            ring = 1.0 - _smoothstep(1.6, 2.2, distance)
            depth = np.maximum(depth, np.maximum(deep, 0.5 * ring))
            cover = np.maximum(f.coverage * core, _CB_RING_COVERAGE * ring)
            wind = (
                np.asarray(spec.wind, dtype=np.float64)
                if spec.wind is not None
                else upper_wind_at(spec.weather, centre)
            )
            shelf = np.maximum(shelf, anvil_spread(dirs, centre, angle, wind))
        else:
            depth = np.maximum(depth, GENUS_DEPTHS.get(f.genus, 0.5) * core)
            cover = f.coverage * core
        convective = np.maximum(convective, cover)
    planes = np.stack([convective, depth, stratiform, shelf], axis=-1)
    return planes.reshape(6, size, size, 4).astype(np.float32)


def depth_shift_tables(
    volume: np.ndarray, field_: DensityField, radius: float, octaves: int = 4
) -> np.ndarray:
    """The convective field's fbm shift by coverage, one table per depth of
    :data:`DEPTHS` (a deep column may grow taller, so it needs less shift
    for the same coverage).

    :param field_: the resolved field (its threshold set at depth 0)
    :returns: (len(DEPTHS), SHIFT_TABLE_SIZE) float64
    """
    return np.array(
        [
            shift_table(
                sphere_column_peaks(volume, field_, radius, octaves, depth=d),
                field_.threshold[0],
            )
            for d in DEPTHS
        ]
    )


def depth_shift(
    coverage: np.ndarray, depth: np.ndarray, tables: np.ndarray
) -> np.ndarray:
    """The fbm shift giving *coverage* at convective *depth*: the tables'
    shifts at the two depths around it, interpolated.

    :param tables: :func:`depth_shift_tables`
    """
    coverage = np.asarray(coverage, dtype=np.float64)
    depth = np.clip(np.asarray(depth, dtype=np.float64), 0.0, 1.0)
    at = np.stack([coverage_to_shift(coverage, t) for t in tables], axis=-1)
    position = depth * (len(DEPTHS) - 1)
    low = np.minimum(np.floor(position).astype(int), len(DEPTHS) - 2)
    frac = position - low
    pick = np.take_along_axis
    lo = pick(at, low[..., None], axis=-1)[..., 0]
    hi = pick(at, low[..., None] + 1, axis=-1)[..., 0]
    return lo + (hi - lo) * frac


def genus_cube(
    spec: GeneraSpec, kind: str, tables: tuple, size: int = GENUS_SIZE
) -> np.ndarray:
    """The primary genus's weather map (:func:`weather.build_weather_texture`
    uploads it), as :func:`weather.weather_cube` is the single field's. The
    other genera read a coverage cube (:func:`coverage_cube`).

    :param kind: :data:`CONVECTIVE` (the primary genus's kind)
    :param tables: its shift tables, one per depth of :data:`DEPTHS`
        (:func:`depth_shift_tables`)
    :returns: (6, size, size, 4) float32: coverage, the fbm shift giving it,
        the convective depth, the anvil shelf's spread
    """
    if kind != CONVECTIVE:
        raise ValueError(f"unknown primary genus kind {kind!r}")
    planes = genus_planes(spec, size)
    cover, depth = planes[..., 0], planes[..., 1]
    shift = depth_shift(cover, depth, np.asarray(tables))
    channels = [cover, shift, depth, planes[..., 3]]
    return np.stack([c.astype(np.float32) for c in channels], axis=-1)


def coverage_cube(
    spec: GeneraSpec, cube: str = "A", size: int = GENUS_SIZE
) -> np.ndarray:
    """A coverage cube of the union march: each genus of *spec* whose
    coverage is a channel of *cube* (:attr:`GenusPreset.coverage`) has its
    coverage map there (:func:`genus_coverage`: its own features'), and cube
    B the high genera's in channels 1-3 (:data:`HIGH_CHANNELS`), the
    others' channels 0. The shift comes from each genus's own table
    (:func:`genus_weather_at`, ``genusShift`` on the GPU).

    :param cube: ``"A"`` or ``"B"``
    :returns: (6, size, size, 4) float32
    """
    out = np.zeros((6, size, size, 4), dtype=np.float32)
    for name in spec.genera:
        source, channel = GENUS_PRESETS[name].coverage
        if source == cube:
            out[..., channel] = genus_coverage(spec, name, size)
    if cube == "B":
        # The high genera's (channels 1-3), those a feature covers
        for name, channel in high_genera(spec, covered=True):
            out[..., channel] = genus_coverage(spec, name, size)
        if spec.cirrus_weather and "cirrus" in spec.high:
            # The weather's cirrus cover too, under the features
            dirs = cube_directions(size).reshape(-1, 3)
            weather = sample_cube(cirrus_cover_map(spec.weather), dirs)
            channel = HIGH_CHANNELS["cirrus"]
            out[..., channel] = np.maximum(
                out[..., channel], weather.reshape(6, size, size)
            )
    return out


def genus_weather_at(
    cube: np.ndarray, channel: int, table: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A genus's weather at *points* from a coverage cube (the shader's
    ``genusWeather``): the coverage filtered off its channel, then the shift
    its table gives that coverage.

    :param cube: (6, n, n, 4) :func:`coverage_cube`
    :param table: its shift table (:func:`weather.shift_table`)
    :param points: (..., 3) planet frame
    :returns: (coverage, shift), each (...)
    """
    coverage = sample_cube(cube[..., channel], points)
    return coverage, coverage_to_shift(coverage, np.asarray(table))
