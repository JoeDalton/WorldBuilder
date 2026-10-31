"""
High cirrus over a planet: one sheet, fibres along the upper wind.

Cirrus is thin ice cloud, a few hundred metres to a kilometre deep and tens of
kilometres across, optically thin (it hides little of the sky behind it). The
billboards that draw the cumulus (:mod:`sphere`) would give it no volume worth
their cost, and their puffs fight its fibres. So it is a **surface**: a sphere at
the slab's mid-height whose fragments evaluate a fibrous density through the
slab's depth (``thickness / |cos|``), like the cumulus's far deck.

**The fibres** follow the upper wind (:mod:`weather`: the jet streams, bent
around the cyclones). Its stream function psi is constant along the wind, so
noise sampled at ``(psi / spacing, 0, 0) + p / length`` varies fast across the
wind and slowly along it: streaks exactly along the flow, wherever it turns,
with no frame to rotate (a stretched noise turned to follow a direction field
shears). Where the wind is weak, psi varies slowly and the same noise turns
isotropic: a veil instead of streaks, and that air is where the weather puts
little cirrus anyway. Finer fibrils (the same construction at a smaller
spacing) give the hair inside a streak; a slow noise added to psi makes the
fibres meander rather than draw the contour lines.

**How much** follows the cirrus cover map (:func:`weather.cirrus_cover_map`),
turned into the density's threshold as the cumulus's coverage is
(:func:`calibrate`): the share of the sheet with cirrus *is* the map's cover.
Far away, where a pixel spans more than a streak, the sheet fades to that
cover (no speckle from orbit).

**Draw order**, against the cumulus below and anything else translucent. The
cirrus lies above the cumulus slab, and a ray's height along it is convex (it
falls, then rises), so where a ray *enters* the cirrus sphere it is above every
cumulus (the cirrus is in front of all of them), and after it *exits* it never
comes down again (the cirrus is behind all of them). For any camera: so the
same mesh is drawn twice, each node keeping one kind of crossing, found per
fragment from the pass's own camera (the sea's mirror included):

* ``behind`` (exits) in the sky's bin after the sky dome and the stars, before
  every cloud bin;
* ``front`` (entries) in Panda's "transparent" bin, after every cloud bin;
  sorted there by its centre (the planet's), so a game's translucent effects
  nearer than that still go on top, as they do over the cumulus.

**The noise** is computed, not fetched: hashed value noise with quintic
interpolation (C2). A streak magnifies its noise ~50 times (a lattice cell is
2 km across the wind, tens of pixels), and through the density's threshold
a 3D texture's linear filter showed its creases as a lattice of facets, and
a smoothed fraction its flat texel edges as one of bands.

Lit as the cumulus are (the ice's phase function, the planet's air reddening
the sunlight and twilight at the cirrus's own height: it stays pink after the
cumulus below have gone grey), seen through the air in front of it per pixel.
No sun march: the sheet barely shadows itself.

Design notes: docs/source/atmosphere.md, "Cirrus".
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import (
    ColorBlendAttrib,
    GeomNode,
    LVecBase3f,
    LVecBase4f,
    NodePath,
    PTA_LVecBase3f,
    PTA_LVecBase4f,
    SamplerState,
    Shader,
    Texture,
    TransparencyAttrib,
    Vec3,
    Vec4,
)

from world_builder.atmosphere import Atmospheres, bind_defaults
from world_builder.clouds.cloud import PRESETS, CloudSettings, CloudType
from world_builder.clouds.noise import (
    OCTAVE_OFFSETS,
    CloudOptics,
    clamped_anisotropy,
    field_offset,
    phase_weights,
)
from world_builder.clouds.sphere import deck_mesh
from world_builder.clouds.weather import (
    WeatherSpec,
    build_weather_texture,
    cirrus_cover_map,
    sample_cube,
    stream_function,
    stream_table,
    stream_vortices,
)
from world_builder.depth import REFLECTION_CAMERA_MASK
from world_builder.sky.sky import SKY_BIN, ensure_sky_bin
from world_builder.terrain.planet import Planet, PlanetSpec

if TYPE_CHECKING:
    from world_builder.terrain.workers import ChunkExecutor

_SHADERS = Path(__file__).parent / "shaders"
#: Quads per face edge of the sheet: its facets sag ~45 m under the sphere on
#: a 600 km planet (~480 m on an Earth-sized one), which only the depth test
#: against the terrain sees (the shading is on the true sphere).
CIRRUS_QUADS = 64
#: The most weather systems the shader bends the fibres around
#: (``uVortexA/B`` in cirrus.frag).
MAX_VORTICES = 16
#: Coverages the threshold table is tabulated at, evenly over [0, 1].
THRESHOLD_TABLE_SIZE = 256
#: Noise samples the threshold is calibrated from: its quantiles to ~0.2 %.
CALIBRATION_SAMPLES = 1 << 18
#: The behind node's draw order in the sky's bin: after the dome (0) and the
#: stars (1).
BEHIND_DRAW_ORDER = 2
#: Panda's own bin for translucent objects, sorted back to front.
FRONT_BIN = "transparent"
#: Weights of the streak's two octaves (the second at twice the frequency).
STREAK_WEIGHTS = (0.65, 0.35)
#: How much of x the fibres' slow coordinates take (:func:`along`).
ALONG_MIX = 0.7


@dataclass(frozen=True)
class CirrusSpec:
    """What a planet's cirrus looks like. Lengths in metres; the fibres' are
    the cloud's own, not scaled with the planet.

    :ivar base: the slab's base above sea level (it must clear the cumulus)
    :ivar thickness: the slab's depth (a sight line crosses thickness / |cos|)
    :ivar amount: the weather's cirrus cover is multiplied by it
    :ivar streak_spacing: between streaks across the wind where the jet blows
        its fastest (wider where it is slower)
    :ivar streak_length: a streak's length along the wind
    :ivar veil_size: the features of the veil the streaks are mixed with
    :ivar directionality: the streaks' share against the veil, in [0, 1]
    :ivar fibril_spacing: between fibrils (the hair inside a streak)
    :ivar fibril_length: a fibril's length
    :ivar fibril_strength: how much density the fibrils take out between them
    :ivar meander: how far the fibres wander off the wind's contours, in
        stream-function metres (metres across the wind at the jets' speed)
    :ivar meander_wavelength: over what distance they wander
    :ivar edge_softness: the density's rise from its threshold, in standard
        deviations of the raw noise (narrow = crisp streak edges)
    :ivar extinction: optical depth per metre of fully dense cirrus
    :ivar max_optical_depth: the most a sight line through the sheet takes
    :ivar optics: the phase function and the light's fill (the ``CIRRUS``
        preset's: ice, strongly forward)
    """

    base: float = 8000.0
    thickness: float = 1000.0
    amount: float = 1.0
    streak_spacing: float = 2000.0
    streak_length: float = 25000.0
    veil_size: float = 6000.0
    directionality: float = 0.75
    fibril_spacing: float = 250.0
    fibril_length: float = 3000.0
    fibril_strength: float = 0.7
    meander: float = 6000.0
    meander_wavelength: float = 60000.0
    edge_softness: float = 1.2
    extinction: float = 6e-4
    max_optical_depth: float = 1.2
    optics: CloudOptics = PRESETS[CloudType.CIRRUS].optics

    @property
    def top(self) -> float:
        """The slab's top above sea level, metres."""
        return self.base + self.thickness

    @property
    def mid(self) -> float:
        """Where the sheet is drawn, above sea level, metres."""
        return self.base + 0.5 * self.thickness


# ── The density, on the CPU as the shader evaluates it ─────────────────────────

#: Noise-space offsets decorrelating the streaks, the veil, the fibrils and the
#: meanders.
OFFSETS = {
    name: np.array(field_offset(seed))
    for name, seed in (("streak", 8), ("veil", 9), ("fibril", 10), ("meander", 11))
}


@dataclass(frozen=True)
class Calibration:
    """The density's threshold by coverage, from the noise's own statistics.

    :ivar thresholds: (THRESHOLD_TABLE_SIZE,) the raw value whose excess
        has each coverage ``i / (size - 1)``: its (1 - c) quantile
    :ivar edge: the smoothstep's width above the threshold, raw units
    :ivar mean_density: the mean density over the sheet's cloudy part (the far
        sheet's, at a coverage of one half)
    :ivar octave_mean: the noise's mean (a faded fibril counts as it)
    """

    thresholds: np.ndarray
    edge: float
    mean_density: float
    octave_mean: float


#: The hash's constants: MUST match cirrus.frag's ``hashCell``.
_HASH_AXES = tuple(np.uint32(v) for v in (0x8DA6B343, 0xD8163841, 0xCB1AB31F))
_HASH_MIX = tuple(np.uint32(v) for v in (0x2C1B3C6D, 0x297A2D39))


def _cell_value(cells: np.ndarray) -> np.ndarray:
    """A lattice cell's value in [0, 1), as cirrus.frag's ``cellValue``.

    :param cells: (..., 3) int64 lattice coordinates
    """
    q = cells.astype(np.uint32)  # wraps as GLSL's uvec3(ivec3) does
    h = (q[..., 0] * _HASH_AXES[0]) ^ (q[..., 1] * _HASH_AXES[1])
    h ^= q[..., 2] * _HASH_AXES[2]
    h ^= h >> np.uint32(15)
    h *= _HASH_MIX[0]
    h ^= h >> np.uint32(12)
    h *= _HASH_MIX[1]
    h ^= h >> np.uint32(15)
    return (h >> np.uint32(8)).astype(np.float64) / 16777216.0


def value_noise(coords: ArrayLike) -> np.ndarray:
    """Hashed value noise with quintic interpolation, as cirrus.frag's
    ``hashNoise`` (to float32 rounding): one feature per unit, in [0, 1).

    :param coords: (..., 3) noise coordinates
    """
    c = np.asarray(coords, dtype=np.float64)
    i = np.floor(c)
    f = c - i
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    cell = i.astype(np.int64)
    out = 0.0
    for dz in (0, 1):
        wz = u[..., 2] if dz else 1.0 - u[..., 2]
        for dy in (0, 1):
            wy = u[..., 1] if dy else 1.0 - u[..., 1]
            for dx in (0, 1):
                wx = u[..., 0] if dx else 1.0 - u[..., 0]
                corner = _cell_value(cell + np.array([dx, dy, dz]))
                out = out + wx * wy * wz * corner
    return out


def along(points: np.ndarray) -> np.ndarray:
    """The slow coordinates along the fibres, as cirrus.frag's ``along``:
    *points* moved into the noise's y and z, so its x carries the stream
    function alone (added to it, they tilt the fibres off the wind). Blind
    along (1, -0.7, -0.7), which is never an east-west wind.

    :param points: (N, 3) metres
    :returns: (N, 3), x zero
    """
    out = np.zeros_like(points)
    out[:, 1] = points[:, 1] + ALONG_MIX * points[:, 0]
    out[:, 2] = points[:, 2] + ALONG_MIX * points[:, 0]
    return out


def _streak(coords: np.ndarray) -> np.ndarray:
    return STREAK_WEIGHTS[0] * value_noise(coords) + STREAK_WEIGHTS[1] * value_noise(
        coords * 2.0 + np.array(OCTAVE_OFFSETS[1])
    )


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@functools.cache
def calibrate(spec: CirrusSpec, seed: int = 0) -> Calibration:
    """The threshold giving each coverage, and what the far sheet needs
    (cached).

    The noise is stationary, so the raw density's distribution is the same
    anywhere on the sheet: the coverage c is ``P(raw > threshold)`` and the
    threshold its (1 - c) quantile over random coordinates, the inverse of
    the coverage function, as for the cumulus (:func:`noise.column_peaks`).

    :param seed: the sampling's seed
    """
    rng = np.random.default_rng(seed)
    n = CALIBRATION_SAMPLES
    span = 1e4
    streak = _streak(rng.uniform(-span, span, (n, 3)))
    veil = value_noise(rng.uniform(-span, span, (n, 3)))
    fibril = value_noise(rng.uniform(-span, span, (n, 3)))
    raw = spec.directionality * streak + (1.0 - spec.directionality) * veil
    coverage = np.linspace(0.0, 1.0, THRESHOLD_TABLE_SIZE)
    thresholds = np.quantile(raw, 1.0 - coverage)
    # The ends a hair past the extremes: 0 is clear, 1 overcast.
    thresholds[0] += 1e-4
    thresholds[-1] -= 1e-4
    edge = spec.edge_softness * float(raw.std())
    half = float(np.quantile(raw, 0.5))
    cloudy = raw > half
    density = _smoothstep(half, half + edge, raw[cloudy]) * (
        1.0 - spec.fibril_strength * (1.0 - fibril[cloudy])
    )
    return Calibration(
        thresholds=thresholds,
        edge=edge,
        mean_density=float(density.mean()),
        octave_mean=float(veil.mean()),
    )


def threshold_for(coverage: ArrayLike, calibration: Calibration) -> np.ndarray:
    """:returns: the raw threshold giving each coverage, linear in the table"""
    table = calibration.thresholds
    return np.interp(coverage, np.linspace(0.0, 1.0, len(table)), table)


@dataclass(frozen=True)
class CirrusWeather:
    """What the cirrus draws from, made once in the pool.

    :ivar cube: (6, n, n, 2) float32: cover, and the raw threshold giving it
    :ivar stream: the zonal stream function by latitude (:func:`weather.stream_table`)
    :ivar vortices: the weather systems' terms (:func:`weather.stream_vortices`),
        at most :data:`MAX_VORTICES`
    :ivar calibration: the threshold's (:func:`calibrate`)
    """

    cube: np.ndarray
    stream: np.ndarray
    vortices: np.ndarray
    calibration: Calibration


@functools.cache
def cirrus_weather(spec: CirrusSpec, weather: WeatherSpec) -> CirrusWeather:
    """Everything the cirrus needs from its weather and its noise (about a
    second, then cached per process). Top-level and picklable: run in the
    pool.

    :param spec: the cirrus
    :param weather: the planet's weather (the cumulus's, for the same systems)
    """
    calibration = calibrate(spec)
    cover = np.clip(cirrus_cover_map(weather) * spec.amount, 0.0, 1.0)
    threshold = threshold_for(cover, calibration).astype(np.float32)
    rows = stream_vortices(weather, Planet(weather.planet))
    # The strongest bend the fibres most
    rows = rows[np.argsort(-np.abs(rows[:, 4]), kind="stable")][:MAX_VORTICES]
    return CirrusWeather(
        cube=np.stack([cover, threshold], axis=-1),
        stream=stream_table(weather),
        vortices=rows,
        calibration=calibration,
    )


def fibre_coordinates(
    spec: CirrusSpec,
    data: CirrusWeather,
    points: np.ndarray,
) -> np.ndarray:
    """The stream function with its meanders at sheet *points*, as the
    shader's ``fibreStream``: what the fibres run along.

    :param points: (N, 3) planet frame, metres
    :returns: (N,) metres
    """
    dirs = points / np.linalg.norm(points, axis=1, keepdims=True)
    psi = stream_function(data.stream, data.vortices, dirs)
    wander = value_noise(points / spec.meander_wavelength + OFFSETS["meander"])
    return psi + spec.meander * 2.0 * (wander - data.calibration.octave_mean)


def cirrus_density(
    spec: CirrusSpec,
    data: CirrusWeather,
    points: np.ndarray,
) -> np.ndarray:
    """The density at sheet *points* as cirrus.frag evaluates it close up
    (no fading with distance).

    :param points: (N, 3) planet frame, metres
    :returns: (N,) in [0, 1]
    """
    points = np.asarray(points, dtype=np.float64)
    psi = fibre_coordinates(spec, data, points)
    slow = along(points)
    across = np.zeros_like(points)
    across[:, 0] = psi / spec.streak_spacing
    streak = _streak(across + slow / spec.streak_length + OFFSETS["streak"])
    veil = value_noise(points / spec.veil_size + OFFSETS["veil"])
    raw = spec.directionality * streak + (1.0 - spec.directionality) * veil
    threshold = sample_cube(data.cube, points)[:, 1]
    density = _smoothstep(threshold, threshold + data.calibration.edge, raw)
    across[:, 0] = psi / spec.fibril_spacing
    fibril = value_noise(across + slow / spec.fibril_length + OFFSETS["fibril"])
    return density * (1.0 - spec.fibril_strength * (1.0 - fibril))


# ── Drawing ────────────────────────────────────────────────────────────────────


@functools.cache
def _shader() -> Shader:
    return Shader.load(
        Shader.SL_GLSL,
        vertex=_SHADERS / "cirrus.vert",
        fragment=_SHADERS / "cirrus.frag",
    )


def _stream_texture(table: np.ndarray) -> Texture:
    """The stream function table as a 1-row float32 texture (read with
    texelFetch and interpolated in the shader, in full float precision)."""
    tex = Texture("cirrus_stream")
    tex.setup_2d_texture(len(table), 1, Texture.T_float, Texture.F_r32)
    tex.set_ram_image(np.ascontiguousarray(table, dtype=np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_minfilter(SamplerState.FT_nearest)
    return tex


class CloudSheet:
    """A planet's high cirrus as a surface: drawn behind and in front of the
    clouds below it (see the module). The sheet layer of
    :class:`planet.PlanetClouds`, which a game builds instead, above a
    :class:`sphere.CloudVolume`.

    Usage::

        sheet = CloudSheet(render, planet_radius, executor,
                           atmospheres=atmospheres, atmosphere=0)
        sheet.set_sun(sun_direction)
        # every frame, camera in the world frame (float64):
        sheet.update(camera_world)

    :ivar root: the node both of its nodes hang under (hide it to hide it)
    :ivar behind: the node drawing where sight lines leave the sheet
    :ivar front: the node drawing where they enter it
    :ivar spec: the cirrus
    :ivar data: what it draws from (:class:`CirrusWeather`); None until the
        set-up is done
    :ivar stats: ``pending`` 1 while the set-up runs in the pool
    """

    #: The settings it takes (``WorldSettings.apply``, ``settings_for``): the
    #: clouds' (only ``reflected`` applies).
    SETTINGS_TYPE: ClassVar = CloudSettings

    def __init__(
        self,
        parent: NodePath,
        radius: float,
        executor: ChunkExecutor,
        position: ArrayLike = (0.0, 0.0, 0.0),
        atmospheres: Atmospheres | None = None,
        atmosphere: int | None = None,
        spec: CirrusSpec = CirrusSpec(),
        planet: PlanetSpec | None = None,
        weather: WeatherSpec | None = None,
        weather_seed: int = 11,
        settings: CloudSettings | None = None,
        sun_brightness: float = 2.8,
        sky_strength: float = 0.35,
        exposure: float = 0.9,
    ):
        """
        :param parent: usually ``render``
        :param radius: the planet's sea-level radius, metres
        :param executor: the world's :class:`terrain.workers.ChunkExecutor`
            (the set-up runs in it)
        :param position: the planet centre in the world frame (float64)
        :param atmospheres: the world's atmospheres (the cirrus is seen
            through them)
        :param atmosphere: the planet's own atmosphere index (reddens the
            cirrus's sunlight); None = none
        :param spec: the cirrus (:class:`CirrusSpec`)
        :param planet: the planet the weather follows; None = an earthlike
            one of this radius
        :param weather: the weather in full (the cumulus's, for the same
            systems); None = the defaults for *planet* with *weather_seed*
        :param weather_seed: the weather systems' seed
        :param settings: :class:`cloud.CloudSettings` (``reflected``); None =
            HIGH
        :param sun_brightness: direct-term gain (as the cumulus's)
        :param sky_strength: ambient sky weight (live: :meth:`set_sky_light`)
        :param exposure: radiance scale before the sky's tonemap
        """
        settings = settings or CloudSettings()
        ensure_sky_bin()
        self.radius = float(radius)
        self.position = np.asarray(position, dtype=np.float64)
        self.spec = spec
        self.stats = {"pending": 0}
        if weather is None:
            planet = planet or PlanetSpec(name="clouds", radius=self.radius)
            weather = WeatherSpec(planet, seed=weather_seed)
        self._job = executor.submit_once(cirrus_weather, spec, weather)
        self.data: CirrusWeather | None = None
        self._settings = settings
        self._pixel_angle = 0.001
        self._sky_light: float | None = None

        self.root = parent.attach_new_node("cirrus")
        self.root.stash()  # out of every camera until the set-up is done
        sheet = self.radius + spec.mid
        mesh = deck_mesh(sheet, CIRRUS_QUADS, name="cirrus_behind")
        twin = GeomNode("cirrus_front")
        twin.add_geom(mesh.modify_geom(0))  # the same vertices, drawn again
        twin.set_bounds(mesh.get_bounds())
        twin.set_final(True)
        self.behind = self.root.attach_new_node(mesh)
        self.front = self.root.attach_new_node(twin)
        self.behind.set_bin(SKY_BIN, BEHIND_DRAW_ORDER)
        self.front.set_bin(FRONT_BIN, 0)
        self.behind.set_shader_input("uFront", 0)
        self.front.set_shader_input("uFront", 1)

        root = self.root
        root.set_shader(_shader())
        root.set_two_sided(True)
        root.set_transparency(TransparencyAttrib.M_none)
        root.set_depth_test(True)
        root.set_depth_write(False)
        root.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add,
                ColorBlendAttrib.O_one,
                ColorBlendAttrib.O_one_minus_incoming_alpha,
            )
        )
        bind_defaults(root)
        if atmospheres is not None:
            atmospheres.bind(root)
        self._sun = PTA_LVecBase3f.empty_array(1)
        self._sun[0] = LVecBase3f(0.0, 0.0, 1.0)
        o = spec.optics
        forward, backward, isotropic = phase_weights(o)
        root.set_shader_inputs(
            planetRadius=self.radius,
            uSheetRadius=sheet,
            uThickness=float(spec.thickness),
            uOwnAtmosphere=-1 if atmosphere is None else int(atmosphere),
            sunDir=self._sun,
            sunColor=Vec3(1.0, 1.0, 1.0),
            skyColor=Vec3(0.45, 0.6, 0.85),
            sunBrightness=float(sun_brightness),
            skyStrength=float(sky_strength),
            exposure=float(exposure),
            uPixelAngle=self._pixel_angle,
            uCloudDebug=0,
            uScales=Vec4(
                1.0 / spec.streak_spacing,
                1.0 / spec.streak_length,
                1.0 / spec.veil_size,
                1.0 / spec.meander_wavelength,
            ),
            uFibril=Vec4(
                1.0 / spec.fibril_spacing,
                1.0 / spec.fibril_length,
                spec.meander,
                spec.directionality,
            ),
            uOffsetStreak=Vec3(*OFFSETS["streak"]),
            uOffsetVeil=Vec3(*OFFSETS["veil"]),
            uOffsetFibril=Vec3(*OFFSETS["fibril"]),
            uOffsetMeander=Vec3(*OFFSETS["meander"]),
            uPhase=Vec4(forward, backward, isotropic, clamped_anisotropy(o)),
            uLight=Vec4(
                o.multiple_scattering, o.shadow_strength, spec.fibril_strength, 0.0
            ),
            uVortexCount=0,
        )
        self._apply_live(settings)

    # ── settings ──────────────────────────────────────────────────────────────

    @property
    def settings(self) -> CloudSettings:
        """What the clouds cost (:meth:`set_settings`)."""
        return self._settings

    def set_settings(self, settings: CloudSettings) -> None:
        """Take the clouds' settings: only ``reflected`` applies (live).

        :param settings: the new settings
        """
        self._settings = settings
        self._apply_live(settings)

    def _apply_live(self, settings: CloudSettings) -> None:
        if settings.reflected:
            self.root.show(REFLECTION_CAMERA_MASK)
        else:
            self.root.hide(REFLECTION_CAMERA_MASK)

    # ── live ──────────────────────────────────────────────────────────────────

    def set_sun(
        self,
        direction: ArrayLike,
        sun_color: ArrayLike | None = None,
        sky_color: ArrayLike | None = None,
    ) -> None:
        """Move the sun and/or restate its colours (uniform writes only).

        :param direction: toward the sun, world frame (need not be unit)
        :param sun_color: RGB of direct sunlight, or None to keep; white by
            default, since the planet's own air colours it
        :param sky_color: RGB of the ambient sky light, or None to keep
        """
        d = np.asarray(direction, dtype=float)
        self._sun[0] = LVecBase3f(*(d / np.linalg.norm(d)))
        if sun_color is not None:
            self.root.set_shader_input("sunColor", Vec3(*sun_color))
        if sky_color is not None:
            self.root.set_shader_input("skyColor", Vec3(*sky_color))

    def set_sky_light(self, strength: float) -> None:
        """The sky's (ambient) light on the cirrus, as ``sky_strength``."""
        if strength != self._sky_light:
            self._sky_light = strength
            self.root.set_shader_input("skyStrength", float(strength))

    def set_pixel_angle(self, radians: float) -> None:
        """The angle one pixel spans (vertical fov / window height): how far
        the fibrils, then the streaks, are drawn before fading to their
        average (0.001 until set).

        :param radians: the angle per pixel
        """
        if radians != self._pixel_angle:  # set every frame by World
            self._pixel_angle = radians
            self.root.set_shader_input("uPixelAngle", float(radians))

    def set_debug(self, flags: int) -> None:
        """Profiling switches (``uCloudDebug`` bits): 2 no atmosphere, 8 flat
        colour after the density.

        :param flags: the bits, 0 = normal
        """
        self.root.set_shader_input("uCloudDebug", int(flags))

    def update(
        self, camera_world: ArrayLike, camera_render: ArrayLike = (0.0, 0.0, 0.0)
    ) -> None:
        """Per frame: finish the set-up when it is in, and place the sheet.

        :param camera_world: camera in the world frame, metres (float64)
        :param camera_render: camera in the render frame (the origin when
            the scene is drawn camera-relative)
        """
        if self._job is not None:
            if not self._job.done():
                self.stats["pending"] = 1
                return
            self._finish()
        cam = np.asarray(camera_world, dtype=np.float64) - self.position
        self.root.set_pos(*(np.asarray(camera_render, dtype=np.float64) - cam))

    def remove(self) -> None:
        """Take the cirrus off the scene."""
        self.root.remove_node()

    def _finish(self) -> None:
        """The set-up's results become the shader's inputs, and the sheet
        is shown."""
        data = self._job.result()
        self._job = None
        self.data = data
        c = data.calibration
        vortex_a = PTA_LVecBase4f.empty_array(MAX_VORTICES)
        vortex_b = PTA_LVecBase4f.empty_array(MAX_VORTICES)
        for i, (x, y, z, k, a) in enumerate(data.vortices):
            vortex_a[i] = LVecBase4f(x, y, z, k)
            vortex_b[i] = LVecBase4f(a, 0.0, 0.0, 0.0)
        self.root.set_shader_inputs(
            cirrusMap=build_weather_texture(data.cube),
            streamTable=_stream_texture(data.stream),
            uVortexA=vortex_a,
            uVortexB=vortex_b,
            uVortexCount=len(data.vortices),
            uShape=Vec4(
                c.edge,
                self.spec.extinction,
                self.spec.max_optical_depth,
                c.mean_density,
            ),
            uFar=Vec4(
                self.spec.streak_spacing,
                self.spec.fibril_spacing,
                c.octave_mean,
                0.0,
            ),
        )
        self.stats["pending"] = 0
        self.root.unstash()

    def __repr__(self) -> str:
        return f"CloudSheet(radius={self.radius:g}, base={self.spec.base:g})"
