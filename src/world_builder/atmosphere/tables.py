"""
The atmospheres' precomputed tables (Hillaire 2020, "A Scalable and
Production Ready Sky and Atmosphere Rendering Technique"; RENDERING_PLAN.md
part B).

* **Transmittance** (:data:`TRANSMITTANCE_SIZE`): the sunlight reaching radius
  r from a direction at cosine mu from the vertical, through the whole air:
  exactly :meth:`~world_builder.atmosphere.Atmospheres.transmittance_to_sun`
  (:func:`~world_builder.atmosphere.depth_toward`, with the
  ``sunset_radius`` curvature), sampled in Bruneton's (r, mu) mapping, which
  crowds texels toward the horizon, where it changes fastest. Rays into the
  ground are not in it (0).
  The ozone layer's absorption (a tent at 25 km, Hillaire's) is not in the
  Chapman function: its optical depth is marched per texel (alpha, km) and
  taken in (the twin reads it back from here, :func:`ozone_toward`).
* **Multiple scattering** (:data:`MULTIPLE_SIZE`): Hillaire's infinite-order
  isotropic approximation, the light scattered more than once reaching a
  point at radius r with the sun at cosine mu_s from its vertical, per unit
  of scattering and of sunlight (Psi_ms): the second order gathered from 64
  directions around the point (the ground's bounce with them,
  :attr:`~world_builder.atmosphere.AtmosphereParams.ground_albedo`), and the
  higher orders as the geometric series of the share the air scatters
  back, ``1 / (1 - f_ms)``. A view ray's sample adds ``Psi_ms *
  scattering`` (no phase function) to its single scattering: the planet's
  shadow lit blue-grey at dusk, the deep sky, the haze's glow by day.

The tables depend on an atmosphere's parameters only (fixed once added), so
they are computed here, on the CPU, from the CPU twin's own formulas, and
uploaded: they match the twin by construction, need no window, and cost ~0.5 s per
atmosphere (the multiple scattering; kept per planet and air). Both atmospheres
(``MAX_ATMOSPHERES``) share one texture, stacked along v; the GLSL reads it with
``atmoTransmittance`` (atmosphere.glsl).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike
from panda3d.core import SamplerState, Texture

#: Transmittance table texels per atmosphere: (mu, r).
TRANSMITTANCE_SIZE = (256, 64)
#: Multiple-scattering table texels per atmosphere: (mu_s, r).
MULTIPLE_SIZE = (32, 32)
#: Directions and steps per texel of the multiple-scattering table: 64 as
#: Hillaire's; the steps a uniform 40 (his 20 are placed with care).
MULTIPLE_DIRECTIONS = 64
MULTIPLE_STEPS = 40


def _unit_to_texel(x: ArrayLike, size: int) -> np.ndarray:
    """[0, 1] to texture coordinates through the first and last texel's
    centres (the table's ends are exact, not half-texel off)."""
    return 0.5 / size + np.asarray(x, float) * (1.0 - 1.0 / size)


def _texel_to_unit(u: ArrayLike, size: int) -> np.ndarray:
    return (np.asarray(u, float) - 0.5 / size) / (1.0 - 1.0 / size)


def transmittance_uv(
    r: ArrayLike, mu: ArrayLike, ground: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    """Where (r, mu) is in one atmosphere's transmittance table: Bruneton
    2017's mapping (x_mu: the distance to the top between its least and
    most at r, x_r: the horizon distance's share), as texture coordinates
    in [0, 1] within that atmosphere's rows.

    :param r: radius from the centre, clamped to [ground, top]
    :param mu: cosine of the direction from the vertical, from the horizon's
        (``-sqrt(1 - (ground/r)^2)``) up
    """
    r = np.clip(np.asarray(r, float), ground, top)
    mu = np.clip(np.asarray(mu, float), -1.0, 1.0)
    h = np.sqrt(top * top - ground * ground)
    rho = np.sqrt(np.maximum(r * r - ground * ground, 0.0))
    d = np.maximum(
        -r * mu + np.sqrt(np.maximum(r * r * (mu * mu - 1.0) + top * top, 0.0)), 0.0
    )
    d_min = top - r
    d_max = rho + h
    x_mu = (d - d_min) / np.maximum(d_max - d_min, 1e-9)
    x_r = rho / h
    width, height = TRANSMITTANCE_SIZE
    return _unit_to_texel(np.clip(x_mu, 0, 1), width), _unit_to_texel(x_r, height)


def transmittance_r_mu(
    u: ArrayLike, v: ArrayLike, ground: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of :func:`transmittance_uv`: the (r, mu) at texture
    coordinates (u, v)."""
    width, height = TRANSMITTANCE_SIZE
    x_mu = _texel_to_unit(u, width)
    x_r = _texel_to_unit(v, height)
    h = np.sqrt(top * top - ground * ground)
    rho = h * x_r
    r = np.sqrt(rho * rho + ground * ground)
    d_min = top - r
    d_max = rho + h
    d = d_min + x_mu * (d_max - d_min)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(d == 0.0, 1.0, (h * h - rho * rho - d * d) / (2.0 * r * d))
    return r, np.clip(mu, -1.0, 1.0)


#: Steps of the ozone's optical depth along a texel's ray.
OZONE_STEPS = 512


def ozone_density(altitude: ArrayLike, params) -> np.ndarray:
    """The ozone layer's density over its peak's at *altitude*: a tent
    (Hillaire 2020's, Bruneton's), 1 at ``ozone_altitude``, 0 beyond
    ``ozone_width`` either side."""
    return np.maximum(
        0.0,
        1.0
        - np.abs(np.asarray(altitude, float) - params.ozone_altitude)
        / params.ozone_width,
    )


@lru_cache(maxsize=8)
def ozone_depth_table(radius: float, params) -> np.ndarray:
    """One atmosphere's ozone optical depth toward the sun (metres at peak
    density), (rows r, columns mu) at the transmittance table's texels: a
    march along the ray, over the planet the ``sunset_radius`` curvature
    takes (the Chapman function's slant, ``depth_toward``), through the
    ground's tangent as it does. Kept per planet and air (read only)."""
    from world_builder.atmosphere import _curve

    curve = _curve(radius, params)
    top = radius + params.height
    width, height = TRANSMITTANCE_SIZE
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    u[-1] = _unit_to_texel(1.0 - 1e-6, width)
    r, mu = np.broadcast_arrays(
        *transmittance_r_mu(u[None, :], v[:, None], radius, top)
    )
    big = radius * curve  # the curved planet: same altitudes, wider
    start = big + (r - radius)
    outer = big + params.height
    out = np.empty(r.shape)
    for row in range(height):
        r0, m = start[row], mu[row]
        # out of the curved shell's top along the ray
        length = -r0 * m + np.sqrt(
            np.maximum(r0 * r0 * (m * m - 1.0) + outer * outer, 0.0)
        )
        dt = length / OZONE_STEPS
        t = (np.arange(OZONE_STEPS) + 0.5)[:, None] * dt[None, :]
        x = r0 * np.sqrt(np.maximum(1.0 - m * m, 0.0)) + 0.0 * t
        z = r0 * m + t
        altitude = np.sqrt(x * x + z * z) - big
        out[row] = np.sum(ozone_density(altitude, params), axis=0) * dt
    out.setflags(write=False)
    return out


def ozone_toward(r: ArrayLike, mu: ArrayLike, radius: float, params) -> np.ndarray:
    """The ozone's optical depth toward the sun from radii *r* at cosines
    *mu* (:func:`ozone_depth_table`, filtered as GL does), metres at peak."""
    top = radius + params.height
    u, v = transmittance_uv(r, mu, radius, top)
    return bilinear(ozone_depth_table(float(radius), params)[..., None], u, v)[..., 0]


def transmittance_table(radius: float, params) -> np.ndarray:
    """One atmosphere's transmittance table, (rows r, columns mu, rgba):
    rgb :meth:`~world_builder.atmosphere.Atmospheres.transmittance_to_sun`
    at each texel's centre (0 where the planet is in the way), a the ozone's
    optical depth, km (``atmoSunlight`` takes it, with the analytic rest).

    :param radius: the planet's ground radius, metres
    :param params: its :class:`~world_builder.atmosphere.AtmosphereParams`
    """
    from world_builder.atmosphere import _curve, depth_toward

    top = radius + params.height
    width, height = TRANSMITTANCE_SIZE
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    # The last column is the horizon itself, where the twin's "into the
    # ground" test (tangent < ground) can round either way: just above it.
    u[-1] = _unit_to_texel(1.0 - 1e-6, width)
    r, mu = transmittance_r_mu(u[None, :], v[:, None], radius, top)
    heights = np.array([params.rayleigh_height, params.mie_height])
    depth = depth_toward(
        r[..., None], mu[..., None], radius, heights, _curve(radius, params)
    )
    ozone = ozone_depth_table(float(radius), params)
    through = np.exp(
        -(
            np.asarray(params.rayleigh) * depth[..., :1]
            + params.mie_extinction * depth[..., 1:]
            + np.asarray(params.ozone) * ozone[..., None]
        )
    )
    rgb = np.where(depth[..., :1] >= 1e9, 0.0, through)
    return np.concatenate([rgb, ozone[..., None] / 1e3], axis=-1)


def _sun_through(r, cos_sun, radius: float, params) -> np.ndarray:
    """The twin's sunlight through the air (``transmittance_to_sun``'s
    formula) at radii *r* (inside the air) with the sun at *cos_sun*: (...,
    3), 0 where the planet is in the way."""
    from world_builder.atmosphere import _curve, depth_toward

    r = np.asarray(r, float)
    cos_sun = np.asarray(cos_sun, float)
    heights = np.array([params.rayleigh_height, params.mie_height])
    depth = depth_toward(
        r[..., None], cos_sun[..., None], radius, heights, _curve(radius, params)
    )
    ozone = ozone_toward(r, cos_sun, radius, params)
    through = np.exp(
        -(
            np.asarray(params.rayleigh) * depth[..., :1]
            + params.mie_extinction * depth[..., 1:]
            + np.asarray(params.ozone) * ozone[..., None]
        )
    )
    return np.where(depth[..., :1] >= 1e9, 0.0, through)


def sphere_directions(n: int) -> np.ndarray:
    """*n* unit vectors spread evenly over the sphere (a Fibonacci
    lattice), (n, 3)."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + 5.0**0.5) * i
    ring = np.sqrt(1.0 - z * z)
    return np.stack([ring * np.cos(phi), ring * np.sin(phi), z], axis=-1)


def multiple_uv(r: ArrayLike, mu_s: ArrayLike, ground: float, top: float):
    """Where (r, mu_s) is in one atmosphere's multiple-scattering table:
    linear in the sun's cosine and the altitude, through the end texels'
    centres."""
    width, height = MULTIPLE_SIZE
    x = np.clip(np.asarray(mu_s, float) * 0.5 + 0.5, 0.0, 1.0)
    y = np.clip((np.asarray(r, float) - ground) / (top - ground), 0.0, 1.0)
    return _unit_to_texel(x, width), _unit_to_texel(y, height)


@lru_cache(maxsize=8)
def multiple_table(radius: float, params) -> np.ndarray:
    """One atmosphere's multiple-scattering table (rows altitude, columns the
    sun's cosine, rgb Psi_ms; see the module), from 64 directions of 40 steps
    a texel: ~0.5 s, kept for the same planet and air (don't change the
    array returned).

    :param radius: the planet's ground radius, metres
    :param params: its :class:`~world_builder.atmosphere.AtmosphereParams`
    """
    top = radius + params.height
    width, height = MULTIPLE_SIZE
    mu_s = np.linspace(-1.0, 1.0, width)
    # The bottom row a metre up (the ground's own rays all leave upward).
    altitudes = np.maximum(np.linspace(0.0, params.height, height), 1.0)
    dirs = sphere_directions(MULTIPLE_DIRECTIONS)
    steps = MULTIPLE_STEPS
    beta_r = np.asarray(params.rayleigh)
    isotropic = 1.0 / (4.0 * np.pi)
    out = np.empty((height, width, 3))
    for row, altitude in enumerate(altitudes):
        r0 = radius + altitude
        origin = np.array([0.0, 0.0, r0])
        b = dirs @ origin
        out_top = -b + np.sqrt(np.maximum(b * b - (r0 - top) * (r0 + top), 0.0))
        ground = b * b - (r0 - radius) * (r0 + radius)
        hits = (ground > 0.0) & (b < 0.0)
        to_ground = np.where(hits, -b - np.sqrt(np.maximum(ground, 0.0)), np.inf)
        dt = np.minimum(out_top, to_ground) / steps  # (D,)
        t = (np.arange(steps) + 0.5)[None, :] * dt[:, None]  # (D, S)
        p = origin + dirs[:, None, :] * t[..., None]  # (D, S, 3)
        rp = np.linalg.norm(p, axis=-1)
        rayleigh = np.exp(-(rp - radius) / params.rayleigh_height)[..., None]
        mie = np.exp(-(rp - radius) / params.mie_height)[..., None]
        scattering = beta_r * rayleigh + params.mie * mie  # (D, S, 3)
        extinction = (
            beta_r * rayleigh
            + params.mie_extinction * mie
            + np.asarray(params.ozone) * ozone_density(rp - radius, params)[..., None]
        )
        segment = np.exp(-extinction * dt[:, None, None])
        # Each step's light integrated over it (Hillaire's analytic form),
        # through the air back to the point.
        weight = (np.cumprod(segment, axis=1) / segment) * (1.0 - segment) / extinction
        # The share of light the air scatters back from all around, once.
        f_ms = np.mean(np.sum(weight * scattering, axis=1), axis=0) * isotropic
        floor = origin + dirs * np.where(hits, to_ground, 0.0)[:, None]
        floor /= np.maximum(np.linalg.norm(floor, axis=-1, keepdims=True), 1.0)
        through = np.prod(segment, axis=1)  # (D, 3) to the ground
        for column, cosine in enumerate(mu_s):
            sun = np.array([math.sqrt(max(0.0, 1.0 - cosine * cosine)), 0.0, cosine])
            lit = _sun_through(rp, (p @ sun) / rp, radius, params)
            light = np.sum(weight * scattering * lit, axis=1) * isotropic
            # the ground's bounce, Lambertian, where a direction meets it
            facing = np.clip(floor @ sun, 0.0, 1.0)
            on_ground = _sun_through(
                np.full(len(dirs), radius + 1.0), facing, radius, params
            )
            bounce = (
                on_ground * through * (facing * params.ground_albedo / np.pi)[:, None]
            )
            light = light + np.where(hits[:, None], bounce, 0.0)
            out[row, column] = np.mean(light, axis=0) / (1.0 - f_ms)
    out.setflags(write=False)
    return out


def inscatter_reference(
    radius: float,
    params,
    origin: ArrayLike,
    direction: ArrayLike,
    distance: float,
    sun: ArrayLike,
    steps: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """The CPU's reference for a view ray through one atmosphere (no other
    body's eclipse, no ring): a fine march of single scattering (the twin's
    sunlight, the phase functions) plus the multiple-scattering table's
    light, as atmosphere.glsl's ``atmoSegment``.

    :param origin: the eye relative to the planet's centre, metres
    :param direction: the view direction
    :param distance: how far the ray goes (the ground stops it too)
    :param sun: toward the sun
    :returns: (transmittance, inscatter), rgb each
    """
    top = radius + params.height
    o = np.asarray(origin, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    s = np.asarray(sun, float)
    s = s / np.linalg.norm(s)
    b = float(o @ d)
    disc = b * b - (float(o @ o) - top * top)
    if disc <= 0.0:
        return np.ones(3), np.zeros(3)
    t0 = max(-b - math.sqrt(disc), 0.0)
    t1 = -b + math.sqrt(disc)
    ground = b * b - (float(o @ o) - radius * radius)
    if ground > 0.0 and -b - math.sqrt(ground) > 0.0:
        t1 = min(t1, -b - math.sqrt(ground))
    t1 = min(t1, distance)
    if t1 <= t0:
        return np.ones(3), np.zeros(3)
    ds = (t1 - t0) / steps
    p = o + d * (t0 + (np.arange(steps) + 0.5) * ds)[:, None]
    r = np.linalg.norm(p, axis=-1)
    rayleigh = np.exp(-(r - radius) / params.rayleigh_height)[:, None]
    mie = np.exp(-(r - radius) / params.mie_height)[:, None]
    beta_r = np.asarray(params.rayleigh)
    extinction = (
        beta_r * rayleigh
        + params.mie_extinction * mie
        + np.asarray(params.ozone) * ozone_density(r - radius, params)[:, None]
    )
    view = np.exp(-(np.cumsum(extinction, axis=0) - 0.5 * extinction) * ds)
    cos_sun = (p @ s) / r
    mu = float(d @ s)
    g = params.mie_g
    phase_r = 3.0 / (16.0 * np.pi) * (1.0 + mu * mu)
    phase_m = (
        3.0
        / (8.0 * np.pi)
        * (1.0 - g * g)
        * (1.0 + mu * mu)
        / ((2.0 + g * g) * (1.0 + g * g - 2.0 * g * mu) ** 1.5)
    )
    single = _sun_through(r, cos_sun, radius, params) * (
        beta_r * rayleigh * phase_r + params.mie * mie * phase_m
    )
    u, v = multiple_uv(r, cos_sun, radius, top)
    psi = bilinear(multiple_table(radius, params), u, v) * params.multiple_scattering
    multiple = psi * (beta_r * rayleigh + params.mie * mie)
    inscatter = params.sun_intensity * np.sum(view * (single + multiple), axis=0) * ds
    return np.exp(-np.sum(extinction, axis=0) * ds), inscatter


def bilinear(table: np.ndarray, u: ArrayLike, v: ArrayLike) -> np.ndarray:
    """GL's linear filter of a (rows, columns, c) *table* at texture
    coordinates (u, v), clamped."""
    rows, columns = table.shape[:2]
    x = np.clip(np.asarray(u, float) * columns - 0.5, 0, columns - 1)
    y = np.clip(np.asarray(v, float) * rows - 0.5, 0, rows - 1)
    x0 = np.minimum(np.floor(x).astype(int), columns - 2)
    y0 = np.minimum(np.floor(y).astype(int), rows - 2)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    upper = table[y0, x0] * (1 - fx) + table[y0, x0 + 1] * fx
    lower = table[y0 + 1, x0] * (1 - fx) + table[y0 + 1, x0 + 1] * fx
    return upper * (1 - fy) + lower * fy


def table_texture(
    name: str, size: tuple[int, int], count: int, fill: float = 1.0
) -> Texture:
    """An RGBA16F table texture for *count* atmospheres stacked along v,
    linearly filtered, clamped; *fill* until filled (1: a clear sky's
    transmittance, 0: no light)."""
    width, height = size
    tex = Texture(name)
    tex.setup_2d_texture(width, height * count, Texture.T_float, Texture.F_rgba16)
    tex.set_ram_image(np.full((height * count, width, 4), fill, np.float32).tobytes())
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


def write_rows(tex: Texture, index: int, table: np.ndarray) -> None:
    """Put one atmosphere's (rows, columns, rgb[a]) *table* in its rows of a
    :func:`table_texture`."""
    height, width = table.shape[:2]
    image = (
        np.frombuffer(tex.get_ram_image(), np.float32)
        .reshape(tex.get_y_size(), tex.get_x_size(), 4)
        .copy()
    )
    rgba = np.ones((height, width, 4), np.float32)
    rgba[..., : table.shape[2]] = table
    image[index * height : (index + 1) * height] = rgba[..., [2, 1, 0, 3]]  # BGRA
    tex.set_ram_image(image.tobytes())
