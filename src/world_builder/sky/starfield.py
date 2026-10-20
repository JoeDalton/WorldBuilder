"""
A procedural star field, as a cube map: stars, a Milky Way band, faint nebulae.

Generated once with numpy (well under a second at 1024 px per face), so there
are no assets to ship; :func:`load_cube_map` takes a painted one instead (e.g.
SpaceFlight's Spacescape skyboxes, ``models/skyboxes/purple_#.png``).

Faces follow OpenGL's cube map convention (what ``texture(samplerCube, dir)``
reads), so :func:`cube_directions` gives the direction of every texel. Texels
are stored gamma-encoded (value ** (1 / 2.2)) in 8 bits; the sky shader decodes
and scales them.
"""

from __future__ import annotations

import math

import numpy as np
from panda3d.core import Filename, SamplerState, Texture
from scipy import ndimage

from world_builder.terrain.noise import fbm

#: (major axis, s axis, t axis) per face, OpenGL order +X -X +Y -Y +Z -Z:
#: direction = major + sc * s_axis + tc * t_axis, with sc, tc in [-1, 1].
_FACES = np.array(
    [
        [(1, 0, 0), (0, 0, -1), (0, -1, 0)],
        [(-1, 0, 0), (0, 0, 1), (0, -1, 0)],
        [(0, 1, 0), (1, 0, 0), (0, 0, 1)],
        [(0, -1, 0), (1, 0, 0), (0, 0, -1)],
        [(0, 0, 1), (1, 0, 0), (0, -1, 0)],
        [(0, 0, -1), (-1, 0, 0), (0, -1, 0)],
    ],
    dtype=np.float64,
)


def cube_directions(size: int) -> np.ndarray:
    """:returns: (6, size, size, 3) unit directions of every texel centre,
    [face, row (t), column (s)]"""
    c = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    tc, sc = np.meshgrid(c, c, indexing="ij")
    out = np.empty((6, size, size, 3))
    for face, (major, s_axis, t_axis) in enumerate(_FACES):
        d = major + sc[..., None] * s_axis + tc[..., None] * t_axis
        out[face] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return out


def directions_to_texels(dirs: np.ndarray, size: int):
    """The cube map texel under each direction (the inverse of
    :func:`cube_directions`).

    :param dirs: (N, 3) directions
    :returns: face, row, column (N,) float (texel units, centres at .5)
    """
    dirs = np.asarray(dirs, dtype=np.float64)
    axis = np.argmax(np.abs(dirs), axis=1)
    rows = np.arange(len(dirs))
    face = 2 * axis + (dirs[rows, axis] < 0.0)
    major = np.abs(dirs[rows, axis])
    s_axis = _FACES[face, 1]
    t_axis = _FACES[face, 2]
    sc = np.einsum("ij,ij->i", dirs, s_axis) / major
    tc = np.einsum("ij,ij->i", dirs, t_axis) / major
    return face, (tc + 1.0) * 0.5 * size, (sc + 1.0) * 0.5 * size


def make_starfield(
    size: int = 1024,
    seed: int = 3,
    stars: int = 14000,
    galactic_pole=(0.25, -0.55, 0.8),
    nebula_strength: float = 0.35,
) -> np.ndarray:
    """The star field's linear radiance, before encoding.

    :param size: face resolution, texels
    :param seed: random seed
    :param stars: how many stars (more crowd the Milky Way)
    :param galactic_pole: normal of the Milky Way's plane
    :param nebula_strength: scale of the diffuse glow (0 = stars only)
    :returns: (6, size, size, 3) float32 linear values, mostly in [0, 1]
    """
    rng = np.random.default_rng(seed)
    pole = np.asarray(galactic_pole, dtype=np.float64)
    pole /= np.linalg.norm(pole)

    # ── diffuse light: the band, its dust lanes, faint nebulae (low res) ──
    low = max(32, size // 4)
    dirs = cube_directions(low).reshape(-1, 3)
    latitude = dirs @ pole  # sine of the galactic latitude
    band = np.exp(-((latitude / 0.16) ** 2))
    radius = 1000.0  # noise is evaluated on a sphere of this size
    clumps = 0.5 + 0.5 * fbm(dirs * radius, 220.0, 5, seed * 13 + 1)
    dust = np.clip(fbm(dirs * radius, 120.0, 4, seed * 13 + 2) * 2.2, 0.0, 1.0)
    glow = band * clumps * (1.0 - 0.75 * dust * np.exp(-((latitude / 0.06) ** 2)))
    milky = glow[:, None] * np.array([0.10, 0.095, 0.085])
    nebula = np.clip(fbm(dirs * radius, 400.0, 5, seed * 13 + 3), 0.0, None) ** 2
    tint = 0.5 + 0.5 * fbm(dirs * radius, 700.0, 2, seed * 13 + 4)
    colour = np.stack(
        [0.06 + 0.05 * tint, 0.02 + 0.02 * (1 - tint), 0.05 + 0.04 * (1 - tint)], 1
    )
    diffuse = (milky + nebula[:, None] * colour * nebula_strength).reshape(
        6, low, low, 3
    )
    image = np.stack(
        [
            ndimage.zoom(diffuse[f], (size / low, size / low, 1), order=1)
            for f in range(6)
        ]
    )

    # ── stars: more of them toward the band; a power law of brightness ──
    count = int(stars)
    picked = []
    while sum(len(p) for p in picked) < count:
        d = rng.normal(size=(count, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        keep = rng.uniform(size=count) < 0.35 + 0.65 * np.exp(
            -(((d @ pole) / 0.25) ** 2)
        )
        picked.append(d[keep])
    d = np.concatenate(picked)[:count]
    brightness = np.clip(0.04 * rng.pareto(1.6, count) + 0.03, 0.0, 6.0)
    temperature = rng.uniform(0.0, 1.0, count)  # blue-white to orange
    hue = np.stack(
        [0.75 + 0.35 * temperature, 0.85 + 0.1 * temperature, 1.1 - 0.45 * temperature],
        axis=1,
    )
    face, row, col = directions_to_texels(d, size)
    # A small gaussian per star, wider for the bright ones, one face only.
    for dr in (-2, -1, 0, 1, 2):
        for dc in (-2, -1, 0, 1, 2):
            r = np.floor(row).astype(int) + dr
            c = np.floor(col).astype(int) + dc
            ok = (r >= 0) & (r < size) & (c >= 0) & (c < size)
            sigma = 0.45 + 0.25 * np.clip(brightness, 0.0, 2.0)
            dist2 = (r + 0.5 - row) ** 2 + (c + 0.5 - col) ** 2
            weight = np.exp(-dist2 / (2.0 * sigma * sigma)) * brightness
            np.add.at(
                image,
                (face[ok], r[ok], c[ok]),
                (weight[ok, None] * hue[ok]),
            )
    return image.astype(np.float32)


def make_starfield_texture(size: int = 1024, seed: int = 3, **kwargs) -> Texture:
    """:returns: :func:`make_starfield` as a mipmapped cube map texture"""
    image = make_starfield(size, seed, **kwargs)
    encoded = np.clip(image, 0.0, 1.0) ** (1.0 / 2.2)
    data = (encoded * 255.0 + 0.5).astype(np.uint8)
    tex = Texture("starfield")
    tex.setup_cube_map(size, Texture.T_unsigned_byte, Texture.F_rgb8)
    # Panda's RAM layout is BGR, pages in face order, rows bottom-up = t up.
    tex.set_ram_image(np.ascontiguousarray(data[..., ::-1]).tobytes())
    _filtered(tex)
    return tex


def load_cube_map(pattern: str, loader) -> Texture:
    """A painted cube map (six images, ``#`` = 0-5 in OpenGL face order).

    :param pattern: e.g. ``".../skyboxes/purple_#.png"``
    :param loader: the ShowBase loader
    """
    tex = loader.load_cube_map(Filename.from_os_specific(pattern))
    _filtered(tex)
    return tex


def _filtered(tex: Texture) -> None:
    tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)


def angular_texel(size: int) -> float:
    """:returns: the angle a texel spans at a face centre, radians"""
    return 2.0 * math.atan(1.0 / size)
