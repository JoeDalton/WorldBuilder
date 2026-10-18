"""
Biome materials: PBR texture sets packed into two 2D texture arrays.

Drop one folder per biome into ``world_builder/terrain/textures/`` (names as in
:data:`biomes.BIOMES`)::

    textures/
        textures.yaml            # optional per-biome tuning, see below
        snow/   rock/   grass/   forest/   desert/   beach/
            <anything>albedo|basecolor|diffuse|diff|color|col<anything>.png|jpg
            <anything>normal|nor|nrm<anything>          (OpenGL convention;
                                                         "dx" in the name flips it)
            <anything>height|disp|displacement|bump<anything>
            <anything>rough<anything>                   (optional)
            <anything>ao|ambientocclusion|occlusion<anything>   (optional)

``textures.yaml``::

    max_size: 2048            # layers are resized to one power-of-two size
    grass: {tile_m: 2, normal: gl, tint: [1, 1, 1]}
    rock:  {tile_m: 4}

``tile_m`` (metres per texture repeat) must be a power of two: UVs are kept
precise by wrapping chunk origins modulo 4096 m, which every tile size and its
8x macro scale must divide. A biome with no folder, or a missing map, gets a
procedural placeholder, so real textures can be dropped in one at a time with
no code change.

Packing: ``albedo_height`` = sRGB albedo + linear height in alpha, and
``normal_rough_ao`` = normal xy, roughness, ambient occlusion (all linear).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from world_builder.terrain.biomes import BIOME_COLORS, BIOMES

LOGGER = logging.getLogger(__name__)

TEXTURES_PATH = Path(__file__).parent / "textures"
#: World period UVs are wrapped to, metres; every tile size must divide it.
UV_PERIOD = 4096.0
#: Macro scale: a second, larger repeat mixed in to hide tiling.
MACRO_SCALE = 8.0
PLACEHOLDER_SIZE = 512

DEFAULT_TILE_M = {
    "snow": 4.0,
    "rock": 4.0,
    "grass": 2.0,
    "forest": 8.0,
    "desert": 8.0,
    "beach": 4.0,
}
DEFAULT_ROUGHNESS = {
    "snow": 0.45,
    "rock": 0.75,
    "grass": 0.9,
    "forest": 0.95,
    "desert": 0.85,
    "beach": 0.8,
}

_MAP_PATTERNS = (
    # Checked in order: AO and normal before colour, since "col" is short.
    (
        "ao",
        re.compile(
            r"(^|[_\-. ])(ao|ambientocclusion|occlusion)([_\-. ]|$)|ambientocclusion"
        ),
    ),
    ("normal", re.compile(r"normal|(^|[_\-. ])(nor|nrm)([_\-. ]|$)|nor_gl|nor_dx")),
    ("height", re.compile(r"height|disp|bump")),
    ("roughness", re.compile(r"rough")),
    (
        "albedo",
        re.compile(
            r"albedo|basecolou?r|diffuse|colou?r"
            r"|(^|[_\-. ])diff([_\-. ]|$)|(^|[_\-. ])col([_\-. ]|$)"
        ),
    ),
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".exr"}


@dataclass
class MaterialSet:
    """The packed layers, ready for upload.

    :ivar albedo_height: (6, S, S, 4) uint8, RGB = sRGB albedo, A = height
    :ivar normal_rough_ao: (6, S, S, 4) uint8, RG = normal xy, B = roughness,
        A = ambient occlusion
    :ivar tile_m: (6,) metres per repeat
    :ivar mean_albedo: (6, 3) linear mean colour, drawn instead of the
        textures far away
    :ivar sources: per biome, a description of what was loaded
    """

    albedo_height: np.ndarray
    normal_rough_ao: np.ndarray
    tile_m: np.ndarray
    mean_albedo: np.ndarray
    sources: dict[str, str]


def load_materials(root: Path = TEXTURES_PATH) -> MaterialSet:
    """Load every biome's textures, filling gaps with placeholders.

    :param root: folder holding one sub-folder per biome
    :returns: the packed set
    """
    config = {}
    if (root / "textures.yaml").exists():
        config = yaml.safe_load((root / "textures.yaml").read_text()) or {}
    max_size = int(config.get("max_size", 2048))

    found = {biome: _find_maps(root / biome) for biome in BIOMES}
    real_sizes = [
        _image_size(path) for maps in found.values() for path in maps.values()
    ]
    size = (
        _pow2_floor(min([max_size, *[min(s) for s in real_sizes]]))
        if real_sizes
        else PLACEHOLDER_SIZE
    )

    albedo_height = np.zeros((len(BIOMES), size, size, 4), dtype=np.uint8)
    normal_rough_ao = np.zeros_like(albedo_height)
    tile_m = np.zeros(len(BIOMES))
    mean_albedo = np.zeros((len(BIOMES), 3))
    sources = {}
    for layer, biome in enumerate(BIOMES):
        settings = config.get(biome, {}) or {}
        tile = float(settings.get("tile_m", DEFAULT_TILE_M[biome]))
        if tile <= 0 or UV_PERIOD % (tile * MACRO_SCALE) != 0:
            raise ValueError(
                f"{biome}: tile_m {tile} must be a power of two dividing "
                f"{UV_PERIOD / MACRO_SCALE:g}"
            )
        tile_m[layer] = tile
        maps = found[biome]
        placeholder = make_placeholder(layer, size)
        albedo = _load(maps.get("albedo"), size, 3)
        if albedo is None:
            albedo = placeholder["albedo"]
        tint = np.array(settings.get("tint", (1.0, 1.0, 1.0)), dtype=np.float32)
        albedo = np.clip(albedo * tint, 0.0, 1.0)
        height = _load(maps.get("height"), size, 1)
        height = placeholder["height"] if height is None else height[:, :, 0]
        normal = _load(maps.get("normal"), size, 3)
        if normal is None:
            normal = placeholder["normal"]
        elif (
            settings.get("normal", "dx" if _is_directx(maps["normal"]) else "gl")
            == "dx"
        ):
            normal = normal.copy()
            normal[:, :, 1] = 1.0 - normal[:, :, 1]
        rough = _load(maps.get("roughness"), size, 1)
        rough = placeholder["roughness"] if rough is None else rough[:, :, 0]
        ao = _load(maps.get("ao"), size, 1)
        ao = placeholder["ao"] if ao is None else ao[:, :, 0]

        albedo_height[layer] = _to_bytes(np.dstack([albedo, height]))
        normal_rough_ao[layer] = _to_bytes(
            np.dstack([normal[:, :, 0], normal[:, :, 1], rough, ao])
        )
        mean_albedo[layer] = np.mean(albedo.reshape(-1, 3) ** 2.2, axis=0)
        sources[biome] = (
            ", ".join(f"{kind}={path.name}" for kind, path in sorted(maps.items()))
            or "placeholder"
        )
    return MaterialSet(albedo_height, normal_rough_ao, tile_m, mean_albedo, sources)


# ── placeholders ───────────────────────────────────────────────────────────────


def tileable_noise(
    size: int, beta: float, seed: int, low_cut: float = 1.0
) -> np.ndarray:
    """Periodic 1/f^beta noise, by shaping white noise in the frequency domain.

    :param size: side, pixels
    :param beta: spectral slope; 0 white, 1 pink, 2 smooth
    :param seed: random seed
    :param low_cut: frequencies below this (cycles per tile) are removed
    :returns: (size, size) in [0, 1], tiling seamlessly
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(size, size))
    fx = np.fft.fftfreq(size) * size
    f = np.hypot(fx[None, :], fx[:, None])
    shape = np.where(f >= low_cut, 1.0 / np.maximum(f, 1e-6) ** beta, 0.0)
    noise = np.real(np.fft.ifft2(np.fft.fft2(white) * shape))
    noise -= noise.min()
    return noise / max(noise.max(), 1e-9)


def make_placeholder(layer: int, size: int) -> dict[str, np.ndarray]:
    """A procedural stand-in material for one biome.

    :param layer: biome index
    :param size: side, pixels
    :returns: albedo (size, size, 3) sRGB, height, normal (size, size, 3)
        encoded, roughness, ao; all floats in [0, 1]
    """
    biome = BIOMES[layer]
    base = BIOME_COLORS[layer]
    seed = 100 + layer
    coarse = tileable_noise(size, 1.6, seed, low_cut=2.0)
    fine = tileable_noise(size, 0.6, seed + 1, low_cut=4.0)
    if biome == "rock":
        crack = 1.0 - np.abs(2.0 * tileable_noise(size, 1.2, seed + 2, 3.0) - 1.0)
        height = 0.55 * coarse + 0.3 * crack**4 + 0.15 * fine
        variation = 0.55 + 0.9 * height
    elif biome == "forest":
        # Tree crowns: a blobby field, dark gaps between crowns.
        crowns = tileable_noise(size, 2.2, seed + 2, low_cut=6.0)
        height = np.clip((crowns - 0.35) * 2.2, 0.0, 1.0) * 0.8 + 0.2 * fine
        variation = 0.45 + 0.9 * height
    elif biome == "desert":
        v = np.arange(size) / size
        ripples = 0.5 + 0.5 * np.sin(2.0 * np.pi * (6.0 * v[:, None] + 1.5 * coarse))
        height = 0.6 * ripples + 0.3 * coarse + 0.1 * fine
        variation = 0.85 + 0.3 * height
    elif biome == "grass":
        height = 0.4 * coarse + 0.6 * fine
        variation = 0.7 + 0.6 * fine * coarse
    else:  # snow, beach
        height = 0.7 * coarse + 0.3 * fine
        variation = 0.9 + 0.2 * height
    hue = tileable_noise(size, 2.0, seed + 3, low_cut=1.0)[:, :, None] - 0.5
    albedo = (
        base * variation[:, :, None] * (1.0 + 0.25 * hue * np.array([1.0, 0.6, -0.4]))
    )
    albedo = np.clip(albedo, 0.0, 1.0)
    normal = normal_from_height(height, strength=6.0 if biome != "snow" else 2.0)
    roughness = np.clip(DEFAULT_ROUGHNESS[biome] + 0.1 * (fine - 0.5), 0.0, 1.0)
    ao = np.clip(0.55 + 0.45 * height / max(height.max(), 1e-6), 0.0, 1.0)
    return {
        "albedo": albedo,
        "height": height,
        "normal": normal,
        "roughness": roughness,
        "ao": ao,
    }


def normal_from_height(height: np.ndarray, strength: float) -> np.ndarray:
    """:returns: a tiling OpenGL-convention normal map (encoded in [0, 1])"""
    size = height.shape[0]
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * size / 2.0
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * size / 2.0
    # Rows run down the image while v runs up, hence +dy.
    n = np.dstack([-dx * strength / size, dy * strength / size, np.ones_like(dx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return 0.5 + 0.5 * n


def noise_rgba(size: int = 256, seed: int = 7) -> np.ndarray:
    """:returns: (size, size, 4) uint8, four independent smooth tiling noises"""
    channels = [tileable_noise(size, 1.8, seed + c, low_cut=1.0) for c in range(4)]
    return _to_bytes(np.dstack(channels))


# ── file handling ──────────────────────────────────────────────────────────────


def _find_maps(folder: Path) -> dict[str, Path]:
    """:returns: map kind -> file, for the image files in *folder*"""
    maps: dict[str, Path] = {}
    if not folder.is_dir():
        return maps
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        stem = path.stem.lower()
        if any(skip in stem for skip in ("preview", "metal", "spec", "opacity")):
            continue
        for kind, pattern in _MAP_PATTERNS:
            if pattern.search(stem):
                maps.setdefault(kind, path)
                break
    return maps


def _is_directx(path: Path) -> bool:
    return bool(
        re.search(r"(^|[_\-. ])dx([_\-. ]|$)|normaldx|nor_dx", path.stem.lower())
    )


def _image_size(path: Path) -> tuple[int, int]:
    from panda3d.core import Filename, PNMImageHeader

    header = PNMImageHeader()
    if not header.read_header(Filename.from_os_specific(str(path))):
        raise OSError(f"cannot read {path}")
    return header.get_x_size(), header.get_y_size()


def _load(path: Path | None, size: int, channels: int) -> np.ndarray | None:
    """Read an image resized to size x size, top row first.

    :returns: (size, size, channels) float32 in [0, 1], or None if *path* is None
    """
    if path is None:
        return None
    from panda3d.core import Filename, PfmFile, PNMImage

    image = PNMImage()
    if not image.read(Filename.from_os_specific(str(path))):
        raise OSError(f"cannot read {path}")
    if image.get_x_size() != size or image.get_y_size() != size:
        resized = PNMImage(size, size, image.get_num_channels(), image.get_maxval())
        resized.gaussian_filter_from(1.0, image)
        image = resized
    pfm = PfmFile()
    pfm.load(image)
    channels_in = pfm.get_num_channels()
    # The buffer can run a few floats past the image; keep the image.
    data = np.frombuffer(memoryview(pfm), dtype=np.float32)
    data = data[: size * size * channels_in].reshape(size, size, channels_in)
    if data.shape[2] >= channels:
        return data[:, :, :channels]
    return np.repeat(data[:, :, :1], channels, axis=2)


def _to_bytes(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def _pow2_floor(value: int) -> int:
    return 1 << (int(value).bit_length() - 1)


# ── Panda upload ───────────────────────────────────────────────────────────────


def make_texture_array(name: str, layers: np.ndarray, srgb: bool):
    """Upload (L, S, S, 4) uint8 RGBA layers (top row first) as a 2D array.

    :param name: texture name
    :param layers: the layers
    :param srgb: whether RGB is sRGB-encoded (decoded to linear on sampling)
    :returns: a mipmapped, repeating, anisotropic Texture
    """
    from panda3d.core import SamplerState, Texture

    count, size = layers.shape[0], layers.shape[1]
    tex = Texture(name)
    tex.setup_2d_texture_array(
        size,
        size,
        count,
        Texture.T_unsigned_byte,
        Texture.F_srgb_alpha if srgb else Texture.F_rgba8,
    )
    # Panda keeps RAM images bottom row first and in BGRA order.
    tex.set_ram_image(np.ascontiguousarray(layers[:, ::-1, :, [2, 1, 0, 3]]).tobytes())
    _set_sampling(tex, SamplerState)
    return tex


def make_noise_texture():
    """:returns: the tiling RGBA noise that breaks up biome boundaries"""
    from panda3d.core import SamplerState, Texture

    data = noise_rgba()
    size = data.shape[0]
    tex = Texture("terrain_noise")
    tex.setup_2d_texture(size, size, Texture.T_unsigned_byte, Texture.F_rgba8)
    tex.set_ram_image(np.ascontiguousarray(data[::-1, :, [2, 1, 0, 3]]).tobytes())
    _set_sampling(tex, SamplerState)
    return tex


def _set_sampling(tex, sampler_state):
    tex.set_wrap_u(sampler_state.WM_repeat)
    tex.set_wrap_v(sampler_state.WM_repeat)
    tex.set_minfilter(sampler_state.FT_linear_mipmap_linear)
    tex.set_magfilter(sampler_state.FT_linear)
    tex.set_anisotropic_degree(8)
