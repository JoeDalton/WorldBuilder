"""
The cumulonimbus anvil's smoothness (noise.anvil_weight): under a deep
cloud's top, where the threshold falls and the cloud spreads into its anvil,
the fine octaves (and in the raymarch the billows) fade to their means, so
the anvil is smooth, not the tower's cauliflower spread sideways. The
lowered threshold itself stays: it is what spreads the tower, and the
coverage by depth is calibrated on it.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from world_builder.clouds.cloud import PRESETS, CloudType
from world_builder.clouds.field_setup import _noise
from world_builder.clouds.genera import depth_shift, depth_shift_tables
from world_builder.clouds.noise import (
    ANVIL_BAND,
    ANVIL_DEPTH,
    ANVIL_SMOOTH_BAND,
    density_on_sphere,
    fbm_sigma,
    sphere_column_peaks,
    top_fraction,
    with_shelf,
    with_threshold,
)

R = 600e3
SHADERS = Path(__file__).parent.parent / "src/world_builder/clouds/shaders"


@pytest.fixture(scope="module")
def convective():
    raw = PRESETS[CloudType.CONVECTIVE].field
    volume = _noise(7)
    peaks = sphere_column_peaks(volume, raw, R)
    field = with_shelf(with_threshold(raw, peaks, fbm_sigma(volume, raw)), volume, R)
    return field, depth_shift_tables(volume, field, R)


def test_anvil_weight_is_zero_without_an_anvil_and_at_shallow_depth(convective):
    from world_builder.clouds.noise import anvil_weight

    h = np.linspace(0.0, 1.0, 21)
    assert np.all(anvil_weight(PRESETS[CloudType.CUMULUS].field, h, 1.0) == 0.0)
    field, _ = convective
    assert np.all(anvil_weight(field, h, ANVIL_DEPTH[0]) == 0.0)
    top = top_fraction(field, 1.0)
    assert anvil_weight(field, ANVIL_SMOOTH_BAND[1] * top, 1.0) == pytest.approx(1.0)
    assert anvil_weight(field, ANVIL_SMOOTH_BAND[0] * top, 1.0) == pytest.approx(0.0)
    # smooth before the threshold falls
    assert ANVIL_SMOOTH_BAND[1] <= ANVIL_BAND[0] + 0.05


def _fine_energy(field, tables, local, patches=24, size=2000.0, step=50.0):
    """The density's energy finer than ~500 m (variance of it less its
    500 m box mean), over horizontal patches at a share *local* of a depth-1
    column's top, coverage 0.6."""
    cube = np.empty((6, 4, 4, 4), dtype=np.float32)
    cube[..., 0] = 0.6
    cube[..., 1] = depth_shift(np.full((6, 4, 4), 0.6), np.full((6, 4, 4), 1.0), tables)
    cube[..., 2] = 1.0
    cube[..., 3] = 0.0
    base, thickness = field.slab
    height = R + base + local * top_fraction(field, 1.0) * thickness
    rng = np.random.default_rng(5)
    n = int(size / step) + 1
    offsets = (np.arange(n) - n // 2) * step
    k = int(500.0 / step) // 2
    energies = []
    for _ in range(patches):
        up = rng.normal(size=3)
        up /= np.linalg.norm(up)
        east = np.cross([0.0, 0.0, 1.0], up)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)
        grid = (
            up * height + offsets[:, None, None] * east + offsets[None, :, None] * north
        )
        d = density_on_sphere(_noise(7), cube, field, grid, R)
        padded = np.pad(d, k, mode="edge")
        box = (
            sum(
                padded[k + i : k + i + n, k + j : k + j + n]
                for i in range(-k, k + 1)
                for j in range(-k, k + 1)
            )
            / (2 * k + 1) ** 2
        )
        energies.append(np.var(d - box))
    return float(np.mean(energies))


def test_the_anvil_band_is_smooth(convective):
    """Fine energy in the anvil band (0.9 of the local top) at most 0.3 of
    the tower's at 0.4."""
    field, tables = convective
    anvil = _fine_energy(field, tables, 0.9)
    tower = _fine_energy(field, tables, 0.4)
    assert anvil <= 0.3 * tower, (anvil, tower)


def test_the_shader_fades_the_fine_octaves_and_billows_in_the_anvil():
    field = (SHADERS / "cloud_field.glsl").read_text()
    body = field[field.index("float anvilWeight(Field f, float h, float depth)") :]
    body = body[: body.index("}")]
    numbers = re.search(
        r"smoothstep\(([\d.]+), ([\d.]+), depth\)"
        r" \* smoothstep\(([\d.]+), ([\d.]+), local\)",
        body,
    )
    assert tuple(map(float, numbers.groups())) == (*ANVIL_DEPTH, *ANVIL_SMOOTH_BAND)
    # both density functions take the faded high octaves
    assert field.count("anvilColumnValue(f, coord, v, low, mean, aw)") == 2
    march = (SHADERS / "cloud_march.frag").read_text()
    assert "1.0 - anvilWeight(" in march


def _cube_at(tables, coverage, depth, spread):
    cube = np.empty((6, 4, 4, 4), dtype=np.float32)
    cube[..., 0] = coverage
    cube[..., 1] = depth_shift(
        np.full((6, 4, 4), coverage), np.full((6, 4, 4), depth), tables
    )
    cube[..., 2] = depth
    cube[..., 3] = spread
    return cube


def _energy_at(field, cube, height, scale, patches=12, size=8000.0, step=200.0):
    """Mean variance, over horizontal patches at *height* above sea level, of
    the density less its box mean over *scale* metres."""
    rng = np.random.default_rng(11)
    n = int(size / step) + 1
    offsets = (np.arange(n) - n // 2) * step
    k = int(scale / step) // 2
    out = []
    for _ in range(patches):
        up = rng.normal(size=3)
        up /= np.linalg.norm(up)
        east = np.cross([0.0, 0.0, 1.0], up)
        east /= np.linalg.norm(east)
        north = np.cross(up, east)
        grid = (
            up * (R + height)
            + offsets[:, None, None] * east
            + offsets[None, :, None] * north
        )
        d = density_on_sphere(_noise(7), cube, field, grid, R)
        padded = np.pad(d, k, mode="edge")
        box = (
            sum(
                padded[k + i : k + i + n, k + j : k + j + n]
                for i in range(-k, k + 1)
                for j in range(-k, k + 1)
            )
            / (2 * k + 1) ** 2
        )
        out.append(np.var(d - box))
    return float(np.mean(out))


def test_the_anvil_is_smooth_at_the_kilometre(convective):
    """The anvil's energy finer than 2 km (its coarse octaves' lumps) at most
    0.4 of the tower's: anvils are smooth over kilometres."""
    field, tables = convective
    cube = _cube_at(tables, 0.6, 1.0, 0.0)
    base, thickness = field.slab
    top = top_fraction(field, 1.0) * thickness
    anvil = _energy_at(field, cube, base + 0.9 * top, 2000.0)
    tower = _energy_at(field, cube, base + 0.4 * top, 2000.0)
    assert anvil <= 0.4 * tower, (anvil, tower)


def test_the_shelf_is_smooth_at_the_kilometre(convective):
    """The shelf spread over shallow columns (depth 0.3, spread 1), in its
    middle: energy finer than 2 km at most 0.4 of a tower's."""
    from world_builder.clouds.noise import SHELF_BAND

    field, tables = convective
    base, thickness = field.slab
    shelf = _energy_at(
        field,
        _cube_at(tables, 0.3, 0.3, 1.0),
        base + 0.5 * (SHELF_BAND[1] + SHELF_BAND[2]) * thickness,
        2000.0,
    )
    top = top_fraction(field, 1.0) * thickness
    tower = _energy_at(field, _cube_at(tables, 0.6, 1.0, 0.0), base + 0.4 * top, 2000.0)
    assert shelf <= 0.4 * tower, (shelf, tower)
