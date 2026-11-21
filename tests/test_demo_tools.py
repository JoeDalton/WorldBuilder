"""The terrain demo's tools for looking at the clouds: the burst capture's
image helpers (B), the march block's keys (- / +), and frame_profile.py's
bench scenes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from panda3d.core import Filename, Texture

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import frame_profile  # noqa: E402

import demo_terrain  # noqa: E402


def test_a_burst_png_reads_back_as_written(tmp_path):
    """_write_png writes [0, 1] values top row first (grey as RGB);
    _texture_array reads a texture back the same way round, RGB order."""
    rows, columns = 4, 8
    image = np.zeros((rows, columns, 3), np.float32)
    image[0, :, 0] = 1.0  # the top row red
    image[:, 0, 2] = 0.5  # the left column half blue
    path = tmp_path / "rgb.png"
    demo_terrain._write_png(image, path)
    texture = Texture()
    assert texture.read(Filename.from_os_specific(str(path)))
    back = demo_terrain._texture_array(texture)
    assert back.shape[:2] == (rows, columns)
    assert np.allclose(back[..., :3], image, atol=1.0 / 255.0)

    grey = np.linspace(0.0, 1.0, rows * columns, dtype=np.float32)
    grey = grey.reshape(rows, columns, 1)
    path = tmp_path / "grey.png"
    demo_terrain._write_png(grey, path)
    texture = Texture()
    assert texture.read(Filename.from_os_specific(str(path)))
    back = demo_terrain._texture_array(texture)
    for channel in range(3):
        assert np.allclose(back[..., channel], grey[..., 0], atol=1.0 / 255.0)


def test_a_burst_crops_a_buffer_to_its_regions_corner():
    """A buffer may be padded past its region, which starts at its bottom
    left (GL): _corner keeps those rows and columns of a top-first image."""
    image = np.arange(6 * 5).reshape(6, 5)
    corner = demo_terrain._corner(image, (2, 3))
    assert corner.shape == (2, 3)
    assert (corner == image[4:, :3]).all()


def _block_demo(block):
    """A stand-in TerrainDemo for _step_march_block: its world's clouds'
    block, and the settings it is given."""
    demo = SimpleNamespace(clouds=object(), given=[])

    def with_value(section, name, value):
        assert (section, name) == ("clouds", "march_block")
        return SimpleNamespace(clouds=SimpleNamespace(march_block=value))

    def world_with(value):
        world = SimpleNamespace(clouds=SimpleNamespace(march_block=value))
        world.with_value = with_value
        return world

    demo.world = world_with(block)

    def set_world(world):
        demo.given.append(tuple(world.clouds.march_block))
        demo.world = world_with(world.clouds.march_block)

    demo._set_world = set_world
    return demo


def test_the_minus_and_plus_keys_step_the_share_of_pixels_traced():
    """- traces fewer pixels a frame, + more, through every block by the
    share it traces (1/16, 1/9, 1/8, 1/4, all), stopping at either end."""
    step = demo_terrain.TerrainDemo._step_march_block
    demo = _block_demo((4, 2))
    for _ in range(5):
        step(demo, 1)
    assert demo.given == [(2, 2), (1, 1)]  # then held at all of them
    for _ in range(6):
        step(demo, -1)
    assert demo.given[2:] == [(2, 2), (4, 2), (3, 3), (4, 4)]
    assert demo.world.clouds.march_block == (4, 4)
    shares = [1 / (a * u) for a, u in [(4, 4), (3, 3), (4, 2), (2, 2), (1, 1)]]
    assert shares == sorted(shares)


def test_the_block_keys_do_nothing_without_clouds():
    demo = _block_demo((4, 2))
    demo.clouds = None
    demo_terrain.TerrainDemo._step_march_block(demo, 1)
    assert demo.given == []


def test_the_bench_scenes_are_the_level_simplified():
    """--scene bench-sea / bench-dry: the level's planet with no vegetation
    and 600 m mountains, its sea kept or not; the rest as loaded."""
    from world_builder.terrain.level import DEMO_LEVEL, load_level

    spec, _ = load_level(DEMO_LEVEL)
    for scene, ocean in frame_profile.BENCH_SCENES.items():
        bench = frame_profile.bench_spec(ocean)(spec)
        assert bench.has_vegetation is False
        assert bench.has_ocean is ocean
        assert bench.elevation.mountain_height == 600.0
        assert bench.radius == spec.radius and bench.seed == spec.seed
    assert set(frame_profile.BENCH_SCENES) == {"bench-sea", "bench-dry"}
    assert frame_profile.BENCH_POSES == ("ground", "bench_1500", "bench_10000")
