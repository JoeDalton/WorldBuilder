"""
Tests for the command-line tools and level files: the terrain preview, the
generation benchmark, level loading and the sample stamps. Each tool runs end
to end at a tiny size.
"""

import sys

import numpy as np
import pytest

from world_builder.terrain import bench, preview
from world_builder.terrain.level import (
    _camera_pose,
    cliff_lake,
    load_level,
    write_sample_stamps,
)
from world_builder.terrain.planet import PRESETS, Planet
from world_builder.terrain.pngio import read_image as read_png

# ── preview ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def planet():
    return Planet(PRESETS["earthlike"])


def test_preview_maps_are_images(planet):
    square = preview.render_map(planet, 20e3, 24)
    assert square.shape == (24, 24, 3) and square.dtype == np.uint8
    globe = preview.render_globe(planet, 12)
    assert globe.shape == (12, 24, 3)
    linear = preview.render_globe(planet, 12, linear=True)
    assert linear.dtype != np.uint8 and linear.max() <= 1.0 + 1e-6
    # The reference point is marked in red.
    assert np.any(np.all(globe == (255, 0, 0), axis=-1))


def test_preview_weather_shows_the_globe_and_its_coverage(planet):
    image = preview.render_weather(planet, 12)
    assert image.shape == (12, 4 * 12 + 4, 3)
    coverage = image[:, 12 * 2 + 4 :]
    assert coverage.min() < coverage.max()  # not uniform


def test_preview_command_writes_sheets(tmp_path, monkeypatch):
    argv = ["preview", "--out", str(tmp_path), "--size", "8", "--preset", "desert"]
    monkeypatch.setattr(sys, "argv", argv)
    preview.main()
    sheet = read_png(tmp_path / "preview_seed1_r600km.png")
    assert sheet.shape[0] == 8 and sheet.shape[2] == 3  # one preset, one row
    # The weather sheet skips airless presets.
    argv = ["preview", "--out", str(tmp_path), "--size", "8", "--weather"]
    monkeypatch.setattr(sys, "argv", [*argv, "--preset", "moon", "desert"])
    preview.main()
    weather = read_png(tmp_path / "weather_seed1.png")
    assert weather.shape[0] == 8  # one row: the moon has no weather


# ── generation benchmark ───────────────────────────────────────────────────────


def test_bench_runs_end_to_end(monkeypatch, capsys):
    """With threads and a small planet, so it stays a few seconds."""
    real = bench.ChunkExecutor

    def threaded(**kwargs):
        kwargs["backend"] = "thread"
        return real(**kwargs)

    monkeypatch.setattr(bench, "ChunkExecutor", threaded)
    monkeypatch.setattr(bench, "LOD_ALTITUDES", (1000.0,))
    monkeypatch.setattr(sys, "argv", ["bench", "--workers", "2", "--radius", "60000"])
    bench.main()
    out = capsys.readouterr().out
    assert "chunk generation" in out and "chunks/s" in out


# ── levels ─────────────────────────────────────────────────────────────────────


def test_sample_stamps_are_written(tmp_path):
    write_sample_stamps(tmp_path)
    for name in ("crater", "mesa", "pad", "lake"):
        height = read_png(tmp_path / f"{name}.png")
        assert height.shape[2] == 1 and 0.0 < height.max() <= 1.0  # grey
    for name in ("crater_paint", "pad_paint", "lake_paint"):
        assert read_png(tmp_path / f"{name}.png").shape[2] == 3  # colour


def test_cliff_lake_has_its_waterline_and_cliff():
    height, paint = cliff_lake(128)
    assert height.min() >= 0.0 and height.max() <= 1.0
    assert height.min() < 0.1 < height.max()  # below and above the sea
    assert paint.shape == (128, 128, 3)


def test_level_poses_and_stamps_in_either_coordinates(tmp_path):
    write_sample_stamps(tmp_path)
    level = tmp_path / "level.yaml"
    level.write_text(
        """
planet: {preset: desert, seed: 4, reference: {lat: 10, lon: 20}, has_ocean: true}
stamps:
  - heightmap: crater.png
    center: {lat: 10.01, lon: 20.02}
    size: 3000
paints:
  - mask: crater_paint.png
    center: {x: 0, y: 0}
    size: [3000, 2000]
    colors: {"#808080": rock}
camera: {x: 1000, y: -2000, z: 450, heading: 30}
"""
    )
    spec, pose = load_level(level)
    assert spec.has_ocean and spec.seed == 4
    assert spec.stamps[0].placement.lat == pytest.approx(10.01)
    assert spec.stamps[0].placement.width == spec.stamps[0].placement.height == 3000
    assert spec.paints[0].placement.height == 2000
    assert pose["alt"] == 450 and pose["heading"] == 30
    assert pose["lat"] < 10.0  # 2 km south of the reference


def test_camera_pose_by_latitude_and_none():
    assert _camera_pose(None, None) is None
    pose = _camera_pose({"lat": 1.5, "lon": -2.0, "pitch": -10}, None)
    assert pose == {
        "lat": 1.5,
        "lon": -2.0,
        "alt": 800.0,
        "heading": 0.0,
        "pitch": -10.0,
    }


def test_empty_level_is_the_preset(tmp_path):
    level = tmp_path / "empty.yaml"
    level.write_text("")
    spec, pose = load_level(level, preset="forest", seed=5)
    assert spec.name == "forest" and spec.seed == 5
    assert pose is None and spec.stamps == ()
