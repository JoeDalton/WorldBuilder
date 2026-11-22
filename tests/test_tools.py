"""
Tests for the command-line tools: the terrain preview, the generation
benchmark and the sample stamps. Each tool runs end
to end at a tiny size.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from world_builder.terrain import bench, preview
from world_builder.terrain.planet import Planet
from world_builder.terrain.pngio import read_image as read_png
from world_builder.world_file import load_planet

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from make_sample_stamps import cliff_lake, write_sample_stamps  # noqa: E402

# ── preview ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def planet():
    return Planet(load_planet("earthlike"))


def test_preview_maps_are_images(planet):
    square = preview.render_map(planet, 20e3, 24)
    assert square.shape == (24, 24, 3) and square.dtype == np.uint8
    globe = preview.render_globe(planet, 12)
    assert globe.shape == (12, 24, 3)
    linear = preview.render_globe(planet, 12, linear=True)
    assert linear.dtype != np.uint8 and linear.max() <= 1.0 + 1e-6
    # The reference point is marked in red.
    assert np.any(np.all(globe == (255, 0, 0), axis=-1))


def test_preview_weather_shows_the_globe_its_genera_and_a_chart(planet):
    image = preview.render_weather(planet, 12)
    assert image.shape == (12, 6 * 12 + 2 * 4, 3)
    for k in range(3):
        panel = image[:, k * (2 * 12 + 4) : k * (2 * 12 + 4) + 2 * 12]
        assert panel.min() < panel.max()  # not uniform


def test_preview_command_writes_sheets(tmp_path, monkeypatch):
    argv = ["preview", "--out", str(tmp_path), "--size", "8", "--planet", "desert"]
    monkeypatch.setattr(sys, "argv", argv)
    preview.main()
    sheet = read_png(tmp_path / "preview_seed1_r600km.png")
    assert sheet.shape[0] == 8 and sheet.shape[2] == 3  # one preset, one row
    # The weather sheet skips airless presets.
    argv = ["preview", "--out", str(tmp_path), "--size", "8", "--weather"]
    monkeypatch.setattr(sys, "argv", [*argv, "--planet", "moon", "desert"])
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


# ── sample stamps ─────────────────────────────────────────────────────────────────────


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


def test_find_landforms_finds_places_on_the_desert():
    """find_landforms.py runs against the current terrain: each landform gets
    places, with a rise measured for the cliffs."""
    import find_landforms

    from world_builder.world_file import load_planet

    planet = Planet(load_planet("desert"))
    found = find_landforms.find(planet, 400, 1)
    assert {"dunes", "mesas", "canyons"} <= set(found)
    for name in ("mesas", "canyons"):
        lat, lon, ground, score = found[name][0]
        assert -90.0 <= lat <= 90.0 and score > 0.0
