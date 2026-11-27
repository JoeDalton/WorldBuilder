"""frame_profile.py's bench scenes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import frame_profile  # noqa: E402


def test_the_bench_scenes_are_the_world_simplified():
    """--scene bench-sea / bench-dry: the world's first planet with no
    vegetation and 600 m mountains, its sea kept or not; the rest as loaded."""
    from world_builder.world_file import parse_world

    path = frame_profile.DEFAULT_WORLD
    world = parse_world(path)
    for scene, ocean in frame_profile.BENCH_SCENES.items():
        bench_path = Path(frame_profile._bench_world(str(path), ocean))
        try:
            bench = parse_world(bench_path).bodies[0]
        finally:
            bench_path.unlink()
        spec = world.bodies[0].spec
        assert bench.spec.has_vegetation is False
        assert bench.spec.has_ocean is ocean
        assert (bench.ocean is not None) is ocean
        assert bench.spec.elevation.mountain_height == 600.0
        assert bench.spec.radius == spec.radius and bench.spec.seed == spec.seed
    assert set(frame_profile.BENCH_SCENES) == {"bench-sea", "bench-dry"}
    assert frame_profile.BENCH_POSES == ("ground", "bench_1500", "bench_10000")


def test_the_terrain_passes_are_the_cameras_bits():
    """--terrain-passes hides the terrain from one camera at a time: the
    view's camera (both its bits) and the sea's reflection."""
    from panda3d.core import BitMask32

    from world_builder import depth

    assert frame_profile.TERRAIN_PASSES == {
        "t_view": BitMask32.bit(depth.NEAR_BIT) | BitMask32.bit(depth.FAR_BIT),
        "t_refl": depth.REFLECTION_CAMERA_MASK,
    }


def test_the_pipelined_summary_is_median_and_p95_in_ms():
    """The pipelined frame: the median and 95th percentile of the free-running
    steps, seconds in, milliseconds out."""
    frame, p95 = frame_profile._summary([0.004] * 19 + [0.010])
    assert frame == 4.0
    assert abs(p95 - 4.3) < 1e-9


@pytest.mark.parametrize("platform, clip_control", [("linux", True), ("darwin", False)])
def test_the_window_asks_for_clip_control_but_on_macos(
    monkeypatch, platform, clip_control
):
    """The reversed camera's 0-1 depth (gl-depth-zero-to-one): every platform
    but macOS, whose OpenGL has none."""
    import tools

    loaded = []
    monkeypatch.setattr(tools, "loadPrcFileData", lambda _, text: loaded.append(text))
    monkeypatch.setattr(tools.sys, "platform", platform)
    tools.window_config("t", "64 64", True, vsync=False)
    assert ("gl-depth-zero-to-one #t" in loaded[0]) is clip_control
