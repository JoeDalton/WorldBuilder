"""frame_profile.py's bench scenes."""

from __future__ import annotations

import sys
from pathlib import Path

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
