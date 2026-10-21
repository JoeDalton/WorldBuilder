"""
Unit tests for the small helpers: world_builder.utils, the worker pool's
CPU counting and backends (world_builder.terrain.workers), and the
atmospheres' bookkeeping (world_builder.atmosphere).
"""

import os

import numpy as np
import pytest

from world_builder import utils
from world_builder.atmosphere import MAX_ATMOSPHERES, AtmosphereParams, Atmospheres
from world_builder.terrain import workers
from world_builder.terrain.planet import PRESETS
from world_builder.terrain.pngio import read_image, write_png
from world_builder.terrain.workers import ChunkExecutor

# ── utils ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "vector, expected",
    [
        ((3.0, 4.0), 5.0),
        ((1.0, 2.0, 2.0), 3.0),
        ((1.0, 1.0, 1.0, 1.0), 2.0),
        (np.array([2.0, 3.0, 6.0, 0.0, 0.0]), 7.0),  # any other length
    ],
)
def test_magnitude_of_every_length(vector, expected):
    assert utils.magnitude(vector) == pytest.approx(expected)


def test_clamp_and_powers_of_two():
    assert utils.clamp(5.0, 0.0, 1.0) == 1.0
    assert utils.clamp(-5.0, 0.0, 1.0) == 0.0
    assert utils.clamp(0.25, 0.0, 1.0) == 0.25
    assert utils.compute_next_power_of_2(1000) == 1024
    assert utils.compute_next_power_of_2(1024) == 1024
    assert utils.compute_next_power_of_2(0.2) == 1  # at least 1


# ── worker pool ────────────────────────────────────────────────────────────────


def test_usable_cpus_falls_back_when_the_platform_says_less(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    assert workers.usable_cpus() >= 1  # through sched_getaffinity
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert workers.usable_cpus() == 1  # nothing known: one


def test_physical_cores_where_topology_is_unknown(monkeypatch):
    def unreadable(*args, **kwargs):
        raise OSError("no sysfs here")

    monkeypatch.setattr("builtins.open", unreadable)
    assert workers.physical_cores() is None
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    assert workers.physical_cores() is None


def test_default_workers_leaves_a_core_for_the_renderer():
    assert 1 <= workers.default_workers() <= workers.usable_cpus()


def test_lower_priority_tolerates_a_refusal(monkeypatch):
    def refuse(increment):
        raise OSError("not permitted")

    # raising=False: Windows has no os.nice (the helper checks for it).
    monkeypatch.setattr(os, "nice", refuse, raising=False)
    workers._lower_priority(10)  # no exception: runs at normal priority
    workers._lower_priority(0)


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="backend"):
        ChunkExecutor(backend="carrier pigeon")


def test_thread_backend_generates_chunks():
    executor = ChunkExecutor(backend="thread", workers=2)
    try:
        version = executor.set_spec(PRESETS["moon"])
        assert executor.version == version
        assert executor.version_of("elsewhere") == 0
        key = (8, 0, 0, 0)
        executor.request([key])
        done = []
        for _ in range(2000):
            done += executor.collect()
            if done:
                break
            import time

            time.sleep(0.005)
        assert [chunk.key for chunk in done] == [key]
        assert done[0].version == version
        assert executor.pending == 0
    finally:
        executor.shutdown()


def test_requests_before_a_spec_are_ignored():
    executor = ChunkExecutor(backend="inline")
    executor.request([(0, 0, 0, 0)], "nothing yet")
    assert executor.collect("nothing yet") == []
    assert executor.pending_for("nothing yet") == 0


# ── atmospheres ────────────────────────────────────────────────────────────────


def test_atmospheres_bookkeeping():
    atmospheres = Atmospheres()
    params = AtmosphereParams(height=40e3)
    index = atmospheres.add(500e3, params, position=(1.0, 2.0, 3.0))
    assert len(atmospheres) == 1
    assert atmospheres.params(index) is params
    atmospheres.set_position(index, (10.0, 0.0, 0.0))
    atmospheres.update(np.array([10.0, 0.0, 600e3]))
    cam = atmospheres._arrays["uAtmoCam"][index]
    assert cam[2] == pytest.approx(600e3)  # relative to the moved centre
    for _ in range(MAX_ATMOSPHERES - 1):
        atmospheres.add(500e3, params)
    with pytest.raises(ValueError, match="at most"):
        atmospheres.add(500e3, params)


def test_sunlight_above_the_air_and_beyond_the_horizon():
    atmospheres = Atmospheres()
    atmospheres.add(600e3, AtmosphereParams())
    high = np.array([0.0, 0.0, 600e3 + 100e3])  # above the 60 km shell
    np.testing.assert_array_equal(
        atmospheres.transmittance_to_sun(0, high, (0.0, 0.0, 1.0)), 1.0
    )
    # Above the air, but the sun is low and its light grazes the air.
    grazing = atmospheres.transmittance_to_sun(0, high, (1.0, 0.0, -0.37))
    assert np.all(grazing < 1.0) and np.all(grazing > 0.0)


# ── png files ──────────────────────────────────────────────────────────────────


def test_png_errors(tmp_path):
    with pytest.raises(TypeError, match="dtype"):
        write_png(tmp_path / "f.png", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(OSError, match="cannot read"):
        read_image(tmp_path / "missing.png")


def test_png_round_trip_in_8_and_16_bits(tmp_path):
    grey = np.array([[0, 65535], [32768, 1000]], dtype=np.uint16)
    write_png(tmp_path / "g.png", grey)
    np.testing.assert_allclose(read_image(tmp_path / "g.png")[..., 0], grey / 65535.0)
    colour = np.zeros((2, 3, 3), dtype=np.uint8)
    colour[0, 1] = (255, 128, 0)
    write_png(tmp_path / "c.png", colour)
    back = read_image(tmp_path / "c.png")
    np.testing.assert_allclose(back[0, 1], (1.0, 128 / 255, 0.0), atol=1e-6)
