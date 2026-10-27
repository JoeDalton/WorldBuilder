# Plan: fix flaky test_terrain_reports_loading (issue #28)

## Context
`test_terrain_reports_loading` asserts `not terrain.loading` after a fixed 40 frames.
`Terrain._upload` (src/world_builder/terrain/terrain.py ~L918) is bounded by wall-clock
`upload_ms_per_frame`, so a slow macOS x86 runner uploads fewer chunks per frame and
needs more than 40 frames. Behaviour is correct; the test measures machine speed.
The whole module is currently skipped on darwin (pytestmark, tests/test_terrain_multibody.py L20-24).
Decisions: option A (settle until done, capped), narrow scope, remove the skip,
document the possible wider fix in both `_settle` helpers.

## Edits (tests only; nothing else touched)

### 1. tests/test_terrain_multibody.py
- Remove the `pytestmark = pytest.mark.skipif(sys.platform == "darwin", ...)` block (L20-24).
  Drop `import sys` only if nothing else uses it (grep first).
- Extend `_settle` docstring with a note: frame count is a speed assumption, since
  `Terrain._upload` stops at `upload_ms_per_frame` of wall-clock time; on slow machines
  a fixed count can stop before refinement ends (issue #28). Possible fix for the
  callers that need a finished terrain: use `_settle_until_loaded` (or give `_settle`
  an `until_loaded` option); callers that rely on a fixed count (e.g. the eviction test's
  15 frames per stop) must be reviewed one by one.
- Add below `_settle`:

```python
def _settle_until_loaded(terrain, camera, timeout=60.0, stable_frames=5):
    """Update until ``terrain.loading`` has stayed False for ``stable_frames``
    frames. Waits on the state, not a frame count (the uploads' wall-clock budget
    makes the frames needed depend on the machine). Fails after ``timeout`` s."""
    deadline = time.perf_counter() + timeout
    stable = 0
    while stable < stable_frames:
        assert time.perf_counter() < deadline, "terrain still loading"
        terrain.update(camera)
        stable = 0 if terrain.loading else stable + 1
```
  (`time` is already imported.) `stable_frames` follows the `Terrain.loading` docstring advice.

- `test_terrain_reports_loading`: keep `assert terrain.loading` at the start and the
  `terrain.update(elsewhere); assert terrain.loading` step (single update after a camera
  jump must still wanted-ground: deterministic, no timing). Replace the two
  `_settle(..., frames=40); assert not terrain.loading` pairs with
  `_settle_until_loaded(terrain, camera)` / `(terrain, elsewhere)` followed by
  `assert not terrain.loading`.

### 2. tests/test_terrain_collision.py
- Add the same documentation note to its `_settle` (L309) (docstring currently absent),
  pointing at the multibody helper / issue #28. No behaviour change.

## Risks
- Removing the module-wide darwin skip re-enables every test in the module on macOS;
  other fixed-frame tests there (store > 64 after 120 frames, moon shown <= 30 chunks)
  could flake on slow runners. Not verifiable locally; watch CI and follow up if so.
- Possible edge: first assertion `assert terrain.loading` right after one `update(elsewhere)`
  relies on the pass finding new wanted chunks; unchanged from today.

## Verification
- `pytest tests/test_terrain_multibody.py -q` (whole module, since the skip is gone on mac only; on Linux confirms nothing else changed).
- Repeat the target test ~20x: `pytest tests/test_terrain_multibody.py::test_terrain_reports_loading --count 20` if pytest-repeat exists, else a shell loop.
- Stress for slowness: run it with `upload_ms_per_frame` simulated tiny (temporary local monkeypatch to 0.01 ms, not committed) to confirm the old test fails and the new one passes.
- No commit (user commits).
