# Frame Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the demo's frame at 2560 x 1440, HIGH, from ~6.2 ms to ≤ 5 ms at the
low poses, with no visible change, by cutting the CPU work that now bounds it.

**Architecture:** The frame is CPU-bound (diagnosis below), so the levers are CPU
first: an asteroid field that does its whole per-frame work from 60,000 km away,
terrain chunks drawn into passes where they draw nothing, and Panda's per-chunk
draw overhead (a transform and state change per chunk, in five passes). GPU-side
terrain shading comes after: it only matters once the CPU is below ~5 ms.
Phase 2 (#57) then replaces the near/far split with one reversed-Z camera, and
lets the clouds read its single depth: the terrain drawn in two passes, not five.

**Tech Stack:** Python 3.14, Panda3D 1.10 (GLSL 330), numpy, pytest.

**Spec:** this document's Diagnosis section (no separate spec); the measurements
behind it are reproducible with Task 1's tool.

## Diagnosis (2026-11-26, RTX 3070, 5600X, 2560 x 1440, HIGH, offscreen, GPU at P0)

Pipelined frame (no per-frame GPU sync, as a game runs) and the CPU step
(`taskMgr.step()`, GPU-synced), median ms:

| pose | pipelined frame | CPU step | terrain hidden |
|---|---|---|---|
| ground | 6.15-6.29 | 6.13-6.28 | 3.36-3.43 |
| coast | 5.66-5.68 | 5.65-5.67 | 3.26-3.28 |
| sunset | 6.45-6.52 | - | 3.32 |
| high | 3.91 | - | 3.05 |

- **CPU-bound.** Stubbing the asteroid field's update (pure CPU) took the
  pipelined frame from 6.15 to 5.02 ms, the same as the CPU step (6.20 to 5.04):
  the GPU is below ~5 ms at `ground`. CPU savings pay in full down to about there.
  (frame_profile.py's numbers are CPU + GPU serialized by its per-frame sync:
  11 ms at `ground`. They rank GPU parts but overstate the frame.)
- **The asteroid ring: 1.15 ms a frame**, around the ice planet 60,000 km away,
  nothing of it drawn from the home planet (`AsteroidField.update`:
  population windows, haze cells, view planes, rotations, every frame).
- **The terrain: 2.4-2.8 ms of the frame, almost all CPU** (Panda's cull and
  draw), additive by pass (terrain hidden from one camera at a time, pipelined):

  | pass | ground | coast |
  |---|---|---|
  | clouds' far depth | 0.95 | 0.85 |
  | main, far range | 0.73 | 0.73 |
  | sea reflection | 0.60 | 0.55 |
  | main, near range | 0.29 | 0.11 |
  | clouds' near depth | 0.23 | 0.11 |

- **Panda's draw, all passes (PStats, `ground`):** Draw 3.9 ms, of it Set State
  1.85 (shader parameters 0.77, textures 0.58), primitives 0.58; Cull 1.6 ms.
  Per frame: 531 Geoms, 535 state changes, **1045 transform changes** (each
  chunk node carries its own transform; the shader's matrices are re-sent per
  draw).
- **The rest of the Python update:** ~2.2 ms in all, of which the asteroid field
  1.15, `World._light` ~0.2, the demo's altimeter ~0.15 (demo only), terrain
  update ~0.15.
- **GPU, terrain shading** (serialized frames, so indicative only): its fragment
  stage 1.1-1.8 ms (flat-shaded debug view); textures 0.5-0.75 (`anisotropy`
  8 to 2: 0.35-0.5; `detail_far` 4000 to 1500: 0.55-0.75); PBR, far forest ~0.
  `split_factor` 2 to 1.5: 0.6-1.3 ms (CPU 0.4-0.75 of it), a visible change.

## Global Constraints

- No visible change in any task here, but Phase 2's macOS look, which is
  accepted as untuned (it must run, not match). A setting that changes the look
  (anisotropy, `detail_far`, `split_factor`, the reflection's reach) is the
  user's decision, listed under "Decisions for the user", not a task.
- Measure with Task 1's `--pipelined` mode: 2560 x 1440, HIGH, offscreen, the
  machine idle, GPU at P0 (`nvidia-smi --query-gpu=pstate,clocks.gr --format=csv`
  during the run). Two runs of each tree; run-to-run noise is ~0.1-0.15 ms
  pipelined.
- Python env: `../SpaceFlight/.venv`, `PYTHONPATH=src`.
- Skip the `jupiter_rocky` preset in every check.
- No commits: the user commits. At the end of each task, update CHANGELOG.md's
  `[Unreleased]` section, and `docs/source/performance.md` (a new section, §38)
  with the task's before/after numbers.

## Review Focus

1. **A camera flying from far away into a field** (the early-out of Task 2):
   rocks must be back on screen within the same frames as a cold start at that
   spot. Test `test_a_field_out_of_reach_fills_again_on_return`.
2. **The ring's haze seen from afar**: the haze still follows the sun and the
   camera while the rocks are skipped. Test
   `test_the_haze_still_updates_out_of_reach`.
3. **A chunk whose grid is all below sea level next to a shore** (Task 3): its
   skirts and its swell must not show in the reflection either (they are clipped
   too: `terrain.vert`'s `gl_ClipDistance`). Shot check at `coast`.
4. **A planet without a sea** (`has_ocean` false) with ground below 0 m: its
   chunks are dry land and must stay in every pass. Test
   `test_dry_ground_below_zero_stays_in_the_reflection`.
5. **`Terrain.set_spec` to a planet with or without a sea**: the reflection
   flags follow the new spec (chunks are rebuilt; check no cached node keeps
   the old flag). Test `test_set_spec_rebuilds_the_reflection_flag`.

---

### Task 1: A pipelined frame mode in frame_profile.py

**Files:**
- Modify: `scripts/frame_profile.py`
- Modify: `docs/source/frame_profile.md` (How it was measured)

**Interfaces:**
- Produces: `frame_profile.py --pipelined [--terrain-passes]`, printing per
  pose `frame` (pipelined median), `p95`, `step` (GPU-synced CPU median), and
  per part the pipelined frame saved. `--terrain-passes` adds the parts
  `t_near`, `t_far`, `t_refl`, `t_cldnear`, `t_cldfar`: the terrain's root
  hidden from one camera bit (`NEAR_BIT`, `FAR_BIT`, `REFLECTION_CAMERA_MASK`,
  `CLOUD_NEAR_BIT`, `CLOUD_FAR_BIT` from `world_builder.depth`). Also a part
  `asteroids`: every field's `update` replaced by a no-op (CPU only).

- [ ] **Step 1: Add `_time_pipelined(demo, probe, frames) -> tuple[float, float, float]`**
  next to `_time`: 30 warm-up steps, then `frames` steps timed without a
  read-back (median and p95, ms), one read-back to drain, then 100 GPU-synced
  steps for the step median. The scratch prototype is
  `/tmp/claude-1000/-home-joe-development-WorldBuilder/887dcbd7-8d8b-403b-b818-0b812ef24e6a/scratchpad/cpu_profile.py` (`--pipelined`,
  `--passes`, `--no-asteroids`), if still there.
- [ ] **Step 2: `--pipelined` uses it for the frame and every part; default 400 frames.**
- [ ] **Step 3: Run it**

  Run: `PYTHONPATH=src ../SpaceFlight/.venv/bin/python scripts/frame_profile.py --offscreen --pipelined --terrain-passes --pose ground coast sunset high`
  Expected: the frame within ~0.2 ms of the Diagnosis table; the `t_*` parts
  summing to within ~0.3 ms of `terrain`.
- [ ] **Step 4: Document the mode** in frame_profile.md ("How it was
  measured": serialized against pipelined, which to read for what), and
  update CHANGELOG `[Unreleased]`.

### Task 2: An asteroid field out of reach does nothing per frame

**Files:**
- Modify: `src/world_builder/asteroids/field.py` (`AsteroidField.update`, ~357-475)
- Test: `tests/test_asteroid_field.py`

**Interfaces:**
- Consumes: `FieldSpec.shape.bounds() -> (lo, hi)` (field frame, `asteroids/spec.py`),
  `Population.ranges(...)`, `Population.count`, `AsteroidField.to_field`.
- Produces: `AsteroidField.out_of_reach: bool` (property, last update's
  verdict); `stats["out_of_reach"]`.

Out of reach = the camera (field frame) is farther from the shape's bounding box
than `max(ranges) + max class hi` **and** the population holds no rock. Then
`update` skips the population, the visible pass, terrain rocks, collision,
`_planes` and the epoch's table rewrite; it still writes the frame inputs and
runs `haze.update` (the haze can be seen from afar). The first out-of-reach
frames still run the population update until its windows are empty, so leaving
evicts as today.

- [ ] **Step 1: Write the failing tests** (use a `built_field`-style fixture of
  its own, module scope, the same 3 km box):

```python
def test_a_far_field_skips_its_population(window, far_field, monkeypatch):
    field, _ = far_field
    far = (1.0e7, 0.0, 0.0)
    for _ in range(20):
        field.update(far, far)
    assert field.population.count == 0 and field.out_of_reach
    calls = []
    monkeypatch.setattr(field.population, "update",
                        lambda *a, **k: calls.append(1) or (np.empty(0, int), np.empty(0, int)))
    field.update(far, far)
    assert calls == [] and field.stats["out_of_reach"]

def test_a_field_out_of_reach_fills_again_on_return(window, far_field):
    field, _ = far_field
    for _ in range(20):
        field.update((1.0e7, 0.0, 0.0))
    for _ in range(10):
        field.update((0.0, 0.0, 0.0))
    assert not field.out_of_reach and field.population.count > 0

def test_the_haze_still_updates_out_of_reach(window, far_field, monkeypatch):
    field, _ = far_field
    for _ in range(20):
        field.update((1.0e7, 0.0, 0.0))
    calls = []
    real = field.haze.update
    monkeypatch.setattr(field.haze, "update", lambda *a, **k: calls.append(1) or real(*a, **k))
    field.update((1.0e7, 0.0, 0.0))
    assert calls == [1]
```

- [ ] **Step 2: Run them, expect FAIL** (`AttributeError: out_of_reach`).

  Run: `PYTHONPATH=src ../SpaceFlight/.venv/bin/python -m pytest tests/test_asteroid_field.py -k "reach or far_field" -v`
- [ ] **Step 3: Implement** the out-of-reach test (`_out_of_reach(camera, ranges) -> bool`,
  the box distance in numpy on the 3-vector) and the early branch in `update`.
- [ ] **Step 4: Run the field's and haze's tests, expect PASS.**

  Run: `PYTHONPATH=src ../SpaceFlight/.venv/bin/python -m pytest tests/test_asteroid_field.py tests/test_asteroid_haze.py -v`
- [ ] **Step 5: Check what is left of the far field's frame.** If
  `stats["update_ms"]` at the `ground` pose is still over 0.1 ms, profile
  `haze.update` from afar (`haze.missing` was ~0.37 ms of profiled time) and
  give it the same early-out when no haze cell is in range. Then measure:
  `frame_profile.py --offscreen --pipelined --pose ground coast high` against
  the Task 1 baseline. Expected: about −1.1 ms everywhere on the home planet.
- [ ] **Step 6: Asteroid scenes unchanged.** Run `scripts/asteroid_bench.py`
  (its default) before and after: same frame within noise.
- [ ] **Step 7: CHANGELOG `[Unreleased]`, performance.md §38.**

### Task 3: All-sea chunks out of the sea's reflection

The reflection draws each chunk's coarse mesh; a chunk whose whole grid is
below sea level on a planet with a sea is all water, and `terrain.vert` clips
water in the mirror pass (`gl_ClipDistance = -1`): it costs the pass's cull and
draw and puts nothing on screen. Expected gain: part of the reflection's
0.55-0.6 ms, most at `coast` and over open sea.

**Files:**
- Modify: `src/world_builder/terrain/mesh.py` (`build_chunk_node`, ~93-148)
- Modify: `src/world_builder/terrain/terrain.py` (the call to `build_chunk_node` in the upload path)
- Test: `tests/test_terrain_mesh.py`, `tests/test_terrain_multibody.py` (or the terrain test that already builds a `Terrain` with `set_spec`)

**Interfaces:**
- Produces: `build_chunk_node(chunk, margin=0.0, slot=None, reflected=True) -> PandaNode`;
  with `reflected=False` the coarse child is shown to `HIDE_FROM_CLOUD_PASS`'s
  bits only (not `REFLECTION_CAMERA_MASK`).
- Produces: `in_reflection(chunk: ChunkData, has_ocean: bool) -> bool` in `mesh.py`.
- Terrain passes `reflected=in_reflection(chunk, spec.has_ocean)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_an_all_sea_chunk_is_left_out_of_the_reflection():
    spec = replace(load_planet("desert"), seed=3).resolved()
    node = build_chunk_node(generate_chunk(spec, (2, 4, 1, 1)), reflected=False)
    fine, coarse = (child.node() for child in NodePath(node).get_children())
    assert not _shown_to(coarse, REFLECTION_CAMERA_MASK)
    assert not _shown_to(fine, REFLECTION_CAMERA_MASK)
    for bit in (CLOUD_NEAR_BIT, CLOUD_FAR_BIT):
        assert _shown_to(coarse, BitMask32.bit(bit))

def test_dry_ground_below_zero_stays_in_the_reflection():
    chunk = generate_chunk(replace(load_planet("desert"), seed=3).resolved(), (2, 4, 1, 1))
    deep = replace(chunk, h_max=-5.0)
    assert in_reflection(deep, has_ocean=False)
    assert not in_reflection(deep, has_ocean=True)
    assert in_reflection(replace(chunk, h_max=0.5), has_ocean=True)

def test_set_spec_rebuilds_the_reflection_flag():
    # Terrain.set_spec regenerates every chunk (terrain.md): the flag is
    # computed at upload from the current spec, never cached on the key.
    # Assert in the upload path's test (or a Terrain with the inline
    # executor): after set_spec(has_ocean False), a re-uploaded deep chunk's
    # coarse child is shown to REFLECTION_CAMERA_MASK.
```

  The flag is a pure function in `mesh.py`,
  `in_reflection(chunk: ChunkData, has_ocean: bool) -> bool`
  (`not (has_ocean and chunk.h_max < 0.0)`), so the first two tests need no
  window. If `ChunkData` is not a dataclass, build the deep chunk by
  setting `h_max` on a copy instead of `replace`.
- [ ] **Step 2: Run, expect FAIL** (`unexpected keyword argument 'reflected'`).

  Run: `PYTHONPATH=src ../SpaceFlight/.venv/bin/python -m pytest tests/test_terrain_mesh.py -v`
- [ ] **Step 3: Implement** the parameter, the helper and the call.
- [ ] **Step 4: Run the terrain tests, expect PASS.**

  Run: `PYTHONPATH=src ../SpaceFlight/.venv/bin/python -m pytest tests -k "terrain or ocean" -q`
- [ ] **Step 5: No visible change.** `scripts/terrain_shots.py` at `coast`
  and an open-sea pose before and after, `scripts/compare_shots.py`: within
  two shots of the same tree's own difference.
- [ ] **Step 6: Measure** `t_refl` and the frame (`--pipelined --terrain-passes --pose coast ground`).
- [ ] **Step 7: CHANGELOG, performance.md §38.**

### Task 4: Spike: per-chunk draw overhead (decide, then plan)

Not an implementation: a measured go/no-go. Panda changes a transform per
chunk per pass (1045 a frame) and re-sends the shader's matrices with it;
Set State is 1.85 ms of the 3.9 ms draw. Two candidate fixes, to try in a
throw-away branch or worktree (`../WorldBuilder-chunkdraw`), measuring
`--pipelined --terrain-passes` and PStats' Draw/Set State and
State changes/Transforms:

- [ ] **Step 1: Shared transform.** Chunk nodes with no transform of their
  own under the terrain root; `terrain.vert` builds the position from the
  chunk table's centre (already texel 0) relative to the camera, the centre
  and camera split into high and low floats (relative-to-eye), and a
  camera-relative view-projection uniform. Bounds in the root's frame. The
  trees' impostor nodes (attached to chunk nodes, `trees.py` ~394) then need
  their own transform or the same scheme. Measure the gain before solving
  the impostors: hide the trees for the spike.
- [ ] **Step 2: Fewer nodes per chunk.** One GeomNode per chunk instead of a
  node plus two GeomNodes (the fine and coarse Geoms in one GeomNode, chosen
  per pass by a tag state or two index ranges), so each pass culls one node
  a chunk, not three.
- [ ] **Step 3: Write the result** into performance.md §38 and
  `docs/source/performance_backlog.md`: gain per pass, cost, risks
  (precision far from the origin, the impostors, SpaceFlight's own use of
  chunk nodes). **Go** if either saves ≥ 0.4 ms pipelined at `ground`; then
  write its implementation plan as a new task list in this file.

### Task 5: Terrain textures: no normal-map fetch where it is faded out

GPU only; worth doing once Tasks 2-4 bring the CPU near ~5 ms (check with
Task 1's tool which side bounds the frame first: if the pipelined frame
falls by less than the step, the GPU has become the limit).

`layeredMaterial` (`terrain/shaders/terrain_material.glsl`) samples every
layer's normal/roughness/AO texture at every distance, though the normal
is faded out past 600 m (`normalFade`) and AO past `far`; only the roughness is
still read. Past the fade, take each layer's mean roughness from a new
uniform `uMeanRoughness[6]` (computed with `uMeanAlbedo` in
`terrain/materials.py`) and skip the fetch.

**Files:**
- Modify: `src/world_builder/terrain/shaders/terrain_material.glsl`
- Modify: `src/world_builder/terrain/materials.py` (the means), `src/world_builder/terrain/terrain.py` (bind `uMeanRoughness`)
- Test: `tests/test_terrain_materials.py` or the test that covers `uMeanAlbedo` (find with `grep -rn uMeanAlbedo tests`)

- [ ] **Step 1: Failing test:** the material means carry a roughness per
  layer, each equal to the mean of the layer's roughness channel (`nra.b`)
  within 1/255.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the means, the uniform, and in the shader:
  `if (normalFade <= 0.0)` take `uMeanRoughness[i]` and AO 1 instead of the
  `uNormRA` fetch.
- [ ] **Step 4: Run the terrain tests, expect PASS.**
- [ ] **Step 5: No visible change:** `terrain_shots.py` at `ground`,
  `sunset`, `high`, `compare_shots.py` within two shots' own noise.
- [ ] **Step 6: Measure** serialized (`frame_profile.py --offscreen --pose ground sunset`)
  and pipelined. Keep it only if the serialized frame drops ≥ 0.15 ms.
- [ ] **Step 7: CHANGELOG, performance.md §38.**

---

## Phase 2: one camera, one depth ([#57](https://github.com/JoeDalton/WorldBuilder/issues/57))

After Task 4. Replace the near/far split cameras (`world_builder/depth.py`) with
one camera and a reversed-Z float depth buffer, then let the clouds read that
single depth instead of drawing their own passes. Together they take the
terrain from five passes to two (the main view and the reflection).

**Why it pays (Diagnosis numbers, terrain only, pipelined):**
- The split's own share: the main view's near camera 0.11-0.29 ms plus the
  clouds' near depth pass 0.11-0.23 ms. These are upper bounds: a single
  camera still draws the near chunks; what goes is the second cull over every
  chunk and the objects drawn twice across the split. Not measured: the haze's
  two passes and the clouds' near-range layers.
- The clouds' far depth pass: 0.85-0.95 ms, removed by Task 7.
- Code that goes: `split_clear_of`, objects drawn twice across the split, the
  haze's two passes, the clouds' near and far layers (`clouds/half_res.py`), the
  pairs of depth textures (`uSceneDepthNear` / `uSceneDepthFar`) in the clouds'
  resolve and ground shadow shaders.

**Platforms: one camera everywhere, the convention differs by platform.**
- **Desktop (Linux, Windows):** `gl-depth-zero-to-one` on (glClipControl),
  depth from 0 to 1, a 32-bit float depth buffer, the projection reversed (near
  at 1, an infinite far plane at 0), depth tests "greater", depth cleared to 0.
  Precision about 1e-7 of the distance at every range.
- **macOS:** Apple's OpenGL (frozen at 4.1) has no clip control
  ([Rosenzweig, "Clip control on the Apple GPU"](https://rosenzweig.io/blog/asahi-gpu-part-6.html)).
  The same single camera, the same passes and the same reversed projection,
  written for OpenGL's default [-1, 1] depth. It runs correctly. Its precision
  is that of one 24-bit range: about 6e-8 x d² / near (near 1 m: ~6 m at
  10 km, ~600 m at 100 km). Its look is not tuned: distant ridges may flicker
  at their silhouettes, and far clouds' edges against mountains are a little
  off. The sea cannot fight the coast, because the terrain's shader draws both
  on one mesh. From orbit, a fixed 1 m near plane cannot separate the moon from
  the planet behind it (~24,000 km of error at 20,000 km). A near plane that
  grows with altitude (1 % of the height above the nearest surface, at most
  1 km) brings that to ~24 km. It is harmless on desktop, so it applies to every
  platform.
- **The branches, and only these:**
  1. the projection matrix, written for [0, 1] or for [-1, 1];
  2. one shared GLSL depth-decode function (the clouds' march, resolve,
     composite and ground shadow; the haze; the sea), its convention a define;
  3. the startup's `gl-depth-zero-to-one`.
  Stock Panda objects keep working: reversed-Z lives in the projection and the
  depth test, not in each shader. Log depth was considered and dropped: every
  shader would have to write it, Panda's stock pipeline could not, and
  `gl_FragDepth` costs the early depth test.
- **The macOS branch is testable on Linux:** run with `gl-depth-zero-to-one`
  off. The tests run both conventions.

### Task 6: Spike: a reversed-Z single camera (go/no-go)

In a throw-away worktree (`../WorldBuilder-reversez`). The one unknown is
whether Panda 1.10's lenses and culling take a reversed projection cleanly
(a custom `MatrixLens`, or a reversed near/far on the lens).

- [ ] **Step 1: The desktop path, sky and terrain only.** A reversed lens
  with an infinite far plane, a float depth buffer
  (`FrameBufferProperties.set_float_depth(True)`), `DepthTestAttrib`
  "greater" on render, depth cleared to 0, `gl-depth-zero-to-one 1`. Every
  other renderer hidden.
- [ ] **Step 2: Check the culling.** The frustum culls as the split's
  cameras did: the chunk count drawn per frame at `ground`, `high` and `orbit`
  is the same as the split's within ~1 %.
- [ ] **Step 3: Check precision.** Shots at `ground` (distant ridges),
  `high` and `orbit` (moon against planet), and the user's tower pose (pass its
  `--start`), against the split's.
- [ ] **Step 4: The macOS path.** The same with `gl-depth-zero-to-one 0` and
  the [-1, 1] projection, with and without the altitude-scaled near plane. The
  same shots. Expected: ridge flicker far away on the ground, the moon
  separated from the planet only with the scaled near plane.
- [ ] **Step 5: Measure** with Task 1's `--pipelined --terrain-passes`
  (sky and terrain only, split against single camera).
- [ ] **Step 6: Write the result** into #57 (as a comment, if the user
  agrees), performance.md and `docs/source/performance_backlog.md`. **Go** if
  the culling and the desktop precision hold. Then write the migration as
  tasks in this file, one per renderer (depth.py's replacement, sky and stars,
  terrain, sea and its reflection camera, models, trees, asteroids and haze,
  clouds), each with its depth-convention tests in both conventions, and
  SpaceFlight's switch from the split cameras as the last.

### Task 6 result: GO (2026-11-27, performance.md §39)

Panda 1.10.16 takes `lens.set_near_far(inf, near)` as a reversed infinite
projection; culling holds (the same Geoms from 20 km and orbit, fewer at the
ground: no chunk drawn twice); desktop precision matches the split; the macOS
path speckles only from orbit, fixed by the altitude-scaled near plane. Only
sky + terrain: 0.2 ms less at the ground. Two findings shape the migration:
- **Float depth needs the output made for it.** Panda's default framebuffer
  properties cannot be changed from code: the scene's output (the game's window
  via `openMainWindow(fbprops=...)`, or a scene buffer) must ask for
  `float_depth` and 32 depth bits. `World` checks and logs when it did not
  (then: 24-bit reversed, about the macOS path's precision).
- **"Greater" goes on the perspective cameras' initial states, not on
  render**: the objects' caster map (`shadows/casters.py`, orthographic, its own
  buffer cleared to 1) draws the same graph and keeps "less".

### Phase 2 migration tasks (6.1-6.8)

Global for these tasks:
- **Both conventions under test.** `tests/conftest.py` sets
  `gl-depth-zero-to-one` from `WB_DEPTH_ZERO_TO_ONE` (default 1) before the
  session's window; CI and the final check run the suite twice (`=1`, `=0`).
- **One source of the convention:** `world_builder.depth.zero_to_one() -> bool`
  (the PRC variable as the GSG took it) in Python, and in GLSL
  `shaders/depth_convention.glsl` (`DEPTH_FAR_Z`, `float depthDistance(float
  depth, float near)`), its `#define DEPTH_ZERO_TO_ONE` prepended by the
  shader loader. No other file writes or decodes depth by itself.
- Each task: failing test first, `pytest -n auto` and `invoke quality` green
  under both conventions, CHANGELOG, a shot comparison (`terrain_shots.py`,
  `compare_shots.py`) at ground, sunset, high, orbit against the split.

#### Task 6.1: The single reversed camera (`world_builder/depth.py`)
- `ReversedDepthCamera(app, near: float = 1.0)`: takes over `app.cam`
  (mask: every camera bit the split's two cameras had), initial state
  "greater", its display region clearing depth to 0;
  `update(far: float, height_above_ground: float) -> None` sets the lens to
  `(inf, clamp(0.01 * height, near, 1000))` and the cull bounds to
  `frustum_bounds` of a finite copy reaching *far*; attributes `far`, `near`;
  `float_depth: bool` (the output's properties); `destroy()` restores the
  camera.
- `reversed_lens(lens) -> Lens` and `zero_to_one() -> bool` helpers.
- Tests: the projection maps near to 1 and far to 0 (or -1 under `=0`); the
  cull bounds reach *far*; the near plane follows the height within [1 m,
  1 km]; `destroy` restores mask and clear.

#### Task 6.2: Sky and stars
- Found in the spike: stars are kept off a planet by `stars.vert`
  (`atmoHitsGround`), but off an airless body (the moon) only by the depth
  test: drawn without depth they cover it. Pinned at depth 0 with "greater or
  equal", after the ground, they are right everywhere (moon shot identical to
  the split's). Test: a star direction behind the moon from orbit stays dark.
- `sky.vert`, `stars.vert`: `gl_Position.z = DEPTH_FAR_Z * gl_Position.w`
  (just inside the far plane in either convention), depth test "greater or
  equal" (`sky.py`), the sky still drawn after the terrain.
- Tests: the sky's state's depth test; shaders include
  `depth_convention.glsl` and pin nothing else.

#### Task 6.3: Terrain and the sea's reflection
- Found in the spike: the reflection must switch in the same step as the main
  camera. Left standard with "greater" anywhere above it, its ground and sky
  fail their depth tests and the far sea (which reflects them) darkens by up
  to 25 levels. The reversed state goes on each camera's initial state, never
  on render.
- The terrain needs nothing (Panda's projection). `ocean/sea.py`'s reflection
  camera: a reversed copy of the lens, initial state "greater", its buffer
  cleared to 0 and asking for float depth; its clip plane unchanged.
- Tests: the reflection camera's lens and state; a coast shot against the split.

#### Task 6.4: Objects, trees, casters
- Models and trees draw through Panda's projection: no change. The caster
  map keeps "less" and its own clear (test: its camera's state is not the
  main one's).
- Test: a model drawn in front of the terrain at 5 m and at 50 km sorts right
  (offscreen render, read back).

#### Task 6.5: Asteroids and their haze
- `haze.vert`'s far clamp through `DEPTH_FAR_Z`; `haze.py`'s depth test
  "greater or equal"; the haze's near-camera passes (`_near_passes`,
  `set_split`, `uHazeSplit`, `uHazeToFar`) deleted.
- Tests: the haze has one pass per bin; the rocks' veil without a split.

#### Task 6.6: The clouds on one depth range
- `clouds/march.py`: one depth pass (the far and near passes merged), one
  card; `cloud_march.glsl`, `cloud_resolve.frag`, `cloud_composite.frag`
  (its `gl_FragDepth`), `ground_shadow.frag` decode through `depthDistance`
  and read one texture (`uSceneDepth`); `clouds/half_res.py`'s two layer
  cameras become one; `use_split_depth` becomes `use_depth_camera`.
- Tests: the existing march tests in both conventions; the depth pass's
  decode against known distances (offscreen: a quad at 10 m, 10 km, 1000 km).

#### Task 6.7: World wiring and removal of the split
- **Split in two (2026-11-27):** 6.7a wires the world to the reversed camera
  (done: `World.use_depth_camera`, `World.view_ranges`, `GameWorld`, a
  float-depth scene buffer presented in the window, clip control in
  `tools.window_config`); 6.7b removes the split (`SplitDepthCameras`, the
  haze's near passes, the clouds' dormant near range). **Order: 6.7a, then
  6.8 (SpaceFlight off the split), then 6.7b**, or 6.7b breaks SpaceFlight.
- **6.7b done (2026-11-27), before 6.8:** SpaceFlight is pinned on a
  WorldBuilder tag, so removing the split cannot break it; 6.8 moves it to
  the reversed camera when it takes a new tag.
- `GameWorld` builds `ReversedDepthCamera`; `World.depth_ranges` becomes
  `World.far_distance` plus the height above the nearest ground;
  `SplitDepthCameras`, `split_clear_of`, `camera_span`, the NEAR/FAR masks'
  split meaning, and `MIN_SPLIT_M` deleted (the masks kept as aliases of the
  one camera's for a release, logged as deprecated); docs (`world.md`, the
  depth module's), CHANGELOG under "Changed" (breaking: games that built
  `SplitDepthCameras` themselves).
- Tests: `test_depth.py` rewritten for the new camera; `test_world.py`'s
  split tests replaced.

#### Task 6.8: SpaceFlight's switch (last; its repo)
- Its window or scene buffer asks for float depth; its own uses of the
  split (cameras, masks, post-process) move to the one camera; its
  `bench_mission3.py` before and after (performance_backlog.md's Mission 3
  table) and its shots.

### Task 7: The clouds read the single depth (after Task 6's migration)

With one camera there is one depth buffer. The clouds' far depth pass
(0.85-0.95 ms of terrain CPU alone, plus its GPU) can go: the main view renders
into a buffer whose depth is a texture, the depth is shrunk to the march's
resolution, the march runs, the clouds are composited after the scene, and the
result goes to the window. Needs its own design (brainstorm first). Two issues
it must solve:
- **The ground shadow pass** (§33) is read by the terrain's shader *during*
  the main view and is computed from this same depth. Either it uses last
  frame's depth (reprojected), or the ground goes back to the per-pixel shade
  (+0.4-0.7 ms GPU, §33).
- **SpaceFlight's post-process.** The game owns the window. `World` renders
  into a buffer the game hands it, and the game's post-process takes its colour
  and depth from there. Do it with Task 6's SpaceFlight switch, so the game
  changes once.

## Decisions for the user (measured, visible, not tasks)

| Change | Gain (measured, `ground`/`sunset`) | Look |
|---|---|---|
| Terrain `anisotropy` 8 to 4 at HIGH | up to 0.35-0.5 ms GPU (8 to 2 measured) | softer ground at grazing angles |
| Terrain `detail_far` 4000 to 2500 m | up to 0.55-0.75 ms GPU (to 1500 measured) | textures fade to flat colour sooner |
| Terrain `split_factor` 2 to 1.5 | 0.6-1.3 ms (0.4-0.75 of it CPU) | less distant detail |
| The reflection's reach (chunks past N km out of it) | up to 0.55-0.6 ms CPU | far shores no longer reflected |
| A split by roughness (backlog: flat ground and seabed stay coarse) | most of `split_factor`'s gain where the ground is flat | none intended; a larger change in `lod.c` and the selector |
