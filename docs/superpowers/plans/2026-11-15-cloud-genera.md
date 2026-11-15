# Cloud Genera Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the cumulonimbus anvil, congestus and cirrus looks, then model all ten WMO genera in
one raymarch, with a separated-genera showcase demo.

**Architecture:** Phases 1–2 are planned in detail:

- Phase 1 fixes the convective field (`noise.py` and `cloud_field.glsl`, kept in lock-step).
- Phase 2 removes the billboard renderer. It keeps the deck (the far field and the sea's
  reflection) and the march's helpers by moving them out of `sphere.py`.

Phases 3–7 are planned at task level. Each is refined into steps when it starts, using what the
previous phase measured.

**Tech Stack:** Python 3, NumPy, Panda3D, GLSL 330, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-cloud-genera-design.md`; research:
`docs/source/cloud_genera_research.md`.

## Global Constraints

- **Python:** `../SpaceFlight/.venv/bin/python` with `PYTHONPATH=src` (tests:
  `PYTHONPATH=src ../SpaceFlight/.venv/bin/pytest tests -q`).
- **Performance:**
  - demo_terrain at HIGH, 2560×1440, RTX 3070: **>160 FPS**;
  - the showcase (`scripts/demo_genera.py`): **>60 FPS**.
- **Shots reproducing the user's view** must pass `--start 29.7159 29.8232` (the genera are laid
  out around the start pose).
- **CPU mirrors:** every NumPy mirror of a GLSL function says `MUST match cloud_field.glsl` and
  changes in the same commit as the shader.
- **Docs:** documentation follows the existing style. `clouds.md` explains each decision and its
  rejected alternative.
- **Commits:** end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Coverage drift.** Changing the anvil or congestus profile must not change how much sky is
   cloud. The depth shift tables (`genera.depth_shift_tables`) are recomputed from the new
   profile, and the coverage test holds within 4 % at depths 0, ½ and 1.
2. **Shadow/view disagreement.** The sun march (`slabSampleDensity`) and the view sample must
   agree where the anvil ends. A smooth anvil in view with a lumpy one in the shadow maps
   shows as blotchy light.
3. **Camera inside the anvil or tower.** Flying through must not pop. The octave fade is a
   function of position only, never of the step or the ray.
4. **Sea reflection and orbit after billboard removal.** The deck must still draw in the
   reflection and from orbit at every quality preset.
5. **Old settings files.** A saved settings file holding `renderer`, `count_scale`,
   `interior_growth` or `half_resolution` must load without error (keys ignored).

---

## Phase 1: Cb anvil and congestus

### Task 1: Baseline shots, poses and timing

**Files:**
- Modify: `scripts/terrain_shots.py` (add poses `genera_cb`, `genera_congestus`,
  `cirrus_overhead`, `cirrus_grazing`; document them in the module docstring)
- Modify: `docs/source/performance.md` (baseline row)

**Interfaces:**
- Produces: the pose names above, used by every later phase's before/after shots.

The poses are relative to the start pose's ground point. The features come from
`genera.DEFAULT_FEATURES`: a Cb 40 km north, congestus around (−30, 20) and (−30, 40) km.

| Pose | Height | Heading | Pitch | Shows |
|---|---|---|---|---|
| `genera_cb` | 300 m | 0° | 8° | the tower and anvil, 40 km away |
| `genera_congestus` | 300 m | 310° | 6° | — |
| `cirrus_overhead` | 300 m | 210° | 75° | — |
| `cirrus_grazing` | 300 m | 210° | 4° | — |

- [ ] **Step 1:** Add the four poses in the existing pose tables (the sun default 210/35 is fine).
- [ ] **Step 2:** Run
  `PYTHONPATH=src ../SpaceFlight/.venv/bin/python scripts/terrain_shots.py --start 29.7159 29.8232 --pose genera_cb genera_congestus cirrus_overhead cirrus_grazing --size "2560 1440" --out shots/baseline`.
  Expected: four PNGs and timings printed. Look at each image before going on: the anvil must
  be visible in `genera_cb`.
- [ ] **Step 3:** Record each pose's mean and p95 frame time in `performance.md` under a new
  "Cloud genera baseline (2026-11-15)" heading.
- [ ] **Step 4:** Commit: `shots: genera and cirrus poses, genera baseline timings`.

### Task 2: Diagnose the missing congestus

**Files:**
- Create: `tests/test_clouds_congestus.py`

**Interfaces:**
- Consumes: `noise.density_on_sphere`, `genera.genus_planes`, `genera.GeneraSpec`.
- Produces: `column_cloud_profile(spec, centre, radius_m, heights) -> np.ndarray`, a test helper
  in this file. It returns the share of columns within `radius_m` of `centre` that are cloud at
  each height.

- [ ] **Step 1:** Write the diagnostic test
  `test_congestus_reaches_its_top`. At the congestus feature (−30 km, 20 km), the share of
  cloudy columns between 4 and 6 km must be at least 0.25 of its value at 2.5–3.5 km (a
  tower: cloud reaches its top). Also write `test_congestus_is_towers_not_scatter`. Over the
  same disc, the cloudy samples above 4 km must sit over columns that are also cloudy at
  3 km in at least 60 % of cases (vertically connected, not scattered blobs).
- [ ] **Step 2:** Run
  `PYTHONPATH=src ../SpaceFlight/.venv/bin/pytest tests/test_clouds_congestus.py -v`.
  Expected: FAIL. The hypothesis, from clouds.md "a cumulonimbus is coverage, not noise", is
  that at coverage 0.45 a depth-½ column holds scattered blobs at every height. Record the
  measured shares in the test's docstring.
- [ ] **Step 3:** If the tests PASS, the field is fine and the problem is visual (haze, or out of
  view). Compare the `genera_congestus` shot against the numbers, and note the finding in
  clouds.md "Genera / Not yet". Task 3 then changes only the shape (taper, dome), not the
  connectivity.
- [ ] **Step 4:** Commit: `test: congestus column diagnostics`.

### Task 3: Congestus as a tapered, connected tower

**Files:**
- Modify: `src/world_builder/clouds/noise.py` (`DensityField`, `threshold_lift`)
- Modify: `src/world_builder/clouds/shaders/cloud_field.glsl` (`thresholdLift`, `Field`,
  `getField`)
- Modify: `src/world_builder/clouds/noise.py` (`pack_layer_params`, the new fields' slots)
- Modify: `src/world_builder/clouds/cloud.py` (`PRESETS[CONVECTIVE]`)
- Modify: `src/world_builder/clouds/genera.py` (`DEFAULT_FEATURES` congestus coverage, if Task 2
  showed scatter)
- Test: `tests/test_clouds_congestus.py`, `tests/test_cloud_noise.py`

**Interfaces:**
- Produces: `DensityField.tower_taper: float = 0.0` (the share of the threshold window the
  threshold rises by at the local top, scaled by the radial distance from the cell octave's
  core) and `DensityField.dome: float = 0.0` (a parabolic lift, `dome * local²`, applied only at
  depth > 0.3). Both are neutral at 0.

Approach, following Nubis's separate top type:

- **Connectivity** comes from raising `cell_weight`'s share at mid depth, so the cell octave
  gathers columns into towers. If Task 2 showed scatter, also raise the congestus features'
  coverage toward 0.7, as was done for Cb.
- **Shape** comes from `dome`: the threshold rises parabolically toward the local top, so only
  the cell's core reaches it and the top is rounded.

- [ ] **Step 1:** Add `test_neutral_profile_unchanged` to `test_cloud_noise.py`.
  `threshold_lift` with `tower_taper=0` and `dome=0` must equal today's values on a grid of
  (h, depth). Use today's function body copied into the test as the reference.
- [ ] **Step 2:** Implement the two fields in `threshold_lift` and in the GLSL `thresholdLift`.
  Pack them into a free slot of `layerParams` (check `LAYER_VEC4S` for a spare component, and
  widen it by one vec4 only if none is free).
- [ ] **Step 3:** Set the `CONVECTIVE` preset's `dome` and `cell_weight` (start at
  `dome=0.15`, `cell_weight=0.7`). Run the Task 2 tests and tune until they pass.
- [ ] **Step 4:** Run the coverage tests:
  `PYTHONPATH=src ../SpaceFlight/.venv/bin/pytest tests/test_clouds_genera.py -k "coverage or shift" -v`.
  Expected: PASS, within 4 % at depths 0, ½ and 1.
- [ ] **Step 5:** Shoot `genera_congestus` and `genera_cb` into `shots/phase1`. Compare with the
  baseline and with WMO photo `4766_main_cumulus-congestus_clouds.JPG`. Towers must be taller
  than wide, with sharp tops.
- [ ] **Step 6:** Commit: `clouds: congestus as tapered connected towers (dome, cell weight)`.

### Task 4: Smooth Cb anvil

**Files:**
- Modify: `src/world_builder/clouds/noise.py`:
  - `threshold_lift`: remove the anvil term;
  - `ANVIL_DEPTH` and `ANVIL_BAND`: keep their names, now driving the octave fade;
  - new `anvil_weight`;
  - `density_on_sphere` and `_slab_density`: fade octaves 3–4 to their mean by
    `anvil_weight`;
  - `with_shelf`.
- Modify: `src/world_builder/clouds/shaders/cloud_field.glsl`:
  - `thresholdLift`;
  - new `anvilWeight`;
  - `cloudDensityAt` and `cloudColumnAt`: `highOctaves` mixed toward its mean by the anvil
    weight;
  - `shelfDensity`: wispy erosion on the underside.
- Modify: `src/world_builder/clouds/shaders/cloud_march.frag`: billows' `window` scaled by
  `1 - anvilWeight`.
- Modify: `src/world_builder/clouds/genera.py`: `anvil_spread` gets a shear offset along the wind
  with height (the anvil leans downwind).
- Test: `tests/test_cloud_noise.py`, `tests/test_clouds_genera.py`

**Interfaces:**
- Produces: `noise.anvil_weight(spec: DensityField, h: ArrayLike, depth: ArrayLike) -> np.ndarray`
  and GLSL `float anvilWeight(Field f, float h, float depth)`. Both return
  `smoothstep(*ANVIL_DEPTH, depth) * smoothstep(*ANVIL_BAND, local)`, where `local` is
  `h / top_fraction`. It is 0 for a field with `anvil == 0`. Later phases (Cb in the genus
  framework) reuse it.

Approach:

- The anvil stops being a lowered threshold on the cauliflower fbm. Where `anvilWeight > 0`, the
  fine octaves and the billows fade to their means. The tower's top then has the shelf's
  smoothness, and the shelf (already 2-octave) carries the spread.
- Inside the anvil band the noise is stretched horizontally: noise coordinates scaled by
  `1 / (1 + 3 * anvilWeight)` on the two horizontal axes of the local frame. Use the up vector
  `normalize(p)`, which is already available.
- Erosion is wispy and carves only the shelf's underside: the billow window applies where
  `shelfWeight < 0.5`, at a quarter strength.

- [ ] **Step 1:** Write `test_anvil_band_is_smooth`. Sample `density_on_sphere` over a depth-1
  column's anvil band (local 0.8–0.95) on a 2 km × 2 km grid at 50 m. The density's
  high-frequency energy (variance of the field minus its 500 m box blur) must be at most 0.3
  of the same measure at local 0.4 in the same column. Also write `test_anvil_weight_neutral`:
  `anvil_weight` is 0 everywhere for `PRESETS[CUMULUS].field`.
- [ ] **Step 2:** Run both. Expected: the first FAILs (the band is as lumpy as the tower), the
  second fails with `AttributeError`.
- [ ] **Step 3:** Implement `anvil_weight` and the fade in `noise.py`. Remove the anvil term from
  `threshold_lift`.
- [ ] **Step 4:** Mirror it in `cloud_field.glsl` (`anvilWeight`, the octave fade in
  `cloudDensityAt` and `cloudColumnAt`, and the sun march's `slabSampleDensity`, which already
  uses two octaves and needs only the stretch). Then add the billow scaling in
  `cloud_march.frag`.
- [ ] **Step 5:** Add the shear: the anvil's spread map is sampled offset downwind by
  `ANVIL_SHEAR * (h - SHELF_BAND[0]) * slab_thickness` metres along the cell's wind. Start
  with `ANVIL_SHEAR = 1.5` in `genera.py`. The convective cube's four channels are all used, so
  the wind goes in one uniform, `uAnvilWind` (vec3, planet frame, unit), set by `CloudMarch`
  from `GeneraSpec.wind`, or from `upper_wind_at` at the focus. That is enough for the
  placeholder maps, which have one weather region. The NumPy mirror takes it as a
  `wind: np.ndarray` argument.
- [ ] **Step 6:** Run `tests/test_cloud_noise.py tests/test_clouds_genera.py -v`. Expected: PASS,
  including the coverage-by-depth tests (recomputed tables).
- [ ] **Step 7:** Shoot `genera_cb` from the ground and one pose inside the anvil (with `--at`, the
  Cb at 9 km altitude, pitch 0). Run `cloud_bench.py --stability 120` on the `genera_cb` pose:
  steadiness no worse than the baseline. Compare with WMO photo
  `5993_main_cumulonimbus-capillatus-praecipitatio-incus-mamma_clouds.jpg`.
- [ ] **Step 8:** Update `clouds.md` "The anvil's shelf" and "Not yet": the anvil is now an
  octave fade plus a shear, and the earlier threshold-lowering band goes in the "Tried on the
  way" list with why it was lumpy.
- [ ] **Step 9:** Commit: `clouds: smooth anvil (octave fade, stretch, shear), not a lowered threshold`.

---

## Phase 2: Remove the billboard renderer

### Task 5: Move the kept pieces out of `sphere.py`

**Files:**
- Create: `src/world_builder/clouds/field_setup.py`. It takes from `sphere.py`, unchanged:
  `_noise`, `_weather`, `_genus_weather`, `_shift_plane`, `_peaks`, `_cached_peaks`,
  `_depth_tables`, `_deck_light`, `octave_offset`, `planet_field`, `planet_weather`,
  `shell_bands`, `ShellSpec`, `DECK_MAX_OPTICAL_DEPTH`. The march's octave bands still use
  `ShellSpec` and `shell_bands`; rename both in Task 7, not here.
- Create: `src/world_builder/clouds/deck.py`. It takes `deck_mesh`, `CLOUD_BIN`,
  `_CLOUD_BIN_SORT`, `_ensure_bin`, `_vertex_format` (if the deck uses it) and the deck half of
  `_shaders()` as `deck_shader() -> Shader`.
- Modify: `march.py`, `planet.py`, `cloud_bench.py`, and the tests importing these names. Update
  only the imports.

**Interfaces:**
- Produces: `field_setup.*` and `deck.deck_mesh`, `deck.deck_shader`, `deck.CLOUD_BIN`. Their
  signatures are identical to today's `sphere.*`.

- [ ] **Step 1:** Move the names. `sphere.py` re-imports them from the new modules, so nothing
  else breaks yet.
- [ ] **Step 2:** Run the full suite. Expected: PASS, the same count as before.
- [ ] **Step 3:** Point every importer outside `sphere.py` at the new modules
  (`grep -rn "clouds.sphere import" src scripts tests`).
- [ ] **Step 4:** Run the full suite. Expected: PASS.
- [ ] **Step 5:** Commit: `clouds: move the march's helpers and the deck out of sphere.py`.

### Task 6: Delete the billboards

**Files:**
- Delete:
  - `CloudVolume` and the tile code (`generate_cloud_tile`, `interior_levels`, `_Tile`,
    `_tile_node`, `_sort_billboards`, `CloudTile`, `SphereCloudSpec`, the shell bins), and so
    `src/world_builder/clouds/sphere.py` once empty;
  - `shaders/cloud_sphere.vert`;
  - `half_res.py`;
  - `tests/test_clouds_sphere.py`, except the deck tests: move `deck_mesh`'s test to a new
    `tests/test_clouds_deck.py`.
- Modify:
  - `shaders/cloud.frag`: keep only the deck's path (it is the deck's fragment shader);
  - `march.py`: `sun_visibility = CloudVolume.sun_visibility` becomes a module function in
    `field_setup.py`, moved verbatim;
  - `planet.py`: drop `RENDERERS`, the stratus `CloudVolume`, the half-res pass, and every
    `isinstance(..., CloudVolume)` branch;
  - `cloud.py` `CloudSettings`: drop `renderer`, `count_scale`, `interior_growth`,
    `half_resolution`, and the billboard-only `CloudSpec` fields `radius`, `aspect` and
    `deep_radius` with their scaling functions;
  - `quality.py` and `settings.py`;
  - `terrain/workers.py`, `terrain/query.py` and `terrain/trees.py`: drop the cloud tile jobs;
  - `world.py`, `depth.py`, `shadows/__init__.py`, `shadow_map.py`, `shadow_defaults.py`,
    `light_volume.py`;
  - `scripts/demo_terrain.py`, `scripts/cloud_bench.py` (the M renderer switch),
    `scripts/flight_bench.py`, `scripts/terrain_shots.py` (`--cloud-renderer`);
  - tests: `test_clouds_march.py`, `test_clouds_genera.py`, `test_quality.py`,
    `test_cloud_shadows.py`, `test_clouds_planet.py`, `test_scene.py`,
    `test_cloud_light_volume.py`, `test_cloud_noise.py`, `test_helpers.py`.

**Interfaces:**
- Consumes: Task 5's modules.
- Produces: `CloudSettings` without the four billboard fields. `CloudSettings.from_dict`, or
  whatever the settings loader is, ignores unknown keys (Review Focus 5).

- [ ] **Step 1:** Write `test_old_settings_keys_ignored` in `tests/test_quality.py`. Load a
  settings dict holding `renderer="billboards"`, `count_scale=0.5`, `interior_growth=True` and
  `half_resolution=True` through the project's settings loader. It must return valid
  `CloudSettings` equal to the defaults for every remaining field.
- [ ] **Step 2:** Run it. Expected: PASS today, which pins the behaviour before the fields go.
- [ ] **Step 3:** Delete the code and files listed above. For each test that exercised only
  billboards, delete it. For each that compared billboards with the march, keep the march half.
- [ ] **Step 4:** Run `grep -rn "CloudVolume\|billboard\|half_res\|count_scale\|interior_growth\|renderer" src scripts tests`.
  Expected: only docstrings explaining history, which Task 7 rewrites, and unrelated uses (the
  word "renderer" for Panda's).
- [ ] **Step 5:** Run the full suite. Expected: PASS.
- [ ] **Step 6:** Shoot `clouds_noon`, `clouds_grazing`, `night_sea` (the reflection's deck),
  `mountains_orbit` (the deck from orbit) and `genera_cb` at LOW and HIGH. Expected: identical
  to the phase 1 shots, apart from the march's noise.
- [ ] **Step 7:** Commit: `clouds: remove the billboard renderer`.

### Task 7: Rename the octave bands, rewrite the docs

**Files:**
- Modify: `field_setup.py` and `march.py`: `ShellSpec` becomes `OctaveBand`, `shell_bands`
  becomes `octave_bands`, and `CloudSettings.shells` becomes `octave_bands`. The settings loader
  maps the old `shells` key onto it.
- Modify: `docs/source/clouds.md` (rewrite "The parts", "The types and their cost" and
  "Lighting" without billboards; keep the march sections), `docs/source/quality.md`,
  `docs/source/performance.md` (mark billboard sections historical),
  `docs/source/atmosphere.md`, `docs/source/index.md` (the clouds line).

- [ ] **Step 1:** Rename, with the old-key mapping, and add a test: `shells=2` in a settings dict
  loads as `octave_bands=2`.
- [ ] **Step 2:** Run the full suite. Expected: PASS.
- [ ] **Step 3:** Rewrite the docs.
- [ ] **Step 4:** Time demo_terrain from the user's start at HIGH, 2560×1440. Expected: >160 FPS.
  Add the row to `performance.md`.
- [ ] **Step 5:** Commit: `clouds: octave bands, docs without billboards`.

---

## Phase 3: The union-march genus framework (refined 2026-11-15)

Design (Plan-agent review of the code, accepted):
- **Sky bands.** Genus slabs merge in Python into at most 4 disjoint sky
  bands (`sky_bands`). The shader keeps today's segment code per *band*
  (≤ 2 segments each, ≤ 8, insertion-sorted), so `SEGMENT_SAMPLES`,
  `stretch`, `boundary` and the jitter are unchanged and gaps between bands
  cost no steps. Inside a segment, each genus whose slab holds the sample and
  whose coverage > 0 adds its density.
- **One light volume and one shadow-map pair** of the summed density
  (`SLICES` 64, the chart over the union slab). Each genus has a
  `far_shadow` weight: the light volume's depth beyond the genus's sun window
  is added (stratus 1: today's full chord from the Cb; convective 0).
- **Weather.** Genus 0, the primary genus, keeps the full weather cube
  (coverage, shift, depth, spread): planets without genera are one genus and
  unchanged. The others read the coverage cubes (`genusCoverA`: St, Sc, Ns,
  As; later `genusCoverB`: Ac, Ci, Cs, Cc), RGBA16F, each turned into an fbm
  shift by a row of `genusShift` (256 × `MAX_GENERA`, R32F).
- **Parameters.** There is one `layerParams` entry per genus
  (`LAYER_VEC4S` 16; row 13 is `(coverageSource, coverageChannel, shiftRow,
  farShadow)`). The octave bands become detail levels: `uDetailFar`,
  `uDetailOctaves` and `genusLevelOffset[genus*8+level]`. There is also a
  deck slot at `MAX_GENERA`. `MAX_GENERA` is 10, about 205 vec4 in all.
- **Deck.** Genus 0's only, as today.
- **API.** `PlanetClouds.shadow_families` and `uCloudShadowOther` go. One
  caster (`_UnionCaster`) computes `sun_visibility = exp2(-Σ depth_g)`.
- **Performance.** Expected within ±0.2 ms. Later phases must skip genera at
  zero coverage before any noise, and light once per sample.

### Task 8: Genus model and CPU geometry (pure Python)
- `noise.GenusField` is frozen: `name`, `field`, `optics`, `coverage`
  (`("weather", 0)` or `("A", ch)`), `far_shadow`, `noise_kind="value"`.
- `noise.pack_genus_params(genus, extinction, octave_mean, octaves)` packs
  rows 0–12 as now, plus row 13.
- `genera.GENUS_PRESETS` holds `GenusPreset(cloud_type, coverage,
  far_shadow, slab=None)` for `"convective"` and `"stratus"`;
  `GeneraSpec.genera = ("convective", "stratus")`.
- In march.py, `sky_bands(slabs) -> list[(base, top)]` and
  `band_segments(...)` replace `family_segments`; there are per-level
  threshold offsets.
- Tests:
  - band merging for overlapping, disjoint and nested slabs;
  - `band_segments` equals the old `family_segments` on disjoint slabs;
  - the row contract of `pack_genus_params`;
  - the uniform total is ≤ 256 vec4.

### Task 9: Coverage cubes, shift texture, deck light array (not wired)
- `genera.coverage_cube(spec, "A")`, `weather.build_shift_texture(tables)`,
  `noise.build_deck_light_array`, `genus_weather_at` (CPU).
- Tests:
  - stratus coverage within 5 % through coverage + table;
  - convective cube unchanged;
  - `test_the_families_cubes` becomes a cube-A test.

### Task 10: Uniform layout swap, no behaviour change
- The new `layerParams` layout and `genusField(g, level)` go into every
  shader and bake. The families and `other*` samplers stay for now.
- Verify: shots at `genera_cb`, `genera_congestus`, `clouds_noon` and
  `night_sea` (`--start 29.7159 29.8232`) are pixel-identical before and
  after, as a numeric image diff. Adapt the layer-packing and family tests.

### Task 11: Union light resources (riskiest)
- Covers: one volume, one map pair, summed bakes (`genusSampleDensity`,
  `skyChord`), `far_shadow`, the receivers (`ground_shadow.frag`,
  `terrain_light.glsl`, `model_light.glsl`, `god_rays.glsl`), the CPU
  `sun_visibility`, and `light_depth_on_sphere` for a list of genera.
- Tests:
  - one bake checked against the CPU over both feature sites;
  - the union map against the CPU twin;
  - the ground-shadow twin;
  - `sun_visibility` equals the old product.
- Shots: a diff where only small stratus-lighting differences are allowed;
  `--cloud-debug 512` and `1024`.

### Task 12: Union view march (risky)
- Sky-band segments, the genus loop, the stratus from cube A + `genusShift`,
  `genusDeckLight` as a 2D array. `otherWeather`, `otherDeckLight`,
  `uFamilies` and `cloudRadianceOther` go.
- Tests:
  - shader constants against the CPU;
  - a GPU smoke test with a synthetic third genus (a stratus copy at 3–4 km,
    inside the convective slab, in cube A's G channel) that renders finite.
- Shots: `--cloud-debug 8`, near-identical except at stratus edges, then
  normal shots.

### Task 13: API cleanup, docs, performance
- `PlanetClouds`, `CloudShadowMaps` and `CloudLightVolumes` become single
  resources. `check_layers` drops the stratus-under rule and checks
  genera ≤ `MAX_GENERA`, sky bands ≤ 4, and the cirrus against the union top.
- Update `clouds.md` ("Genera", "The raymarch", "The passes").
- Add the timing rows to `performance.md` §37.

## Phase 4: Showcase demo

### Task 11: `showcase_features()` and `scripts/demo_genera.py`
In `genera.py`, `showcase_features(focus) -> tuple[Feature, ...]`: an arc 40–70 km from the
start, one strip per genus, 15–25 km wide, each at its real altitude. `Feature.genus` accepts
every genus name.

`scripts/demo_genera.py` builds a `TerrainDemo` with that `GeneraSpec`, a fixed start, and keys
1–0 that toggle each genus. `terrain_shots.py` gets a `genus_<name>` pose per genus. Timing:
>60 FPS with all of them in view.

## Phase 5: Volumetric genera (one task each, in this order)

| Task | Genus | Requires |
|---|---|---|
| 12 | Worley noise (prerequisite) | `noise.worley_volume(size, cells, seed) -> np.ndarray` (F1, F2 in two channels, tileable) and its texture; tests check tiling and the range |
| 13 | St | the existing stratus moved to the genus recipe (fractus base) |
| 14 | Sc | Worley cells, open or closed by a weather channel |
| 15 | Ns | fbm, thick, ragged base, virga tail |
| 16 | As | stretched fbm, low extinction, striations |
| 17 | Ac | thin Worley lens, few samples |

Each task covers:

- the recipe from the spec;
- a coverage test;
- a showcase shot compared with its WMO photo;
- a `--stability` run;
- a timing row.

## Phase 6: The high layer

| Task | What |
|---|---|
| 18 | Thin-layer sampling in the march: 1–2 jittered samples per crossing of the high slab, with `cirrus.frag`'s thin-slab lighting moved into `cloud_shading.glsl` |
| 19 | Cs veil |
| 20 | Cc grains with a footprint fade |
| 21 | Ci flow streaks: the upper-wind warp plus curl, Worley tufts, 4–8 taps along the flow for tails, sheared fall streaks. A tuning preview in the style of `cirrus_preview.py`, procedural, top-down beside the WMO photo |
| 22 | Delete `CloudSheet`, `cirrus_atlas.*`, `bake_cirrus_atlas.py`, `cirrus_preview.py` and `cirrus.frag`/`.vert`; keep `cirrus_cover_map` and the upper wind (rewired as the Ci channel's source) |

## Phase 7: Docs and performance

### Task 23
Update `clouds.md` and `atmosphere.md` (cirrus) for the genus framework. Finish the
`performance.md` rows: demo_terrain >160 FPS, showcase >60 FPS. Republish the research artifact
if the findings changed.
