# Marched clouds: stable, both genera families, lit from one sun-transmittance map

## Context

The raymarch (`CloudSettings.renderer = "march"`; `clouds/march.py`, shaders `cloud_march.frag`,
`cloud_resolve.frag`, `cloud_composite.frag`) should cut frame time when big or many clouds fill the view:
billboards pay for the volume on screen as overdraw (`docs/source/performance.md` §21–22: 60–150 ms near
cumulonimbus at HIGH). Open problems of the march:

- **Unstable** (jitter, dot grid): Phase 1, partly fixed, see "Open issues".
- **No genera**: it samples `cloudDensityShifted` through one slab, so `PlanetClouds` forces billboards when genera are on.
- **No shadows**: it casts none on ground, sea or models, and takes no eclipse, mountain or ring shadow (`PlanetClouds._bind_layer` skips `CloudMarch`).
- **Per-sample lighting cost**: each cloudy sample runs the full sun march (`sunOpticalDepthAt`).

Insight: cloud shadow maps and the march's lighting are one problem, transmittance toward the sun at a point.
Extending `clouds/shadow_map.py` from "optical depth of the whole chord" to "cumulative depth at several heights"
gives one GPU-baked map that lights the march, shades ground/sea/models, and covers both families.

**Decisions:** the march becomes the default renderer (billboards as fallback), tuned for its own look, not
billboard parity · one march for both genera families (bands never overlap; family picked by height) · no godrays ·
target GTX 1070 at 1600×900, p95 first, mean must not rise · stop for the user's demo check after each phase.

## Phase 0: baseline and tools (done, performance.md §23)

`scripts/cloud_bench.py` has `--stability N` (per-pixel spread over still frames), a grid score (absolute, in
8-bit steps, still and `--move`), `--cloud-debug` bits (no sun march, 16 steps/pixel, no billows).
Findings on the 1070 that set the priorities:

- Over fair-weather cumulus billboards are cheaper (1.6–4.9 ms vs 2.5–6.5 at HIGH); the march wins in dense cloud
  (6.9 vs 10.9) and near genera (billboards 12–27 ms). Low flight: march 11.2 / 19.1 ms mean / p95 vs 12.2 / 21.4.
- The march costs a fixed ~2.2 ms (depth passes, full-res resolve, composite, near range) plus traced pixels
  (`inside`: 3.1 / 6.5 / 18.9 ms tracing 1/16, ¼, all).
- Lighting is not the lever (sun march 0.4–1.4 ms, billows 0.4–1.4, air ~0). Phase 2 stays for shadows, both
  families and the far segment; it moves after Phase 3, kept for lighting only if the genera's tall sun marches need it.
- Levers in order: reconstruction good enough to trace 1/16 of pixels a frame at HIGH (≈ billboards' cost); the fixed
  cost; the genera.

## Phase 1: temporal pipeline (committed 38364b9; **not closed**, see "Open issues")

Implemented in `cloud_march.frag`, `cloud_resolve.frag`, `march.py`:

- Jitter indexed per pixel visit (`frame / block²`, own `uFrame` per layer), not per frame.
- Resolve: every pixel blends the 4 nearest fresh samples (bilinear, times a Gaussian weight of the nearest sample,
  `REACH` 0.3 block) into its history; history reprojected in two steps by the pixel's own distance; clamped to
  ±1.25 σ of the 3×3 fresh samples; per-pixel sample count in the distance target's `g`; fresh share
  `near · max(1/(count+1), uMinWeight 0.05)`; the clamp's removal resets the count.
- Far off, the two finest octaves and the billows fade to their means by pixel footprint (`uPixelAngle / uBlock`).
- Slab segments take ≥ 8 samples (`SEGMENT_SAMPLES`); the near range gathers over frames too.
- Tried and dropped: clamp from 4 bilinear samples instead of 3×3 (2× the spread, no faster); halving `STEP_GROWTH`
  (+0.4–0.9 ms, long rays hit the budget).

Result (HIGH, same session): spread 2–5× lower, grid in motion 2–4× lower, same cost. At `march_resolution` ¼
(1/16 of pixels a frame): 1.5–3.8 ms, steadier than the old ½. Presets untouched pending a flight check.
Tests 819 green. Detail in performance.md §23 "Steadier" and clouds.md "The raymarch".

### Open issues (reported after viewing; second pass done, see below)

1. **A light jitter remains, even at `march_resolution` 0.25** (worst in far clouds, per the commit message).
2. **A grid pattern is still visible in the clouds** in a screenshot (`terrain_shots`-style still: far cumulus
   deck and the big cumulus at upper right; blocky/dotted texture at the block period).

Suspects to check, none verified yet:

- The Gaussian `near` weight is itself block-periodic: pixels on a sample take a fresh share ≈1/(count+1), pixels
  2–3 px away take ~0.25–0.06. Counts and converged noise then differ by position in the block, so a grid is baked
  into the history. The gather scores measured a bench pose, not a few-frames-old screenshot.
- A screenshot taken after few frames has low counts: the 1/16 reconstruction (4×4 bilinear) shows its raw block
  structure before the history fills in. Check how many frames `terrain_shots` renders and whether the history is
  warm (or settle N frames first).
- The grid score is absolute RMS over a pose; it may not capture a faint high-frequency pattern on the far deck.
  Add a metric on far-cloud crops (spectral peak at the block period, still and moving) before tuning.
- Jitter residual: far clouds trace 5–9 coarse steps per slab segment, so each ray lands on or misses thin cloud;
  the count cap (1/`uMinWeight` = 20 visits, i.e. 320 frames at 1/16) may be too low or the min weight too high for
  noise that averages slowly. Candidates: per-pixel R2/blue-noise offset on the jitter, lower `uMinWeight` for far
  pixels, density filtering by footprint before the step.

**Second pass (done, uncommitted; performance §23 "Steadier, second pass").** (1) The grid/streaks the user saw in the
demo's start view (`terrain_shots --pose ground_up`) were the **billows' value noise** (lattice planes shown by the
ridged billows): fixed by sampling it through a rotation (`BILLOW_TURN`, `cloud_march.frag`). The field's own octaves
share the lattice planes (fainter; also in the billboards): not touched. (2) In `cloud_resolve.frag`: the Gaussian
per-pixel weight was a lattice (removed); the clamp floor (0.04 + 15 % of the mean) only while the view is still
(`uStill`), since in motion it left clouds thin as the camera approached (measured: last moving frame vs converged);
count kept ≥ 4 after a clamp; `MIN_WEIGHT` 0.03. A bicubic history read was tried and dropped (biased to transparent).
RTX 3070, 2560×1440, ¼: still spread p99 −25–40 %, the 20 km grid 0.184 → 0.045, same cost; in flight the lag equals
the old pipeline's. Billows then reworked (demo pose lat 20.80828 lon 28.37786 alt 3267, `terrain_shots --at`): the ridged noise carved
sheets, so they are now two layers of the raw noise (`BILLOW_FINER_*`, `billow_gain`). The user confirmed the
transparency and the sparkle are fine; the remaining jitter is in BACKLOG.md.

Gate to close Phase 1: stability and grid clearly low on far cloud crops, still and moving, in `terrain_shots`
frames and in `cloud_bench.py --interactive`, at ¼, at no frame-time increase. Then the user flies it.

## Phase 2: one sun-transmittance map (lighting, shadows, both families)

**Done, with these changes to the plan** (shadows.md "The cloud shadow maps"; uncommitted):
- **4 levels in one RGBA16F target, not 8 in two**: a texel is (total, ¼, ½, ¾ of the chord). The ground still
  reads r. `LEVELS` could go to 8 with a second target (Panda MRT) if a thick convective slab needs it: only
  cumulus is marched until Phase 3.
- **The march casts a shadow** (`CloudMarch.shadow_layer/shadow_field/bind_shadow_inputs/sun_visibility`;
  `PlanetClouds.shadow_caster` returns it): the maps and the terrain shadow work for it. That part of Phase 3's
  field-object extraction was needed here; the rest (genera, shared object) stays in Phase 3.
- **The march reads the maps** (`PlanetClouds.bind_maps`, `CLOUD_MAP_LIGHTING` in `cloud_march.frag`): 2 local steps +
  the chord's total less the level at the local reach, whole chord (the old sun march stopped after its 3 steps).
- Measured on the RTX 3070, 2560×1440, `terrain_shots` (ground_up / in_clouds / clouds_noon), ms: the march's light
  from the maps 9.08 / 11.45 / 8.49 vs the sun march 8.99 / 11.40 / 8.57 (equal: a cumulus's sun march has 3 steps);
  the image differs by 0.4 of 255 on average. The shadows the march now casts cost 0.3-0.4 ms (the terrain's
  lookup): 9.15 / 11.84 / 8.61 vs 8.80 / 11.43 / 8.17 before. A swinging sun costs +1.2-1.3 ms a frame in all
  (the maps' re-bake and the other shadow maps; not split).
- Not done: the stratus adding the convective map's total (needs genera in the march, Phase 3); billboards'
  `cloudFarDepth` from the map (optional).
- Tests: the GPU levels against the CPU twin (`test_cloud_shadows.py`), map + local march against a fine march of the
  whole chord at 50° and 25° (median shade error 0, p90 0.03-0.05), the march as a caster
  (`test_clouds_march.py`).

Extend `CloudShadowMaps` / `cloud_shadow_map.frag`, keeping the charts (azimuthal equidistant on each family's base
sphere; near 60 m, far 500 m texels).

- A texel holds the **cumulative** optical depth from the base up to K = 8 heights along the sun ray (two RGBA16F
  targets, total last), from the same march with more steps. Ground and models read the total (`cloudMapsDepthIn`);
  a sample at height h reads total − C(h) at its ray's base entry (`cloudShadowChord`, analytic).
- Near detail stays local: the levels are coarse (125 m in cumulus, ~1.3 km in a 10.5 km convective slab), so ≈2
  steps of `sunStepLength` with the full field give close self-shadowing, the map everything sunward. ~4 fetches +
  2 map fetches per sample instead of 12–24. Beyond the far map or with maps off, fall back to `sunOpticalDepthAt`.
- One map per family; a stratus sample adds the convective map's total at its entry on the convective base.
- Bake only when the sun turns, the camera moves ¼ map, or settings change; measure a re-bake with a moving sun,
  slice over frames if too dear.
- CPU twin: `noise.cloud_shadow_on_sphere` samples give cumulative depth (`_cloud_shadow_samples`); `sun_visibility`
  keeps the total. Optional, measured: billboards' `cloudFarDepth` as a map lookup.
- Tests: GPU readback of the levels vs the CPU twin (pattern `tests/test_cloud_shadows.py::test_a_cloud_shadow_map_is_the_march_from_the_base`); NumPy check that map + local march ≈ full sun march (`tests/test_clouds_march.py`).

## Phase 3: the march with genera, and the march's shadows

**Done, with these changes to the plan** (clouds.md "Genera"; uncommitted):
- **No shared field object extracted.** `CloudMarch` holds a list of `_Family` (field, preset, weather, deck field) and
  a `_FamilyCaster` per family; `CloudVolume` is unchanged. Phase 2 already made the march a caster; the rest of the
  extraction would have moved `CloudVolume`'s state for no gain here.
- **Layers: `MAX_LAYERS` kept**, not an octave table: two families take `MAX_LAYERS // 2 - 1` = 3 bands each (HIGH asks
  for more), one layer per band and family and a far layer per family. Octaves per band stay per layer.
- **Segments** from both slabs sorted by start (`family_segments`, 4 at most); the family is the stretch's, not picked
  by height. The weather is read per family (`weatherMap`, `otherWeather`), the light table too; the stratus takes the
  convective maps' total as its shade from above (`cloudSunDepthOther`).
- **`PlanetClouds`**: renderer is free with genera (`stratus` is None with a march; `shadow_families` 2);
  `bind_maps` binds both families' maps.
- Measured (performance.md "The raymarch with genera"): 2-2.4 times cheaper than billboards under and between the
  towers (8.8 vs 21.4 ms, 7.6 vs 16.6), dearer over a distant sky (3.8 vs 1.4).
- Tests: `family_segments` (up to four sorted stretches), the march with genera in `test_clouds_genera.py` (two
  families, layer budget, both shadows, renderer change).
- Not done: the stratus in the sea's reflection (the deck is the convective family's), a cheaper far march (Phase 4).

- A field object both renderers share: move the renderer-independent state out of `CloudVolume` (`sphere.py`):
  field, packed layer params, weather and far-table textures, shadow inputs, `take_shadow_of`,
  `bind_shadow_inputs`, `sun_visibility`, `shadow_layer` / `shadow_field`. One per family; `CloudVolume` and
  `CloudMarch` draw from it (removes `isinstance(CloudMarch)` cases in `clouds/planet.py`).
- `CloudMarch` with genera: segments from both slabs' sphere intervals, merged and sorted (up to 4), clear gap
  skipped; family picked by height; field from `cloudDensityAt` with the weather's full `gba`, fetched once per
  `WEATHER_STRETCH`; two weather samplers; far coverage and far lighting per family.
- Layer budget: `MAX_LAYERS` = 8, `layerParams[104]` must hold both families without a bigger uniform array (weak
  GPUs). Move octaves-per-distance-band into a small uniform table: one layer per family plus the far one.
- `PlanetClouds._make_volume` stops forcing billboards with genera; `_bind_layer` binds the march's card nodes so it
  takes eclipse, mountain and ring shadows (`bind_inputs`).
- Tests: extend `tests/test_clouds_march.py` NumPy mirrors (`slab_segments` with two slabs, family by height); a
  march-with-genera case in `tests/test_clouds_genera.py` (both families shade the ground, rebind on renderer change).

## Phase 4: speed, then the march as default

Measure each lever against Phase 0, keep what pays, record in §23: march resolution/block per preset (¼ at HIGH if
flight confirms), billows as a cost toggle, near-range pass (merge or keep), and an **occupancy grid** experiment
(low-res conservative max of the field baked on the GPU; dropped if < ~20 % of steps are skipped, or it flickers).

Default switch: `CloudSettings.renderer = "march"` at every preset (a `kept` knob), billboards selectable.
Risk to show the user: the sea's reflection with the march shows only the deck; proposal is a cheap mirror march
(¼ res, no history) unless cost says otherwise. Docs: `clouds.md` "The raymarch" and "Genera", `shadows.md` "The
cloud shadow maps", `quality.md`, `CHANGELOG.md`.

## Critical files

`src/world_builder/clouds/march.py` · `clouds/shaders/cloud_march.frag`, `cloud_resolve.frag`,
`cloud_composite.frag` · `clouds/shadow_map.py`, `shaders/cloud_shadow_map.frag` · `clouds/shaders/cloud_field.glsl`
(`cloudShadowChord`, `cloudMapsDepthIn`, `cloudShadowIn`, `slabOpticalDepthCell`), `cloud_shading.glsl`
(`cloudRadianceAt`) · `clouds/sphere.py` · `clouds/planet.py` · `clouds/noise.py` (CPU twins) · `clouds/cloud.py`
(`CloudSettings`) · `scripts/cloud_bench.py` · tests `test_clouds_march.py`, `test_cloud_shadows.py`,
`test_clouds_genera.py`.

## Verification (each phase)

- `python -m pytest tests/ -n auto` green (GPU readback tests cover the shaders; `Shader.load` does not check GLSL).
- `cloud_bench.py` on the 1070 at 1600×900: frame mean / p95, stability and grid, still and `--move`, cumulus and
  `--genera` poses, both renderers; `flight_bench.py --flight low` for the world.
- Shots: `terrain_shots.py --cloud-renderer march` (`in_clouds`, `clouds_noon`, `clouds_low`, `sunset`, `cirrus_*`),
  with and without `--genera`; checked by Claude first, then confirmed by the user in
  `cloud_bench.py --interactive` and `demo_terrain.py --cloud-renderer march [--genera]`.
- Memory rules: never commit unless asked; stop after each phase for the user's visual check; shut the worker pool
  down in every bench path.
