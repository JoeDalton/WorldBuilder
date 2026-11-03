# Marched clouds: stable, both genera families, lit from one sun-transmittance map

## Context

The raymarch (`CloudSettings.renderer = "march"`, `clouds/march.py`, `shaders/cloud_march.frag`,
`cloud_resolve.frag`, `cloud_composite.frag`) was meant to cut frame time when big or many clouds
fill the view. Billboards pay for the clouds' volume on screen as overdraw (§21–22 of
`docs/source/performance.md`: 60–150 ms near cumulonimbus at HIGH). Today the march has these problems:

- **Jitter and a dot grid.** Read from the code, there are three causes:
  1. Each pixel is traced every block²-th frame, but the golden-ratio jitter advances every frame. At a 2×2 block each visit jumps the jitter by 4·0.618 ≈ 0.47 of a step, so the samples never stratify.
  2. Untraced pixels are reprojected with their block's fresh distance, not their own, which smears edges.
  3. Pixels with no history copy their block's sample as is (2×2 blocks). Fresh pixels get a fixed ⅓ weight and no spatial filtering, so every frame 1 pixel in 4 differs from the history around it: the grid.
- **No genera.** It samples `cloudDensityShifted` (no convective depth, no shelf) through one slab, so `PlanetClouds` forces billboards whenever genera are on.
- **No shadows either way.** It casts no shadow on the ground, sea or models (`PlanetClouds._shade` falls back to defaults). It also takes no eclipse, mountain or ring shadow, because `PlanetClouds._bind_layer` skips `CloudMarch`.
- **Its cost is per-sample lighting.** Each cloudy sample runs the full sun march (`sunOpticalDepthAt`, 6–12 steps × 2 fetches), on top of 2–4 density fetches and the billows.

Your insight is right: the cloud shadow maps and the march's lighting are one problem, transmittance toward the sun at a point. Extending the existing maps (`clouds/shadow_map.py`) from "optical depth of the whole chord" to "cumulative optical depth at several heights" gives one GPU-baked map that lights the march's samples (a fetch instead of a sun march), shades the ground, sea and models, and covers both families.

**Decisions:**
- The march becomes the default renderer, with billboards as fallback. It is tuned for its own look, not billboard parity.
- One march covers both genera families: the bands never overlap, so the family is picked by height.
- No godrays.
- Target: GTX 1070 at 1600×900. p95 first, and the mean must not rise.

Phases stop for your demo check after each one (memory: stop-after-each-phase, visual-confirmation).

## Phase 0 findings (done: `docs/source/performance.md` §23)

Measured on the GTX 1070. They change the plan's priorities:

- **Over fair-weather cumulus the billboards are cheaper** (HIGH cloud cost 1.6–4.9 ms vs the march's 2.5–6.5). The march wins in a dense cloud (6.9 vs 10.9 ms) and near the genera, where the billboards cost 12–27 ms at HIGH. In the world it already wins on the main thread: low flight 11.2 / 19.1 vs 12.2 / 21.4 ms mean / p95, the billboards' tiles and sorts spared.
- **The march's cost is a fixed ~2.2 ms a frame** (depth passes, full-resolution resolve, composite, near range) **plus its traced pixels.** At `inside`: 3.1 / 6.5 / 18.9 ms tracing 1/16, ¼ and all of them per frame.
- **Lighting is not the lever here.** The per-sample sun march is 0.4–1.4 ms, the billows 0.4–1.4 ms, the air nothing measurable. So Phase 2's sun-transmittance map stays for its other reasons (shadows, both families, the far segment). It moves after Phase 3, and is kept only if the genera's tall sun marches make lighting matter.
- **Instability, quantified.** Still frames flicker (spread 1.2–3 mean, p99 8–20 levels; 8.6 / 57 at 20 km). The block grid is 0.04–0.13 still and **0.13–0.29 in motion**, against the billboards' 0.01–0.02.
- **The levers, in order:** Phase 1's reconstruction, good enough to trace 1/16 of the pixels a frame at HIGH (≈2.5–3.4 ms, the billboards' cost); then the fixed cost; then the genera (Phase 3).

## Phase 0: Baseline and measuring tools

- `scripts/cloud_bench.py` gains:
  - `--stability N`: per-pixel brightness spread over N still frames, mean / p99 over cloud pixels. This is the §20 metric, now in the script.
  - A **grid** score: the image's spectral energy at the block period (FFT at ½ and ¼ of Nyquist), still and with `--move`.
  - The existing `--cloud-debug` bits for breakdowns: 1 no sun march, 16 steps per pixel, plus a new bit for no billows.
- Shut the worker pool down in every bench path (memory: bench-scripts-pool-shutdown).
- Baseline on the 1070, one session, mean and p95: billboards vs march at MID/HIGH, the cumulus poses and the genera poses (billboards only today). Also `flight_bench.py --flight low` with each renderer. Written up as `docs/source/performance.md` §23.

## Phase 1: Temporal pipeline (the jitter and the grid)

In `cloud_march.frag`, `cloud_resolve.frag` and `march.py` (`_MarchLayer`, `bayer_offset`, `_FRESH_WEIGHT`):

1. **Jitter per visit, not per frame.** Index the sequence by `frame / block²`, with a per-pixel R2 or blue-noise offset, so each pixel's visits stratify its step.
2. **Reproject each pixel by its own last distance.** Reproject with the history distance texture (two steps: guess with the nearest fresh distance, then refine with the history's own). Fresh pixels use their own.
3. **Spatial reconstruction.** Untraced pixels with no history get a bilateral blend of the 4 nearest fresh samples, weighted by distance similarity, instead of a nearest copy.
4. **Variance clamp and a sample count.** Clamp the history to mean ± γσ of the 3×3 fresh samples (YCoCg). A per-pixel accumulated count (a spare channel of the distance target) makes the fresh weight `max(1/n, floor)` instead of a fixed ⅓. The average converges, and fresh pixels stop standing out.
5. Keep the composite's depth logic. Check the edge flicker it causes with `--move`.

**Gate:** the stability and grid scores clearly better than the baseline, still and moving, with no frame-time increase. Then you look at it in `cloud_bench.py --interactive` (M toggles the renderer).

## Phase 2: One sun-transmittance map (lighting, shadows, both families)

Extend `CloudShadowMaps` and `cloud_shadow_map.frag`. Keep the charts: an azimuthal equidistant chart on each family's base sphere, near 60 m and far 500 m texels.

- **What a texel holds.** Today it is the optical depth of the whole chord, from where the sun ray enters the base. Now it holds the **cumulative** optical depth from the base up to K heights along that ray (K = 8 in two RGBA16F targets, the total in the last). It is the same march, run once per texel with more steps, writing at the levels.
  - Ground and models below the base read the total, as today (`cloudMapsDepthIn`).
  - A sample at height h reads total − C(h), interpolated between levels, at its ray's base entry (`cloudShadowChord`, analytic).
- **Near detail stays local.** The levels are coarse: 125 m in a cumulus slab, about 1.3 km in a 10.5 km convective one. A short local sun march (≈2 steps of `sunStepLength`, the full field) gives the self-shadowing close to the sample, and the map gives everything sunward of it.
  - This mirrors the billboards' split into a near march plus the far segment.
  - Per sample: ~4 fetches plus 2 map fetches, instead of 12–24.
  - Beyond the far map, or with the maps off, the march falls back to today's `sunOpticalDepthAt`.
- **Both families.** One map per family, as the ground maps already are. A stratus sample adds the convective map's total at its ray's entry on the convective base, so a cumulonimbus shades the stratus under it.
- **Cost control.** Bake only when the sun turns, the camera moves a quarter of a map, or the settings change, as today. Measure the cost of a re-bake with a moving sun; if it is too high, re-bake slices over several frames.
- **CPU twin.** `noise.cloud_shadow_on_sphere`'s samples already give cumulative depth, so add the levels (`_cloud_shadow_samples`). `sun_visibility` keeps using the total.
- **Billboards (optional, measured).** `cloudFarDepth` per vertex could become a map lookup.

**Tests:** a GPU readback of the map's levels against the CPU twin (pattern: `tests/test_cloud_shadows.py::test_a_cloud_shadow_map_is_the_march_from_the_base`). A NumPy check that the map plus the local march matches the full sun march within a tolerance (`tests/test_clouds_march.py`).

## Phase 3: The march with genera, and the march's shadows

- **A field object both renderers share.** Move a volume's renderer-independent state out of `CloudVolume` (`sphere.py`):
  - the field, the packed layer params, the weather and far-table textures, the shadow inputs;
  - `take_shadow_of`, `bind_shadow_inputs`, `sun_visibility`, `shadow_layer` / `shadow_field`.

  Each family is one such object; `CloudVolume` and `CloudMarch` draw from it. That removes "a raymarched volume casts none yet" and the `isinstance(CloudMarch)` special cases in `clouds/planet.py`.
- **`CloudMarch` with genera.**
  - Segments come from both slabs' sphere intervals, merged and sorted (up to 4), so the clear gap between the bands is skipped analytically.
  - Per sample, the family is picked by height. The field comes from `cloudDensityAt` with the weather's full `gba` (depth, shelf), still fetched once per `WEATHER_STRETCH`.
  - The weather is two samplers: the convective 4-channel cube and the stratus one.
  - Far coverage and the averaged far lighting are taken per family.
- **Layer budget.** The march packs a layer per shell octave band (`MAX_LAYERS` = 8, `layerParams[104]`). Both families' bands must fit without raising the uniform array, which weak GPUs' fragment uniform limits would not take. So octaves per distance band move to a small uniform table, leaving one layer per family plus the far one.
- `PlanetClouds._make_volume` stops forcing billboards with genera.
- `_bind_layer` binds the march's card nodes, so the march takes the eclipse, mountain and ring shadows (`bind_inputs`).

**Tests:** extend `tests/test_clouds_march.py`'s NumPy mirrors (`slab_segments` with two slabs, family by height). Add a march-with-genera case to `tests/test_clouds_genera.py` (both families shade the ground, rebinding on a renderer change).

## Phase 4: Speed, then the march as the default

Measure each lever against Phase 0, keep what pays, and record it in §23:

- the march's resolution and block at each preset, now that reconstruction holds;
- billows as a cost toggle;
- the near-range pass (merge it or keep it);
- an **occupancy grid** around the camera, as an experiment with a gate: a low-resolution conservative max of the field, baked on the GPU, for empty-space skipping. §20's analytic ceiling never proved a cell clear; a sampled max with a slope margin at a finer cell might. It is dropped if it proves under ~20 % of the steps clear, or flickers.

**Default switch:**
- `CloudSettings.renderer` defaults to `"march"` at every preset (a `kept` knob), with billboards selectable.
- **Risk, to measure and show you:** the sea's reflection with the march shows only the deck. The proposal is a cheap mirror march (¼ resolution, no history; ripples hide the rest), unless its cost says otherwise.
- Docs: `clouds.md` "The raymarch" and "Genera", `shadows.md` "The cloud shadow maps", `quality.md`, `CHANGELOG.md`.

## Critical files

`src/world_builder/clouds/march.py` · `clouds/shaders/cloud_march.frag`, `cloud_resolve.frag`, `cloud_composite.frag` · `clouds/shadow_map.py`, `shaders/cloud_shadow_map.frag` · `clouds/shaders/cloud_field.glsl` (`cloudShadowChord`, `cloudMapsDepthIn`, `cloudShadowIn`, `slabOpticalDepthCell`), `cloud_shading.glsl` (`cloudRadianceAt`) · `clouds/sphere.py` (field state to extract) · `clouds/planet.py` · `clouds/noise.py` (CPU twins) · `clouds/cloud.py` (`CloudSettings`) · `scripts/cloud_bench.py` · tests `test_clouds_march.py`, `test_cloud_shadows.py`, `test_clouds_genera.py`.

## Verification (each phase)

- `python -m pytest tests/ -n auto` stays green. The GPU readback tests cover the shaders, since `Shader.load` succeeds without a context and does not check the GLSL.
- `cloud_bench.py` on the 1070 at 1600×900: frame mean and p95, stability and grid scores, still and `--move`, cumulus and `--genera` poses, both renderers. `flight_bench.py --flight low` for the full world.
- Shots: `terrain_shots.py --cloud-renderer march` (`in_clouds`, `clouds_noon`, `clouds_low`, `sunset`, `cirrus_*`), with and without `--genera`. I check them first, then you confirm in `cloud_bench.py --interactive` and `demo_terrain.py --cloud-renderer march [--genera]`.
- Nothing is committed (memory: never-commit).
