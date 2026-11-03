# Last conversation, compacted

(The full 5650-line log is in git history: `git show HEAD:last_convo.md`. Plan: `MARCH_PLAN.md`.)

## 1. Rebase (done, pushed by the user)

Branch `11-cloud-and-mountain-shadows` (11 commits: eclipses, ring demo, object shadows, cloud shadows, terrain
shadow map, local maps, custom model shader, ring shadow + GPU shadow maps, ...) replayed onto `origin/main`
b825304 (which had reworked the clouds: `clouds/field.py` deleted, `cloud_shading.glsl`, cirrus back, genera/"cumulus
to cumulonimbus continuum"). Decisions: replay all 11 commits; both genera families cast shadows on ground/sea/
objects; cirrus only receives; check visually at the end. Ports made: `bind_shadows(shadows, body)` on
`PlanetClouds`, `clouds/shadow_defaults.py` replacing the flat-field defaults, the sun march generalised through
`sunOpticalDepthAt`, CPU twin `_march_density` in `noise.py`. Each rebased commit passes the suite (752 → 813 tests).
Known leftovers: `ring_ground` pose needs re-aiming (#42 moved the start); the stratus's far flat layer isn't darkened
by convective clouds above; cirrus casts no shadow. Tower/anvil lumpiness is expected with billboards.

## 2. Marched clouds: request and decisions

User: fix the march's jitter and point-grid pattern, unify stratus and convective, asked whether godrays come free
(no) and whether shadow maps are the same problem (yes: transmittance toward the sun). Decisions: march becomes the
default with billboards as fallback; one march, both families; no godrays; target GTX 1070 at 1600×900.
Plan written to `MARCH_PLAN.md` (the user asked for it in a root file, not plan mode).

## 3. Phase 0 (go for 0): measured, committed as 01c6376

`cloud_bench.py` got p95, `--stability`, an absolute grid score, steps/pixel readout, pool shutdown on every exit,
and a "no billows" debug bit. Findings (performance.md §23): billboards are cheaper over fair-weather cumulus, the
march wins in dense cloud and near genera; fixed march cost ≈ 2.2 ms + traced pixels; lighting is not the lever;
still flicker 1.2–3 mean (8.6 at 20 km), grid 0.13–0.29 in motion vs billboards 0.01–0.02.

## 4. Phase 1 (go for 1): committed as 38364b9

Changes: per-visit jitter; new `cloud_resolve.frag` (4-nearest bilinear reconstruction × Gaussian nearest-sample
weight `REACH` 0.3, two-step reprojection by own distance, ±1.25 σ clamp of the 3×3 fresh samples, per-pixel count in
`g` of the distance target, fresh share `near·max(1/(count+1), 0.05)`); fade of fine octaves/billows to their means
by pixel footprint; ≥ 8 samples per slab segment; near range gathers too. Tests 819 green, ruff clean.

Numbers (HIGH, same session, spread mean/p99 old → new ½ → new ¼): below 2.8/18 → 0.9/8 → 0.8/4; start 2.0/14 →
0.7/7 → 0.6/3; inside 1.3/15 → 0.5/5 → 0.4/3; above 2.8/15 → 0.9/6 → 0.8/4; 20 km 8.6/57 → 1.6/20 → 1.4/11.
Cost at ¼: 1.5–3.8 ms (≈ billboards). Failed experiments: 4-sample clamp, half `STEP_GROWTH`. A grid score that
was relative misled; it is absolute now. The sea's glints keep the "below" grid at 0.3–0.5.
At 1/16 the near range showed 4×4 blocks until it gathered too.

## 5. Where it stands (this message)

The user reviewed it: Phase 1 is **not** fully done. A light jitter remains even at `march_resolution` 0.25, and a
screenshot (far cumulus deck and a big cumulus, two test spheres, sea at the bottom) still shows a grid pattern in
the clouds. Open issues and suspects are in `MARCH_PLAN.md` ("Open issues"). Next step: reproduce on far-cloud crops,
find the cause (suspect the Gaussian per-pixel weight and low counts in a few-frames-old shot), fix, then the Phase 1
gate again before Phase 2.

## 6. Standing rules

Never commit unless asked; stop after each phase for the user's visual check; bench scripts must shut the worker pool
down; the user flies the result in `cloud_bench.py --interactive` and `demo_terrain.py --cloud-renderer march`.
