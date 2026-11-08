# Rendering plan: the reference profile, the atmosphere's tables, the clouds' light volume and god rays

Replaces ATMOSPHERE_LUT_PLAN.md and LIGHT_VOLUME_PLAN.md (both written after the EVE-Redux / Scatterer
salvage, performance.md §24), with the user's decisions of 2026-11-07. Order: **A**, then **B**, then **C**.

Rules: never commit unless asked; stop for the user's visual check after each part (and after B.2, the look
change); numbers in docs/source/performance.md and the profile file; CHANGELOG, clouds.md, atmosphere.md,
shadows.md as things close. Machine: RTX 3070, 2560 x 1440, HIGH, GPU at P0
(`nvidia-smi --query-gpu=pstate,clocks.gr --format=csv`); VS Code runs (Claude runs in it), nothing else. Two
runs at least of anything compared; trees compared alternated in one session (a scratch worktree for the old one).
**Scope: WorldBuilder only.** SpaceFlight's frame (its draw, its bots) is out of scope.

## Decisions taken

- Pre-baking is **change-driven**: a bake redone when what it depends on changed (the sun's turn past a
  tolerance, the camera's recentring, settings), not a "static solar system" mode, so a moving sun still works.
  A sun turning by a day's rate (0.25 degrees a minute) re-bakes the cloud bakes every ~12 s, sliced over frames.
- **Rotating a bake instead of re-baking it:** exact for the atmosphere's sky-view table (built in the sun's
  azimuth frame, it depends on the sun's elevation and the camera's altitude only, B.3); not for the clouds'
  shadows or light: the clouds are fixed on the planet, so a sun turning relative to them changes the line its
  light takes through them, which changes the shadows, not their place (EVE rotates its light volume with a
  rotating cloud layer, an approximation it states). Not done for the clouds.
- **An alternate march-lighting path from the light volume**, benchmarked against today's (C.3).
- **No baked density volume** for the clouds (the field is static, but at the march's detail it is ~200M voxels
  around the camera; a clipmap is a project of its own, the payoff unsure).
- **The clouds' second order of scattering** (0.7-0.85 ms with genera, A) is kept for now and folded into the
  light volume (C.3): the march reads one stored value for both orders instead of computing the octave per
  sample.
- The atmosphere's tables are done even though the GPU is not the only limit (B); the light volume next, with
  **god rays, toggleable** (C.5).

## A. The reference profile (WorldBuilder only)

Goal: where a frame's time goes now, GPU and CPU, per part, in WorldBuilder's own scenes, so B and C are
measured against it and the biggest culprits are known. Output: **docs/source/frame_profile.md** (a new file,
linked from docs/source/index.md and performance.md).

- Tools: `scripts/flight_bench.py --offscreen --flight low --trace` (CPU parts, slow frames), `terrain_shots.py`
  poses timed with each layer hidden in turn (the performance_backlog.md method: a layer's cost is the median
  frame with it shown less with it hidden, GPU-synced), `cloud_bench.py --passes`, PStats (`want-pstats`,
  `text-stats`) for draw/cull, and the frame at 2560 x 1440 against 1280 x 720 (GPU-bound or not).
- Poses: the demo's start (1.3 km), low flight over terrain, 20 km, orbit, sunset, in clouds, with genera, over
  the sea (its reflection), near the test balls (models), a ring.
- Per pose: frame mean / p95; GPU parts (terrain, sea and its reflection, sky, atmosphere's share via
  `AtmosphereSettings.steps` 2 against 8 and the ground's, clouds' march and passes, cirrus, shadows' bakes,
  models, asteroids if a level has them); CPU parts (cull, draw, `World.update`, terrain selection and uploads,
  clouds' Python).
- Settle the open question of §24: the second order of scattering's cost (compiled out against 0.3, idle-ish).
- Close: the ranked culprits, each with its measured cost and the plan item or backlog entry that addresses it.

## B. The atmosphere's tables (Hillaire 2020)

Replace the per-pixel single-scattering march (`atmosphere.glsl` `atmoSegment`, 4 / 6 / 8 / 12 samples at
LOW / MID / HIGH / ULTRA) and its flat multiple-scattering share (`uAtmoExtra.x` 0.12, "guesses" in
atmosphere.md) with tables. Hillaire's, not Scatterer's Bruneton tables: they rebuild in a fraction of a
millisecond when the atmosphere changes (settings, `sunset_radius`, two planets).

Tables per atmosphere (`MAX_ATMOSPHERES` 2):

| Table | Size, format | Over | Rebuilt |
|---|---|---|---|
| Transmittance | 256 x 64 RGBA16F | radius r, view cosine mu | when the atmosphere's parameters change |
| Multiple scattering | 32 x 32 RGBA16F | r, sun cosine mu_s | when they change |
| Sky view | 192 x 108 RGBA16F | directions around the camera, latitude warped toward the horizon, **in the sun's azimuth frame** | when the camera's altitude or the sun's elevation changes past a tolerance (a sun moving in azimuth only is a rotation of it) |
| Aerial perspective | 32 x 32 x 32 (an atlas of slices, GL 3.3), in-scatter RGB + transmittance | the view frustum, exponential distance slices over both split-camera ranges | every frame (camera-dependent) |

- Transmittance takes the `sunset_radius` curvature (`uAtmoExtra.y`), matching `atmoDepthToward` and the CPU's
  `depth_toward`; the analytic Chapman `atmoSunlight` stays where it is cheaper than a fetch.
- Multiple scattering: Hillaire's isotropic infinite-order approximation.
- Sky view and aerial perspective take the planet's shadow analytically, the other bodies' eclipses
  (`shadows.glsl`) and the rings per bake texel instead of per pixel.
- Interface kept (atmosphere.md's plug points): `applyAtmospheres`, `atmosphereAlong`, `atmoSunlight`.
  `sky.frag` and the sea reflection's sky read the sky view inside the air, a march with the transmittance and
  multiple-scattering tables from space; the terrain, sea and models read the aerial perspective within its reach
  and march beyond it, blended over the last slice; the stars take transmittance only. The clouds still put no air
  in front of themselves (atmosphere.md "Clouds at the horizon", `test_clouds_march.py`).
- Panda: the GPU bake pattern of `terrain/shadow_bake.py` / `clouds/shadow_map.py` (`make_output`, float FBO,
  a fragment pass), bound like `deckLight` with 1 x 1 fallbacks (`shadow_defaults.py` pattern); the live
  `PTA_int` step knobs stay for the fallback march and LOW.
- CPU twin: `transmittance_to_sun` / `depth_toward` against a transmittance readback (1 %); a NumPy in-scatter
  reference (single plus the multiple-scattering approximation) for daytime rays, the table path within tolerance;
  twilight compared against the reference, not the old flat share. Existing tests stay green.

Steps: **B.0** the atmosphere's share from A · **B.1** transmittance table + readback test · **B.2**
multiple-scattering table in place of the flat share (the look change: dusk, the planet's shadow, the upper sky;
user's check) · **B.3** sky view for `sky.frag` and the reflection's sky, re-baked on change, the from-space path ·
**B.4** aerial perspective for the terrain, sea and models (limb and terminator banding, the split seam) · **B.5**
presets (sizes; LOW may keep the march), docs.

Gates: the atmosphere's total, bakes included, <= 0.5 ms at HIGH, and below A's measured share; no banding
(`sunset`, orbit, `dusk_in_clouds`, a descent from orbit); the user's dusk and orbit check.

**Done (2026-11-07), B.1-B.5.** The per-pixel atmosphere's share is ~0 wherever the camera is in the air
(0.4-1.2 ms in A); frames 0.3-0.9 ms faster at HIGH (performance.md §25). B.5: both tables on at every preset
(they save at LOW's 4 samples too); no banding in the gate poses nor across the air's top (61 / 59 km). Left:
from orbit the march stays, +0.24 ms on the cheapest frame (3.6 ms).

## C. The clouds' light volume, and god rays

The march lights its clouds from the cloud shadow maps: 4 cumulative levels along the sun's chord per texel plus 2
local steps (`cloudMapSunDepth`). BACKLOG.md: bands in towers (levels 2.6 km apart in a 10.5 km slab), the map
depth counts only to the sun march's reach, anvils too coarse, the low-sun stratus's hard edges, sky light from a
height ramp. EVE-Redux's light volume (`Atmosphere/RaymarchedClouds/LightVolume/LightVolume.cs`): 256 x 256 x 32
slices through the cloud shells, R16F, direct and ambient stacked, a paraboloid mapping covering out to the horizon
(`MoveLightVolumeIfNeeded`), a 50-step sun march per voxel, 4 direct and 1 ambient slice a frame, double-buffered
and reprojected when the camera moves > 3 % of its radius; read by the march (one fetch, plus 2 local steps) and by
Scatterer for god rays.

- **C.1 Mapping and storage.** The cloud shadow maps' azimuthal-equidistant chart around the camera (readers have
  it) with a radial warp (finer near), covering to the horizon by EVE's formula, capped; 32 shells between the
  lowest base and highest top (per family if the convective slab's 330 m slices show steps). A 2D atlas of slices
  (8 x 4 tiles of 256^2, RG16F: direct optical depth, sky optical depth; read with a manual blend of two slices;
  depth not transmittance, as `cloudLevelDepth` learned); two atlases for the recentring.
- **C.2 Change-driven bake.** The pattern of `terrain/shadow_bake.py` (no clear, a few tiles a frame): a full fill
  on a new pose, settings or weather; then only when the sun turns past `utils.SUN_TOLERANCE` (all slices
  re-baked over the next frames, 4-8 a frame) or the camera moves past 3 % of the radius (a reprojection pass, A
  to B, then the slices refreshed). With a still sun and camera: nothing a frame. Direct: a march toward the sun,
  steps growing, through both families (the stratus shaded by the towers), the calibrated two-octave field. Sky:
  a march up (plus 3 tilted directions if it shows).
- **C.3 Readers, two paths benchmarked.** (a) the volume in place of the maps' levels in `cloudSunDepthIn` (2
  local steps + the volume's depth at the local reach's end, the whole chord; `sunOpticalDepthAt` beyond it);
  the second order of scattering taken from the volume too (the bake stores the octave's depth scaling, the
march's per-sample octave goes: A measured it at 0.7-0.85 ms with genera); (b) **the alternate path**: no local steps, the volume alone (one fetch pair a sample) where the camera is far
  enough that its voxels resolve the clouds; both against today's at HIGH (`--passes`, `--jitter`, genera), kept
  per distance band if (b) pays. Sky reach from the sky channel in place of the height ramp. Ground, sea and
  models keep the maps' `.r`. A debug bit to compare with the maps.
  **Status (2026-11-07):** C.1-C.3 built. (a) The march's sun depth from the volume (2 local steps, then the
  difference of two volume reads over the sun march's window: the whole chord, debug 1024, blackens towers);
  each voxel marched with the weather per step (taken once at the chord's start, voxels on one sun ray
  disagreed: streaked towers); the reads jittered by up to half a texel and slice (the texels' grid showed).
  Cost: 0 with all still, 0.1-0.45 ms re-baking; as the maps' lighting within noise, +0.1-0.2 ms with the sky
  read. Sky reach from the volume's up depth (`mix(skyOcclusion, 1, exp2(-depth))`) in place of the ramp
  (debug 512: maps and ramp, the old look). (b) The volume alone past a view footprint of 0.04
  (`march.LIGHT_LOCAL_FOOTPRINT`): 0-0.2 ms with genera, nothing plain; a little softer far towers. The second
  order: compiled out against in, alternated, now ~0 (A's 0.7-0.85 ms gone): not folded, nothing to gain.
- **C.4 Presets and docs.** Done (2026-11-07): 128 MID, 256 HIGH/ULTRA, off at LOW;
  `cloud_light_slices` 2/1/2 (re-baking 0.3-0.4 ms at 256 and one slice); shadows.md, clouds.md,
  performance.md §26, BACKLOG. Atlas size and slices per preset (off at LOW: the maps stay its light);
  shadows.md "The cloud light volume", clouds.md, CHANGELOG; BACKLOG items it closes.
- **C.5 God rays, toggleable** (`AtmosphereSettings.god_rays`, off by default at LOW/MID). The aerial-perspective
  bake (B.4) marches each frustum cell's ray through the air: its in-scatter takes the clouds' shadow at each
  sample: the cloud shadow maps' `.r` below the base (the sun ray's entry), the light volume's direct depth inside
  the slab, unshadowed above. **The mountains' shadow too** (the user's request, 2026-11-07): at each sample
  `shadows.glsl`'s `terrainShadow(planetDir, altitude)`, which answers for any point, not only the ground (its
  altitude against the ceiling the mountains set toward the sun there: the global cube map everywhere, the local
  map's finer one near the camera), so valleys at a low sun take shafts from the ridges as well as the clouds;
  the bake's root binds the body's `shadow_map` inputs as the terrain does. A test: a cell behind a ridge (a
  ceiling above it) darker than one beside it, as `shadow_map.terrain_visibility` on the CPU says. The 32 x 32 x 32
  volume is too coarse for crisp shafts: with god rays on, a finer one (e.g. 160 x 90 x 64) or a separate
  half-resolution in-scatter pass with a temporal blend (Scatterer's `historyGodrayOcclusionBuffer` way); measured.
  The sky's own path takes the same shadowed in-scatter where the view ray passes under the clouds.
  **Built (2026-11-07):** both bakes (aerial cells, sky-view table) take `god_rays.glsl`'s shadows at each
  sample's direct sun (the mountains' `terrainShadow`, both families' clouds from the light volume, the maps
  beyond); 0.09-0.17 ms, 0.31 with genera. The finer grid not built: shafts softly banded, no blocks at
  1440p; to revisit after the user's look.

CPU: `lights_at` / `sun_visibility` keep the maps and the CPU march. Tests: slices read back against a fine
NumPy march (`noise.cloud_shadow_on_sphere` along the sun and the zenith, `test_cloud_shadows.py:817` pattern);
recentring continuity; god rays: a cell under a cloud's shadow darker than a lit one, one behind a ridge darker
than one beside it, the toggle off equal to B.

Gates (RTX 3070, 2560 x 1440, HIGH): bake <= 0.4 ms a frame while re-baking, ~0 with sun and camera still,
<= 1.5 ms on a recentre or first fill; the march's cost not up with (a), down with (b) where kept; no tower bands in
`--genera` shots; the anvil and low-sun poses from BACKLOG fixed; no popping on recentre in flight; god rays
<= 0.5 ms on, 0 off.

Risks: Panda partial updates of a no-clear target (the terrain bake's bands suggest it works); the convective
slab's vertical resolution; mixed slices during a sun swing (blend by age); the aerial-perspective volume's slices
at planet scale; two atmospheres; memory (two 8 MB atlases at HIGH).

## After C: look items logged on the way

- **The cumulonimbus's underside looks too bright** (logged 2026-11-07, after B.3, the user's flight check;
  addressed 2026-11-08: the fill and the second order fade with the column above, clouds.md "Lighting"):
  to look at once the light volume (C) lights the towers; candidates: the sky fill from below (the ground's
  bounce, B.2's brighter low sky), the second order of scattering in the base, the height ramp of the sky
  reach (`skyReach`) that C.3 replaces.
- **Grain on grey clouds in motion** (logged 2026-11-08, after the base's fade, the user's flight check; before
  the bands below): darker cloud shows the march's sampling noise (performance.md §28). Lead: the sun term's
  noise at its source (more local sun steps, a smoother volume read).
- **Stacked horizontal bands on a tower seen level** (logged 2026-11-07, during C.3, the user's flight check;
  pose `--genera --at 22.46569 30.31014 2831 0 -1`). Cause: a level ray crosses tens of kilometres of the
  slab's clear air before the tower and spends ~97 % of its step budget there (`--cloud-debug 128`: green
  6-10/255, every pixel over budget), so the tower is drawn from 2-4 samples, whose fall on its smooth face
  bands. ULTRA's 96 steps make them faint; the light volume plays no part (the sun march alone shows them).
  Done meanwhile: the fine octaves' fade by the step's length from its distance alone (not the deepened, cut
  step), the steps' boundaries moved per pixel and visit (each segment's first step a random share): both
  weaken the bands a little. Tried and reverted: blind empty-space skipping (steps doubling in clear air to 4x,
  back on entering cloud): in-cloud steps 3 % -> 12-16 %, bands weaker, but the tower see-through, its edge
  speckled (thin cloud jumped), vertical smears. Lead: skip only where the weather says the column cannot
  hold cloud (`cloudShadowMarch`'s "most the octaves could reach" test), so clear sky is crossed fast and thin
  edges are not jumped (clouds.md "No longer steps in clear air": blind skipping failed the same way before, and
  the fbm admits no bound that proves a cell clear; the weather's column test is coarser, so it may); measure
  with `cloud_bench --stability/--jitter` and the frame's cost.

## Critical files

A: `scripts/flight_bench.py`, `terrain_shots.py`, `cloud_bench.py`, new `docs/source/frame_profile.md`.
B: `atmosphere/shaders/atmosphere.glsl`, `atmosphere/__init__.py`, new `atmosphere/tables.py` and bake shaders,
`sky/shaders/sky.frag`, `terrain/shaders/terrain_frag.glsl`, `objects/shaders/model.frag`, the ocean's reflection,
`tests/test_atmosphere_sky.py`, `docs/source/atmosphere.md`.
C: `clouds/shadow_map.py` (pattern, chart), new `clouds/light_volume.py` and its shader, `cloud_field.glsl`
(`cloudSunDepthIn`), `cloud_shading.glsl` (`skyReach`), `clouds/march.py`, `world.py` (update order), the
aerial-perspective bake (god rays), `clouds/noise.py` (twin), `tests/test_cloud_shadows.py`.
