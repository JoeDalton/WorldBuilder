# A light volume for the marched clouds (from EVE-Redux)

Not built. This plan follows the EVE/Scatterer salvage plan (cheap lighting route first: multiple-scattering
octaves and a zenith sky-occlusion map). Build it only if the towers' bands, the anvils or the low-sun stratus
still need it after that.

## Context

The march lights its clouds from the cloud shadow maps (`clouds/shadow_map.py`, `cloud_shadow_map.frag`). A
texel holds the cumulative optical depth along the sun's chord at 4 levels (total, ¼, ½, ¾), on azimuthal
equidistant charts (near 60 m, far 500 m texels) per family. A cloudy sample takes 2 local steps of the field,
then the levels' depth out to the sun march's reach (`cloudMapSunDepth`, `cloud_field.glsl`). BACKLOG.md lists
what that cannot do:

- **Faint horizontal bands in towers**: 4 levels over a 10.5 km convective chord (about 2.6 km apart), linear
  between them.
- **The map-lit depth counts out to the sun march's reach only**: the whole chord left towers black, and the
  levels are too coarse for anvils.
- **Low sun (about 4°)**: the stratus under the convective clouds is shaded with hard-looking edges
  (`cloudSunDepthOther`, the convective map's total).
- **Sky light from a height ramp** (`skyReach` in `cloud_shading.glsl`): a cloud's base is as dark under a
  thin cloud as under a tower.

EVE-Redux solves the same problem with a **light volume**. It is a 3D texture around the camera holding the
light that reaches each point of the cloud layers, for the sun and for the sky. It is filled a few slices a
frame and read by the march in one fetch. Scatterer also reads it for god-rays. This plan ports the idea to
WorldBuilder's charts, bakes and twins.

It is a look feature, not a speed one: the march's lighting is 0.57 ms today (performance.md, Phase 4.2).

## EVE's design (`Atmosphere/RaymarchedClouds/LightVolume/LightVolume.cs`, branch `raymarched-volumetrics`)

Only the C# is public; the shaders (`EVE/RaymarchCloud` passes 2 and 3, `ReprojectLightVolume`) ship compiled.
What the C# sets:

- **Size**: 256 × 256 horizontally, 32 slices vertically through [inner, outer] cloud radius. Half floats
  (`RHalf`), bilinear, clamped. The direct volume's 32 slices and the ambient volume's 32 are stacked in one 3D
  texture (one sampler less, for Macs).
- **Mapping**: a paraboloid projection from an origin on the camera's vertical. The origin's height is chosen
  so the paraboloid covers everything out to the horizon (`MoveLightVolumeIfNeeded`):
  - `d_h = sqrt(alt² − R²)`, the distance to the planet's horizon (alt is clamped ≥ R + 2.5 km);
  - `cos = d_h / alt`;
  - `d = d_h + sqrt(outer² − R²)`, to the cloud sphere beyond it;
  - origin at `alt − cos·d`, covered radius `sqrt(d² − (cos·d)²)`, optionally capped by `maxLightVolumeRadius`.
- **Fill**: 50 light-march steps a voxel (`stepCount`).
  - Direct: `32 / directLightTimeSlicing (8)` = 4 slices a frame, so a full refresh every 8 frames. Under time
    warp > 2× this goes faster, up to `timewarpRateMultiplier` (3).
  - Ambient: `32 / 32` = 1 slice a frame.
  - Each layer adds to the volume (`clearExistingVolume` on the first layer only).
- **Moving**: double-buffered.
  - When the new origin moves more than 3 % of the covered radius, the radius changes by more than 3 %, or the
    layers' radii change, a reprojection pass resamples the whole old volume into the new one. It is a compute
    shader, or a blit per slice without compute.
  - Otherwise the volume stays put. It is rotated with the slowest-rotating cloud layer ("good enough in most
    cases").
- **Readers**:
  - the march: `LIGHT_VOLUME_ON`, `lightVolume`, `paraboloidPosition`, `worldToParaboloid`, the inner and outer
    radii. `lightMarchSteps` (2) and `lightMarchDistance` (400 m) remain for the local self-shadowing;
  - Scatterer (`scattererDirectLightVolume`) for the atmosphere's god-rays.

## WorldBuilder design

### Mapping

Use the existing **azimuthal equidistant chart** around the camera (`TangentFrame`, `uMapChart`, as in
`CloudShadowMaps._Map.draw`), not EVE's paraboloid: every reader already has the chart's code, and the shadow
maps' tests duplicate it (`test_cloud_shadows.py`).

- Add a **radial warp** (texel radius ∝ sqrt of the distance, or a log) so texels are finer near the camera,
  as EVE's paraboloid is.
- Coverage out to the horizon by EVE's formula above, capped at a preset distance (the far map's reach today).
- **Vertical**: 32 shells between the lowest base and the highest top of the families present.
  - With genera the convective slab is about 10.5 km: about 330 m a slice. That is 8× finer than today's
    levels, but still coarser than a cumulus's 125 m octave. The local steps cover that.
  - If anvils need more, take 32 shells per family (two atlases), not 64 shared.

### Storage (GL 3.3, Panda3D 1.10)

Layered rendering into a 3D texture needs a geometry shader and `RTM_bind_layered`. Use a **2D atlas of
slices** instead:

- 8 × 4 tiles of 256², so 2048 × 1024 `RG16F` (r: direct optical depth toward the sun, g: optical depth toward
  the sky), 8 MB.
- A reader interpolates between two slices by hand: two bilinear fetches and a mix.
- **Optical depth, not transmittance**: the levels' tail is linear in depth for a reason (`cloudLevelDepth`: a
  line in transmittance blew up near a level that held almost all the depth).
- **Two atlases (A, B)** for the reprojection; the readers bind the current one.

### Bake

Follow the existing GPU bakes:

- `terrain/shadow_bake.py:160-200`: `make_output` on the main GSG, sort −300 (before the cameras, after the
  terrain bake at −500), float FBO, `RTM_bind_or_copy`, no colour clear.
- A fullscreen-card fragment pass, as `cloud_shadow_map.frag`, but **drawn over a few tiles a frame**. One card
  per tile, shown or hidden, or one card with a scissor (`DisplayRegion` per tile).

Fill:

- **Direct**: from the voxel's centre, march toward `sunDir` through both families' fields. Use the convective
  field and the stratus, as `cloudSunDepthOther` does today: the stratus under the towers is shaded by them.
  - Step lengths grow (50 steps, EVE's count) and stop at the clouds' own top (`cloudChordToTop`).
  - The field is the calibrated two-octave field the maps already march (`slabOpticalDepthCell`), not the
    billows.
- **Sky**: march straight up (the zenith). Optionally add 3 directions tilted 45° and average their
  transmittances; measure whether it shows.
- **Rate**: 4 direct slices and 1 sky slice a frame (EVE's).
  - When `utils.sun_moved` trips, direct slices go at 8 a frame until a full cycle is done.
  - Mixed old and new slices within a cycle: blend each slice by its age, or accept it if a swinging-sun test
    (`cloud_bench.py --move` with a turning sun) shows nothing.
- **First fill** (new pose, settings, weather): all slices in one frame, as the maps' one-shot bake does now.

### Moving with the camera

- Recentre when the camera's origin moves more than 3 % of the covered radius, or the radius changes more than
  3 % (altitude change). Draw a **reprojection pass** that samples the old atlas at each new voxel's position
  (old chart and warp), writes the new atlas, then swaps A and B. Voxels outside the old coverage take the
  march (a fresh slice) or the far map's level as a first guess.
- The world does not rotate under the clouds here (no wind, `noiseOffset` is 0), so EVE's per-layer rotation is
  not needed. Add it if wind comes.

### Readers

- **The march** (`cloudSunDepthIn`, `cloud_field.glsl`, with `CLOUD_MAP_LIGHTING`): keep 2 local steps of the
  full field for close self-shadowing, then add the volume's direct depth at the local reach's end point. This
  replaces `cloudMapSunDepth`'s levels and its reach window: the whole chord counts, as it should once the
  levels are fine.
  - Outside the volume: `sunOpticalDepthAt` (today's fallback), blended over the edge.
  - A debug bit (as `--cloud-debug 64`) switches between volume and maps to compare.
- **Sky reach** (`cloud_shading.glsl`): `skyReach = mix(skyOcclusion, 1, exp2(−k · skyDepth))` from the g
  channel. It replaces the salvage plan's zenith map (Phase 4), which then goes.
- **The ground, sea and models** keep the existing maps' `.r` (`planetCloudShadow`): they are right already and
  cover the far map's reach. The volume's lowest slice could serve them later.
- **The billboards** are unchanged (they read neither).

### CPU side

- `lights_at` and `sun_visibility` keep the maps and the CPU march (`noise.cloud_shadow_on_sphere`): the volume
  is GPU-only.
- **Twin for the tests**: a NumPy fine march at a voxel's centre, along the sun and the zenith
  (`noise.cloud_shadow_on_sphere` accepts per-point directions).

## Phases

1. **The bake, static.** Atlas, chart and warp, the direct channel, a full fill in one frame. Readback test
   against the NumPy fine march at a set of voxels (median and p90 error, as `test_cloud_shadows.py:510`).
2. **The march reads it.** `cloudSunDepthIn` with the volume, the debug switch, the edge blend. Shots:
   `--genera` towers (bands), the anvil and the low-sun pose from BACKLOG
   (`terrain_shots.py --pose at --at 24.48163 33.77925 5019 -49.1 8.3 --sun 270 4 --genera`). User's check.
3. **Recentring.** Reprojection pass, A/B swap, coverage by EVE's horizon formula. Flight test for popping
   (`flight_bench.py --flight low`, descent from orbit).
4. **Time slicing.** 4 + 1 slices a frame, the sun-move policy, the age blend if needed. Measure a swinging sun.
5. **The sky channel.** Zenith march (plus tilted, if it shows), `skyReach` from it. Remove the zenith map.
6. **Presets.** Atlas size and slices per preset (off at LOW: the maps stay its lighting), `quality.md`.
7. **Docs.** `shadows.md` ("The cloud light volume"), `clouds.md`, `performance.md`, `CHANGELOG.md`; close the
   BACKLOG items it fixes.

## Gates (RTX 3070, 2560 × 1440, HIGH)

- Bake ≤ 0.4 ms a frame steady, ≤ 1.5 ms on a recentre or first fill.
- The march's cost not up (one fetch pair replaces two map fetches and the levels' arithmetic).
- No bands in `--genera` towers. The anvil and the low-sun stratus look right to the user.
- No popping when the volume recentres in flight; no visible lag of the light behind a swinging sun.
- `python -m pytest tests/ -n auto` green.

## Risks

- **Partial updates of a render target in Panda**: confirm that a buffer with no clear keeps the tiles not
  drawn this frame (the terrain bake's bands suggest it does).
- **The convective slab's vertical resolution**: 330 m a slice may still show steps in an anvil's shelf. Fall
  back to per-family atlases.
- **Mixed slices during a sun swing**: blend by age.
- **Memory**: two 8 MB atlases (16 MB) at HIGH; ULTRA at 512² tiles would be 64 MB.

## Critical files

`clouds/shadow_map.py` (pattern, chart) · new `clouds/light_volume.py` · new
`clouds/shaders/cloud_light_volume.frag` · `clouds/shaders/cloud_field.glsl` (`cloudSunDepthIn`) ·
`clouds/shaders/cloud_shading.glsl` (`skyReach`) · `clouds/march.py` (binding) · `world.py` (update order) ·
`clouds/noise.py` (twin) · `tests/test_cloud_shadows.py` · `scripts/cloud_bench.py`.
