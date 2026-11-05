# Precomputed atmosphere tables (Scatterer's idea, Hillaire's method)

Not built. This plan replaces the per-pixel atmosphere march with precomputed tables. It is separate from the
EVE/Scatterer cloud salvage plan, which stays on the clouds.

## Context

The atmosphere (`atmosphere/shaders/atmosphere.glsl`) is single scattering, marched per pixel:

- `atmoSegment` takes up to `uAtmoSteps` samples per atmosphere along the view ray: 4 / 6 / 8 / 12 at LOW /
  MID / HIGH / ULTRA, at most 16 (`ATMO_MAX_STEPS`). Short sight lines take fewer.
- The optical depth toward the sun is the analytic Chapman function. The planet's shadow is found analytically
  along the ray, with eclipses by other bodies and rings' shadows.
- Multiply scattered light is a **flat share** (`uAtmoExtra.x` = 0.12) that fades through twilight. Its colour
  and fade are, in `atmosphere.md`'s words, "guesses" ("Left out: true multiple scattering (the Bruneton
  option)").

It runs per pixel in:

- `sky.frag` (`atmosphereAlong`);
- the terrain and the sea (`terrain_frag.glsl`: `applyAtmospheres` at the land and the near sea's early-out);
- the models (`model.frag`).

It runs per vertex in the stars. The clouds put **no air in front of themselves** (`atmosphere.md`, "Clouds at
the horizon"; enforced by `test_clouds_march.py`); they only take `cloudSunTint` and `cloudTwilight` (analytic
Chapman).

Measured costs (`performance_backlog.md`, GTX 1070, 1600 × 900, before the #20 rounds):

- the **sky** pass 0.5–1.4 ms (1.4 at the start pose, 0.5 in orbit);
- the **terrain's per-pixel atmosphere** ~2.0 ms;
- the **sea reflection's sky** 0.2–0.7 ms.

There are no numbers yet on the RTX 3070 at 2560 × 1440, the target.

**Scatterer**, EVE-Redux's sister mod, uses Bruneton & Neyret's precomputed tables (Proland's preprocessor:
transmittance 512 × 128, 4 scattering orders, a 4D in-scatter table packed in 3D). They are precomputed per
planet once, which suits a fixed atmosphere. WorldBuilder's atmospheres change at run time (settings,
`sunset_radius`, two planets), so the better fit is **Hillaire 2020**, "A Scalable and Production Ready Sky and
Atmosphere Rendering Technique" (EGSR). Its small tables are rebuilt in a fraction of a millisecond: a
transmittance table and a multiple-scattering table on parameter change, a sky-view table and an
aerial-perspective volume every frame. It gives multiple scattering of every order (twilight colour, a
deep-blue upper sky) at a constant per-pixel cost.

## Tables, per atmosphere (`MAX_ATMOSPHERES` = 2)

| Table | Size, format | Over | Rebuilt |
|---|---|---|---|
| Transmittance | 256 × 64, RGBA16F | radius r, view cosine μ | on parameter change |
| Multiple scattering | 32 × 32, RGBA16F | r, sun cosine μₛ | on parameter change |
| Sky view | 192 × 108, RGBA16F | direction around the camera (latitude warped toward the horizon) | every frame |
| Aerial perspective | 32 × 32 × 32, RGBA16F (in-scatter RGB, transmittance) | the view frustum, exponential depth slices | every frame |

- **Transmittance** takes `sunset_radius`'s curvature scale (`uAtmoExtra.y`) so it matches `atmoDepthToward`
  and the CPU's `depth_toward`. The analytic Chapman `atmoSunlight` may stay where it is cheaper than a fetch
  (models, terrain sunlight).
- **Multiple scattering** follows Hillaire's isotropic infinite-order approximation (a geometric series over
  the second order's integral on the sphere). It replaces the flat 0.12 share.
- **Sky view** is baked with the planet's shadow and the eclipses (`shadows.glsl`) per bake texel, and the
  rings' shadow as now (twice per segment): 20k texels instead of 3.7M pixels.
- **Aerial perspective** must span both split-camera ranges (`depth.SplitDepthCameras`): directions from the
  near lens, slices by **distance**, exponential out to the far range's reach. Beyond the last slice the
  readers fall back to the march. It is stored as an **atlas** (32 tiles of 32 × 32) because Panda 1.10 at
  GL 3.3 renders into a 3D texture only with a geometry shader and layered targets; it is read with a manual
  blend between two slices.

## Interface

Keep the three functions that `atmosphere.md` names as the plug points: `applyAtmospheres`, `atmosphereAlong`,
`atmoSunlight`.

- **`sky.frag`**: inside an atmosphere, the sky-view table (one fetch per atmosphere). From space (the camera
  above the top), a march with the transmittance and multiple-scattering tables, with fewer samples than now.
  The sea's reflection of the sky takes the same.
- **Terrain, sea, models** (`applyAtmospheres`): the aerial-perspective volume within its reach, the march
  beyond it, blended over the last slice.
- **Stars** (per vertex): transmittance only.
- **Clouds**: unchanged. They keep no air in front; `cloudSunTint` may read the transmittance table. Revisiting
  the "Clouds at the horizon" decision is a separate choice: the volume would make it one fetch per pixel.
- The live `PTA_int` step knobs (`AtmosphereSettings.steps`, `vertex_steps`) stay for the fallback march and
  for LOW.

## Panda3D

- **Bakes**: the GPU bake pattern of `terrain/shadow_bake.py` and `clouds/shadow_map.py`. That means
  `make_output` on the main GSG, sort before the cameras, a float FBO, `RTM_bind_or_copy`, and a fullscreen
  card with a fragment shader per table. Transmittance and multiple scattering are one-shot
  (`set_one_shot(True)`) on change; sky view and aerial perspective draw every frame.
- **Binding**: like `deckLight` (`clouds/sphere.py`: `set_shader_input` on the nodes), with 1 × 1 fallbacks
  (`clouds/shadow_defaults.py` pattern) so shaders compile where no atmosphere is bound.
- **Two atmospheres**: one table set each. The sky-view and aerial-perspective bakes loop over the atmospheres
  the camera sees (`uAtmoCount`).

## CPU twin and tests

- `transmittance_to_sun` and `depth_toward` (`atmosphere/__init__.py`) against a readback of the transmittance
  table: within 1 % over (r, μ), with `sunset_radius` ≠ 1.
- **New: a NumPy in-scatter reference** (single scattering plus the multiple-scattering approximation) for a
  set of view rays. The table path (sky view, aerial perspective) must be within tolerance of it by day.
  Twilight is *expected* to differ from today's flat share; compare it against the reference, not against the
  old image.
- `light_at` is unchanged: it does not use in-scatter.
- The existing tests stay green: `test_atmosphere_sky.py`, `test_scene.py`'s shared steps,
  `test_quality.py`'s settings, `test_world.py`'s `light_at`.

## Phases

0. **Measure first** on the RTX 3070 at 2560 × 1440, HIGH: the sky pass, the terrain's atmosphere (steps 8
   against 2), the sea reflection's sky, at the start, 20 km, dusk and orbit poses (`terrain_shots.py`,
   `flight_bench.py`). If the total is under ~1 ms, stop here and keep the plan for the look alone.
1. **Transmittance table** and its readback test; `cloudSunTint` and `atmoSunlight` may read it.
2. **Multiple-scattering table**, replacing the flat share in the existing march (the march stays). This is
   the look change; check dusk, the planet's shadow and the upper sky (`sunset`, `dusk_in_clouds`, orbit).
   User's check.
3. **Sky-view table** for `sky.frag` and the reflection's sky; the from-space path.
4. **Aerial-perspective volume** for the terrain, sea and models, with the fallback and the blend. Check the
   limb and terminator for banding, and the split-camera seam.
5. **Presets**: table sizes per preset; LOW may keep the march (`quality.md`).
6. **Docs**: `atmosphere.md` (replace "Left out: true multiple scattering"), `performance.md`, `CHANGELOG.md`.

## Gates (RTX 3070, 2560 × 1440, HIGH)

- The atmosphere's total, bakes included, ≤ 0.5 ms a frame.
- No banding at the limb or the terminator (`sunset`, orbit, `dusk_in_clouds`, a descent from orbit).
- The user's look check at dusk and from orbit.
- `python -m pytest tests/ -n auto` green.

## Risks

- **The aerial-perspective volume at planet scale**: a frustum reaching hundreds of kilometres from altitude.
  The slice distribution must keep the near terrain fine and still reach the horizon; the fallback march covers
  the rest.
- **Eclipses and rings in a per-frame bake**: their cost moves into the sky-view bake (cheaper per texel, but
  every frame).
- **Two atmospheres** (a moon's air seen through the planet's): two table sets, twice the bakes.
- **Look change**: true multiple scattering changes dusk and the planet's shadow (today's blue-grey fill). The
  user decides on the look in phase 2 before anything else depends on it.

## Critical files

`atmosphere/shaders/atmosphere.glsl` · `atmosphere/__init__.py` (`Atmospheres`, `AtmosphereSettings`, twin) ·
new `atmosphere/tables.py` and `atmosphere/shaders/atmo_*.frag` (bakes) · `sky/shaders/sky.frag` ·
`terrain/shaders/terrain_frag.glsl` · `objects/shaders/model.frag` · `ocean` (reflection's sky) ·
`tests/test_atmosphere_sky.py` · `docs/source/atmosphere.md`.
