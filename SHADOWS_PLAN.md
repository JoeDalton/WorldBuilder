# World shadows: eclipses, terrain, clouds and objects (issues #11, #13)

## Context

Today the only shadow in the world is the planet's own: `atmoSunlight` per pixel
(terrain, clouds, sky). Mountains, clouds, moons and rings cast nothing, and an
airless moon stays lit inside the planet's shadow. Models get one Panda3D
`DirectionalLight` evaluated **at the camera**. Issue #11 asks for cloud and
mountain shadows on the ground and the sea. Issue #13 asks for eclipses, for game
objects that respond to world shadows, and for god rays. Bloom and lens flare
stay in #13, outside this plan.

Design, agreed in discussion:

- **One sun-visibility product.** Every shader, and a CPU twin, multiplies its
  sunlight by independent terms: eclipses (closed form), rings (closed form),
  cloud shadow (read from the cloud density field), and terrain shadow
  ("shadow ceiling" maps).
- **No Panda3D shadow mapping for world shadows.** It offers one map, no
  cascades, and the custom shaders would ignore it. A sun-space depth map is used
  only for shadows that **objects cast**.
- **Sun size.** The game's sun is a direction plus an angular radius, so the drawn
  disc *is* its physical size. It becomes one value, `SunLook.angular_radius`
  (today `sky.SUN_ANGULAR_RADIUS` = 0.55°), read by both the sky disc and every
  penumbra.
- **Static sun for now.** The terrain-shadow bakes depend on the sun. They are
  rebuilt in the background on `set_sun`, which also gives a slow sun. A fast sun
  would later need horizon maps, behind the same receiver functions.
- **Moving bodies are a separate issue.** The eclipse term reads the current body
  positions every frame, so a shadow moves with any motion. The demo checks
  motion by swinging the sun (J/K).

User choices:
- Demo: the moon moves to about 21,000 km, so it looks as wide as the sun.
- Near terrain detail: a local ceiling map around the camera, not a per-vertex
  bake.
- A generic model shader is part of this plan.
- Terrain drawn into the caster map ("hero" shadows) is an optional last part.

## Conventions for every part (each one a sub-issue for a fresh session)

- **Leave everything uncommitted.** The user commits and branches.
- **Visual confirmation.** Render PNGs with `scripts/terrain_shots.py`, inspect
  them, and ask the user to confirm the look before closing the part. Use
  `scripts/compare_shots.py BEFORE AFTER --diff` to show that unrelated views did
  not change.
- **CPU twin for every GLSL term.**
  - CPU tests: limits, monotonicity, and agreement with a brute-force reference.
  - One GPU read-back test per new GLSL function, following
    `test_the_shader_swell_is_the_cpus` (`tests/test_scene.py:372-482`): paste
    the include, write one point per pixel, compare with the numpy twin.
- **Inputs bound once, rewritten in place.** Use PTA arrays, as `Atmospheres`
  does (`atmosphere/__init__.py:150-232`). Every owner calls `bind_defaults`
  before `bind`, so every shader links without the feature.
- **Wire both drivers.** `World` (`world.py`) and the hand-wired `TerrainDemo`
  (`scripts/demo_terrain.py`), which `terrain_shots.py` uses.
- **Cost.** Report GPU and frame ms before and after on the existing poses and
  `scripts/flight_bench.py`. Benches shut the worker pool down.
- **Quality knobs.** A `ShadowSettings` section (`quality.py` pattern, registered
  in `settings.py` `SECTIONS`) is created by the first part that needs a knob.
- **Docs.** Each part updates a new `docs/source/shadows.md` (linked from
  `index.md`) and the pages it touches (`atmosphere.md` "Not yet", `clouds.md`,
  `object_lighting.md`, `world.md`), plus the `CHANGELOG.md` entry.

## Order and dependencies

| # | Part | Needs | Issue |
|---|---|---|---|
| A | Shadow foundation and eclipses | none | #13 |
| B | Lighting objects per node (CPU) | A | #13 |
| C | Cloud shadows on ground and sea | A | #11 |
| D | Terrain shadows: global ceiling map | A (C for the near sea) | #11 |
| E | Terrain shadows: local ceiling map, and the cloud shadow map | D, C | #11 |
| F | Long-range cloud-on-cloud shadows | C (its slab-march helper) | #11 |
| G | Model shader with per-pixel world shadows | A, B (C, D, E for full effect) | #13 |
| H | Objects cast shadows (caster map) | B, G | #13 |
| I | Ring shadows | A, and #14 P7 (rings) | #13 |
| J | God rays | D, E (C) | #13 |
| K | Optional: terrain in the caster map | H | #13 |

C, D and F can run in any order after A. Recommended order: A, B, C, D, E, F, G,
H, then I when #14 P7 lands, then J, then K only if wanted.

---

## A. Shadow foundation and eclipses

**Goal.** A shared `Shadows` uniform block and a `shadows.glsl` include, with the
first term: eclipses by spherical bodies, with penumbra, on every receiver. One
sun size everywhere. Demo eclipses.

- **New package `src/world_builder/shadows/`.**
  - `__init__.py` holds `Shadows`, a copy of the `Atmospheres` pattern. It keeps
    shared PTA arrays:
    - `uShadowBodies[MAX_SHADOW_BODIES]`, one vec4 per body: its centre minus
      the camera in the render frame, computed in float64, and its radius;
    - `uShadowCount`;
    - `uSunAngularRadius`;
    - `uAtmoBody[MAX_ATMOSPHERES]`, the body that owns each atmosphere, so the
      atmosphere excludes its own planet.
  - Its methods are `add(radius, position)`, `bind(node)`, `bind_defaults(node)`
    and `update(camera_world)`.
  - Each receiver node also gets `uShadowSelf`, its own body index, because a
    body never eclipses itself: `atmoSunlight` and `day` already handle that.
- **`shadows/shaders/shadows.glsl`.**
  - `float discCoverage(float sunR, float occR, float sep)`: the covered share of
    a uniform disc, from the circle-overlap area.
  - `float eclipseVisibility(vec3 rel, vec3 L, int self)`, where `rel` is the
    point minus the camera, in the render frame. It loops over the bodies and
    returns the product of `1 - coverage`.
  - Later parts add `sunShadow(...)` as the product of all terms.
- **CPU twin**: `disc_coverage` and `Shadows.eclipse(points_world, exclude)`,
  vectorised.
- **Receivers.** Multiply the sunlight by the eclipse term:
  - Terrain: `terrain_frag.glsl:169-174`, which covers land and far water.
  - Billboards: `cloud_sphere.vert:118-124`, into `vSunTint`, at the billboard
    centre (`rel = p - camPos`).
  - The deck: `cloud.frag:207-217`, per fragment.
  - The air: `atmosphere.glsl` `atmoSegment`, where the per-sample
    `direct = exp(...) * lit` (`:208`) is multiplied by the eclipse term at `ps`
    (`rel = ps - uAtmoCam[i].xyz`). The sky darkens, and so does the aerial
    perspective over the ground.
  - Sky light during an eclipse: terrain `skylight` and the clouds' sky term are
    multiplied by `mix(ECLIPSE_SKY_FLOOR, 1, visibility)`, a tunable constant.
  - The near sea and Panda's lights go through `World._light_at`
    (`world.py:544-555`), which gains the CPU eclipse term. This also fixes the
    airless moon inside the planet's shadow, a lunar eclipse.
- **Sun size.** Add `SunLook.angular_radius` (`world.py:94-104`); the `Sky` takes
  it instead of `SUN_ANGULAR_RADIUS` (`sky/sky.py:49-51`, `:187`).
- **Demo.**
  - `MOON_DISTANCE_M` becomes about 21,000 km (`demo_terrain.py:137-139`), so the
    moon and the sun look the same width.
  - New `terrain_shots.py` poses:
    - `eclipse_ground` (looking up, sun just off the moon's limb: a partial
      eclipse);
    - `eclipse_total`;
    - `eclipse_orbit` (the umbra spot on the planet, seen from above);
    - `lunar_eclipse`.
  - A pose can aim its sun at a body (the direction from the ground point to
    `body`, plus an offset); `SUNS` (`:85-89`) gains that form.
- **Tests.**
  - `disc_coverage`: 0 and 1 limits, symmetry, and a Monte-Carlo check.
  - The eclipse from `Shadows` against geometry.
  - GPU read-back of `eclipseVisibility`.
  - Defaults link (`test_defaults_bind_every_input` pattern).
  - World light: extend `tests/test_world.py:195-231`.
- **Done when** the new poses show a soft-edged umbra, the sky darkens, swinging
  the sun with J/K slides the shadow, and the existing poses are unchanged
  (`compare_shots`).

## B. Lighting objects per node (CPU)

**Goal.** Games light each model from the light **at its own position**, with
world shadows, instead of the light at the camera.

- **`World.sun_visibility(points_world) -> (n,)`.** It is the product of the CPU
  terms that exist so far; A provides the eclipse. Parts C, D and E each add
  their own twin to it.
- **`World.light_at`** takes visibility into account.
- **`World.track_light(node, cast=False)` / `untrack_light(node)`.**
  - A tracked node gets its own `DirectionalLight` and `AmbientLight`, with
    `set_light_off` for the world's lights on that node.
  - `World.update` evaluates all tracked nodes in one vectorised call (after
    `_light`, `world.py:666`) and scales them by `light_scale`.
  - `cast` is stored for part H.
- **Demo.** The test ball (`demo_terrain.py:517-532`, `:571`) becomes a tracked
  node.
- **Tests.** Tracked nodes follow `light_at` at their own position. Untracking
  restores the world's lights.
- **Done when** the ball goes dark inside an eclipse while the camera is outside
  it.

## C. Cloud shadows on the ground and the sea

**Goal.** The planet's clouds shade the terrain, the far water and the near sea,
read directly from the density field. The result depends on world position only.

- **`cloud_field.glsl`.**
  - Factor the inner loop of `sunOpticalDepth` (`:200-227`) into
    `slabOpticalDepth(f, start, dir, length, steps)`; `sunOpticalDepth` becomes
    a call to it, with an unchanged result.
  - Add `cloudShadowDepth(f, p, dir, maxSteps)`. It intersects the sun ray with
    the slab's two spheres (`planetRadius + slab`), then marches the chord with
    a step count proportional to its length, capped. The weather is fetched at
    the entry point.
  - It uses the deck layer's field (`SphericalClouds._deck_layer`, the full field
    with per-metre extinction), so the shadow matches the drawn clouds.
- **CPU twin.** `cloud_shadow_on_sphere` in `clouds/noise.py`, built from
  `_sun_march_samples` and `_sun_depth` (`:737-772`) and `sample_cube`.
- **Binding.** `SphericalClouds.bind_shadow_inputs(node)` binds `cloudNoise`,
  `layerParams`, `weatherMap`, `planetRadius`, `sunDir`, `uSpherical`,
  `noiseSize`, `noiseOffset` and `uCloudShadowLayer`. Today the textures are only
  reachable through `root.get_shader_input`, so keep handles to them. Defaults
  give `uCloudShadowLayer = -1`. `World` binds the body's clouds to the body's
  terrain root.
- **Terrain.**
  - `terrain_frag.glsl` includes `cloud_field.glsl`, after `atmosphere.glsl`; no
    name clashes were found.
  - Its sunlight gets `exp2(-cloudShadowDepth(...))` at the planet-frame position
    (`uAtmoCam[own].xyz + vRelPos`; clouds only exist with air).
- **Near sea.**
  - `sea.glsl` is lit only by the CPU's `iWaterLight` (`:26`, `:423`, `:451`).
    Split it into a direct part and an ambient part (`Sea.set_light`,
    `World._light_ocean`, `world.py:694-715`).
  - `seaShade` takes a per-pixel `sunShade`, which multiplies the direct part.
    D's terrain shadow reuses it.
- **Knob.** `ShadowSettings.cloud_shadow_steps` (0 = off, LIVE). Creating it
  includes the `settings.py` registration and the `tests/test_quality.py` rules.
- **World.** `sun_visibility` gains the cloud term.
- **Tests.**
  - The refactor leaves `sunOpticalDepth` bit-identical (CPU twin and existing
    tests).
  - The shadow is zero with clear sky or the sun overhead of a clear column, and
    grows with coverage.
  - CPU twin vs GPU read-back.
- **Visual.** Noon and afternoon low-altitude poses, shadows offset from the
  clouds by `altitude / tan(elevation)`, a view from orbit, and the near-sea
  band. Cost measured on `flight_bench`.

## D. Terrain shadows: global shadow-ceiling map

**Goal.** Mountain shadows at planetary scale, with the correct penumbra, on the
terrain, the sea, the clouds (mountains at sunset) and objects. Baked once for
the static sun.

- **Bake.** In `terrain/shadow_map.py`, numpy first; port the march to C in
  `terrain/native/` only if it misses its budget.
  - **Heights.** `Planet.sample_dirs(cube_directions(N), min_wavelength ≈ 2
    texels)` (`planet.py:455-476`; `sky/starfield.py:50`), with stamps included.
  - **Map contents.** For each texel, march toward the sun from `D_NEAR` (about 2
    texels) out to where the ray clears the planet's highest terrain, including
    curvature.
    - The **ceiling** is the lowest altitude from which the sun clears every
      occluder.
    - The **distance** is the distance to the occluder that sets the ceiling.
    - Store both as an RG cube map. The ceiling is well defined because shadow
      on a heightfield is monotonic in altitude.
  - **Scheduling.** The bake runs in the pool through `executor.submit_once`,
    tiled per face and cached per process, as `clouds/weather.py:511-536` does.
  - **Loading and the sun.** It counts toward `World.loading`. On `set_sun` the
    map is rebuilt asynchronously and swapped in.
  - **Size.** N is 1024 by default, about 0.9 km texels at 600 km and 25 MB in
    RG16F. It is set by `ShadowSettings.terrain_map_size`.
- **`shadows.glsl`: `terrainShadow(planetPos, altitude)`.** One cube fetch,
  then a smoothstep of `(altitude - ceiling) / distance` over
  `±uSunAngularRadius`. That gives the physical penumbra.
- **Receivers.**
  - **Terrain** per pixel. An airless body has no planet-frame position in the
    fragment shader today, so add a varying from `terrain.vert` (`uPlanetCentre`
    is there). The moon gets crater-rim shadows too.
  - **Near sea** through C's `sunShade`.
  - **Clouds**: billboards per vertex (`cloud_sphere.vert`) and the deck per
    fragment. `World` binds the body's map to its clouds.
  - **Objects**: a CPU twin `Terrain.sun_visibility(points)` with `sample_cube`
    (`clouds/weather.py:154`), fed into `World.sun_visibility`.
- **Clouds' terminator.** Check `grazingLight` (`cloud_field.glsl:242`) at sunset
  with the real ceiling. Keep it unless it darkens the clouds twice; if it does,
  reduce its ground part. `cloudTwilight` stays as it is.
- **Expected gap.** Nearby small ridges (closer than about 2 km) cast nothing
  until part E.
- **Tests.**
  - A flat planet gives no shadow anywhere.
  - A synthetic peak (a stamp such as `mesa.png`) casts a shadow of length about
    `h / tan(elevation)`.
  - The ceiling is monotonic.
  - CPU twin vs GPU read-back.
  - Bake time is reported.
- **Visual.** Sunset over the mountains from the ground and from orbit, clouds
  lit and shadowed by peaks, the moon's terminator, and bake time at load.

## E. Terrain shadows: local ceiling map around the camera, and the cloud shadow map

**Goal.** Nearby shadow detail (about 50 m texels within about 50 km), with no
change to chunk generation or the LOD.

- **Bake.** The same bake code over a camera-centred square, about 100 km at
  2048², with heights at about 50 m spacing. It marches only up to the global
  map's `D_NEAR` plus a margin, so the two maps multiply without counting an
  occluder twice.
- **Rebuild.** In the pool when the camera has moved a quarter of the map, and
  only below an altitude limit. The new map is swapped in when ready, with an
  edge fade to the global map.
- **`terrainShadow`** multiplies the local term and the global term.
- **CPU twin.** Added to `Terrain.sun_visibility`.
- **Knobs.** `local_map_size` and `local_map_spacing`, both off at LOW.
- **Main risk: height-sampling cost.** About 4M heights per rebuild; measure it
  first. If too slow, use a coarser spacing, a smaller extent, or band-limited
  heights.
- **Tests.** Seams between the two maps (no double shadow, no gap), rebuild
  hysteresis, and the twin vs a brute-force march.
- **Visual.** Valley walls and small ridges at low sun, while flying, with no
  pops on rebuild.

### The cloud shadow map (added after C)

C marches the clouds per pixel: 0.8–1.7 ms at HIGH near the ground
(`performance_backlog.md`, "A cloud shadow map"). The march depends on the
sun ray alone, so it can be stored once per ray and shared by every pixel
on it. It shares this part's machinery: a map around the camera, its
re-centring and rebuild rules, and the fade to a coarser term.

- **The map.** Keyed by where the sun ray enters the slab's base sphere
  (`cloudShadowChord`'s `start`, analytic per pixel), on a tangent chart
  around the camera. A render-to-texture pass marches each texel once
  (`slabOpticalDepth`), on the GPU.
- **Two extents.** Near detail at about 60 m texels, and a far map at about
  500 m out to the horizon. They fade into each other and then into the
  mean table (`ground_shadow_table`) where the march gives way today.
- **Rebuild.** When the camera has moved a fraction of the map, or the sun
  has turned more than a texel's worth: every frame while the sun swings,
  which is still one march per texel, not per pixel. Same hysteresis as the
  local ceiling map.
- **Kept per pixel.** Ground inside the slab (mountain tops in the clouds),
  where the ray starts at the point and not on the base sphere.
- **Receivers.** Terrain, far water and the near sea's `sunShade`. The CPU
  twin (`SphericalClouds.sun_visibility`) does not change.
- **Knob.** Map sizes in `ShadowSettings`. `cloud_shadow_steps` becomes the
  march's per texel.
- **Tests.** The map against `cloud_shadow_on_sphere` at texel centres. The
  near/far/mean seams. Rebuild on a moving camera and a moving sun.
- **Visual and cost.** The C poses unchanged within a texel's blur;
  `clouds_grazing` still without bands. Target 0.2–0.3 ms where C costs
  0.8–1.7.

## F. Long-range cloud-on-cloud shadows

**Goal.** At low sun, clouds shade clouds kilometres away, beyond the 640 m
per-fragment march.

- **Far segment.** Per billboard, in `cloud_sphere.vert`, a far segment from the
  current reach (`sunSteps × sunStepLength`) out to a few km, in about 6 coarse
  steps of 1–2 octaves. It uses C's `slabOpticalDepth` and is multiplied into
  `vSunTint`. The detail march per fragment does not change.
- **What stays as it is.** The far deck keeps its averaged table
  (`deck_light_table`), which already accounts for a low sun's long path.
- **Watch for** the earlier rejected per-vertex march (far billboards lit flat,
  `performance_backlog.md`). Only the slowly varying far part moves to the
  vertex here; check it visually.
- **Knob.** `ShadowSettings.cloud_far_reach` (0 = off).
- **Tests and visual.** Dusk poses inside and above the deck. Cost per
  `flight_bench`.

## G. Model shader with per-pixel world shadows

**Goal.** Models are lit and displayed like the world, with every world shadow
per pixel. This implements `docs/source/object_lighting.md`.

- **Package.** `src/world_builder/objects/`, with `model.vert`, `model.frag` and
  `apply_world_lighting(node)` / `World.track_light(..., shader=True)`.
- **Lighting.**
  - `objectLight` from `terrain/shaders/object_light.glsl`.
  - Albedo from `p3d_Texture0` decoded with `pow(2.2)`, or the material colour.
  - The sun's colour through the air with `atmoSunlight`, at the fragment.
  - `sunShadow`: eclipse, cloud, terrain global and local terms.
  - `applyAtmospheres`, then `uExposure` and `aces_tonemap` (`display.glsl`).
- **Inputs.** The nearest body's shadow inputs are bound to a shared models root
  each frame. B's per-node lights give the sky light at the node.
- **#14.** The asteroid shaders include the same `shadows.glsl`; a small
  follow-up wires them.
- **Tests and visual.**
  - Test ball and a textured model: noon, sunset, crossing a cloud-shadow edge,
    inside an eclipse, and next to a mountain shadow.
  - The ball no longer clips to flat white (`object_lighting.md`).

## H. Objects cast shadows (caster map)

**Goal.** Tracked models with `cast=True` shadow the terrain, the sea, other
models and themselves.

- **The map.**
  - A sun-space orthographic depth buffer and camera on camera bit 20 (unused,
    `depth.py:66-94`), following `clouds/half_res.py:106-143, 214-221`: a
    depth-only initial state and a tag state for the casters.
  - Its lens is fitted every frame to the casters' bounds, not to the camera's
    view. A high ship's shadow lands wherever it falls.
  - Hardware depth comparison with 3×3 PCF.
- **Receivers.** Terrain (land, far water, near sea through `sunShade`) and the
  model shader. Fragments outside the map's bounds skip it.
- **Knobs.** `ShadowSettings.caster_map_size` and an on/off.
- **Cost.** About one extra camera pass of main-thread time (the half-res layer
  cameras cost about 0.6 ms each) plus little GPU. Measure it.
- **Visual.** The ship's shadow on a landing pad, at altitude, ship on ship, and
  self-shadowing. Acne and peter-panning tuned with bias.

## I. Ring shadows (after #14 P7)

- **The term.** Add ring occluders to `Shadows`: the plane's normal, the inner
  and outer radius, and a 1D opacity texture taken from #14's ring definition.
  `ringVisibility` in `shadows.glsl` intersects the sun ray with the ring plane
  and reads the opacity.
- **Receivers.** Planet, clouds, air samples and objects, plus a CPU twin.
- **Rings as receivers.** The ring shader (#14) includes `shadows.glsl` and
  `atmoSunlight`, so the planet's shadow falls across the rings.
- **Visual.** The ring shadow band on the planet from orbit and from the ground,
  and the planet's shadow on the rings.

## J. God rays

- **Shadows in the air.** In `atmoSegment`, the per-sample `direct` (`:208`) is
  also multiplied by `terrainShadow` (global and local maps) and by a cheap cloud
  shadow (1–2 taps of the shadow density at mid-slab). That gives shafts and
  mountain shadows in the sky and in the aerial perspective.
- **Binding.** The sky and the clouds take the body's maps.
- **Knob.** `ShadowSettings.volumetric` (off, terrain only, or terrain and
  clouds).
- **Cost.** The per-pixel `ATMO_STEPS` × the extra fetches. Measure it on the sky
  and the terrain.
- **Visual.** Sunset shafts between peaks and under broken clouds.

## K. Optional: terrain in the caster map

Only if D and E look too soft across large ships. Extend H's map along the sun
far enough for terrain occluders, up to about 150 km at low sun, and draw the
terrain chunks inside that narrow box with `depth_only_shader()`
(`terrain.py:101-106`). Terrain depth goes to its own target, so the terrain
never shadows itself twice. It costs about one extra pass per map. Budget one or
two hero maps.

---

## Verification (whole effort)

- `poetry run pytest` (or the repo's `tasks.py` test target) passes after each
  part. The GPU read-back tests run where a GL context exists.
- `terrain_shots.py --offscreen` for each part's poses, compared before and after
  with `compare_shots.py`. Untouched views stay identical unless the part says
  otherwise. The user confirms each look.
- Frame cost per part on `flight_bench.py` and the shots' timed frames, recorded
  in `docs/source/shadows.md`. Load-time bakes (D, E) are reported separately.
