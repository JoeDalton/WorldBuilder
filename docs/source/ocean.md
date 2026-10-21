# Ocean

[`world_builder.ocean`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/ocean/)
is a camera-locked reflective ocean, curved with the planet. It takes a `game`
object by duck typing (see the module docstring for exactly what it reads),
registers its own per-frame update in `game.method_lists`, and exposes
`clean()`. The per-class API is in the [code reference](apidocs/index.rst).

## `ocean/ocean.py` — clipmap-free reflective ocean

[`ocean.py`](../../src/world_builder/ocean/ocean.py)'s `Ocean` (its module
docstring has the overview):

- **Per-pixel waves.** By default the surface is one huge flat quad re-centred
  under the camera every frame. All wave detail is computed in the fragment
  shader from world position, so sliding the quad causes no motion.
  `compute_wave_dirs` precomputes the per-iteration wave directions once on the
  CPU as a uniform array; `OceanSettings.wave_iterations` sets how many.
- **Quality.** Every cost/quality trade is a field of `OceanSettings`, with
  LOW / MID / HIGH / ULTRA presets (`OceanSettings.preset(level)`; see
  [Quality settings](quality.md)). Pass one to `Ocean(settings=...)`, or
  change it with `set_settings()`. Most knobs are live:
  - the octave count;
  - how many pixels a wavelength needs before it fades (`aa_full_px`,
    `aa_gone_px`);
  - the rough fresnel and the reflection's blur;
  - the seabed's cascades;
  - the depth colour, foam and depth-scaled swell switches;
  - the reflection camera's field-of-view margin.

  The reflection's resolution (`reflection_scale`) remakes its buffer on the
  same camera, so the lens and far plane taken at construction are kept.
  The swell grid's tessellation remakes the mesh. The switches are uniforms tested in branches that are the same
  across the screen, so a feature that is off costs almost nothing and needs
  no recompile. The terrain demo cycles the presets with F5;
  `terrain_shots.py` and `flight_bench.py` take `--ocean-quality`.
- **Seabed** (`ocean/seabed.py`). The ocean knows nothing of the terrain:
  `set_seabed(sampler)` gives it a function from points of its plane (in the
  app's world) and a texel size to the sea floor's height there.
  `SeabedMap` calls it on a worker thread over three square cascades centred
  on the camera. With `high`: 1 km at 8 m texels, 8 km (the swell grid) at
  42 m, 64 km at 500 m. Each cascade is uploaded as a float texture, re-baked
  once the camera has moved a quarter of its extent, refreshed every 5 s,
  and baked at most every 0.5 s. A fast camera outruns the finest cascade for
  a moment, and the next coarser one covers. Where the plane sits in that
  world may change every frame (`set_seabed_frame(origin, east, north)`): a
  tangent plane following the camera over a planet turns as it flies. Each
  cascade keeps the frame it was baked in, and every frame the current
  plane's xy is mapped into it in float64, so a bake stays put on the ground.
  Bake corners sit on the texel lattice, so a new bake reads the same ground
  points and the shoreline does not shift when it lands. Without a sampler,
  or beyond the cascades, the sea is deep and looks as before. The terrain
  demo's sampler reads `TerrainQuery.ground_height_at` at the grid level
  matching the texel, from stored chunks only. In the low fast flight
  (`flight_bench.py`) the bakes take 15–36 ms each and leave the frame times
  as they were.
- **Geometric swell** (`geometric_swell=True`, used by SpaceFlight's ocean
  level and the terrain demo; a prototype with known artifacts, see the
  [swell artifact report](https://github.com/JoeDalton/WorldBuilder/blob/main/src/world_builder/ocean/OCEAN_SWELL_ARTIFACT_REPORT.md))
  builds `make_swell_grid_mesh` instead: a dense grid that `ocean.vert`
  displaces by the swell, tapered to flat before its edge, ringed by
  geometrically spaced border rings in the same mesh.
- **Curved over the planet** (swell mode only: four vertices cannot curve).
  Every vertex droops by `d² / 2R`, baked into the mesh. That is exact because
  the mesh is re-centred on the camera, the same origin the cloud shader droops
  from, and costs nothing per frame. With `exact_sphere=True` the mesh lies
  on the sphere itself instead: the two part by `d⁴ / 8R³` (58 m at 100 km
  on a 600 km planet), so water meeting real spherical terrain needs it (the
  terrain demo), while water meeting the drooped flat cloud field keeps the
  paraboloid. The horizon then dips `sqrt(2h/R)` below
  eye level (1.0° at 1 km, 3.05° at 9 km). The border is rings rather than one
  quad because a quad interpolates linearly and would droop into a cone; the
  spacing is geometric because the flat-cell sag is `spacing² / 8R`, and 28
  rings keep it under 1 arcmin (15 would give 2.9) for about 23% more
  triangles. The surface reaches past the horizon seen from 20 km,
  `sqrt(2R·altitude)` ≈ 505 km plus a 10% margin, so the planet's bulge hides
  its edge. Two things stay flat on purpose: a game's collision plane (a ship is
  within metres of the camera, where the droop is micrometres) and the planar
  reflection, which mirrors about z=0 and so drifts by the droop with distance
  — 7.8 m at 10 km, where the normal is already flattened to a mirror.
- **Aerial perspective.** Distant water fades to `haze_color` after the
  tonemap, as the clouds do, so sea and sky meet at the horizon. Its
  `haze_distance` (150 km) is much shorter than the clouds' 400 km: the air
  path to sea level is far denser than to cloud at altitude. In a world with
  [atmospheres](atmosphere.md), bind them to `ocean_node`
  (`Atmospheres.bind`) and call `set_atmosphere_frame(base_to_world,
  sun_direction)` every frame: they replace the haze, applied in linear
  light before the tonemap, as on the terrain.
- **Night.** The water's own colour was chosen in daylight; `set_light(rgb)`
  scales it (uniform `iWaterLight`) and the reflection buffer's clear colour
  by the light on the sea relative to that daylight, so the sea is dark by
  night. The terrain demo computes it with `Atmospheres.light_at`. The
  reflected sky darkens by itself. The clear colour is stored tonemapped
  (`aces_tonemap`, the numpy twin of the shader's), like everything else in
  the buffer.
- **Far plane.** SpaceFlight raises it to 600 km (`CAMERA_FAR_M`), set before
  the scene is built because the reflection camera copies the lens. Panda's
  default 100 km clips the outermost cloud shell and keeps the ocean horizon in
  view only up to 785 m of altitude; 600 km clears the 512 km shell and holds
  the horizon to about 28 km. It costs almost no depth precision, which goes as
  `z² / (near · 2^bits)` and is set by the near plane.
- **Planar reflections.** `make_reflection_buffer` builds an offscreen texture
  buffer and a reflection camera that `mirror_camera` (called from `update`)
  mirrors about the water plane every frame. Its lens is a copy of the main
  camera's taken then, with 1.4× its field of view (so turning quickly never
  culls what the mirror should show at its edges) and 4× its far distance.
  A clip plane restricts it to geometry above the water. Its camera mask is
  bits 0–19 less bit 1, the bit the ocean hides itself from; since Panda
  draws a node for a camera if any of the camera's bits is shown on it (see
  `world_builder.depth`), that hide does not exclude the ocean: what keeps
  the surface out of its own mirror is that the camera below sees only its
  back faces, which are culled. The terrain is reflected or not with
  `TerrainSettings.reflected` (all of bits 0–19). The
  buffer is sized from the main camera's display region (`app.cam`) when
  it has one (the window size otherwise) times `OceanSettings.reflection_scale`
  (0.5 at HIGH),
  so it tracks the internal render resolution rather than
  the window. `uReflUVScale` corrects for power-of-two texture padding and is
  refreshed from the real texture size once the buffer is realized: the default
  pipeline pads, simplepbr (not used) does not.
  In the terrain demo the mirror shows the sky (its dome is drawn on the
  far plane, so the reflection camera's shorter range keeps it) and the
  clouds. Shaders drawn through the mirror write `gl_ClipDistance[0]`
  (terrain: the real distance; sky, stars, clouds: always kept, they are
  above the water). Its resolution is `reflection_scale` (half the view by
  default); lowering it saves little in the demo, where the mirror's cost
  is per billboard, not per pixel.
- **No collision of its own.** Physics is the game's business: SpaceFlight's
  `space_flight.scenes.ocean.Ocean` subclass adds a flat terrain collision
  plane at Z=0 so ships can crash into the water.

## Shaders

Driven by [`ocean/ocean.py`](../../src/world_builder/ocean/ocean.py) (above):

- **[`seabed.glsl`](../../src/world_builder/ocean/shaders/seabed.glsl)**
  (included by both) reads the seabed: the finest cascade covering a point,
  blended into the next coarser one across the outer 10% of its side, so
  mostly one texture read. Its `swellDepthFactor` takes the swell from
  nothing at `swell_shallow` (5 m) to all of it at `swell_deep` (60 m).
  **[`water.glsl`](../../src/world_builder/ocean/shaders/water.glsl)**
  holds the colour by depth, shared with the terrain's sea surface.
- **[`ocean.vert`](../../src/world_builder/ocean/shaders/ocean.vert)** is a
  passthrough plus the optional `uGeometricSwell` mode (which SpaceFlight
  enables): it displaces vertices vertically by the same `swellField` the
  fragment shader samples, tapered to zero near the dense grid's edge so the
  displaced centre joins the undisplaced border rings. Over the seabed the
  swell dies out in the shallows, so it does not climb the beaches (the demo
  can keep the game's 20 m swell). At the shore the water also sinks, to
  1 m under the shoreline. It meets the ground at 0.25 m of depth, four times
  as steeply as the seabed rises, so a flat beach barely above the water
  stops z-fighting with it in the distance. Under land it stays well below
  the ground, so the drawn waterline keeps to the seabed's. Its world position comes from
  `uModelMatrix` (refreshed by `Ocean.update` after the plane is re-centred),
  not `p3d_ModelMatrix`, so the waves stay anchored to the world rather than
  to the moving render origin.
- **[`ocean.frag`](../../src/world_builder/ocean/shaders/ocean.frag)**:
  - **Wave field.** `getwaves`/`waveGradient` accumulate Gerstner-like octaves
    (`wavedx`) along per-iteration directions `iWaveDirs`, precomputed in
    Python by `compute_wave_dirs`. `waveGradient` computes the slope
    analytically in one pass instead of a three-sample finite difference.
  - **Filtering by size on screen** (#12). Sky reflection and water colour
    are far apart, so the waves alias as soon as they go sub-pixel. Camera
    distance is the wrong measure: what counts is how much water a pixel
    covers, and that depends on the grazing angle too. So the shader takes
    the pixel's footprint from the lens: across the view, the distance times
    the pixel's angle (`uPixelAngle`, narrower off the screen's centre);
    along it, that over the sine of the elevation. `dFdx`/`dFdy` of the
    world position would jump at every edge of the displaced swell mesh, and
    each triangle would fade its waves differently. Each octave fades by
    how far its phase moves across the pixel along its own direction:
    `uAAStart`..`uAAEnd`, from `aa_full_px` (8) down to `aa_gone_px` (4)
    pixels per wavelength. A wave running across the view therefore stays
    longer than one running along it. Octave i's frequency is 1.18^i, so the
    footprint's bounds give two octave ranges up front: those resolved in
    every direction run in a plain loop, as cheap as unfiltered, and only
    the octaves in between pay for the fade. Past the last of them nothing
    is evaluated, and the small waves are skipped only once even the longest
    one running across the view is gone (cutting them off by the
    along-the-view footprint drew a hard line). In the coast pose that leaves 0.1 octaves per pixel
    where there were 2, and 32.7 where there were 34 at 3 m over the sea.
    Every pixel normalises by the full weight sum (`uWaveWeightSum`), so
    dropping fine octaves does not inflate the coarse ones.
  - **What the filter takes away is kept** as slope variance. A fine
    octave's slope is nearly as large as a coarse one's (`frequency ×
    weight = 0.944^i`), so dropping it without compensation would turn
    distant water into a too-bright mirror. The faded octaves' variance
    (`C_WAVE` per octave, the tail from `uTailVar`, both computed on the
    CPU) is added to that of the faded swell and of the noise mask, which
    also fade to their means. That spread, flattened like the resolved
    normal, does two things:
    - It picks the fresnel from `uFresnelLUT`. That table is Schlick
      averaged over Gaussian facets, weighted by their area seen from the
      eye (`rough_fresnel_table`), so distant water settles on a steady
      blend of sky and water colour.
    - It blurs the reflection through its mipmaps (`uReflBlur`): one
      sample, at the level the ripples would have spread it over.
    The domain warp and `detailAngle`'s flattening toward a mirror at the
    horizon are kept.
  - **Swell + frequency modulation.** A large-scale `swellField` (the product
    of two drifting value-noise layers, so it changes shape rather than
    scrolling) tilts the normal, and also shifts the small-wave phase
    (`fmPhase`) so the dominant octave's tiling drifts. Its tilt follows
    the geometry's height: `swell_amplitude / 20` and the depth factor.
    `set_swell_amplitude()` changes the height live. The
    field mostly spans 0.1 to 0.5, so 20 m of amplitude stands crests about
    7 m above the troughs, over a swell some 250 m across (`swell_scale`).
  - **Colour by depth.** Over shallow seabed the water's own colour goes
    from `shallow_color` to the deep `water_color`. That is Beer–Lambert
    along the light's way down and the view's back up, with an e-folding
    depth `shallow_depth` (`deepWaterShare` in `water.glsl`). The terrain's
    sea surface, drawn above the 14–16 km handover, applies the same tint
    on top of its own 3,000 m shelf gradient (`Terrain.set_shallow_water`,
    defaulting to the same shared `SHALLOW_WATER_COLOR` and
    `SHALLOW_WATER_DEPTH_M`), so coasts keep their colour across it.
  - **Shoreline foam.** Under `foam_depth` metres of water:
    - solid at the waterline;
    - then lines along the depth contours, `foam_spacing` metres of depth
      apart, running shoreward at `foam_speed` and shifted along the coast
      by the swell;
    - broken up by noise.
    It is drawn wherever the seabed says land too: water drawn there is at
    the waterline, and where the terrain stands above, it hides the foam.
    Lines closer than a pixel fade to their mean coverage. The foam fades
    out where the seabed's texels pass 12–48 m, since coarser ones cannot
    place a band a metre or two deep. The foam is lit by `iWaterLight`.
  - **Reflection.** The reflection UV is recomputed per fragment from the
    flat (z = 0) surface position: interpolating a vertex clip-space value
    would skew the projective divide per triangle once the swell displaces
    vertices. Sampling at that flat footprint still slides against the
    displaced geometry (see the
    [ocean swell artifact report](https://github.com/JoeDalton/WorldBuilder/blob/main/src/world_builder/ocean/OCEAN_SWELL_ARTIFACT_REPORT.md)).
    The ripple-perturbed UV is clamped to the rendered region
    (`uReflUVScale`) so it never samples the power-of-two padding. The
    buffer holds *display* colours (the sky and terrain tonemap themselves,
    with the same exposure), so the sample is decoded back to linear light
    (`aces_inverse`, divided by `uExposure`) before the blend. Otherwise the
    output tonemap applies twice and lifts the dark: a night sky's
    reflection came out about twice as bright as the sky. A full mirror now
    shows exactly the sky's own colour.
  - **Output.** The fresnel blend of reflection and water colour (the
    latter scaled by `iWaterLight`) is ACES-tonemapped (`aces_tonemap`), then
    hazed toward `uHazeColor` over `uHazeDistance` — after the tonemap, as
    the cloud field does, so sea and sky meet cleanly at the horizon. With
    atmospheres bound (`uAtmoCount > 0`), `applyAtmospheres` replaces the
    haze, in linear light before the tonemap, along the view ray turned into
    world axes by `uBaseToWorld`, toward `uAtmoSunDir`.
  - **Debug.** `uDebugMode` 1–12 (`Ocean.set_debug_mode`; 0 is normal
    rendering; `terrain_shots.py --ocean-debug`) visualise intermediate
    quantities: normal, reflection UV, fresnel, world grid, reflection-UV
    clamp, raw reflection sample (display colour), pre-tonemap clipping, FM
    phase, share of the octaves resolved, unresolved slope spread, octaves
    evaluated (the loop's cost), depth (black at the waterline to white at
    50 m; red where the seabed says land, blue beyond the cascades).
