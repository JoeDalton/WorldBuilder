# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `world_builder.gravity.Gravity` and `World.gravity_field(g_constant)`: the
  bodies' raw gravity vector at many points at once, from point masses
  (`g = -G sum m (x - p) / |x - p|^3`, capped inside a body at its surface
  value). A fused native kernel (`native/gravity.c`, 4-9x faster) with a numpy
  fallback.
- `world_builder.asteroids`: asteroid fields from a spec, stored nowhere
  (docs/source/asteroids.md). A `FieldSpec` sets the extent (box or ball, soft
  edges), the density and its clumping (noise above a threshold set from a
  coverage), the size distribution (`Distribution`: uniform, normal,
  lognormal, power law, U-shaped, table), spins (axis distribution, size
  scaling, free symmetric-top tumble), prototypes, ice and keep-outs.
  `Scatter` gives the rocks of any part of it, the same every time: Poisson
  cells per size class, density thinning and a Matern hard core (no two
  rocks overlap). A native scatter (`native/scatter.c`) gives numpy's rocks
  ~100x faster. `python -m world_builder.asteroids.preview` draws a field's
  rocks and statistics. (#14)
- Rock prototypes (`asteroids.shape`, `asteroids.prototype`,
  `asteroids.mesh`): an ellipsoid with lumpy noise, ridges and craters,
  meshed on the terrain's cube-sphere grid at four levels of detail (band
  limited, crack-free, with each level's error), with a full-detail normal
  map and a radius grid for collision; built by `ChunkExecutor` workers.
  `scripts/asteroid_gallery.py` renders them. (#14)
- `world_builder.asteroids.field.AsteroidField`: an asteroid field drawn by
  hardware instancing around the camera (one draw per prototype and level
  of detail), rocks held per size class within their drawing distance,
  levels of detail by pixel error, spin and tumble in the vertex shader at
  the game's clock, the terrain's materials and lighting. `AsteroidSettings`
  (the `asteroids` section of `WorldSettings`) sets its costs.
  `scripts/demo_asteroids.py` flies through one; `scripts/asteroid_bench.py
  --field` times it: 3-5 ms p95 frames against SpaceFlight's 10-37 ms for
  the same 15 km cube. (#14)
- `AsteroidField.enable_collision`: the rocks near the camera solid to a
  game's CollisionTraverser, through a native solid
  (`native/asteroid_collision.cpp`, in the collision library) testing spheres
  and segments against each rock's finest mesh as drawn, turned to the
  field's time; one entry per collider. Its owner gives the material
  (`material_at`) and the surface's speed (`velocity_at`) at a hit.
  `scripts/asteroid_bench.py --collision` times it: 2.4 ms a traverse
  against 32 ms for SpaceFlight's per-rock spheres. (#14)
- An `AsteroidField` without a camera (headless) keeps its rocks and their
  collision up to date and draws nothing. (#14)
- `scripts/asteroid_bench.py`: frame times of an asteroid field flown along
  fixed paths; `--legacy` rebuilds SpaceFlight's current field (one node and
  collision sphere per rock) as the reference to beat. (#14)
- `scripts/compare_shots.py`: compares two folders of screenshots pixel by
  pixel, for shader changes meant to change nothing. (#14)

- Big rocks near the camera drawn as terrain (`asteroids.promotion`,
  `asteroids.rock.RockSpec`): the terrain's level of detail and collision
  within ~0.8 radii of their surface, turning with the field's time; pooled
  `Terrain`s on the field's workers and textures. `AsteroidSettings` gains
  `terrain_min_radius` and `max_terrain_rocks`.
  `scripts/asteroid_bench.py --flight approach` flies into one, and
  `--terrain-rocks N` overrides their number. (#14)
- Crooked asteroid shapes (`ShapeSpec.family`): besides round rocks (now
  with big bulges), contact binaries (two lobes at an angle, a waist, as
  Toutatis), bent and twisted rocks (as 54509 YORP) and angular fragments
  (planes clipping the body). Small rocks are mostly crooked, big ones
  mostly round (`ShapeSpec.round_share`, 15 % to 85 %), chosen by the
  scatter in numpy and C alike (native interface 13). Lumps are evaluated
  on the body's surface. `asteroid_gallery.py --family NAME --views N`.
  (#14)
- Ice in patches (`asteroids.shape.IcePatches`, `MaterialSpec.patch_scale`):
  a rock's ice covers its share of the surface where a smooth noise ranks
  highest, a few patches of ice on a rocky rock, a few of rock on an icy
  one, each pure (no rock/snow blend, no beige). The rank is the normal
  map's fourth channel (RGBA8); the rock's terrain and `material_at` read
  the same function. Far rocks keep the patches' mean colours. (#14)
- The terrain draws any body (`spec.evaluator()`; a `Planet` or a rock), with
  per-body height bounds for the LOD pass (a small body was mostly culled
  by the planets' -7 km), a texture period per grid (`grid_for(radius,
  uv_period)`: no stretched textures on small bodies), an orientation
  (`Terrain(orientation=)`, `set_orientation`: a turning body), lighting as
  an object (`spec.radial_light`), a plain display encoding
  (`Terrain(tonemap=False)`) and shared texture arrays (`Terrain(textures=)`).
  (#14)
- `World` takes no body for open space: the sky, the sun and the lights
  alone, `space_ambient` the sky light away from any body. (#14)
- Planet rings: `asteroids.spec.Annulus` (inner and outer radius,
  thickness, soft faces and rims, a radial profile of ringlets and gaps),
  in the numpy and native scatter (native interface 11), and a `ring`
  preset 220,000 km across. (#14)
- The haze of the rocks not drawn (`asteroids.haze`, `shaders/haze.frag`):
  along each pixel's ray, the cross-section of the rocks under the drawn
  size (the hard core's thinning included) and past the field's `far`, plus
  `Density(dust=)`, lit as Lambert spheres. A ring is a band of light from
  afar; rocks appear out of it on approach. Drawn by the far camera of a
  split from its near plane, on a hull pulled inside the far plane.
  `AsteroidField(haze=)`. The haze averages the clumps (backlog). (#14)
- `World(fields=[FieldLook(...)])` and `World.field(name)`: asteroid fields
  and rings in a world, on its pool, settings, sun and clock, part of its
  loading. `scripts/demo_asteroids.py` is built on it (split depth, the sky,
  render's origin at the camera) and gains `--preset ring` and a far view
  (Home). (#14)

### Changed

- Asteroid fields without thresholds (#14):
  - each size class holds a ball of cells around the camera (not a box),
    moved by its difference and grown over frames (about 30,000 new cells a
    frame); `AsteroidField.loading` waits for every window to be within 10 %
    of its range. Crossing a cell in a ring cost 230 ms, now a few ms;
  - `far` defaults to None (each rock as far as its size draws it); a far,
    or a window still growing, fades its class's rocks out over its last
    fifth, and the haze takes them in, class by class (no square, no
    fronts);
  - the haze between the camera and a rock veils it (`haze.glsl`, shared by
    the haze and the rocks): far rocks melt into the haze they leave to;
  - a native visible pass (`native/visible.c`, interface 12) over a compact
    copy of the rocks held, which also lists the big rocks near for the
    terrain rocks (they cost 19 ms a frame in a ring);
  - `instance_budget` counts rocks drawn in view (HIGH 80,000), and a new
    `held_budget` (HIGH 300,000) bounds the rocks held in every direction:
    a ring is drawn down to 2 pixels (it was 12);
  - a coarser mesh level (2 quads a face side) for rocks a few pixels wide;
  - what the haze misses per metre is tabulated each frame (one lookup a
    sample: the ring's frames 23 ms to 12);
  - only new rows of the instance table are written on the CPU (an upload
    cost 11 ms with 45,000 rocks);
  - `asteroid_bench.py` flies a ring (`--field ring`), reports rocks held
    and in range, and takes `--setting NAME=VALUE`.

- The terrain's fragment shader is split into includes other shaders can
  share, with no pixel changed: `terrain_material.glsl` (the biome texture
  blend, `layeredMaterial`) and `object_light.glsl` (sun, sky light and
  highlight, `objectLight`); it takes `aces_tonemap` from `display.glsl`
  instead of keeping its own copy. (#14)
- **Breaking:** `BodyLook.mass` is required (keyword only), for the gravity
  field; `Body.mass` reads it.
- Terrain chunks generate about 3x faster (~8 ms instead of ~24): the noise
  sums its octaves in a small native library (`native/noise.c`, plain C over
  ctypes), shipped in the platform wheels next to the collision library and
  built by `invoke native`. Without it the numpy noise runs, with the same
  results to ~1e-12. The weather map, the Milky Way glow and the sea level
  are computed faster through it too. (#20)
- Wanted chunks are generated by benefit: ground nothing covers yet first,
  then the chunks largest on screen, those in view before those behind
  (`Terrain.update(..., camera_forward=)`, passed by `World`). (#20)
- The worker processes start with the pool, during the caller's own
  start-up, and keep glibc's freed memory (Linux): the demo's first frame
  comes in ~3.8 s instead of ~6.9, its ground is loaded in ~12 s instead of
  ~22. (#20)
- The sky is drawn after the opaque ground, only where it shows (its own
  cull bin, `sky.SKY_BIN`, before the clouds'); `CloudSettings.shells` takes
  at most 6 shells (was 7). (#20)
- `OceanSettings.reflection_fov_scale` defaults to 1.1 (was 1.4): fewer
  chunks and billboards in the reflection, no clamped edges measured. (#20)
- Short sight lines through the air take fewer atmosphere samples (from 2,
  for nearby ground), within one 8-bit level of colour. (#20)
- Cloud tiles generate 2-10x faster: the clouds' volume noise runs in the
  native library too (bit for bit), and each tile draws as many candidate
  points as it needs. The clouds' set-up (weather map, deck light) runs in
  the worker pool, and the workers share the weather map instead of each
  building it. The demo's first frame comes in ~2.1 s, its world is loaded
  in ~9 s. (#20)
- Per-frame shader inputs of the ocean are bound once and written in place;
  the clouds' pixel angle and sky light are set only when they change. (#20)
- The sea is drawn by the terrain's own chunks, on the true sphere and with
  the terrain's LOD, instead of by a separate mesh under the camera (#20):
  - `world_builder.ocean.Sea` gives it its near look (the old ocean's waves,
    reflection, depth colour and foam) below `OceanLook.near_band`
    (12-18 km), crossfaded into the terrain's flat water above: the 15 km
    handover and its first-time shader compile are gone. Only chunks with
    water pay for the near look's shader.
  - The swell is a field of the planet-frame point, the same for every
    camera, drifting with the wind by slow rotations (no seams, no anchor
    jumps). It fades out with each chunk's resolution, so it flattens with
    distance through the LOD.
  - Collision and `TerrainQuery` follow the drawn water, swell included,
    instead of the sea-level sphere (`Terrain.set_swell`, which `Sea` calls
    every frame). The swell's CPU copy is `terrain/native/swell.h`, in the
    noise library (ABI 3) and the collision solid, with a numpy fallback.
  - Depth colour and foam read each vertex's own depth instead of seabed
    textures baked on a tangent plane, so they reach every distance the
    chunks resolve; the waves' phases are computed in float64 on the CPU.
  - The coastline is one mesh, without the depth bias that hid two meshes'
    seam.
- The terrain's chunks draw front to back in their own cull bin
  (`terrain.TERRAIN_BIN`, before the opaque bin): 1-1.6 ms less a frame low
  over the sea. Chunks that are all water stay out of the sea's reflection,
  and the near look's swell tilt comes from the vertices instead of being
  evaluated per pixel. (#20)
- The LOD pass runs in the native library (`lod.c`,
  `quadtree.NativeLodSelector`, picked by `make_selector`; the Python
  `LodSelector` stays, as the reference and the fallback): 0.5 ms a pass
  instead of 2.4 at the start pose, and in low flight 1.5 ms off the frame
  and a fifth off the refinement lag. Kept chunks are marked as recently used
  only when they are let go, not on every pass. (#20)
- A chunk's geometry is computed in the native library too (`chunk.c`: face
  directions, normals and slope, the vertex buffer and its box): a chunk
  takes 6.7 ms instead of 8 (median), and the refinement lag in low flight
  drops by another fifth. The planet's rules (curves, biomes, stamps, paints)
  stay in Python only. (#20)
- Terrain tiles hang under cell nodes about 32 km wide (`terrain.CELL_SIZE`),
  so an origin snap moves the cells instead of every tile: the 5-13 ms frame
  every 4 km of low flight is gone (p95 18.7 → 17.4 ms, the mean unchanged).
  `flight_bench.py` takes `--trace` (what the slowest frames spend) and
  `--no-hud`. (#20)
- The planet clouds' back-to-front re-sort runs in the native library
  (`billboards.c`, a radix sort writing the triangle indices): 1.45 → 0.75 ms
  a frame in low flight, p95 17.7 → 16.5 ms, the mean 13.3 → 12.5 ms. The
  numpy fallback sorts stably over the previous order too, so both give the
  same order. (#20)
- The planet clouds' tiles are built in the native library too, straight
  into Panda's vertex buffer (`wb_billboard_vertices`): 0.3-1 ms a tile
  instead of 0.6-3.6, the low flight's p95 16.25 → 15.9 ms and its update
  p95 8.5 → 7.8 ms. (#20)
- `CloudSettings.half_resolution` (off by default): the clouds drawn at half
  the view's resolution into one layer per split-depth range and laid over
  the scene (`clouds.half_res.HalfResClouds`; `World.use_split_depth(depth)`
  gives the world the split cameras it needs). 1-1.7 ms less GPU time low
  over the planet, ~1.2 ms more main-thread time: for GPU-limited setups.
  The demos take `--half-res-clouds`. (#20)
- The terrain's chunks share one render state: their own shader values are
  in a chunk table instead of inputs per chunk node, which made Panda draw
  each chunk in a state of its own. Low flight: mean frame 12.45 → 10.55 ms,
  p95 16 → 14.1 ms, refinement lag ~190 → ~150 ms (median); the images are
  unchanged (#20):
  - The table is a float texture on the terrain's root (`uChunkTable`), one
    slot per uploaded chunk, three texels each: centre and swell band; uv
    origin, uv step and skirt depth; level and face. Slots are freed when
    chunks are evicted, and the table doubles when it runs out.
  - Every vertex names its chunk's slot in `biome_b`'s last two bytes,
    written at upload (`mesh.build_chunk_node(chunk, margin, slot)`). The
    skirt flag is now `biome_b`'s top bit (the alpha byte >= 128); the
    generated chunk data keep 255 there.
  - The chunk nodes' `uChunk`, `uChunkCentre`, `uLevel` and `uFace` inputs
    are gone; `terrain.vert` fetches the values and hands the level and the
    vertex step to the fragment stage as flat varyings.
- The demos and scripts load `transform-cache false`: the camera-relative
  world gives every drawn node a new net transform every frame, so Panda's
  composed-transform cache only made garbage to collect. Low flight: mean
  10.65 → 9.6 ms, p95 14.25 → 13.05 ms. Recommended for games
  (docs/source/world.md). (#20)
- The terrain's store and node trims move a kept chunk they meet to the end
  of the LRU order instead of walking past every kept chunk each frame
  (0.42 ms each), and the chunk table copies only the new slots: low flight
  mean 9.75 → 8.65 ms, p95 13.3 → 11.65 ms. (#20)
- `Sea.update` costs 0.2 ms a frame instead of 0.47: the swell's drift has
  its constant terms computed once (`ocean.SwellDrift`, which
  `swell_rotations` now uses) and is skipped while the clock stands, the
  nadir frame is placed in plain floats, the wave phases are written in one
  go, and the reflection's fov and the water's light only on change. (#20)
- Planet clouds share the sky's tonemap (`displayColour` in the new
  `atmosphere/shaders/display.glsl`, which now holds `aces_tonemap` too;
  `sky_common.glsl` includes it). `CloudLook.exposure` and
  `SphericalClouds(exposure=)` default to 0.9, the sky's (was 1.15), and the
  clouds' sunlight takes the air's whole reddening (was half), so sunsets
  stay orange. From orbit, the clouds along the terminator are a paler
  orange, and twilight-lit cloud shows faintly inside it. The flat
  `CloudField` keeps its knee and its 1.15.

### Removed

- `world_builder.ocean.Ocean`, its swell mesh, `ocean/seabed.py` and the
  `ocean.vert` / `ocean.frag` / `seabed.glsl` shaders; use `Sea`, which a
  `World` builds from an `OceanLook`. (#20)
- `OceanSettings.seabed_cascades` and `swell_grid_subdivs`: saved settings
  that name them load with a warning. (#20)
- `OceanLook.handover`, `Terrain.set_sea_surface` and `World`'s
  `BodyOcean`; `Body.ocean` is now the body's `Sea`. `OceanLook.haze_color`
  and `haze_distance` are ignored (the atmosphere hazes the sea). (#20)

### Fixed

- Planet clouds toward the horizon, and toward the limb from above, looked
  orange even with the sun overhead. The air in front of them leaves lit
  tops a little warm, and the clouds' own knee (the hue kept at every level,
  no display gamma) turned that into orange. Seen from inside the deck, the
  far clouds' yellow and blue speckle is gone too
  (docs/source/atmosphere.md, clouds at the horizon).
- Chunks uploaded in the frame after an LOD pass waited for the camera to
  move or for the 15-frame fallback before being shown. (#20)
- `flight_bench.py`'s descent stopped only over dry land; over a calm sea it
  flew on to its time limit. (#20)

## [0.2.0] 2026-10-23

### Added

- `SplitDepthCameras.update(split, far, translucent=...)` keeps the split off
  big translucent volumes (a game's shield bubbles), which the two cameras'
  overlap band would otherwise blend twice into a ring (`split_clear_of`,
  `camera_span`). (#16)
- `world_builder.world.World`: every part a game draws around its camera
  (each body's terrain, air, planet clouds and swell ocean, the sky, the sun
  and Panda3D lights), built from the game's looks (`BodyLook`, `CloudLook`,
  `OceanLook`, `SunLook`, `SkyLook`) and updated with one
  `World.update(origin_world, far)` a frame; `scripts/demo_world.py` is the
  terrain demo built on it. (#16)
- Terrain collision for a game's own Panda3D `CollisionTraverser`
  (`Terrain.enable_collision`): a native collision solid over the drawn
  chunks answers ship, sensor and laser colliders with ordinary collision
  entries, one per collider, plus `material_at` for the ground at a hit. The
  Linux x86_64 and aarch64 (manylinux2014), Windows x64 and macOS
  (universal2) wheels ship it built; from a source checkout,
  `invoke native` builds it, and `invoke develop` does so too. (#16)
- `Terrain.loading`: whether the view still waits for ground, for a game's
  loading phase (undrawn ground is not solid either). (#16)
- Terrain on a true spherical planet: a cube-sphere with quadtree LOD,
  float64 positions re-based around the camera, and collision queries on the
  sphere. (#1)
- Several bodies (a planet and its moon) sharing one worker pool, and a
  near/far camera split (`world_builder.depth`) for views from the ground to
  orbit. (#1)
- `world_builder.atmosphere`: single scattering by air and haze, applied by
  every shader, with an analytic planet shadow, a twilight fill, redder
  sunsets on small planets (`sunset_radius`), and the light at any point
  (`Atmospheres.light_at`) for lighting anything else. (#2)
- `world_builder.sky`: a procedural star field (point stars, a Milky Way band
  and nebulae with live palettes) or a painted cube map, and the sun's disc,
  seen through the atmospheres. (#2)
- Planet clouds (`clouds.sphere.SphericalClouds`): billboard shells near the
  camera and a deck far away, one pattern from the ground to orbit, following
  a weather map (`clouds.weather`: circulation, cyclones, climate, terrain),
  their cost set by their settings (`CloudSettings`, #18). (#2)
- `Ocean.set_light`, to light the water as the sea under the camera (dark at
  night), and a setting for its reflection's resolution (now
  `OceanSettings.reflection_scale`). (#2)
- `ocean.aces_tonemap`, the NumPy twin of the shaders' tonemap. (#4)
- `clouds.noise.sun_optical_depth_on_sphere` (the clouds' sun march on the
  CPU) and `deck_light_table` (the far deck's averaged lighting). (#5)
- Scripts: `terrain_shots.py` (fixed poses to PNG, with GPU-bound frame
  times; `night_sea` checks the sea's reflection by night, [#4) and
  `flight_bench.py` (frame times in scripted fast flights). (#2)
- The terrain demo gains the moon, the atmosphere, clouds, the sky and
  Panda3D lights that follow the light at the camera (#2); and a white test
  ball ahead of the start pose, to judge the lighting on models (#5).
- Scene-graph tests on a live ShowBase (`tests/test_scene.py`). (#2)
- Documentation: [Lighting on objects](docs/source/object_lighting.md), how
  models are lit compared with the world and what would make them match.
  (#5)
- The ocean's seabed (`Ocean.set_seabed`, `ocean.seabed.SeabedMap`): sea-floor
  heights from the app, baked around the camera on a worker thread. They
  give the water a colour by depth (shared with the terrain's sea surface,
  `Terrain.set_shallow_water`), shoreline foam, and a
  swell that dies out over the shallows. The terrain demo feeds it from the
  terrain and keeps the game's 20 m swell. (#12)
- `Ocean.set_debug_mode`, with views of the wave filtering and the depth.
  `Ocean.set_swell_amplitude`, the swell's height live. (#12)
- A `beach` pose in `terrain_shots.py`: the shore nearest the reference
  point, up close. (#12)
- Quality settings for every part of a world, for a game's settings menu
  ([Quality settings](docs/source/quality.md)) (#18):
  - one frozen dataclass of knobs per part: `TerrainSettings`,
    `AtmosphereSettings`, `SkySettings`, `CloudSettings`, `OceanSettings`;
  - each with LOW / MID / HIGH / ULTRA presets (the shared `QualityLevel`);
  - each field described for a menu (label, range or choices, unit,
    live / rebuild / restart);
  - `settings=` and `set_settings()` on `Terrain`, `Atmospheres`, `Sky`,
    `SphericalClouds`, `CloudField` / `Clouds` and `Ocean`;
  - `world_builder.settings.WorldSettings` bundles them, with lenient,
    versioned `to_dict()` / `from_dict()` for a settings file;
  - discoverable without naming a knob: `WorldSettings.schema()` describes
    every one as plain data (id, label, kind, range or choices, unit, how
    it applies, default and per-level values), each object declares the
    section it takes (`SETTINGS_TYPE`), and `WorldSettings.apply(*objects)`
    / `settings_for(object)` hand them over; a renamed knob keeps reading
    its old id (`RENAMED`).

  New knobs:
  - **Terrain:** the finest detail (`min_level`: MID drops one LOD level,
    LOW three), the LOD distance, texture distance and filtering, its caches
    and upload budget, texture size, workers.
  - **Atmosphere:** the march's steps per pixel and per vertex (a uniform
    now, not a `#define`).
  - **Sky:** the share of stars drawn, the glow's size.
  - **Clouds:** each of the old level's parts on its own, their
    reflection, their tile budgets.
  - **Ocean:** its reflection's resolution and margin, and the swell mesh,
    which it remakes.

  The demo takes `--quality` and a `--<part>-quality` per part, as do
  `terrain_shots.py` and `flight_bench.py`.

### Changed

- Releases carry the Linux x86_64 and aarch64, Windows x64 and macOS wheels
  (`world_builder-<v>-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`,
  `world_builder-<v>-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl`,
  `world_builder-<v>-py3-none-win_amd64.whl`,
  `world_builder-<v>-py3-none-macosx_11_0_universal2.whl`) and the sdist,
  no longer a `py3-none-any` wheel: a game depending on a wheel's URL
  points at the new names, one per platform with markers. Panda3D is pinned
  to 1.10.16, the version the native library is built against. (#16)
- The atmosphere's step count is a uniform: HIGH frames ~2 ms faster at sea
  level (the loop is no longer unrolled), with the same image. The
  reflection camera's mask is `world_builder.depth.REFLECTION_CAMERA_MASK`.
  The ocean sizes its reflection from the main camera's display region, not
  a game's `graphics_manager`. (#18)
- The terrain demo keeps only its flight controls, the sun (J / K / I / U)
  and each part's quality preset (F1 - F5: terrain, air, sky, clouds,
  ocean; F6 all). (#18)

- Faster frames: worker processes at lower priority, one per physical core
  but one; a cheaper LOD pass, run less often; per-frame shader inputs
  written in place; cloud shaders that skip invisible billboards and clear
  air exactly. (#2)
- The terrain demo and scripts ask for no multisampling (Panda got ×16
  whatever was asked) and turn vsync off. (#2)
- The ocean's mirror shows the sky and the clouds. (#2)
- The sun's disc and the glow around it keep the colour of the light they
  cast: orange at sunset (the disc was white whatever its light), and so does
  their reflection on the sea. A low sun is drawn with a hotter core, a
  redder rim and a glare. (#5)
- The clouds' terminator is gradual instead of a hard line from lit to dark,
  close up and from orbit: their sunlight fades out toward the planet's
  shadow, and the still-lit air above lights them through twilight. (#5)
- The ocean's waves no longer alias. They fade by their size on screen
  instead of by camera distance, and what they lose is kept: the fresnel is
  averaged and the reflection blurred over the unresolved slopes. The
  `wave_fade_near`, `wave_fade_far` and `wave_fade_k2` arguments are gone.
  (#12)
- `TerrainQuery` groups many points by chunk 2.5–3 times faster. (#12)

### Removed

- Every way to set a part's cost but its settings (#18):
  - `CloudQuality`, `QUALITY_SCALES`, `at_quality`, `VOLUME_FRACTIONS` and
    `SHELL_OCTAVES` (use `QualityLevel` and `CloudSettings.preset`);
  - `CloudField(quality=, resort_frames=)` and its read of
    `game.app.graphics_settings.config["clouds"]["quality"]`;
  - `SphericalClouds(quality=, shells=, volume_fractions=)`;
  - `Ocean(wave_iterations=, reflection_scale=, swell_grid_subdivs=)`,
    `set_wave_iterations`, `set_foam`, and its read of
    `graphics_settings.config["render"]["reflection_scale"]`;
  - `Terrain(min_level=, workers=)`, `set_min_level`, `set_textured` and
    `set_reflected`;
  - the module constants that mirrored knobs (`STORE_SIZE`,
    `NODE_CACHE_SIZE`, the upload and LOD budgets, `TILE_CACHE`,
    `REFLECTION_FOV_SCALE`, `DEFAULT_REFLECTION_SCALE`,
    `REFLECTION_CAMERA_BITS`).

### Fixed

- An ocean drawn before its first update (e.g. behind a game's loading
  screen) failed on a missing `uSeabedCount` shader input: the seabed's
  count is now set when it is built. (#16)
- `SplitDepthCameras` drew its far camera into the window even when the
  scene renders into an offscreen buffer (a game's render-scale or
  anti-aliasing pipeline, e.g. `FilterManager`): it now draws into the near
  camera's own output. (#16)
- A chunk generated on the query path (`TerrainQuery` with `generate=True`,
  the default of `sphere_contact` and the point queries) went into the
  terrain's store without its height bounds: the LOD selector saw it as
  unknown, and evicting it later raised `KeyError` in `Terrain.update`. The
  query now hands such chunks to the terrain's own store. (#16)
- `TerrainQuery` point queries (`height_at`, `altitude`, ...) failed on an
  empty array of points. (#16)
- A planet seen from beyond 10,000 km was culled whole (Panda's
  `lens-far-limit`): the far camera gets explicit cull bounds. (#1)
- `hide(HIDE_FROM_NEAR)` did nothing: each split-depth camera now has a
  camera bit of its own. (#1)
- The ocean was not hidden from its own reflection camera (the same Panda
  camera-mask rule): it now hides from the whole `REFLECTION_CAMERA_MASK`.
  (#2)
- The reflection camera follows the main camera's field of view (it was
  copied once). (#2)
- The flat cloud field's vertex shader writes `gl_ClipDistance[0]`, so the
  ocean's reflection pass no longer clips it at random. (#2)
- The planet clouds' cull bins no longer tie with the flat field's (sort 25).
  (#2)
- `flight_bench.py` sampled the ground under the previous camera position
  when placing a low flight. (#2)
- The worker-priority test failed on Windows (no `os.nice`). (#2)
- The sea's reflection by night was about twice as bright as the sky: the
  reflection, already tonemapped, was tonemapped again. It is now decoded to
  linear light first, and the colour it is cleared to is tonemapped like the
  rest. (#4)
- Clouds along the terminator were blue-grey from far away and turned orange
  when closing in: the far deck's lighting is now the average of its
  close-up lighting, by the sun's elevation and the weather's coverage. (#5)

## [0.1.0]

The port from SpaceFlight: procedural terrain with biomes, stamps and LOD,
the reflective ocean and the volumetric billboard clouds, all three curved
by a camera-relative drop.
