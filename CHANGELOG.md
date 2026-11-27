# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Frame time at 2560 x 1440, HIGH, as a game runs it (CPU and GPU overlapping,
`frame_profile.py --pipelined`), on an RTX 3070 / Ryzen 5600X: `ground` 6.20 to
3.44 ms, `coast` 6.02 to 3.46, `sunset` 6.30 to 3.57, 20 km 3.85 to 2.10 (the
steps below; FRAME_COST_PLAN.md, performance.md §38-40).

### Added

- `world_builder.depth.ReversedDepthCamera` (#57): one camera over the whole
  view, reversed and infinite (near at 1, infinity at 0), with a float depth
  buffer's precision (about 1e-7 of the distance at every range): one cull
  and one draw of the scene. Its own camera under `app.camera` (the app's
  lens stays standard, for picking), a "greater" depth test, culling cut at
  the world's far distance, a fixed near plane (1 m; one following the
  altitude cut a game's cockpit and close ships). `update(far)`: the world
  gives it `far_distance(origin)`. A window's framebuffer has no float
  depth, so it draws into a scene buffer of its own (presented in the window, or in a
  game's post-process buffer, under the other regions; `scene_buffer=False`
  for an output with float depth already: `depth.float_depth_fbprops`). The
  scene buffer keeps its output's multisampling (`depth.scene_buffer_props`).
  It copies its depth into `depth_texture` after each frame.
- The depth convention's one source: `shaders/depth_convention.glsl` and
  `depth.depth_mode_input()` (`uDepthMode`, `DEPTH_STANDARD`,
  `DEPTH_REVERSED_01` with `gl-depth-zero-to-one`, `DEPTH_REVERSED_11`
  without: macOS), `depth.far_plane_test`, `depth.view_test`,
  `depth.reversed_lens`, `depth.zero_to_one`. The renderers that pin or read
  depth follow it: `Sky.set_depth_mode`, `Sea.set_depth_mode` (its mirror's
  camera reversed too), `FieldHaze.set_depth_mode`, the clouds.
- `World.use_depth_camera`, `World.far_distance`; `depth.planet_far`, `depth.system_far`.
- `AsteroidField.out_of_reach`; `MaterialSet.mean_rough_ao`.
- `scripts/frame_profile.py --pipelined` (frames as a game runs them; the
  default mode's frame is the CPU and the GPU added) and `--terrain-passes`
  (the terrain's cost in the view, `t_view`, and the sea's reflection,
  `t_refl`); a part `asteroids` (the fields' per-frame update).
- The tests run in either depth convention: desktop's 0-1 by default,
  `WB_DEPTH_ZERO_TO_ONE=0` for macOS's [-1, 1].

### Changed

- The air's march (`atmosphere.glsl`'s `atmoSegment`) runs its samples
  (`atmoStep`) in one of two loops: with the gathered eclipses, or with none,
  the eclipses' array then untouched (it cost 0.2-0.5 ms with no eclipse in
  view).
- Per-chunk Python in a terrain selection made one numpy pass: the
  collision's reach (`TerrainCollision.update`) and the trees' box distances
  (`Trees.update`), most of a selection's ~2 ms in SpaceFlight's Mission 3.
- `World(light_period=N)` spreads the light's reckoning over its frames:
  `lights_at` runs in `LIGHT_STAGES` (3) stages (the air's light; the
  eclipses and the clouds' shadow; the rings' and the mountains' shadows),
  one a frame at N = 3, instead of all of it every N-th frame (SpaceFlight's
  Mission 3: a ~1 ms bump every third frame, now ~0.25-0.42 ms each frame).
  The lights set are as of the job's first frame (two frames old at 3); a
  moved sun or a change in the tracked nodes still reckons them at once.
- `CasterMap.update` tests every caster's shadow size in one numpy pass
  (it made a few small arrays per caster): 197 to 129 µs an update with 16
  ships.
- `performance_backlog.md` cleaned up: the items about the billboard clouds,
  the split cameras and the clouds' depth passes (all gone) and the done ones
  removed; where the frame stands now (the demo, SpaceFlight's Mission 3: 11.7
  ms at 2560 x 1440, ~16 before) and the candidates left, with two new ones
  (the caster map's fit, the light's every-third-frame spike). Mission 3's
  history moved to performance.md §41.
- **One reversed-Z camera instead of the split cameras** (#57):
  `load_world` draws the view with a `ReversedDepthCamera` and hands it to the
  world (`World.use_depth_camera`). Games should set `gl-depth-zero-to-one #t`
  before opening their window (not on macOS: its OpenGL has no clip control;
  the world then keeps about a 24-bit range's precision). The demo and
  scripts do (`tools.window_config`). -0.2 to -0.25 ms low over the ground.
- **The clouds read the view's depth instead of drawing their own**: the
  march, its resolve, the composite and the ground shadow pass read last
  frame's (`ReversedDepthCamera.depth_texture`); their depth pass, which drew
  the scene again, is gone: -1.1 to -1.45 ms a frame low over the ground. In
  motion the clouds' edges against a moving silhouette are a frame late (a
  forward flight: the same error against converged frames as before).
- An asteroid field or ring out of the camera's reach does no per-frame work
  for its rocks once it has let them go (its haze still follows the camera):
  the demo's ring, 60,000 km away, took ~1.1 ms of every frame.
- A field's haze makes its table of the rocks not drawn only when the pixel
  scale, the smallest size drawn or the classes' limits change: 0.3 ms less a
  frame inside a field.
- The ground's materials (terrain, and rocks drawn with its textures) no
  longer read their normal, roughness and occlusion texture past 600 m, where
  the normal map has faded out: each layer's mean roughness and occlusion
  stand in (`uMeanRoughAO`): 0.17-0.36 ms of GPU low over the ground.
- The sea's reflection culls to its own standard range through explicit
  bounds (`depth.frustum_bounds`), whatever its lens.
- `PlanetClouds.use_split_depth` is `use_depth_camera`; the clouds' shaders
  read one depth texture (`uSceneDepth`) through one matrix (`uClipToRender`).
- `frame_profile.py --terrain-passes` measures `t_view` and `t_refl`.

### Removed

- **The split depth cameras** (#57): `depth.SplitDepthCameras`,
  `split_clear_of`, `camera_span`, `MIN_SPLIT_M`, `planet_ranges` /
  `system_ranges` (the far distance: `depth.planet_far` / `system_far`),
  `World.use_split_depth`, `World.depth_ranges`, `SPACE_SPLIT_M`, the haze's
  split passes (`FieldHaze.set_split`, `uHazeSplit`, `uHazeToFar`), the
  clouds' near range (its depth pass, card, `set_near`, `uSceneDepthNear`,
  `uNearDepthOn`, `uClipToRenderFar`). A game building the split cameras
  itself builds a `depth.ReversedDepthCamera` (or uses `load_world`).
- The clouds' depth pass and what served only it: `clouds.passes.pass_states`
  and `pass_camera`, the terrain's depth-only stage (`depth_only_shader`,
  `terrain_depth.frag`, `PASS_TAG`), the trees' `pass_state` and
  `IMPOSTOR_TAG` (and the impostors' coarser-target sprites:
  `uPointScale`), `depth.CLOUD_NEAR_BIT`, `CLOUD_FAR_BIT`,
  `HIDE_FROM_CLOUD_PASS`.
- `cloud_bench.py --passes` has no `-near` / `-depth` columns.

### Fixed

- `load_world` in an app without a window (`window-type none`: a game's
  headless runs) failed on `app.camLens`: there `GameWorld.depth` is None
  and bodies get no sea (both draw through the window); `update` and
  `destroy` work without them.
- `depth.frustum_bounds` is six planes (an `IntersectionBoundingVolume`),
  no longer a `BoundingHexahedron`: Panda carries the cull volume into each
  model's frame, where a hexahedron rebuilt its planes from corners out at
  the far distance in single precision. Past ~10,000 km (a far moon) they
  came out metres off near the camera, and culled a game's cockpit panels at
  the screen's edge, flickering as the far distance moved. The sea's
  reflection and SpaceFlight's mirror build theirs with it too.
- The clouds' ground shadow pass unprojected clip z = 0 for its footprint,
  infinity in reversed 0-1 depth: the cloud shadow's softness on the ground
  came out wrong with the reversed camera.

## [0.5.0] 2026-11-22

### Added

- World files: a whole world (bodies, asteroid fields and rings, sun, sky) in
  YAML, loaded in one line, `world_builder.load_world(app, "my_world.yaml")`,
  then `world.update(camera_in_the_world)` each frame
  (`world_builder.world_file`, [World files](docs/source/world_file.md)).
  A file is a definition only (no quality, camera or clock: those are
  arguments of `load_world`) with every knob of every object, no presets:
  planets (elevation, climate, stamps, paints), atmospheres, clouds with
  their global and pinned weather and genera, seas, the sky, fields (shape,
  density, sizes, spins, rock shapes, keep-outs, axis), mass or density per
  body. Any block may be `{file: other.yaml}`, a file of the world file's
  own folder, with sibling keys overriding it. `parse_world` reads a file
  without a window; `world_to_yaml` writes one.
- `world_builder.config`: the strict, reflective reader and writer behind
  it (`from_config`, `to_config`, `dump_yaml`), for any of the library's
  frozen dataclasses; errors name the key's place.
- Shipped worlds in `src/world_builder/worlds/`: `three_planets.yaml` (the
  home planet and its moon, an ice planet 3000 km across with air and
  clouds, and a ring from 5,000 to 10,000 km around it) and
  `deep_space.yaml` (the 15 km asteroid cube). They are the former presets
  written out in full, and a test keeps them equal.

### Changed

- `scripts/demo_world.py` is the one demo: it flies any world file
  (`--body --lat --lon --alt` or `--position`, `--quality`, `--frames
  N --shot FILE`). It was broken (it inherited from the terrain demo
  without running its constructor).
- The benches and shots (`frame_profile`, `flight_bench`,
  `vegetation_bench`, `terrain_shots`, `motion_shots`, `tree_gallery`,
  `cloud_bench`, the asteroid scripts) take their window settings and quality
  arguments from the new `scripts/tools.py` (which also holds the shared
  `local_sun`, `set_local_sun` and `settle`) and fly `WorldDemo`; the world
  they fly is a world file where it is fixed.

### Removed

- `world_builder.asteroids.spec.PRESETS` (`cube`, `icy_ball`, `ring`) and
  `_ringlets`: `worlds/asteroid_cube.yaml`, `asteroid_icy_ball.yaml` and
  `ring.yaml` define the fields in full; `world_builder.load_field("ring")`
  reads one. `asteroids.preview --field` and `asteroid_bench.py --field` take
  a file or a shipped name (they took a preset name). The tests read the
  files.
- `world_builder.terrain.planet.PRESETS` (`earthlike`, `earth`, `desert`,
  `forest`, `moon`, `ice`, `tiny`, `jupiter_rocky`): the library keeps no
  planet presets. `worlds/earth.yaml`, `desert.yaml` and `forest.yaml` (and
  the existing `kerbin`, `moon`, `ice`) define the planets in full;
  `world_builder.load_planet("desert")` reads one for the tools that draw a
  planet alone (`terrain.preview --planet`, `terrain_metrics.py --planet`,
  `weather_report.py --planet`, `find_landforms.py --planet`). The tests read
  the files too (`earthlike.yaml`, `tiny.yaml` and the others), through
  `load_planet` and `load_field`; a loaded planet is no longer resolved
  (`sea_bias` is computed by `Planet`), so `dataclasses.replace` on one is safe.
- `world_builder.terrain.level` (`load_level`, `level_for`, the level YAML
  format) and `terrain/levels/` (`demo`, `tiny`, `earth`, `jupiter_rocky`):
  a world file's `planet` block replaces them. The sample stamp images moved
  to `worlds/stamps/`; `scripts/make_sample_stamps.py` regenerates them.
- `scripts/demo_terrain.py`, `scripts/demo_asteroids.py` and
  `scripts/demo_genera.py`: the hand-wired demos, replaced by
  `demo_world.py` on world files. With them go the demo's test balls and
  ship, burst capture, march-block keys, genus toggles and sun keys.


## [0.4.2] 2026-11-21

### Removed

- The old cirrus sheet (cloud genera, Task 24): `clouds.cirrus`
  (`CloudSheet`, `CirrusSpec`, the stamps), its atlas
  (`clouds/cirrus_atlas.png` and `.json`), `scripts/bake_cirrus_atlas.py`,
  `scripts/cirrus_preview.py` and `shaders/cirrus.vert` / `cirrus.frag`.
  The march's high layer draws the weather's cirrus instead, 0.13-0.4 ms
  cheaper (performance §37, Task 24). `PlanetClouds.sheet`,
  `PlanetClouds.bind_inputs` (it bound the sheet alone) and its `game`
  argument go with it.
- The temporary `--high-placeholder` of the terrain demo and
  `terrain_shots.py` (Task 20) and the placeholder cirrus, with
  `march.thin_samples` and `THIN_TWO_SAMPLES`: every high genus is read
  once, at the sheet point.
- The billboard cloud renderer: `clouds.sphere` (`CloudVolume`, its tiles,
  shells and sprite atlas), `clouds.half_res` (`HalfResClouds`), the
  native `billboards.c`, and the `CloudSettings` knobs `renderer`,
  `count_scale`, `overlap_near`, `overlap_far`, `interior_growth`,
  `half_resolution`, `uploads_per_frame`, `resorts_per_frame` and
  `tile_cache` (a saved config holding them loads with them skipped), and
  `CloudSpec`'s `radius`, `aspect` and `deep_radius`. The low clouds are
  always raymarched (`clouds.march.CloudMarch`). The demos lose
  `--cloud-renderer` and `--half-res-clouds`, `cloud_bench.py` its
  `--renderer`, `--half-res` and `--interior-growth`.

### Added

- Crisp clouds from orbit: the march's far path draws a coverage as
  opaque masses with sharp, ragged edges and clear gaps (`macroCover`: the
  coverage moved by two octaves of the noise volume at 40 and 14 km, then
  stepped at a half, each octave fading in as the view pixel resolves it),
  their alpha following the weather's coverage under them
  (`macroThickness`), not a translucent slab of alpha 0.92 c through which
  the ground and its coastlines showed. The weather's mesoscale texture
  (`WeatherSpec.meso_wavelength`, 160 km) keeps its physical size on every
  planet: scaled by the planet's feature scale it was 1700 km on Earth.
  Debug bit 16384 turns the far masses off.
- Clear skies (synoptic weather, clear-sky step): the fair-weather
  background goes through a contrast (`weather.contrast`, a smooth step at
  `CLOUD_EDGE` 0.30), so the sky is clear or cloudy, not a thin cover
  everywhere, and `synoptic.rules.zonal_cover` drops its 0.15 floor to 0.10
  (ITCZ 0.55, storm tracks 0.45): the subtropics are clear. Tunables:
  `WeatherSpec.cloud_amount` (bias), `cloud_contrast` (the step's
  half-width, 0.10), `cirrus_amount`. The cirrus cover takes the same
  contrast, its floor 0.02 (was 0.08): patches, not a veil. On Earth the
  drawn low cloud's mean is 0.41-0.43, 37-39 % of the planet clear (13 %
  before), 32-35 % overcast, 70-73 % of the subtropics clear;
  `scripts/weather_report.py` checks these bands too (ISCCP's total 0.675
  counts the high and middle clouds not drawn). Widening the highs'
  subsidence was tried and changed nothing measurable: left as it was.
- Synoptic weather, part 8: `scripts/weather_report.py`, the Earth-scale
  planet's weather against observed climatology over 5 weather seeds and 5
  days (the jets' and the Hadley cell's latitudes, the lows' count and size,
  the subtropical highs on the eastern basins, the tropical cyclones' count
  and genesis, the circulation's sense, the warm front's cloud sequence);
  exit 1 on a failed band. All pass. `terrain_shots.py` takes the demo's
  `--time` and `--pin`.
- Synoptic weather, part 7 (the weather's genera drawn): the stratus genus
  is a family (`noise.GenusField.family`, `RECIPE_FAMILY`), the weather's
  stratus thickening into its nimbostratus where it says so, in one slab
  from 300 m under its base, 1550 m thick (`genera.STRATUS_FAMILY`: under
  the convective clouds' 2.2 km base). The nimbostratus's share, in cube
  A's channel 2 (`noise.FAMILY_SHARE_CHANNEL`, `genera.genus_share`), sets
  the cloud's extent and density (`noise.family_range`, `family_height`):
  the stratus at 0, a deep rain cloud at 1. `CloudSettings.weather_nimbostratus`
  (restart; off at the low and mid presets) draws it. The cirrostratus
  takes the weather's own (cube B's channel 2); the cirrus cover keeps its
  cirrus and cirrocumulus (`weather.CIRRUS_SHEET_GENERA`). The weather's
  altostratus is not drawn: as a genus of its own (altostratus to
  nimbostratus over 0.6-6 km, a first version) it shared both sky bands
  and cost 2-3 ms on the Kerbin-like planet, 6-8 ms on Earth. clouds.md,
  "Genera", has what the frontal shield is and why it is drawn low.
- Synoptic weather on the cloud genera (rebased on #59): the stratus
  genus's coverage is the weather's stratus and stratocumulus under its
  features (`genera.WEATHER_GENERA`, `genus_coverage`, `has_coverage`), the
  convective depth and anvil shelf the weather's, and the upper wind
  (`genera.upper_stream`, `upper_wind_at`) the weather's meandering jets
  (`weather.upper_stream`). The nimbostratus, altostratus and altocumulus,
  and the cirrostratus and cirrocumulus channels of the high layer, stay
  their features' alone (the weather's Ci, Cs and Cc feed the cirrus cover).
- Synoptic weather, part 1 (issue #8): the weather's scales follow the
  planet. `PlanetSpec` gains `day_length_h` (6 h by default) and
  `stratification` (N*H, 100 m/s); `clouds.synoptic.Scales.from_planet`
  derives the cyclones' size (Eady wavelength of the deformation radius),
  the storm-track wavenumber, the Hadley cell's edge (eddy-limited, as
  Omega^-1/2), the jets (Rhines / Williams spacing), one storm band per
  eddy-driven jet and the tropical cyclones' band and size, calibrated on
  Earth. A literature review with every equation of the model:
  `docs/literature/synoptic-weather-review.tex`.
- Synoptic weather, part 2: the weather as a closed-form function of time,
  `WeatherSpec.time_s` (no simulation, no state). `clouds.synoptic.jets`:
  each jet a meandering axis of three Rossby waves (long waves near still,
  short ones moving east), its troughs and ridges, and the upper wind's
  stream function along it. `clouds.synoptic.systems`: lows (Norwegian or
  Shapiro-Keyser), migratory, blocking and subtropical highs (on the
  eastern flanks of the ocean basins), cut-off lows and tropical cyclones
  (born over warm sea, recurving, dying at landfall), each on a
  closed-form track with a life cycle; a train of lows per storm band,
  born downstream of their jet's troughs and riding its wave.
  `clouds.synoptic.fields`: sea-level pressure (a zonal climatology plus a
  kernel per system, Holland's profile for tropical cyclones) and the wind
  (geostrophic, gradient wind in a tropical cyclone, friction at the
  surface, the trades near the equator), culled by cube-face tiles.
- Synoptic weather, part 3: the clouds come from the weather (closes the
  placeholder of issue #8). `clouds.synoptic.fronts`: each low's warm, cold
  and occluded fronts (Norwegian), or its fractured cold front and
  bent-back warm front (Shapiro-Keyser), shaped by its stage.
  `clouds.synoptic.rules`: the genera each zone holds: the warm front's
  cirrus, cirrostratus, altostratus and nimbostratus shield, the cold
  front's cumulonimbus line, the warm sector's stratus and
  stratocumulus, the comma head and the dry slot, the cold air's open
  cells, a high's clear core, stratocumulus deck over cold sea and fog, a
  tropical cyclone's eye, eyewall, spiral bands and outflow cirrostratus;
  all eight other genera are computed. `weather.synoptic_cube` holds
  them; `coverage_map`, `weather_cube` and `genera.genus_planes` read it
  (the convective family's coverage, depth and anvil shelf; the
  stratiform sheet's stratus and stratocumulus) and the cirrus sheet takes
  the systems' cirrus. The cirrus and the anvils follow the meandering
  jets (`weather.upper_stream`); the orographic cloud follows the
  systems' surface wind.
- Synoptic weather, part 4: local weather presets pinned at coordinates
  (`WeatherSpec.pins`, `clouds.synoptic.presets.WeatherPin`): a fair-weather
  high, a warm front approaching, a cold front, a squall line, overcast
  and drizzle, a hurricane, thunderstorms. A preset is a role relative to
  a weather system (the pin on the cold front of a mature low): the
  system is placed so, its track worked back to its birth, the jets
  refitted so a pinned low lies downstream of a trough and a pinned high
  under a ridge, and the free systems form round it, keeping clear of
  it. A pin the planet cannot hold raises with the reason (no ocean,
  latitude, land, cold sea, outside the storm track, dry air, pins too
  close); pins on the same low share it. Stable air under a warm front's
  shield and in its warm sector calms the fair-weather cumulus.
- Synoptic weather, part 5: `python -m world_builder.terrain.preview
  --weather` draws three maps per planet: the globe under all its cloud,
  the genera by colour, and a surface chart (isobars every 4 hPa, cold,
  warm and occluded fronts with their symbols, the jets' axes, the
  systems' centres: `clouds.synoptic.chart`). It and
  `scripts/demo_terrain.py` take `--time SECONDS` and `--pin LAT LON
  PRESET` (repeatable; a bad preset name lists the good ones); the
  demo's showcase features are left out when a pin asks for the weather
  there. The weather docs (atmosphere.md, clouds.md) describe the
  synoptic model.
- Planet presets `earth` (6371 km, a 23.934 h day: Earth's weather scales)
  and `jupiter_rocky` (69 911 km, a 9.925 h day: many narrow jets and small
  storms; too heavy for the demo yet), each with a level of its own
  (`terrain/levels/earth.yaml`, `jupiter_rocky.yaml`: the demo level's
  600 km radius no longer overrides theirs).

### Fixed

- A thin veil of a near cloud recoloured every cloud behind it (at
  sunset, red clouds far behind a yellow one turned yellow): the march took
  the light (the sun's colour through the air, twilight, eclipses:
  `cloudSunAndSky`) once a pixel, at its first cloud. It takes it again as
  the ray goes on, at a cloud sample past `max(growth × where it was last
  taken, that + a least distance)` (`lightStale`, the uniform
  `uLightRefresh`): `CloudSettings.light_refresh_growth` (presets 2× low,
  1.6× mid, 1.4× high, 1.25× ultra; 0 once a pixel; the demo's L cycles
  them) and `light_refresh_min` (2 km on a 600 km planet, scaled by the
  radius). The takes along a ray grow with the log of its far/near
  distance ratio, not its length: grazing views through a cloud field stay
  cheap.
- Clouds glowed grey on the Earth-scale planet's night side, against a
  black sky and sea: their sky light faded on a curve of the sun's height
  alone (`smoothstep(-0.3, 0.1)`), tuned on the 600 km planet, where the
  air above a cloud stays sunlit for many degrees after sunset. It now also
  takes the sunlit share of the air above the cloud, from the planet's
  shadow height and the air's scale height (`cloudSkyFade`,
  `noise.sky_light_fade`): with the sun 8° down, ~0.1 % on Earth (35 %
  before), ~90 % of the curve on the 600 km planet, whose twilight is
  unchanged. A render of every commit since #59 showed no regression: the
  Earth preset, new on this branch, had never been seen at night.
- Stratiform sheets stayed grey-white at sunset beside orange cumulus: the
  march's samples land 100-300 m into a sheet, under the tens of metres a
  low sun lights at its top, so their sun depth left it in shadow. A recipe
  genus (the stratus, its nimbostratus) takes its sun terms from its far
  table at every distance (`cloudRadianceFrom`; debug bit 8192 restores
  the per-sample ones).
- The march's loop over a band shared by several genera re-read each
  genus's weather at every sample when they alternated, and read the field
  of a genus without coverage: each keeps its weather now
  (`BAND_WEATHER`), and a genus without coverage is skipped first
  (2-4 ms at the ground with a deep slab sharing the bands). A family
  genus reads no noise outside its cloud's extent, and a genus without
  coverage where a band's chord starts no longer sets the band's cut top
  (`skyBandChord`).
- The terrain demo's clouds stayed sunlit through an eclipse (since the
  cloud genera, #59): the demo no longer bound the world's `Shadows` (the
  moon, the rings, the mountains) to the clouds' nodes, which kept their
  defaults (no casters). It binds them again (`PlanetClouds.bind_maps`, as
  `World` does), so `terrain_shots.py`'s eclipse poses show it too.
- The weather's population spacing check kept its systems in arrays
  rebuilt for every candidate: 2 s of a 9 s bake on a planet of thousands
  of systems.

### Changed

- The extratropical lows are smaller: their radius 0.6-1.05 of the
  cyclone scale (a quarter of the Eady wavelength), not 0.75-1.3, so
  1100-1950 km across on Earth, the observed 1000-2000 km (40 % were over
  2000 km: `weather_report.py`). The literature review's equation follows.
- `WeatherSpec` loses the placeholder systems' fields (`storm_tracks`,
  `cyclones`, `tropical` and their radii, swirls and strengths): the
  systems follow the planet. `zonal`, `jets` and `cirrus_zonal` default
  to None (the planet's own); `systems=False` gives the climate's
  background alone (`cloud_bench.py`'s uniform weather); `cirrus_shield`
  is now the share of the systems' cirrus the sheet keeps.
- `weather.vortices` and its comma-shaped `Vortex` are gone (the
  synoptic systems replace them). `stream_table`, `stream_vortices` and
  `stream_function` keep their signatures as the zonal-mean upper wind.

- The terrain demo's `B` key: a burst capture of the next 12 frames into
  `--capture-dir` (default `captures/`, git-ignored), a folder each. Per
  frame: the screen, and per march depth range the clouds' raw march (the
  traced pixels) and the resolve's output, read from their framebuffers
  (reading a render target's texture back froze the march for the rest of
  the run). Each frame's pose, block, still frames and cloud debug bits go
  in `meta.json`. While it runs the clock is held at the frame rate measured
  before it, so the camera moves as far a frame as in play despite the slow
  readbacks. Documented in clouds.md ("Testing"); its image helpers, the
  `-` / `+` block keys and `frame_profile.py`'s bench scenes are tested
  (`tests/test_demo_tools.py`).
- The terrain demo's `-` and `+` keys trace a smaller or larger share of the
  view's pixels a frame in the clouds' march (`CloudSettings.march_block`:
  1/16, 1/9, 1/8, 1/4 or all), live: the march rebuilds its passes, as a
  preset change does. The HUD's clouds line shows the share.
- `frame_profile.py --scene bench-sea|bench-dry`: a simple bench scene (the
  level's planet without vegetation, its mountains 600 m, with its reflective
  sea or without), profiled at the ground, 1.5 km and 10 km up. Results in
  performance.md §37. `TerrainDemo(spec_hook=)` changes the level's planet
  spec before it is built.

- The high layer's cirrus as procedural flow streaks (`clouds.md`, "The
  cirrus"), replacing the placeholder: detached streaks along the upper
  wind in patches, fibres in tapering bundles with fine strands, tufts
  (heads) with tails trailing upwind from 6 taps along the flow, hooked
  across it (the fall streak's shear), the domain warped by the curl
  noise at two scales. The upper wind has a form on the GPU
  (`genera.high_wind_cube`, `upper_wind_field`, `upper_stream`;
  `highWind`, an RGBA32F cube: the wind and its stream function, the
  streaks' across-the-wind coordinate), so the streaks curve with the jets
  and around the lows. Each part fades to its mean where a pixel cannot
  resolve it (`march.cirrus_means`), the fibres by the footprint across
  and along the wind apart (`march.footprint_along`): fibres across a
  grazing view stay sharp. The cirrus has real optics (depth 1.5, a faint
  halo) and is read once at the sheet point, like the cirrostratus and
  cirrocumulus. CPU mirror: `march.cirrus_streaks`, `cirrus_detail`,
  `cirrus_tufts`, `cirrus_patches`. The showcase gains a cirrus field far
  to the north (`genera.SHOWCASE_HIGH`, `SHOWCASE_HIGH_EYE`) and the poses
  `genus_cirrus`, `genus_cirrus_overhead`, `genus_cirrus_grazing`.
- The cirrus's heads are fibrous, not solid blobs (`clouds.md`, "The
  cirrus"). The tails' taps fan out toward the head (`march.CI_FAN`), so
  the fibres splay out there. The head is a wide, frayed window on the tuft
  noise, striated by the strands, at most 1.2 (`CI_HEAD_*`). It costs no
  extra fetch.
- `--cirrus` / `--no-cirrus` (the terrain demo, `terrain_shots.py`): the
  weather's cirrus in the march's high layer (`GeneraSpec.cirrus_weather`:
  `weather.cirrus_cover_map` fills cube B's channel 1), on by default, off
  by default with `--showcase`.
- `genera.weather_cirrus`: the high layer of a march without genera, the
  weather's cirrus alone, so a one-type march (`--no-genera`,
  `demo_world.py`, `cloud_bench.py`) draws the cirrus too.
- `CloudMarch.set_genus_visible` / `genus_visible` (and `PlanetClouds`')
  take a high genus's name (`"cirrus"`): the high genera have bits of
  `uGenusMask` past the low genera's (`march.HIGH_MASK_SHIFT`). The C key
  shows or hides the march's cirrus; the HUD says "march cirrus on/off".
- The high layer's cirrostratus and cirrocumulus (`clouds.md`, "The high
  layer"). The cirrostratus is a nearly uniform veil (optical depth 0.2)
  with slow swells 25 km across stretched 6 times along the upper jet,
  faint fibres, the sun plain through it, milky toward the horizon, and a
  22 degree halo in its phase (`haloRing`). The cirrocumulus is white
  grains of Worley cells 200 m across in rows across the jet, which fade
  to their mean where a pixel cannot resolve them
  (`march.cirrocumulus_grains`). Both read their pattern once where the
  ray crosses the slab's middle (`march.sheet_crossing`), with the pixel's
  footprint stretched by the slant (`march.slab_footprint`). Each high
  genus has optics of its own (`genera.HIGH_OPTICS`, `HighOptics`;
  `uHighExtinction`, `uHighCap`, `march.high_optical_depth`) and the layer
  a stretch axis (`genera.high_jet`). The showcase gains an overhead row
  (`genera.SHOWCASE_HIGH`, `SHOWCASE_HIGH_PITCH`) and the poses
  `genus_cirrostratus` and `genus_cirrocumulus`. `thinSlabRadiance` takes
  a halo strength.
- The clouds' high layer in the raymarch (`clouds.md`, "The high layer"): a
  sky band of its own above the genera (`genera.HIGH_SLAB`, 11-12.5 km;
  `GeneraSpec.high`, `GeneraSpec.high_slab`), crossed in one or two
  jittered samples with no sun march, sorted among the bands' stretches.
  Its genera (cirrus, cirrostratus, cirrocumulus) take coverage cube B's
  channels 1-3 (`genera.HIGH_CHANNELS`, `high_genera`). It is lit as a
  thin slab: `cloud_shading.glsl`'s `thinSlabRadiance`, which the cirrus
  sheet now shares. The shader variant compiles only when a high genus is
  covered (`march.march_fragment`; `cloud_march_high.frag`,
  `cloud_march_recipes_high.frag`). For now its density is a placeholder
  veil. `planet.check_layers` refuses a high slab under the genera's top
  or above the air. The demo and `terrain_shots.py` gain `--no-cirrus` and
  a temporary `--high-placeholder` (poses `high_cb`, `high_overhead`,
  `high_grazing`, `high_above`).
- The cloud genera showcase: `scripts/demo_genera.py` (the terrain demo's
  `--showcase`), every genus alone (`genera.showcase_features`,
  `GeneraSpec.background`) in two rows: the cumulus species along the
  northern horizon (`genera.SHOWCASE_ROW`: humilis, mediocris, congestus,
  cumulonimbus) and the stratiform and mid-level genera along the southern
  (`genera.SHOWCASE_SOUTH`: stratocumulus, stratus, nimbostratus,
  altostratus, altocumulus, 38 km out; `showcase_slots`). Keys 1-9 show or
  hide each genus of the march (`PlanetClouds.set_genus_visible`; the
  cumulus species are one genus), and `terrain_shots.py --showcase` has a
  `genus_<name>` pose per feature (a cumulus from 1 km over the start; a
  low sheet and the altocumulus from under it; the altostratus against the
  sun). Cumulus humilis and mediocris features (`genera.GENUS_DEPTHS`).
- `docs/source/cloud_genera_research.md`: the ten cloud genera and the
  density functions the literature draws them with.
- Recipe genera: every genus but the convective one evaluates one generic
  density (`genusRecipeDensity` in `cloud_field.glsl`,
  `noise.recipe_density_on_sphere` on the CPU) from its preset's knobs
  (`GenusPreset.recipe`, `noise.GenusField`: value fbm, Worley blobs
  `"worley"` or closed cells `"cells"`, horizontal and vertical stretch, a
  profile, billowy or wispy erosion, `erosion_base` to keep it to the
  base), its coverage calibrated on its own column peaks, erosion included
  (`noise.recipe_column_peaks`), its far table its own
  (`noise.recipe_deck_light_table`), and the CPU's cloud shadow marching it
  as the shadow maps do (`noise.recipe_shadow_on_sphere`,
  `field_setup.genus_sun_depth`). A tileable Worley F1/F2 volume
  (`noise.worley_volume`, the `cloudCells` texture) and a second coverage
  cube (`genusCoverB`). A genus preset may override its type's field
  (`GenusPreset.field`) and optics (`GenusPreset.optics`). The march
  compiles the recipe only where a genus needs it
  (`cloud_march_recipes.frag`; the bakes' `CLOUD_RECIPE_SHADOWS`).
- Five recipe genera, all in `GeneraSpec.genera` by default and marched only
  where a feature holds them (each non-convective feature fills its own
  genus's coverage map, `genera.genus_coverage`):
  - stratocumulus (cube A channel 1), 0.8-1.8 km: closed cells 1.5 km
    wide (Worley `F2 - F1`, apart however high the coverage), domed,
    flat-based, billowy;
  - nimbostratus (cube A channel 2), 0.6-5 km: a thick, dark, featureless
    layer that blots out the sun, its base ragged;
  - altostratus (cube A channel 3), 4.7-6 km: a thin grey sheet stretched
    along east, the sun dimly through it;
  - altocumulus (cube B channel 0), 3.5-3.8 km: a thin lens of closed
    cells 350 m apart with billowy erosion: small rounded patches, shaded,
    regularly arranged with small clear gaps, 1-5 degrees across from the
    ground;
  - and the stratus, now on the recipe (Changed).

- `AsteroidField.reach(camera_world)`: how far a field's rocks can still be
  drawn (its largest rock at the settings' `min_pixels`, at most the field's
  far side).

### Changed

- Cloud colours: the stratiform genera and the cumulonimbus were far too
  dark outside.
  - The sheets now scale their sunward depth as the cumulus do: stratus
    preset `shadow_strength` 0.35 (was 1), nimbostratus 0.4 with
    `multiple_scattering` 0.35 (was 0.05). The altostratus keeps a physical 1.
  - The sky's light under a deep column turns neutral grey (it was the blue
    floor that made thick sheets' bases navy).
  - A cumulonimbus's tower takes half its sky as open sky under its anvil.
  - At noon (display levels): stratus top 200 → 226, underside 87 → 132;
    nimbostratus top 190 → 211, face from outside 67 → 99; cumulonimbus sides
    157 → 169. The cumulus are unchanged.

- Far clouds keep their lumps. The march no longer fades features finer
  than a share of the step's length (`uStepFade`, `march.STEP_FADES`, the
  `genusDensity` step-length argument: removed); the view pixel alone
  decides what a sample resolves. The billows and fine octaves now survive
  tens of kilometres further, so towers and cumulus are no longer smooth
  blobs. Far clouds also shimmer less against a full-resolution reference
  (`cloud_bench --jitter`: error 6.15 → 2.51, flicker 3.28 → 2.32 at
  `start`). Costs 0.1-0.25 ms.

- Far clouds' edges: where a pixel no longer resolves the fine octaves, the
  billows or a recipe genus's octaves, cells and erosion, their spread now
  widens the threshold's window (`edgeWidening`: the expected density over
  the noise a pixel spans) on top of the fade to their means. Thresholding
  the mean alone drew far clouds with smooth, sharp outlines; their edges
  are now softer and frayed. The march's view only: the bakes and the CPU
  mirrors keep the plain fade. `uNoiseSigma`, `uBillowSigma`, and the cell
  spreads in the deck entry's row 14 (`noise.cell_spreads`). Debug bit 4096
  turns it off. No measurable cost.
- Far clouds keep their own self-shadowing and powder as far as their shapes
  (footprint 0.15-0.6 of the coarse feature), reading the light volume
  alone out there. Before, they faded to the far table's averages from a
  quarter of that (about 110 km at 2560x1440 for the cumulus). About
  +0.05 ms, within the noise.

- The cirrus defaults: `PlanetClouds(cirrus=True)` (was a `CirrusSpec`)
  draws the weather's cirrus in the march's high layer, with or without
  genera; `CloudLook.cirrus` is a bool (default False; was a `CirrusSpec`
  or None), and `demo_world.py` sets it. `planet.check_layers` takes
  `cirrus` as a bool: the sheet's rule is gone, the high slab's applies
  whenever the cirrus is drawn. `PlanetClouds.high_slab` gives the high
  layer's slab, which `World.depth_ranges` reaches. `cloud_bench.py`'s
  `--no-cirrus` turns off the march's cirrus.
- The march takes the high slab's crossings out of its sorted segments:
  the one on the way down before every band, the one on the way up after
  (a ray's height is convex, the slab above every band). The high variant
  cost the low clouds 0.1-0.2 ms more with them sorted among the bands
  (performance §37, Task 23).
- `CloudSettings.shells` is `octave_bands` (a saved `shells` loads as it);
  `clouds.sphere`'s helpers the march uses moved to `clouds.field_setup`,
  `clouds.deck` and `clouds.passes`.
- Cumulus congestus grow into connected towers that reach their top
  (`DensityField.deep_exponent`, `deep_erosion`, `cell_power`), and the
  cumulonimbus's anvil and shelf are smooth (`noise.anvil_weight`,
  `ANVIL_SCALE`).
- The genera are one union march (`CloudMarch(genera=...)`): every genus is
  marched in one pass over the sky bands (`march.sky_bands`, the genera's
  slabs merged; at most `noise.MAX_GENERA` genera and `MAX_SKY_BANDS` bands),
  so genera may overlap: `planet.check_layers` no longer keeps the stratus
  under the convective clouds; it checks the genera's and bands' counts and
  the cirrus against the union's top (`PlanetClouds.top_radius` too). Every
  genus keeps all its `octave_bands` (no more halving).
- One light volume and one shadow map pair hold all the genera's summed
  depth: `PlanetClouds.shadow_families` is removed; `shadow_caster()` and
  `bind_shadow_inputs(node)` take no family; `CloudMarch.genus_caster(g)` is
  one genus's part on the CPU, `CloudMarch.genera_count` how many (was
  `families`). `CloudLightVolumes.volume` and `CloudShadowMaps.pair` are
  single (were the lists `volumes` and `pairs`).
- `ShadowSettings.cloud_light_slices` counts slices of each sky band a frame.
- The stratus is a recipe genus: a grey, uniform sheet of 10 km value fbm
  stretched three times east, a soft top, a flat base, ragged (wispy) near
  its base only, its outline sharper; its coverage calibrated with its
  erosion in (`noise.recipe_column_peaks`).
- The march takes only the genera holding cloud somewhere
  (`genera.genus_specs(covered=True)`): an empty genus sharing a band
  cost the demo 0.65 ms. `planet.check_layers` still counts them all.
- A slab bridging two groups of slabs (the nimbostratus between the low
  sheets and the convective clouds) splits its sky band at the gap
  (`march.sky_bands`; `band_masks` takes each genus reaching into a band):
  merged, the low sheets took a 9.9 km band's light-volume slices and bake
  steps.

- The far plane reaches the asteroid fields' rocks (#39):
  `World.depth_ranges` returns the larger of the bodies' far distance (or
  `SPACE_FAR_M` in open space) and every field's `reach`. In open space the
  far plane was a fixed 1,000 km, so the biggest rocks were cut while
  still a few pixels wide: a line where the drawn rocks met the haze (the
  `ring` preset seen from 80 km over its plane: now 3,440 km, the line
  gone, no cost seen). With bodies the far plane still follows their
  horizon; it only grows. `demo_terrain.py --ring` does the same.

### Fixed

- Clouds shimmered at rest, worst on thick sheets (nimbostratus) and with
  large march blocks. The cause was the light volume's read jitter (a texel
  and a slice either way, new each visit), also applied to the depth-up read
  behind the sky's reach and the diffused light's fade. A sample near a
  sheet's top or edge read a much deeper column, which made black dots in
  the raw march; they moved every visit and nudged their blocks in the
  resolve. At rest in front of the nimbostratus (4x4 block), unjittering
  that read took the frame-to-frame change from 0.34 to 0.07 levels.
- The jitter is now gone from the sun-depth reads too (the stratus at rest,
  the cumulonimbus in motion still shimmered: with the jitter off alone,
  the stratus's frame-to-frame change 0.22 → 0.09 levels, the cumulus's
  0.45 → 0.26). It hid streaks along the sun on an anvil and on tower faces,
  whose cause was the light window's two reads: the far one, taken where
  the sun ray goes, blended other texels than the near one, and their
  interpolation errors did not cancel in the difference. The far read is
  now taken straight above the near one (`cloudLightSunDepth`): no streaks,
  no jitter; a tower's face a little flatter. The two reads then differ only
  by their slice, and the linear read's kinks at each slice drew bands along
  the heights on an anvil's underside: the volume is now read across its
  slices with a cubic B-spline (`cloudLightAt`, four slices inside the
  band; clouds 4.53 → 4.63 ms at `genera_cb`, 5.54 → 5.37 ms at the anvil
  pose, within noise). The exact fix, a chart sheared
  along the sun, is in BACKLOG.md.

- The clouds traced at every pixel (march block 1x1) were grainy all over
  in motion: the resolve took a quarter of each pixel's raw sample a frame,
  without the smooth mean the larger blocks take. At 1x1 it now takes the
  smooth mean in motion (`march.SMOOTHING`, 1) and a sixteenth of the frame
  at least (`march.MOVING_WEIGHTS`). Flying toward the cumulonimbus
  (`motion_shots.py`, `genera_cb`): grain 4.97 → 1.24, specks 6.5 % →
  0.53 %, error against the converged view 3.02 → 1.43 (HIGH's 4x2: 0.92,
  0.11 %, 1.23). ULTRA's 2x2 (half the smooth mean, a quarter a frame) gets
  the same: grain 1.33 → 0.88, error 1.52 → 1.36. Nothing changes at rest.

- The sea reflected only the cumulus family: the reflection's deck was genus
  0's alone. Every low and middle genus now has its own deck at its slab's
  middle (`deck.deck_nodes`; `cloud.frag` shades it as that genus:
  `uDeckGenus`, its weather, density and far table, none when hidden),
  drawn highest first. The high layer is still not reflected.
- The marched clouds ignored eclipses, the mountains' shadow and the rings':
  only the billboards' shader took them, and the march never did. The
  march's light (`cloudSunAndSky`, once a pixel, and the high layer's
  crossing) now multiplies them in, as every other receiver does. The world
  binds the shadow inputs to the march's passes and decks through
  `PlanetClouds.bind_maps`.
- The march's cirrus, two artefacts seen flying at its height:
  - **A second, offset layer.** Two causes:
    - A crossing of the high slab that never meets its middle (a camera
      inside the slab, looking away from it) read the patterns at the
      crossing's middle, another height for every ray; it now draws nothing
      (`sheetCrossing` returns -1, `march.sheet_crossing` None).
    - The showcase also drew the old cirrus sheet under its own cirrus.
      `--cirrus` now defaults to none with `--showcase`.
  - **Stairs along the fibres up close.** The upper wind's stream function
    went through a cube map's 8-bit filter weights, steps of about 30 m. It
    is now a 2D array of the faces, blended in float (`highWindAt`).
- A faint contour across a thin cloud sheet seen at a grazing angle (the
  altostratus): the march counted a ray opaque at transmittance 0.04 and
  snapped its alpha from 0.96 to 1 there; the light let through now eases to
  0 over 0.04-0.08 (`march.opaque_residual`, `opaqueResidual` in
  `cloud_march.glsl`).
- A field's haze ended abruptly at the depth split (#39): only the far
  camera drew it, from the split on, on the assumption that every rock
  nearer is drawn. A ring's dust never is, and from orbit the split sits
  hundreds of kilometres out (halfway down to the ground: 248 km at 500 km
  up). The near camera now draws the haze up to the split
  (`FieldHaze.set_split`, which the field calls when the app's camera is a
  split's near one); both passes cut the rays at the same plane, given from
  Python (read from the near lens's projection in floats it was off by
  kilometres: a seam).
- Rocks beyond the split were veiled twice over the stretch before it (by
  their own veil, then by the near camera's haze pass): their veil now
  starts at the split (`haze.glsl`'s `hazeVeilStart`).

## [0.4.1] 2026-11-14

### Added

- Physically based shading for the game's models (`World.track_light(node,
  shader=True)`), derived from panda3d-simplepbr's BRDF (GGX, Smith-GGX,
  Schlick) in the world's units, with every world shadow, the air and the
  tonemap as before: glTF's metal and roughness, emission, normal maps (or a
  frame from the screen without tangents), and the sky reflected from the
  sky-view table. A quality knob, `ModelSettings.shading` (the new `models`
  section): LOW to HIGH PBR with normal maps and sky reflections; ULTRA
  (`PBR_FULL`) adds glTF's occlusion maps (read back from the .gltf:
  panda3d-gltf drops them), the sky's reflection blurred by the roughness
  (the sky-view table is now mipmapped) and the light the ground bounces
  up. The cheaper tiers (the terrain's light, PBR, PBR with normal maps)
  stay choices of the knob.
- Physically based ground (`TerrainSettings.pbr`, on at every preset):
  the land lit by the models' BRDF (now `terrain/shaders/pbr_light.glsl`),
  a dielectric with its biome textures' roughness and occlusion; nothing
  measurable in cost. The trees take the same switch (a leaf's waxy
  specular over their translucent diffuse), the asteroids their own
  (`AsteroidSettings.pbr`, on at every preset).
- `QualitySettings.DISTINCT_PRESETS`: a section whose cheaper presets
  coincide reads as the highest level it equals.
- `terrain_shots.py --model PATH`: the `ship` and `ship_fill` poses, a
  game's model whole and filling the screen; the terrain demo's F8 cycles
  the models' preset, and it hangs SpaceFlight's tie-interceptor beside the
  white test ball (`--test-ship`, `--no-test-ship`).

### Changed

- The far sea is no longer a sharp mirror (#55): the rough fresnel and the
  reflection's blur take the unresolved slopes the grazing flattening hid
  (capped at 0.4), so far water reflects less and blurrier, its colour
  showing through, and the seam with the rippled near water fades. LOW
  (no fresnel table) takes them as a steeper Schlick look.

### Fixed

- glTF models were decoded to linear light twice: panda3d-gltf loads their
  colour textures as sRGB, and the model shader decoded them again (too
  dark, too saturated). `objects.mark_model` marks the parts authored in
  linear light (on each geom's state); the base colour is now a PBR
  material's `baseColor` (its `diffuse` is darkened by its metalness).

## [0.4.0] 2026-11-12

### Added

- Vegetation placement, Serious Sam 4's way (VEGETATION_PLAN.md V1,
  `docs/source/vegetation.md`): a vegetation density per vertex (procedural,
  painted over by a level's `density:` paints, `stamps.DensityPaint`), the
  ground's convexity, the trees and the small plants on jittered grids whose
  cells derive from the baked crowns (`trees/crowns.json`), per-group rules
  (instances per hectare by biome, slope, convexity and altitude ramps).
- The trees one by one (`world_builder.terrain.trees`, `terrain.trees`): the
  plants placed on each chunk drawn as instanced meshes near the camera
  (`VegetationSettings.mesh_range`) and as impostors out to `tree_range`
  (one point per plant, drawn from views baked by
  `scripts/bake_tree_impostors.py`), crossing over a dithered band; rocks and
  bushes are meshes only. `scripts/build_tree_assets.py`, `tree_gallery.py`
  and `vegetation_bench.py` make and look at them.
- The far forest (V2): Bruneton and Neyret's shader-map in the ground's
  shader, from tables measured on our own trees
  (`scripts/bake_forest_tables.py`, `terrain.far_forest`), fed by four new
  bytes per terrain vertex (the trees' cover, the far groups' densities); the
  forest floor under the trees, grass in the clearings. The terrain's vertex
  record is now 40 bytes (`chunkgen.VERTEX_DTYPE`, the native `chunk.c`:
  rebuild the noise library).

- `scripts/frame_profile.py --at LAT LON ALT HEADING PITCH` (repeatable):
  more poses, from the demo's printed camera line.

- `World(light_period=N)`: the light at the camera and the tracked nodes
  (`lights_at`, the CPU's march through the clouds' shadow) reckoned every N
  frames, at once when the sun moves or the tracked nodes change. 1 (the
  default) is every frame, as before.
- `world_builder.atmosphere.tables`: the atmospheres' precomputed tables
  (Hillaire 2020): the sunlight's transmittance (256 x 64, read by
  `atmoTransmittance`) and the light scattered more than once (32 x 32,
  `atmoMultiple`), built on the CPU when an atmosphere is added;
  `inscatter_reference`, a CPU reference for a view ray.
- The sky-view table (`world_builder.atmosphere.sky_view.SkyView`,
  `Sky.attach_view`): the air from the camera to space for every direction,
  baked each frame inside an atmosphere; the sky, the sea's reflection of it
  and the stars read it instead of marching (from space they still march).
- The aerial-perspective cells (`world_builder.atmosphere.aerial`,
  `Atmospheres.attach_aerial`, `AtmosphereSettings.aerial`): the air in front
  of the terrain, sea and models from a 32 x 32 x 32 grid over the view baked
  each frame, instead of a march per pixel (within 128 km, the camera inside
  an atmosphere; marched elsewhere, as before).
- `AtmosphereParams.ground_albedo` (0.3): the ground's bounce in the
  multiple scattering.
- An ozone layer in the air (`AtmosphereParams.ozone`, `ozone_altitude`,
  `ozone_width`; Hillaire's Earth values): it absorbs orange and green, so
  the twilight sky stays blue instead of olive. In the view rays, the
  multiple scattering and the sunlight (its depth toward the sun marched
  into the transmittance table's alpha; `transmittance_to_sun` includes it).
- The clouds' light volume (`world_builder.clouds.light_volume`,
  `ShadowSettings.cloud_light_size` and `cloud_light_slices`; RENDERING_PLAN.md
  part C): the clouds' optical depth toward the sun and up for every voxel of
  their slab around the camera, baked a slice or two a frame when the sun turns
  or the camera moves far. The raymarch lights its clouds by it in place of the
  shadow maps' levels, takes the sky's reach from its depth up in place of the
  height ramp, and far away (`march.LIGHT_LOCAL_FOOTPRINT`) its sunlight from it
  alone, without the sun march's local steps (`uCloudDebug` 512 back to the maps
  and the ramp, 1024 the whole chord). Off at LOW, 128 at MID, 256 at HIGH and
  ULTRA.
- God rays (`AtmosphereSettings.god_rays`, on at HIGH and ULTRA): the clouds'
  and the mountains' shadows in the air of the aerial cells and the sky-view
  table (`god_rays.glsl`; `Atmospheres.god_ray_root`, `bind_god_rays`):
  shafts under broken cloud and down valleys at a low sun, across the sky too.
  `frame_profile.py` has a `godrays` column.
- A frame-rate cap (`world_builder.display`: `DisplaySettings.frame_cap`,
  off (0) by default, 90 on a 180 Hz display; `FrameCap`, which `World` and the demo make): the
  `display` section of `WorldSettings`, a choice kept by every preset
  (`QualitySettings.LEVELLED` False; the schema's sections say `levelled`).
  Capped at a divisor of the display's refresh, every frame shows for the
  same time: a compositor held a 6.9 ms frame for one 5.5 ms refresh or two,
  and turning stuttered (performance.md §27). The demos and benchmarks take
  `--frame-cap N` (the benchmarks 0 by default).
- `flight_bench.py --flight turn --at LAT LON ALT HEADING PITCH --sun AZ EL`:
  turning on the spot; the trace times the light volume, the cloud maps, the
  atmospheres, the sky and the shadow maps too, and the report has p99.

### Changed

- HIGH terrain drops one level of the finest LOD (`TerrainSettings.min_level`
  defaults to 1, about 3.6 m of vertex spacing); ULTRA keeps the finest (0).

- Each terrain chunk carries a coarse mesh over its own vertices (every
  other one: `chunkgen.triangle_indices(COARSE_STRIDE)`, a quarter of the
  triangles), drawn instead of the full one by the clouds' depth passes and
  the sea's reflection (`mesh.COARSE_PASSES`); `build_chunk_node` returns a
  node holding the two. Up to 0.3 ms less a frame at 2560 x 1440.

- The clouds' shadow on the terrain and the sea, with the raymarch, from one
  pass at the depth passes' resolution (`clouds.march._GroundShadow`,
  `PlanetClouds.bind_ground_shadow`, which `World` and the demo call): the
  ground reads it back by distance, per pixel where none agrees and in the
  sea's reflection. 0.36-0.70 ms less a frame at 2560 x 1440. The terrain's
  debug view 6 shows the clouds' shadow alone.

- The raymarch's resolve in motion: each frame takes at least about a cycle
  of the block (`march.MOVING_WEIGHTS`: 1/8 at 4 x 2), and the clamp's
  statistics come from a seamless window (the 4 x 4 samples about a pixel,
  weighted by a radial bump that vanishes two samples away;
  `march.stats_window` mirrors it). The 33-frame history had lagged clouds
  whose look changes fast in flight, and the clamp had cut that lag along its
  box's jumps between the block's cells: a mosaic of 4 x 2 blocks and grain on
  shaded cloud. Specks in motion 12.9 to 6.8 % and 17.0 to 6.4 % at the
  user's poses, +0.05 to 0.13 ms (performance.md §30). `STATS_REACH_MOVING`
  is gone.

- The raymarch's steps no longer lengthen as a ray's light is spent in a
  cloud. That made the steps' grid follow what the samples found, and the
  visits' jitter left its contours: bands across a tower's face, strongest
  behind a thin veil. 0.36 ms more over a stratus deck at 2560 x 1440
  (performance.md §29). `march.STEP_DEEPENING` is gone.

- The demo and the scripts sharing its options (`terrain_shots.py`,
  `motion_shots.py`, `flight_bench.py`, `frame_profile.py`) draw the clouds by
  genus by default: `--genera` is on, `--no-genera` for the plain field
  (`cloud_bench.py` keeps its own opt-in `--genera` and pose sets).
  `motion_shots.py` and `frame_profile.py` now honour `--start` (they
  ignored it), and `motion_shots.py` takes the sun from the start pose as
  the demo does.

- The raymarched clouds' multiple-scattering fill and second order fade with
  the cloud above (`1 / (1 + 0.2 * depth)` of the light volume's column up):
  a cumulonimbus's base and its anvil's underside are dark grey at noon (0.57
  of a lit cumulus's brightness, was 0.74); thin cumulus and dusk unchanged.
  `--cloud-debug 2048` for the old look.
- The sea decodes its reflection with the inverse of `displayColour`
  (`displayInverse`), not of the per-channel tonemap: the sky and the clouds
  keep a bright colour's hue, and decoded per channel a lit cloud's slight
  warmth came back far redder (red up to 2.9 times its blue for a true 1.11).
- The light volume's read jitter moves on at each sample along a ray (was one
  draw a ray): the error over grey clouds in motion 1.55 to 1.41 levels.
- The raymarched clouds in motion: the composite's depth is no farther than
  the scene's (the march stops there), the history's distance is held within
  the fresh samples', and the history survives a jump small against the
  clouds' distance (`march.HISTORY_JUMP_SHARE`). Flying forward the lagging
  distance put the clouds behind the ground: 90 % of a stratus cut, its edge
  sliding against the mountains, half a descent's clouds missing; now 0.3 % and
  1.6 % (performance.md §28). The clear-sky early-out tests the 3 × 3 samples.
  In motion the resolve keeps a quarter of the clamp's floor and takes this
  frame's samples as their smooth mean: a tower is no longer grainy while the
  camera moves (the grain 1.47 to 1.11 of the still look's), and spreads its
  samples' statistics wider in motion (`STATS_REACH_MOVING`).
- The light volume's reads are jittered by a texel and a slice either way
  (`LIGHT_JITTER`), not half: the dark specks on far towers in motion (2.7 %
  of the cloud off its still look, 1.07 % now).
- The light volume is smoothed once baked (`cloud_light_smooth.frag`, a
  [1, 2, 1] tent along its three axes, one frame a bake, into a third atlas):
  the remaining specks on far towers' shaded sides gone (0.71 % of the cloud
  off its still look at the first pose, 1.25 % at the second, the rest on the
  deck's edges).
- `CloudSettings.march_refresh_moving` (advanced, off): the raymarch traces
  twice the pixels while the camera moves (`march.MOVING_BLOCKS`).
- `scripts/motion_shots.py`: the clouds in motion against converged, flown from
  a pose with the demo's speed rule at a fixed frame time (`--flicker` too).
- The frame cap is off by default (`DisplaySettings.frame_cap` 0).
- The raymarch's steps: each stretch's first step is a random share of its
  length (per pixel and visit), and the field's fine octaves fade by the step's
  length from its distance alone, not the step deepened in a cloud: both drew
  bands up a tower's face seen level (weaker now; the rest is the step budget,
  RENDERING_PLAN.md "After C").
- The clouds' two finest fbm octaves are turned off the noise's axes
  (`noise.OCTAVE_TURNS`, `OCTAVE_3_TURN` / `OCTAVE_4_TURN` in
  cloud_field.glsl), as the billows already were: the value noise's lattice
  planes drew stacked slices and interference lines across a tower's face seen
  at a grazing angle. The clouds' fine detail is a different pattern; their
  coverage is unchanged (the calibration follows `noise.fbm`).
- The air's light scattered more than once comes from the multiple-scattering
  table instead of a flat share of the sunlight: the planet's shadow at dusk
  is lit by the sky above it, and the low sky by day is brighter (the
  ground's bounce). `AtmosphereParams.multiple_scattering` is now that
  light's scale (1, physical), no longer a share (0.12). The per-pixel march
  reads the sunlight's transmittance from its table (cost even); the
  multiple-scattering read and the ozone cost ~0.2-0.7 ms at 2560 x 1440,
  HIGH, until the sky-view and aerial-perspective tables take the march off.

- The raymarch's two scene-depth passes are drawn at half the view's
  resolution per axis (`march.DEPTH_SCALE` 2): about 0.45-0.55 ms less a
  frame in SpaceFlight's Mission 3 at 2560 x 1440; static shots identical
  but for a few pixels.
- The raymarch's billows are bent by a tileable curl noise
  (`march.CURL_STRENGTH` 2, `noise.curl_noise_volume`): wisps and small
  fragments curl into tufts. +0.0-0.15 ms at 2560 x 1440, HIGH; steadiness
  unchanged.
- The clouds take a second order of scattering (`CloudOptics.octave_gain`
  0.3, `octave_attenuation` 0.5, every cloud type): sunlight that scattered
  once more sees half the sunward depth, its phase half-way to isotropic, as
  EVE-Redux's multiple-scattering lobe. A share of the first order's light,
  so lit tops keep their brightness and shadowed sides open up. The far
  clouds' table (`deckLight`) is RGBA: the second order's column mean in
  alpha. `octave_gain` 0 is the old look.
- `CloudSettings.march_resolution` (per axis: 0.25, 0.5, 1.0) is now
  `march_block` (bx, by): the march traces one pixel of each block a frame.
  Besides 4 x 4, 2 x 2 and 1 x 1 (a quarter, a half, full) it takes 4 x 2 (one
  pixel in 8) and 3 x 3 (one in 9), EVE-Redux's default range, in orders where
  consecutive frames never trace touching pixels (`march.block_order`).
  **HIGH now traces 4 x 2** (it was 2 x 2, which ULTRA keeps): its clouds cost
  about 60 % of what they did (0.4-2.9 ms less at 2560 x 1440), a little softer
  in motion (performance.md §24). Saved settings keep their block
  (`RENAMED`; "Quarter", "Half" and "Full" are the same choices).
  `cloud_bench.py --march-block 4x2` (`--march-resolution` still takes 1, 0.5
  or 0.25).
- The raymarched clouds march once over both depth ranges of the split
  cameras (EVE-Redux's single march), not once per range: a ray runs from the
  near range's near plane to its surface, else the far range's, and stops
  once opaque, so a cloud near the camera spares the march what lies behind
  it. Each range keeps its own step budget. Two cards lay the one result over
  the view (the near range's over the pixels its depth pass drew). Clouds cost
  0.3-1.8 ms less at 2560 x 1440, HIGH, where the camera is in or near cloud
  (performance.md §24); the look is unchanged. `cloud_bench.py --passes` has a
  `-near` column in place of `far` and `near`.
- Fewer render states made per frame for the tracked models: the model
  shader's attrib is only set on a node when it differs (`ModelLighting.apply`).
- The cloud shadow maps and the terrain's shadow bake are not drawn again for
  a sun that moved less than 0.05° since they were (`utils.SUN_TOLERANCE`,
  `sun_moved`): a slowly turning sun redraws them every few frames, not
  every frame.
- The terrain's collision range (`Terrain.enable_collision(max_distance=)`)
  defaults to 20 km, not 100 km: every drawn chunk within it is checked each
  frame.
- The cloud march's per-frame inputs (its matrices, offsets, history flags)
  are bound once and written in place, and `ModelLighting.update` gives the
  shader the body's rotation only when it turned: fewer render states made
  each frame.
- `World.update` reads the tracked nodes' positions once, keeps the bodies'
  centres and radii, and the caster map compares its casters by key (no
  quadratic list scan).

### Fixed

- A stutter every 16 frames in the demo: its HUD, naming each settings
  section's preset, built every preset anew, and each settings object
  resolved its class's type hints (~0.3 ms each, ~6 ms a refresh). The hints
  are now resolved once per class (`quality._type_hints`); the frame's p95 at
  a still pose fell from 17.7 to 14.3 ms.
- Trees in clouds: the raymarch's depth passes draw the trees too (their
  impostors sized in the pass's pixels, `trees.pass_state`), and the resolve
  weighs each sample by how near the scene it stopped at is to the pixel's
  (`uSceneDepthNear`, `uSceneDepthFar`): a tree in a cloud no longer stands
  unfogged in front of it or shows a halo of sky around it. The impostors'
  facing and the artifacts where cloud met terrain are corrected.
- The tree population's tuning (`forest.SPECIES`) changed `TREE_CELL`; the
  far forest's tables are baked again for it.

### Removed

- The ray-marched canopy (`vegetation.Canopy`, `terrain.canopy`, the
  `canopy*` and `vegetation_cells` shaders, the ground's shell variant
  `terrain_veg.frag`): the trees one by one (`terrain.trees`) and the far
  forest draw the vegetation. `VegetationSettings` keeps `enabled`,
  `mesh_range` and `tree_range`; `terrain.vegetation` replaces
  `terrain.canopy`. Saved settings with the old knobs still load (they are
  dropped).

## [0.3.0] 2026-11-04

### Added

- Clouds by genus (`world_builder.clouds.genera`,
  docs/source/clouds.md#genera; #45): stratus, and cumulus growing with the
  convective depth through congestus to cumulonimbus, drawn by billboards.
  `PlanetClouds(genera=GeneraSpec(...))`, `CloudLook.genera`, `--genera` in
  the terrain demo, `terrain_shots.py` and `cloud_bench.py` (its genera
  poses). The maps are a placeholder for the weather: features around one
  point.
  - `CloudType.CONVECTIVE` and `DensityField.min_top`, `anvil`,
    `cell_weight`: a field's top rising with the depth, its anvil, a cell
    octave gathering towers; `CloudSpec.deep_radius`: billboards growing
    with the depth. Neutral by default: the other types are unchanged.
  - The cumulonimbus's anvil spreads as a smooth lens to 3 radii of its
    tower (`noise.SHELF_BAND`, `shelf_lift`, `shelf_fbm`, `with_shelf`,
    `DensityField.shelf_shift`): a deep column's top over its own spread
    map, two octaves only, its top fading in density. It streams downwind
    on the cirrus's upper wind (`genera.anvil_spread`, `upper_wind_at`,
    `GeneraSpec.wind`).
  - The weather cube may carry the depth (a third channel) and the anvil's
    spread (a fourth); the coverage's fbm shift is tabulated by depth.
  - Expensive near the big clouds (docs/source/performance.md §21, §22):
    half resolution makes those frames 2.5-2.9 times faster on a weak GPU.
- `CloudSettings.interior_growth` (on at LOW and MID): billboards deep
  inside a cloud grow and thin as the cube, the same extinction for less
  overdraw.

### Changed

- The raymarch is the default low-cloud renderer at every quality preset
  (`CloudSettings.renderer = "march"`; billboards stay selectable, a
  rebuild). MID now marches at a quarter of the view's resolution (40 steps).
  The sea's reflection still shows the deck alone (no shapes, no stratus);
  a cheap mirror march is described in docs/source/clouds.md#the-raymarch.
- Faster raymarch (docs/source/performance.md "Phase 4.2"): the billows share
  the column's noise, the resolve returns early in clear sky, the march stops
  at 4 % transmittance and its steps grow in dense cloud (about 0.4-1.2 ms at
  1440p, half resolution). A quarter resolution meets the cost target
  (2.1-2.3 ms plain, 2.5-3.1 ms genera); HIGH and ULTRA keep a half.
- `CirrusSpec.base` defaults to 11 km (was 8 km): over the cumulonimbus's
  cap, 10.5 km.
- The LOW and MID cloud presets draw the clouds at half resolution
  (`CloudSettings.half_resolution`): 2.5-2.9 times faster frames near big
  clouds on a weak GPU. HIGH and ULTRA keep full resolution.

- Raymarched low clouds, now the default renderer at every preset
  (`CloudSettings.renderer = "march"`; `"billboards"` stays selectable; `world_builder.clouds.march`,
  docs/source/clouds.md#the-raymarch; #45).
  - `CloudMarch` draws the billboards' field, weather, lighting and far
    field per pixel, from the camera to the scene's depth: a game's model
    inside a cloud is hidden by just the cloud in front of it, and the
    camera inside a cloud sees it all around. It reaches orbit with no deck.
  - One pixel of every 2 × 2 block (4 × 4 at a quarter resolution: LOW and MID) is traced each frame and gathered into
    the full resolution by reprojection (`march_resolution`, `march_steps`,
    stepped by the quality presets). The result is laid over the scene at
    the cloud's own depth, so models cut it at full resolution.
  - 1.3-2 times the billboards' cost over open sky on a weak GPU, the same
    inside the slab, two-fifths in a dense cloud
    (docs/source/performance.md §20).
  - The raymarch draws the genera (`CloudMarch(genera=...)`, one pass of two
    families, `march.family_segments`), and `PlanetClouds` no longer forces
    billboards with them; both families cast their shadows.
  - The raymarch is steadier at every preset: a blue-noise jitter at the march's
    pixel (the gradient noise at the view's stride put a lattice of diagonal
    stripes in motion), features finer than a step fade to their means (the far
    shimmer), the resolve's statistics and reprojection distance are smooth
    across a block, and a pixel on fresh samples takes their smooth mean
    (`march.blue_noise`, `march.SMOOTHING`). `cloud_bench.py --jitter`,
    `--passes` and `--steps-sweep` measure it; debug bits 128 and 256.
  - The raymarch casts a cloud shadow on the ground, sea and models, as the
    billboards do (`CloudMarch.shadow_layer`, `shadow_field`,
    `bind_shadow_inputs`, `sun_visibility`), and reads the cloud shadow maps'
    levels for its own lighting (`PlanetClouds.bind_maps`; the maps' texels
    hold the depth a quarter, a half and three quarters of the way along the
    chord, `noise.cloud_shadow_on_sphere(levels=True)`).
- `Knob.kept`: a settings knob that is a choice rather than a level of
  quality. Moving to a preset (`WorldSettings.with_level`,
  `with_section_level`, `QualitySettings.at_level`) keeps it, and it makes
  nothing custom. The clouds' renderer is one: cycling the quality presets
  no longer switches the raymarch back to billboards.
  - `PlanetClouds.use_split_depth` hands it the split cameras (`World` does
    it in `use_split_depth`); `PlanetClouds.set_visible` hides either
    renderer; a change of renderer rebuilds the volume live.
  - The billboards' lighting moved, unchanged, into
    `clouds/shaders/cloud_shading.glsl`, which both renderers share.
  - `scripts/cloud_bench.py` times and shoots the clouds alone (no terrain)
    with either renderer, still or moving. Its `--interactive` mode flies a
    probe ball in and out of the clouds. `terrain_shots.py` and the terrain
    demo take `--cloud-renderer`.

- High cirrus over a planet's clouds (`world_builder.clouds.cirrus`,
  docs/source/atmosphere.md#cirrus).
  - `CloudSheet`, built by `PlanetClouds(cirrus=CirrusSpec())`, or
    `CloudLook(cirrus=...)` in a `World`: baked patches of cirrus uncinus
    stamped on one sheet at 8-9 km. Each patch has cottony tufts along a
    curved front, comet tails that curve, widen and merge, and strands of
    uneven width and strength along every curve.
  - `scripts/bake_cirrus_atlas.py` bakes the 24 patches into
    `clouds/cirrus_atlas.png/.json`; `scripts/cirrus_preview.py` draws the
    atlas, or the placed field beside a reference photo, for tuning.
  - The stamps are a Poisson field, thinned by a cover map of the cirrus's
    own (anvils over the wet tropics, streaks along the jets, a shield ahead
    of each low), so that the share of sky under a stamp is the map's. Each
    has its tufts downwind along the upper wind (the jet streams, bent
    around the weather's cyclones). Neighbours stay parallel: their heading
    wanders together, and stamps fade out where the wind turns along their
    length.
  - It draws behind every cumulus where sight lines leave its sphere and in
    front of them where they enter it, so the order holds from the ground to
    orbit and in the sea's mirror, with no sorting.
  - `PlanetClouds` (and `World`, before building anything) refuses cirrus
    that is not above the clouds or not inside the air.
  - `weather.py` gains the upper wind (`WeatherSpec.jets`, `upper_wind`,
    `stream_table`, `stream_vortices`) and the cirrus cover
    (`cirrus_cover_map`).
  - The terrain and world demos draw it (C toggles it); `terrain_shots.py`
    gains the `cirrus_up`, `cirrus_between`, `cirrus_above` and
    `cirrus_dusk` poses.
  - Its cost is not measured yet: see docs/source/performance_backlog.md,
    "Cirrus".
- `world_builder.shadows`: what hides the sun from a point, for every
  receiver (docs/source/shadows.md). Its first term is the eclipse: each
  body hides the sun's disc for the others, with a physical penumbra
  (`shadows.glsl`'s `eclipseVisibility`, the covered share of the sun's
  disc). It darkens the terrain and far water, the cloud billboards and
  deck, the air's samples (the sky and the aerial perspective), and the
  sky light (down to `ECLIPSE_SKY_FLOOR`). `Shadows` feeds the shared
  uniforms, as `Atmospheres` does; `Shadows.eclipse` is its CPU twin, which
  `World.light_at` (Panda3D's lights, the sea's near look) now applies: an
  airless moon in its planet's shadow goes dark. `Shadows.set_sun` works
  out which bodies' shadows can reach which, so a view with no eclipse
  costs about what it did (within ~0.2 ms). (#13)
- Cloud shadows (#11): the planet's clouds shade the terrain, the far
  water and the near sea, marched per pixel through the deck's density
  field toward the sun (`cloud_field.glsl`'s `cloudShadowDepth`, its CPU twin
  `noise.cloud_shadow_on_sphere`), so the shadow follows the clouds drawn
  and the sun. `CloudVolume.bind_shadow_inputs(node)` lets a node's
  shaders read them, and `PlanetClouds.bind_shadow_inputs` its volume's,
  again after a change of renderer (`World` binds each body's clouds to
  its terrain; a raymarched volume casts none yet);
  `CloudVolume.sun_visibility` (and `PlanetClouds'`) is the CPU side, in
  `World.sun_visibility` and `lights_at`. The march takes the genera's
  weather (convective depth, cell octave, anvil shelf) where its chord
  enters the slab. `PlanetClouds.bind_inputs(bind)` gives every layer's
  shader more inputs (the eclipses: `World` binds them so). `ShadowSettings`
  (`cloud_shadow_steps`) is a new `shadows` section of `WorldSettings`.
  `Sea.set_light(light, sun=...)` takes the sun's share apart, which the
  shadow dims per pixel. Far away and under a grazing sun the march gives
  way to its mean (`noise.ground_shadow_table`), which neither sparkles
  nor bands. F7 in the terrain demos cycles the shadows' preset.
  `terrain_shots.py` gains `clouds_noon`, `clouds_low` and
  `clouds_grazing`.
- Mountain shadows (#11): each body's shadow ceiling cube map
  (`world_builder.terrain.shadow_map`: per texel the lowest altitude from
  which the sun clears every occluder, and how far the occluder is),
  baked in the worker pool for each sun in one pass over a sun-aligned grid
  (~2.3 s at 1024), its heights sampled once. `ShadowMap` (one per body in
  `World`, `Body.shadow_map`) swaps it in when ready; shadows.glsl's
  `terrainShadow` reads it with a physical penumbra, on the terrain (the
  moon's craters too), the near sea and the clouds; its CPU twin is in
  `World.sun_visibility` and `lights_at`. `ShadowSettings.terrain_map_size`
  sets its size. `terrain_shots.py` gains `mountains_low`,
  `mountains_orbit` and `moon_terminator`.
- Nearby ridges' shadows (#11): `terrain.shadow_map.LocalShadowMap`, a 50 m
  shadow map around the camera (owned by each body's `ShadowMap`) for the
  occluders nearer than the global map's reach, sampled in the pool as the
  camera moves a quarter of it and baked for each sun (~1 s);
  `terrainShadow` takes the lesser of the two maps.
  `ShadowSettings.local_map_size` and `local_map_spacing` set it.
- The clouds' shadow from maps (#11): `clouds.shadow_map.CloudShadowMaps`
  marches it once per texel of two maps around the camera (by where the
  sun's ray enters the clouds' base) instead of once per pixel of ground,
  drawn again when the camera has moved a quarter of a map or the sun has
  turned. `ShadowSettings.cloud_map_size` sets them; `World` makes them for
  a body with clouds (`Body.cloud_maps`).
- Long cloud shadows (#11): with a low sun, each cloud billboard looks
  past its own march toward the sun for the clouds kilometres away that
  shade it (`cloud_field.glsl`'s `cloudFarDepth`, per vertex, its CPU twin
  `noise.far_cloud_depth`). `ShadowSettings.cloud_far_reach` sets how far
  (0: off). `terrain_shots.py` gains `dusk_in_clouds` and
  `dusk_above_clouds`.
- Models drawn as the world is (#13): `world_builder.objects`, a model
  shader that decodes a model's colour, lights it as the terrain lights its
  ground and tonemaps it alike, with the air in front, and shades it per
  pixel by every world shadow (`model_light.glsl`'s `modelSunlight`: the
  eclipses, the body's air, clouds and mountains). `World.track_light(node,
  shader=True)` draws a model with it, by the inputs of the body nearest it
  (`Body.models`, `ModelLighting`). On an airless body the sun sets at its
  own horizon with a penumbra (`shadows.glsl`'s `ownShadow`, CPU twin
  `Shadows.own_shadow`). The demos' test balls (now a white and a textured
  one) use it; M switches them to Panda3D's lighting. `terrain_shots.py`
  gains `ball_noon`, `ball_sunset`, `ball_eclipse`, `ball_cloud_edge`,
  `ball_mountain_edge` and `--panda-lights`.
- The mountains' shadow maps are baked on the GPU (#11): with a window,
  `ShadowMap` (`output=`, which `World` passes) marches each texel of the
  global cube map and each node of the local map toward the sun in a
  fragment shader (`terrain.shadow_bake`), a band of rows a frame: a new
  sun's maps within a few frames, not seconds, and the worker pool left
  to the terrain. The CPU twins march the texels around a point
  (`march_ceiling`, `march_local`). Without a window, the pool bakes as
  before. `sky.starfield.texel_directions`.
- Ring shadows (#13): a ring dims the sun crossing it, on its planet's
  terrain, sea, clouds, air and models: a band of shadow with its ringlets
  in it (`Shadows.add_ring`, `shadows.glsl`'s `ringVisibility`, CPU twin
  `Shadows.ring_visibility`; its depth from its rocks and dust,
  `asteroids.haze.ring_depth`). `World` adds every ring field's. The
  bodies' shadows fall on the rocks and the haze (a planet's across its
  ring). `terrain_shots.py` gains `ring_shadow_ground`,
  `ring_shadow_orbit` and `ring_planet_shadow`.
- Objects cast shadows (#13): a model tracked with
  `World.track_light(node, cast=True)` shadows the terrain, the sea and
  the models the model shader draws, itself too, through a sun-space depth
  map fitted every frame to the casters' bounds
  (`world_builder.shadows.casters.CasterMap`, `World.casters`; read by
  `shadows.glsl`'s `casterShadow`, 3x3 hardware comparisons). Its pass
  draws the casters alone (camera bit 20, `depth.CASTER_BIT`).
  `ShadowSettings.object_shadows` and `caster_map_size` set it. The demos'
  balls cast; `terrain_shots.py` gains `caster_pad`,
  `caster_ship_on_ship`, `caster_self` and `caster_altitude`.
- `World.track_light(node, cast=False)` / `untrack_light(node)`: a game's
  model lit by the light where it is, not at the camera: its own Panda3D sun
  and ambient lights (`world_builder.lights.TrackedLights`), set every
  `update` in one vectorised call. `World.lights_at(points)` and
  `World.sun_visibility(points)` give the light and the world's shadows at
  many points at once (`Atmospheres.light_at`, `transmittance_to_sun`,
  `in_shadow`, `twilight` and `depth_toward` take arrays). The demos' test
  ball is tracked. (#13)
- `SunLook.angular_radius` and `Sky(sun_angular_radius=)`: one sun size, the
  disc the sky draws and every penumbra (`shadows.SUN_ANGULAR_RADIUS`,
  0.55°). (#13)
- `scripts/terrain_shots.py`: a pose's sun can be aimed at a body
  (`AimedSun`), and four eclipse poses: `eclipse_ground`, `eclipse_total`,
  `eclipse_orbit`, `lunar_eclipse`. (#13)
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

- A planet's clouds are one part, `world_builder.clouds.planet.PlanetClouds`,
  with two layers named by how they are drawn:
  `clouds.sphere.SphericalClouds` is now `CloudVolume` (the low clouds:
  billboard shells and the far deck), and the cirrus is
  `clouds.cirrus.CloudSheet`. `PlanetClouds` takes `SphericalClouds`'
  arguments plus `cirrus`, drives both layers (sun, sky light, settings,
  pixel angle, half resolution, update, removal) and checks the cirrus
  against the low clouds and the air (`planet.check_layers`). In a `World`,
  `Body.clouds` is a `PlanetClouds` (the volume: `body.clouds.volume`) and
  the cirrus is `CloudLook.cirrus`.
- More realistic planet relief (#9). The same seed now gives a different
  planet, and the demo level moved to keep a coast to its north and a range
  to its south:
  - tectonic plates (`terrain.plates`, `ElevationParams.plates`, 14 by
    default, none on the moon): mountain belts where plates close, mid-ocean
    ridges and rift valleys where they open, long straight coasts where they
    slide, and continents pulled onto plate edges. The noise mask still
    raises a few ranges inside plates;
  - ranges are massifs: a summit level that rises and falls along a range,
    and a crest profile with broad valleys and sharp ridges instead of
    `crest**3`, so a range is mostly mountain rather than a few spikes;
  - plains: a regional character field scales hills and detail, from flat
    plains to rough country (flat land at 100 m: 0-4 % of earthlike land
    patches before, 30-55 % now);
  - coasts: a regional coast style, smooth (the shore follows the
    continental field) or rugged (islets and inlets);
  - desert landforms on arid land (`terrain.landforms`, each a toggle in
    `ElevationParams` and only computed where it is): sand seas of
    Sandastray's dunes (its nine layered noises, combined by min, add and
    multiply: rounded dunes with sharp crests, 100-150 m high) on every
    arid plain, at any height (most plains of the desert planet), and
    tablelands in some upland regions: tables on a warped Voronoi noise cut
    by valleys that widen with a regional erosion stage, from narrow
    canyons to plains with rounded mesas, buttes and spires, their walls
    shaped by the planet's rock strata (`terrain.strata`: hard layers in
    cliffs with ledges shedding gently outward, soft ones in slopes, a
    talus at the foot; the wall rises straight on from the talus' top) and
    tinted by them in bands (a strata mask in each vertex's sixth biome
    byte). All band limited: coarse chunks keep their mean height.
    Tableland chunks cost about 4 times the others (23 ms against 6 ms);
  - rock starts at 35 degrees, not 30 (`biomes.ROCK_SLOPE`);
  - cost: fine chunks as before; coarse chunks (level 7 and up at 600 km,
    which evaluate the low band directly) about 25 ms instead of 13 ms; the
    pool's throughput in `terrain.bench` 340 chunks/s instead of 399.
    `plates=0` saves most of it.
  `scripts/terrain_metrics.py` measures flatness, slopes, coast roughness
  and peaks per preset. `scripts/find_landforms.py` finds places over each
  desert landform (sand seas, canyons by depth, mesas by relief), and the demos (`demo_terrain.py`, `demo_world.py`,
  `terrain_shots.py`) take `--planet PRESET`, `--seed N` and
  `--start LAT LON`.
- The terrain demo's moon is 21,000 km out (was 10,000 km): as wide in the
  sky as the sun, so it can eclipse it. (#13)
- The terrain demos (`demo_terrain.py`, `demo_world.py`, and the shots and
  flight bench built on them) take `--ring`: a small ring of dust and rocks
  around the planet, in its equator's plane, 1.5 to 2.1 radii out
  (`ring_spec`; off by default, so runs compare with earlier ones).
  `terrain_shots.py --ring` adds its poses, `ring_ground` and `ring_orbit`.
  (#13)
- `sky.SUN_ANGULAR_RADIUS` moves to `world_builder.shadows` (`sky` still
  imports it). (#13)
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

- The flat-world parts: the flat cloud field (`world_builder.clouds.field`:
  `CloudField`, `CloudLayer`, `Clouds`, `lod_shells`; `shaders/cloud.vert`;
  `scripts/demo_clouds.py`) and what only it used:
  - `cloud.sample_field_particles`, `snap_to_noise_period`,
    `volume_fraction` and `CloudSpec.volume_fraction`;
  - `noise.density`, `column_peaks`, `resolve_field`, `measure_coverage` and
    `streak_direction` (the planet's are `density_on_sphere`,
    `sphere_column_peaks` and `with_threshold`);
  - `CloudSettings.resort_frames` (saved settings that name it load with a
    warning);
  - the shaders' flat mode (`uSpherical`, `earthCurvature`, `horizonFade`,
    `hazeColor`, the soft-knee tonemap) and `pack_layer_params`'
    `wrap_radius` and `wrap_fade_band`;
  - `world_builder.PLANET_RADIUS_M`, the flat parts' shared droop radius.

  The planet clouds' shell bins are now the contiguous sorts 23-28.
- The flat sea's leftovers: `OceanLook.haze_color` and `haze_distance`
  (ignored since the sea is drawn on the terrain) and the historical
  `ocean/OCEAN_SWELL_ARTIFACT_REPORT.md`.

- `world_builder.ocean.Ocean`, its swell mesh, `ocean/seabed.py` and the
  `ocean.vert` / `ocean.frag` / `seabed.glsl` shaders; use `Sea`, which a
  `World` builds from an `OceanLook`. (#20)
- `OceanSettings.seabed_cascades` and `swell_grid_subdivs`: saved settings
  that name them load with a warning. (#20)
- `OceanLook.handover`, `Terrain.set_sea_surface` and `World`'s
  `BodyOcean`; `Body.ocean` is now the body's `Sea`. `OceanLook.haze_color`
  and `haze_distance` are ignored (the atmosphere hazes the sea). (#20)

### Fixed
- Clouds no longer turn yellow/orange at the horizon, in the sea's
  reflection and seen at a grazing angle from orbit, even at noon (#50): none
  of them (raymarch, billboards, deck, cirrus) draws any air in front of
  itself any more, as that air removed their blue and its in-scattered light
  was too dim to restore it. Clouds therefore have no distance haze
  (docs/source/atmosphere.md#clouds-at-the-horizon). `uCloudDebug` bit 2
  (no atmosphere) is gone with it.
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
- A ring seen from the ground was drawn over the clouds (their billboards
  write no depth, the ring's haze was drawn after them), veiling and
  dimming them where it passed behind. Its haze is now split at its
  planet's clouds' tops (`FieldHaze.set_clouds_top`,
  `PlanetClouds.top_radius`, the cirrus's included): what lies beyond them
  is drawn before them, and before the cirrus's far side. (#14)
- `CloudVolume.sun_visibility` raised a `KeyError` once its cache of
  shades had grown past its size, on a call with a cell already in it and
  a new one (in flight, after a few thousand 25 m cells): the cache is now
  dropped before the lookup, not in it. (#11)

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
