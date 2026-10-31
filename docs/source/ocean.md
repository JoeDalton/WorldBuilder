# Sea

A planet's sea is drawn by its terrain's own chunks: where the ground is under
sea level, `terrain.vert` lifts the vertices to the water and `terrain.frag`
shades them as sea. There is no separate ocean mesh. So the sea:

- lies on the true sphere and follows the terrain's LOD;
- knows the depth under every vertex (the chunk's own ground height);
- meets the coast inside one mesh.

[`world_builder.ocean`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/ocean/)
gives it its looks. The per-class API is in the
[code reference](apidocs/index.rst).

- `ocean.py`: what the sea costs (`OceanSettings`) and the CPU halves of the
  shading.
- `sea.py`: `Sea`, the near look's inputs, the reflection and the frame at the
  camera's nadir.
- `swell.py`: the swell on the CPU, for collisions.
- Shaders: `sea.glsl`, `swell.glsl` and `water.glsl`.

A `World` body with an `OceanLook` gets a `Sea`. A body with `spec.has_ocean`
and no look gets the far look only.

## Two looks, one surface

- **Far** (the plain `terrain.frag`): flat water lit per pixel, like the ground:
  - tinted by depth, a 3,000 m shelf gradient under the shallow tint;
  - an analytic sun highlight;
  - the sky light reflected at grazing angles.

  It is cheap, and it is what an orbit view shows.
- **Near** (`terrain_sea.frag`, the same source with `SEA_NEAR`): `sea.glsl`'s
  waves, the planar reflection, colour by depth and shoreline foam.
- **The blend.** `Sea.update` blends from the near look below
  `near_band` (12 km) to the far one above 18 km (`uSeaNear`, smoothstep).
  There is no switch, so nothing pops. The reflection pass runs only while
  the near look shows.
- **Who pays.** Chunks with water hang under `Terrain.sea_root`. A `Sea` gives
  that node the near look's shader (`Terrain.set_sea_near`) only while the
  look shows. Land chunks and high views never pay for its code: compiled
  into every terrain fragment, it cost 0.5 ms at 20 km even unused. Water
  fully in its near look skips the land's shading entirely.
- **LOD.** Chunk LOD uses the ground's bounds, seabed included. Deep water
  stays coarse, since its surface has no relief to resolve. The shallows
  refine like land, for the coast and the foam. Culling uses each chunk's own
  box, which holds the water (and its swell).

## Waves

- **Per pixel, filtered by their size on screen** (#12).
  - `waveGradient` sums octaves along per-iteration directions
    (`compute_wave_dirs`, `wave_iterations` of them), their slope taken
    analytically.
  - The pixel's footprint comes from the lens: across the view, the distance
    times the pixel's angle; along it, that over the sine of the elevation.
  - Each octave fades by how far its phase moves across the pixel along its
    own direction (`aa_full_px` 8 down to `aa_gone_px` 4 pixels per
    wavelength).
  - Octave i's frequency is 1.18^i, so the fully resolved octaves run in a
    plain loop and only those in between pay for the fade.
  - Every pixel normalises by the full weight sum, so dropping fine octaves
    does not inflate the coarse ones.
- **What the filter takes away is kept** as slope variance: the faded
  octaves', the tail's (`uTailVar`), the swell's and the noise mask's. That
  spread does two things:
  - it picks the fresnel from a table of Schlick averaged over Gaussian
    facets, seen from the eye (`rough_fresnel_table`), so distant water
    settles on a steady blend;
  - it blurs the reflection through its mipmaps (`reflection_blur`).
- **Anchored to the ground.**
  - The waves live on the tangent plane at the camera's nadir, in the
    coordinates of a chart anchored near it. The chart is re-anchored, with a
    jump in the pattern, once the nadir is `anchor_range` (50 km) away.
  - Each octave's phase at the plane's origin (frequency × direction · chart
    position, plus time) is computed in float64 on the CPU (`wave_phases`,
    `uWavePhase`). The shader adds only the local part, so the finest octaves
    keep their phase tens of kilometres from the anchor, where float32 lost
    them before.
- **Domain warp and grazing.** The domain warp and the flattening toward a
  mirror at grazing angles (`detailAngle`) are kept.

## Swell

- **A field of the planet-frame point** (`swell.glsl`), the same for every
  camera.
  - It is the product of two layers of two-octave value noise. The lattice
    hash is integer arithmetic, so the CPU's copy gives the same bits.
  - Each layer turns slowly about its own axis through the planet's centre
    (`swell_rotations`, float64 on the CPU), so the pattern drifts with the
    wind at the planet's reference point and changes shape instead of
    sliding. It has no seams and no anchor jumps.
- **Geometry.**
  - `terrain.vert` raises every wet vertex by
    `swell_amplitude × (field − 1/4)`, centred on sea level.
  - It dies out over the shallows (`swell_shallow` 5 m to `swell_deep` 60 m
    of the vertex's own depth).
  - Its octaves fade to their mean by the chunk's vertex spacing (the terrain
    noise's own `octave_weight` rule, cut-off `1 / (4 spacing)`, per chunk in
    `uChunkCentre.w`). So the swell flattens with distance through the LOD:
    the main 250 m octave is whole up to level 4 (29 m spacing) and gone at
    level 5, 4–7 km from a low camera.
  - With 20 m of amplitude the water stays within about −4 to +7 m.
- **Shading.**
  - The near look tilts its normal by the gradient of the very field the
    vertices are raised by (`swell_amplitude / 20`, and the depth factor),
    computed analytically per vertex with the height and interpolated
    (`vSwell`). So the shading follows the drawn water, and its detail fades
    with the chunk's level like the geometry's. Evaluating the field again per
    pixel cost 0.8-1.6 ms a frame low over the sea on a GTX 1070 (32 lattice
    hashes a pixel, each five 32-bit integer multiplies).
  - The field also shifts the small waves' phase (`fm_depth`), so the
    dominant octave's tiling drifts.

## Colour by depth and foam

- **Colour by depth.** Over shallow seabed the water goes from `shallow_color`
  to the deep `water_color`: Beer–Lambert down and back up, with an e-folding
  depth `shallow_depth` (`deepWaterShare`, `water.glsl`). Both looks use it.
- **Shoreline foam.** Under `foam_depth` metres of water:
  - solid at the waterline;
  - then lines along the depth contours, `foam_spacing` apart, running
    shoreward and shifted along the coast by the swell;
  - broken up by noise.

  The depth is the chunk's own, interpolated between vertices. So the foam
  fades out on chunks whose vertices are 12–48 m apart, which cannot place a
  band a metre or two deep.

## Reflection

- **The camera.** It is mirrored across the tangent plane at the nadir
  (`Sea.frame`) every frame, and renders into an offscreen buffer sized from
  the main display region times `reflection_scale` (0.5 at HIGH).
  - Its lens is a copy of the main camera's taken at construction, its field
    of view widened by `reflection_fov_scale` (1.1) and its far distance ×4.
  - So build the sea (or the `World`) before a
    `world_builder.depth.SplitDepthCameras` shortens the main lens.
- **What it shows.**
  - A clip plane on the camera keeps only what is above the water.
  - Shaders drawn through the mirror write `gl_ClipDistance[0]`. The terrain
    clips its own wet vertices in that pass, so the sea is not in its own
    reflection. That is clipping, not a discard, which would cost the early
    depth test everywhere.
  - Chunks that are all water are hidden from the reflection camera
    (`REFLECTION_CAMERA_MASK`): it would clip every vertex of them, and their
    bounds, which reach the water, keep the clip plane from culling them.
  - The terrain is reflected or not with `TerrainSettings.reflected`.
- **The lookup.** The sea samples the reflection at each fragment's footprint
  on the mirror plane, perturbed by the ripples and clamped to the rendered
  region (`uReflUVScale`, measured once the buffer is realized).
- **Decoding.** The buffer holds display colours (the sky and terrain
  tonemap themselves), so the sample is decoded back to linear light
  (`aces_inverse`) before the blend. A full mirror then shows exactly the
  sky's own colour.
- **Its limit.** The mirror is a plane: far reflections drift by the planet's
  curvature (about 80 m at 10 km on a 600 km planet), where the normal is
  already flattened toward a mirror. A sphere-correct reflection would be
  another technique (screen-space or hybrid).

## Night, atmosphere, collision

- **Night.** The water's own colour was chosen in daylight. `Sea.set_light`
  scales it near the camera (and the reflection's clear colour) by the light
  on the sea there relative to that daylight; `World` computes it with
  `Atmospheres.light_at`. The far look is lit per pixel.
- **Atmosphere.** Both looks are seen through the atmospheres like the
  terrain is, in linear light before the tonemap.
- **Collision follows the drawn water.** `Terrain.set_swell` hands the swell
  (`world_builder.ocean.swell.Swell`, which `Sea` makes every frame) to the
  query and to the native collision solid.
  - Both raise the wet corners of the triangle under a point exactly as
    `terrain.vert` does, at the chunk's own resolution.
  - A ship stands on the water it sees, crests included, and the water moves
    with time.
  - Material lookups answer "water" wherever the seabed is under sea level.
  - The swell's CPU copy is `terrain/native/swell.h` (in the noise library
    and the collision solid), with a numpy fallback. The tests compare all
    three, the GPU's included: they agree to 3e-4 of the field, a few
    millimetres of water, the limit of float32 planet coordinates.

## Quality

Every cost/quality trade is a field of `OceanSettings`, with LOW / MID / HIGH
/ ULTRA presets (see [Quality settings](quality.md)):

- the octave count and the filter's pixels per wavelength;
- the rough fresnel and the reflection's blur;
- the depth colour, foam and depth-scaled swell switches;
- the reflection's resolution and field-of-view margin.

All are live; `reflection_scale` remakes the buffer on the same camera. The
terrain demo cycles the presets with F5; `terrain_shots.py` and
`flight_bench.py` take `--ocean-quality`.

## Debug

`Sea.set_debug_mode` (`uSeaDebug`, `terrain_shots.py --ocean-debug`) shows, on
water in its near look:

| Mode | Shows |
|---|---|
| 0 | the shading |
| 1 | the normal |
| 2 | the reflection UV |
| 3 | the fresnel |
| 4 | the world grid |
| 5 | the reflection UV's clamp |
| 6 | the raw reflection sample |
| 7 | pre-tonemap clipping |
| 8 | the frequency-modulation phase |
| 9 | the share of the octaves resolved |
| 10 | the unresolved slope spread |
| 11 | the octaves evaluated |
| 12 | the depth (black at the waterline, white at 50 m) |
