# Terrain

[`world_builder.terrain`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/terrain/)
is procedural planetary terrain on a true sphere, seen from orbit down to the
ground: heightmap-based, with climate-driven biomes, hand-made stamps,
cube-sphere quadtree LOD and collision queries. SpaceFlight does not use it
yet.

```
python scripts/demo_terrain.py            # 3D flight: planet, moon, orbit to ground (controls in the script)
python scripts/terrain_shots.py           # the demo from fixed poses, to PNG
python -m world_builder.terrain.preview   # globe and top-down maps of every preset, to PNG
python -m world_builder.terrain.preview --weather   # each globe under its clouds' weather
python -m world_builder.terrain.bench     # chunk, LOD and worker timings
python scripts/flight_bench.py            # frame times in scripted fast flights
python -m world_builder.terrain.level     # regenerate the sample stamp images
pytest tests/test_terrain_*.py tests/test_depth.py
```

The demo, the shots, the preview and the bench take `--radius` (metres).

## Layers

| Layer | Modules | Notes |
|---|---|---|
| Planet definition | `noise`, `chart`, `planet`, `biomes`, `stamps`, `level` | Pure functions of a **direction on the unit sphere**, in float64 numpy. The renderer and the collision queries read the same planet. |
| Chunks | `cubesphere`, `chunkgen`, `quadtree`, `workers` | The six faces of a cube pushed onto the sphere, each a quadtree of tiles with 65×65 vertices plus skirts. Generated coarse-first in spawned worker processes. |
| Rendering | `mesh`, `materials`, `terrain`, `shaders/`, `world_builder.depth` | Panda3D. True geometry, camera-relative placement, depth split between two cameras, biome texture arrays, sunlight through the planet's own air, the ocean's tonemap, and every atmosphere along the sight line (`world_builder.atmosphere`). |
| Queries | `query` | `height_at`, `altitude`, `normal_at`, `sphere_contact`, `segment_intersect`, `material_at` on planet-frame positions, all on the same triangles that are drawn. |
| Collision | `collision`, `native` | `Terrain.enable_collision`: a native Panda collision solid over the drawn chunks, for a game's `CollisionTraverser`. |

### Planet scale

The radius is the one knob (`PlanetSpec.radius`, `planet.radius` in a level,
`--radius` on the tools). It defaults to 600 km, a Kerbin-sized world.

- **Planetary features scale with the radius.** The warp, continents, mountain
  ranges and climate bands are given for a 600 km planet and multiplied by
  `feature_scale` (radius / 600 km unless set). An Earth-sized planet gets
  ten-times-bigger continents rather than ten times as many.
- **Mountain height** follows the square root of that scale, the factor
  kept within 2/4.8 to 7/4.8: 2 to 7 km for the default 4.8 km crests.
- **Ground-level character does not scale:** hills, detail, local climate
  patches and biome rules are in metres, so the ground looks the same at
  any radius.
- **The chunk grid follows the radius** (`cubesphere.grid_for`): enough levels
  for the finest spacing to stay near 2 m. That is 14 levels at 600 km, 17 at 6371 km.
- **The finest tile** is `QUADS` (64) quads of that spacing: 1.8 m spacing and
  115 m tiles at 600 km. The knob is `cubesphere.TARGET_SPACING` (2 m; the grid
  rounds to the nearest power of two). At runtime, `TerrainSettings.min_level`
  (`Terrain(settings=)`, `set_settings`; MID drops one level, LOW three)
  stops refining at a coarser level, and collision follows. That and the
  terrain's other cost knobs (LOD distance, texture distance and filtering,
  caches, upload budget) are listed in [Quality settings](quality.md).
- Below a 150 km radius, features stop shrinking.

### Elevation

The elevation is built up in stages:

1. A domain warp.
2. Continents, with a sea bias set so that `ocean_fraction` of the planet lies below sea level.
3. Ridged mountains, masked into ranges.
4. Hills, then detail.
5. Stamps.

Detail is band-limited to the chunk's vertex spacing: octaves shorter than
about four samples fade out, so coarse chunks never alias. Fields with nothing
shorter than about 10 km (the "low band") are sampled on a lattice of each
cube face (230 m at 600 km, scaled with the features) and interpolated. That
costs under 5 cm of error and saves about half the time.

### Biomes

Temperature and humidity come from low-frequency noise. Each planet preset sets their mean and spread, which is how you bias a planet (desert, forest, ice, …).

On top of that:

- Altitude cools the temperature at 6.5 °C/km, so peaks get snow.
- Slopes steeper than about 30–48° turn to rock. The slope is measured against the local vertical.
- On planets with an ocean, ground up to 6 m above sea level becomes beach.
- Paint masks are applied last.

### Hand-made content

The hand-made pieces are listed in a level YAML. The `world_builder.terrain.level`
module docstring documents the format, and the package's `levels/demo.yaml` is
the demo's level.

- **Stamps** are 16-bit grayscale PNG heightmaps.
- **Paints** are colour PNGs that force biomes.

Each stamp or paint is placed by latitude and longitude, or by chart metres
around the level's reference point, with a size and rotation; a paint can
reuse a stamp's placement. Stamps blend in one of these modes: `replace`,
`add`, `max`, `min`, or `flatten` (level to the ground height at the stamp's
centre), fading in over `falloff` metres from their edges. Coarse chunks read
a box-filtered mip of the image, so a stamp never aliases. The level also
holds the demo's start pose, by latitude, longitude and altitude or by chart
`x`, `y` and `z`.

The demo reloads the level when you press **L**.

## The cube-sphere

Chunk keys are `(level, face, i, j)`.

- Level 0 is the finest. The top level is a whole face, so the planet seen
  from far away is just its six root chunks, or their children.
- The cube is equal-angle: face coordinate `a` maps to `tan(πa/4)`. Grid
  lines are great circles, and cells range from 0.71× to 1× the nominal spacing.
- Samples sit at integer indices along each face. Neighbouring chunks of any
  level share their edge samples exactly.
- Across a face seam, both faces compute bit-identical directions. Heights
  differ by millimetres there, because each face has its own low-band
  lattice; the skirts hide that, as they hide the T-junctions between levels.
- On the four equatorial faces, texture u runs east and v north. Textures
  are rotated across the other seams.

### LOD

- A tile splits when the camera is nearer than K = 2 times its size, and
  merges back only 15 % farther out. The distance is measured to a box in
  the tile's own tangent frame, bounded by its height range.
- Tiles wholly behind the planet's bulge are culled, using exact sphere
  geometry, as are tiles wholly beyond `max_distance` when it is set.
- Away from face seams, a ring holds about 66 tiles per level, as on the
  flat chart. Near a seam, the two faces' grids don't align and both
  contribute partial rings.
- A parent is never replaced before all its children are ready.

## Coordinates and precision

Positions are float64 metres in the **planet frame**: the planet centre is at
the origin, axes are fixed to the planet, and +Z points to the north pole.
Nothing in the renderer depends on that frame being static. Rotating the
terrain root later would rotate the planet.

Panda transforms are float32, so the world moves around the camera rather
than the camera through the world:

- The camera stays at (or near) the render origin.
- The terrain root is offset by `origin − camera`, computed in float64.
  `origin` is the camera snapped to a 3D grid: 4 km up to 64 km of
  altitude, then doubled as often as needed to stay at least altitude / 16,
  so orbital speeds do not re-place every tile each frame.
- Tiles hang under the root at `centre − origin`, and their vertices are
  float32 offsets from their own float64 centre.
- `Terrain.to_render(p)` places any other planet-frame object the same way.

| Tile | Size | float32 vertex resolution |
|---|---|---|
| Leaf (level 0) | 115 m | under 1 mm |
| Level 6 | 7.4 km | about 2 mm |
| Face root | 940 km | about 6 cm (only ever seen from far away) |

### Depth

`world_builder.depth.SplitDepthCameras` draws the view twice into the same
window. A far camera covers [split, far]. Then the usual camera covers
[near, split] after clearing depth. This is KSP's scaled-space trick, without
the scaled space.

The split sits halfway down to the highest possible ground, and the far plane
just past the horizon. That keeps 24-bit depth steps under a metre at the far
end of each range at every altitude.

Nothing else changes:

- the window's depth buffer and MSAA
- the existing shaders
- the terrain's coastline depth bias
- the ocean's reflection camera, which keeps its own copy of the lens

## Lighting

`Terrain(sun_direction=, sun_color=, ambient=)` and `Terrain.set_sun` take
the sunlight *above any air* and the sky light, both linear.

- **Through its own air.** With `atmosphere=` (the index of its own
  atmosphere in `atmospheres=`), the shader passes the sunlight through that
  air (`atmoSunlight`): reddened near the terminator, gone in the planet's
  shadow. An airless body just has a soft day/night line.
- **Sky light fades through twilight**, as the sun goes from about 12°
  above the local horizon to 12° below it, so the night side is dark rather
  than lit by a blue sky.
- **Aerial perspective** comes last, in linear light before the tonemap:
  the terrain is seen through every atmosphere the sight line crosses.

## Several planets

A world can hold several bodies: one `Terrain` each, all updated with the
same camera.

```python
executor = ChunkExecutor()                      # one worker pool for all
atmospheres = Atmospheres()                     # the planet's air
atmospheres.add(spec.radius, AtmosphereParams())
planet = Terrain(render, spec, executor=executor,
                 atmospheres=atmospheres, atmosphere=0)
moon = Terrain(render, replace(PRESETS["moon"], seed=7), executor=executor,
               atmospheres=atmospheres,         # seen through the planet's air
               position=(3.0e6, 9.0e6, 1.0e6), materials=planet.materials)
for terrain in (planet, moon):                  # every frame
    terrain.update(camera_world)                # float64, world frame
atmospheres.update(camera_world)
```

- **Frames.**
  - The world frame is shared, and `Terrain.position` is each body's
    centre in it (float64, settable, so orbits can move it later).
  - Each terrain's queries, chunk centres and `to_render` work in its own
    planet frame (centre at the origin, same axes);
    `terrain.to_planet(p_world)` converts.
  - float64 is fine at solar-system scale: about 30 µm at 1 AU.
- **One worker pool.** `ChunkExecutor` serves one channel per terrain. Each
  channel has its own spec, version, wanted list and results. Each free
  slot goes to the channel with the fewest tasks in flight, so a distant
  moon's handful of coarse tiles is not stuck behind the planet's thousands.
- **Memory follows the view.** A terrain keeps at most
  `TerrainSettings.store_size` / `node_cache_size` tiles (2,000 / 1,200 at
  HIGH), and below that, 3× what its current selection
  needs (at least 256). A planet you fly away from releases its ground; a
  moon seen from the planet costs about 6–15 tiles.
- **Depth.** `SplitDepthCameras.system_ranges` takes the split from the
  nearest ground and the far plane from the farthest visible body. Depth
  steps up close only depend on the split, so reaching a moon 10,000 km
  away costs nothing on the ground; on the moon itself steps are a few km,
  narrower than a pixel at that range.
- **No atmosphere:** leave `atmosphere=` unset (no air of its own). Give
  such a body bare sunlight and almost no ambient light (`set_sun`); it is
  still seen through the other bodies' air when a sight line crosses it.
- **Barren worlds:** `PlanetSpec.has_vegetation=False` removes grass and
  forest, and desert takes whatever snow leaves. The `moon` preset is a
  200 km airless ball of sand, with rock only on cliffs.

A body seen through another's air is dimmed and tinted by it: from the
planet's surface by day the moon is a pale disc in a blue sky (see
[Atmosphere, sky and planet clouds](atmosphere.md)).

## Ocean, sky and clouds in the demo

- **The swell ocean is a flat-world object.** The demo puts it in a frame
  that follows the point under the camera: the tangent plane at sea level.
  Its xy are anchored to the ground through a chart that re-centres every
  50 km, at which point the wave pattern jumps once. Its droop uses the
  terrain's radius (`Ocean(planet_radius=..., exact_sphere=True)`), and it is
  seen through the atmosphere like everything else.
- **Above 15 km the swell ocean is hidden** (handed over between 14 and
  16 km). The terrain draws the sea itself instead
  (`Terrain.set_sea_surface`): seabed vertices are lifted to sea level and
  shaded as water, tinted by depth, with the sky light reflected at grazing
  angles.
- **The swell ocean reflects the terrain.** The ocean's reflection camera
  draws the scene a second time; `TerrainSettings.reflected` off (LOW's)
  hides the terrain from it by `depth.REFLECTION_CAMERA_MASK`. Its clip
  plane at the water is applied by the terrain's vertex shader
  (`gl_ClipDistance`), since a custom shader does not get it for free.
- **Sky, haze and the limb** come from the planet's atmosphere (single
  scattering), and the clouds from one planet-wide density field: see
  [Atmosphere, sky and planet clouds](atmosphere.md). The terrain takes
  `atmospheres=` (the world's) and `atmosphere=` (its own index, for its
  sunlight); there is no separate haze any more.

## Textures

Put PBR sets in `src/world_builder/terrain/textures/<biome>/` for the six biomes: snow, rock, grass, forest, desert and beach. Files are recognised by name, case-insensitively (`*color*` / `*diff*` / `*albedo*`, `*normal*` / `*nor_gl*`, `*displacement*` / `*height*` / `*bump*`, `*rough*`, `*AO*` / `*AmbientOcclusion*`); names with preview, metal, spec or opacity are skipped.

The ambientCG and Poly Haven naming schemes both work, and a DirectX normal map is detected from `dx` in its name. Any biome or map that is missing falls back to a procedural placeholder. `textures/textures.yaml` sets each biome's tile size (`tile_m`, a power of two up to 512 m), normal convention (`normal: gl | dx`) and albedo `tint`, and the layer size cap (`max_size`).

## Collision

`Terrain.enable_collision(into_mask, name="terrain", owner_tag="owner", max_distance=100_000)` makes the drawn terrain solid to a game's own `CollisionTraverser` (traversing `render`). It adds one into-only `CollisionNode` holding a native `CollisionTerrain` solid (`terrain/native/terrain_collision.cpp`).

The solid answers for every drawn chunk within `max_distance` of the camera. Chunks are registered as the LOD selection shows them and dropped as it hides them.

**What the game gets:**

- **Spheres** (ships, ordnance, sensors) and **segments** (lasers) get ordinary `CollisionEntry` objects, so event and queue handlers work unchanged.
- **At most one entry per collider per frame**, with the surface point and the normal of the very triangle drawn there. With the sea, the sea surface is solid and its normal is the local vertical.
- **No holes at LOD seams.** Each direction is answered by exactly one drawn chunk.
- **Material:** the node's owner tag is a `TerrainSurface`. Its `material_at(point, relative_to=None)` names the ground at a hit point: `"water"`, or the dominant biome's name.

**Where the node goes:** it hangs beside `Terrain.root`, not under it, and copies the root's transform each frame. Under the root, its collide mask would send the traverser through every chunk node for every collider.

**Cost:** on the earthlike preset, with 64 ships, 33 active sensors and 300 lasers within 5 km of the camera, the terrain adds about 0.6 ms to a traverse, and 1.1 ms with every ship and laser hugging the ground (`scripts/collision_bench.py`). `scripts/collision_viewer.py` shows the same load live in a window: contacts, normals and the traverse time.

The library ships in the Linux x86_64 wheel, built against Panda3D 1.10.16 (the version world_builder pins). From a source checkout, `invoke native` builds it, and `invoke develop` does so on Linux x86_64 (see `NATIVE_COLLISION_BUILD.md`). `enable_collision` raises `NativeCollisionError` when it is missing or built for another Panda3D series.

`include_ocean=False` makes the seabed solid instead of the sea surface. The drawn chunk meshes themselves are never collision targets: their into-mask is off.

**Loading:** only drawn ground is solid, and the ground around a new camera position takes seconds to load (about 10 s from nothing on the earthlike preset). `Terrain.loading` is True while the last LOD pass still wanted chunks it needs. A game's loading phase should wait for it to stay False for a moment: in flight it turns on briefly whenever detail refines.

`TerrainQuery` stays the pure-numpy path for one-off queries (`altitude`, `height_at`, and so on).

## Paths to SpaceFlight

- **Collision:** `terrain.enable_collision(into_mask)` with SpaceFlight's terrain into-mask. The existing `ship-into-terrain`, `laser-into-terrain` and sensor handlers then see the planet. `munition_into_terrain` picks its sparks with `owner.material_at(hit_point, root_node)`. The flat ocean `CollisionPlane` goes away where there is a terrain. Gravity and a radial "up" for flight are a separate matter, not needed for collision.
- **Loading phase:** a level should wait for `not terrain.loading` (held for a moment) before ships fly near the ground, since ground that is not drawn yet is not solid.
- **Camera:** SpaceFlight already keeps the player at the render origin
  (`recenter_render_origin`). The terrain's root must hang under `render`,
  not the game root, and be updated after that recentring with the camera's
  float64 planet-frame position.
- **Depth:** `SplitDepthCameras` is opt-in. With it, the HUD's
  `lens.project` and `laser.frag`'s `gl_FragDepth` should use the near camera's lens.
- **Ocean and clouds** still use `PLANET_RADIUS_M` (6371 km) and the flat
  world by default. A 600 km terrain needs them built with its radius, and
  placed in a frame that follows the camera's nadir, as the demo does with
  the ocean.
