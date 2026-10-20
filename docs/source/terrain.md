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
python -m world_builder.terrain.bench     # chunk, LOD and worker timings
python -m world_builder.terrain.level     # regenerate the sample stamp images
pytest tests/test_terrain_*.py tests/test_depth.py
```

All of them take `--radius` (metres) where it applies.

## Layers

| Layer | Modules | Notes |
|---|---|---|
| Planet definition | `noise`, `chart`, `planet`, `biomes`, `stamps`, `level` | Pure functions of a **direction on the unit sphere**, in float64 numpy. The renderer and the collision queries read the same planet. |
| Chunks | `cubesphere`, `chunkgen`, `quadtree`, `workers` | The six faces of a cube pushed onto the sphere, each a quadtree of tiles with 65×65 vertices plus skirts. Generated coarse-first in spawned worker processes. |
| Rendering | `mesh`, `materials`, `terrain`, `shaders/`, `world_builder.depth` | Panda3D. True geometry, camera-relative placement, depth split between two cameras, biome texture arrays, the ocean's tonemap, and haze measured through a shell over the planet. |
| Queries | `query` | `height_at`, `altitude`, `normal_at`, `sphere_contact`, `segment_intersect`, `material_at` on planet-frame positions, all on the same triangles that are drawn. |

### Planet scale

The radius is the one knob (`PlanetSpec.radius`, `planet.radius` in a level,
`--radius` on the tools). It defaults to 600 km, a Kerbin-sized world.

- **Planetary features scale with the radius.** The warp, continents, mountain
  ranges and climate bands are given for a 600 km planet and multiplied by
  `feature_scale` (radius / 600 km unless set). An Earth-sized planet gets
  ten-times-bigger continents rather than ten times as many.
- **Mountain height** follows the square root of that scale, kept between 2 and 7 km.
- **Ground-level character does not scale:** hills, detail, local climate
  patches and biome rules are in metres, so the ground looks the same at
  any radius.
- **The chunk grid follows the radius** (`cubesphere.grid_for`): enough levels
  for the finest spacing to stay near 2 m. That is 14 levels at 600 km, 17 at 6371 km.
- Below a 150 km radius, features stop shrinking.

### Elevation

The elevation is built up in stages:

1. A domain warp.
2. Continents, with a sea bias set so that `ocean_fraction` of the planet lies below sea level.
3. Ridged mountains, masked into ranges.
4. Hills, then detail.
5. Stamps.

Detail is band-limited to the chunk's vertex spacing. Fields with nothing
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

The hand-made pieces are listed in a level YAML; `levels/demo.yaml` documents the format.

- **Stamps** are 16-bit grayscale PNG heightmaps.
- **Paints** are colour PNGs that force biomes.

Each stamp or paint is placed by latitude and longitude, or by chart metres
around the level's reference point, with a size and rotation. Stamps blend in
one of these modes: `replace`, `add`, `max`, `min`, or `flatten` (level to the
ground height at the stamp's centre).

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

- A tile splits when the camera is nearer than K = 2 times its size. The
  distance is measured to a box in the tile's own tangent frame, bounded by
  its height range.
- Tiles wholly behind the planet's bulge are culled, using exact sphere geometry.
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
  `origin` is the camera snapped to a 3D grid: 4 km near the ground, and
  altitude / 16 above that, so orbital speeds do not re-place every tile
  each frame.
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

## Several planets

A world can hold several bodies: one `Terrain` each, all updated with the
same camera.

```python
executor = ChunkExecutor()                      # one worker pool for all
planet = Terrain(render, spec, executor=executor)
moon = Terrain(render, replace(PRESETS["moon"], seed=7), executor=executor,
               position=(3.0e6, 9.0e6, 1.0e6), haze_distance=None,
               materials=planet.materials)
for terrain in (planet, moon):                  # every frame
    terrain.update(camera_world)                # float64, world frame
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
- **Memory follows the view.** A terrain keeps at most `STORE_SIZE` /
  `NODE_CACHE_SIZE` tiles, and below that, 3× what its current selection
  needs (at least 256). A planet you fly away from releases its ground; a
  moon seen from the planet costs about 6–15 tiles.
- **Depth.** `SplitDepthCameras.system_ranges` takes the split from the
  nearest ground and the far plane from the farthest visible body. Depth
  steps up close only depend on the split, so reaching a moon 10,000 km
  away costs nothing on the ground; on the moon itself steps are a few km,
  narrower than a pixel at that range.
- **No atmosphere:** `haze_distance=None`. Give such a body bare sunlight
  and almost no ambient light (`set_sun`).
- **Barren worlds:** `PlanetSpec.has_vegetation=False` removes grass and
  forest, and desert takes whatever snow leaves. The `moon` preset is a
  200 km airless ball of sand, with rock only on cliffs.

Not handled yet: one body seen through another's atmosphere. From the
planet's surface the moon is drawn crisp over the sky, untouched by the
planet's haze.

## Ocean and sky in the demo

- **The swell ocean is a flat-world object.** The demo puts it in a frame
  that follows the point under the camera: the tangent plane at sea level.
  Its xy are anchored to the ground through a chart that re-centres every
  50 km, at which point the wave pattern jumps once. Its droop uses the
  terrain's radius (`Ocean(planet_radius=...)`).
- **Above 15 km the swell ocean is hidden.** The terrain draws the sea itself
  instead (`Terrain.set_sea_surface`): seabed vertices are lifted to sea
  level and shaded as water, tinted by depth.
- **The sky dome** is drawn by the far camera only. It follows the local
  vertical and fades to black between 20 and 80 km.
- **The planet's limb** gets its glow from the haze shell. Haze is the length
  of the sight line inside a shell 12 km above sea level. Near the ground
  that is the whole distance, as before; from orbit only the limb hazes.
- Proper ocean, atmosphere and cloud rendering from orbit are future work.

## Textures

Put PBR sets in `textures/<biome>/` for the six biomes: snow, rock, grass, forest, desert and beach. Files are recognised by name (`*Color*` / `*_diff*` / `*albedo*`, `*Normal*` / `*_nor_gl*`, `*Displacement*` / `*height*`, `*Rough*`, `*AO*` / `*AmbientOcclusion*`).

The ambientCG and Poly Haven naming schemes both work, and a DirectX normal map is detected from `dx` in its name. Any biome or map that is missing falls back to a procedural placeholder. Tile sizes are set in `textures/textures.yaml`.

## Paths to SpaceFlight

- **Physics must go spherical.** SpaceFlight's world is flat (Z up, a
  collision plane at z = 0). The terrain is now a sphere, so gravity, "up"
  and the ground under a ship all become radial. `TerrainQuery` already
  answers in the planet frame, ready for that:
  - ships: `sphere_contact` at the hitbox radius
  - bot sensors: `sphere_contact` at 30/50/100 m, served from coarse levels
  - lasers: `segment_intersect`
  - altimeters: `altitude`

  The hit point and normal are what `sensor_into_obstacle` and
  `ship_into_terrain_pushback` read today.
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
