# Terrain

[`world_builder.terrain`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/terrain/)
is procedural planetary terrain: heightmap-based, on a spherical planet, with
climate-driven biomes, hand-made stamps, quadtree LOD and collision queries.
SpaceFlight does not use it yet.

```
python scripts/demo_terrain.py            # 3D flythrough (controls in the script)
python -m world_builder.terrain.preview   # top-down maps of every preset, to PNG
python -m world_builder.terrain.level     # regenerate the sample stamp images
pytest tests/test_terrain_*.py
```

## Layers

| Layer | Modules | Notes |
|---|---|---|
| Planet definition | `noise`, `chart`, `planet`, `biomes`, `stamps`, `level` | Pure functions of a **direction on the unit sphere**, in float64 numpy. The renderer, the collision queries and a future orbital view all read the same planet. |
| Chunks | `chunkgen`, `quadtree`, `workers` | Quadtree tiles on the flat chart, 65×65 vertices each, plus skirts. Generated coarse-first in spawned worker processes. |
| Rendering | `mesh`, `materials`, `terrain`, `shaders/` | Panda3D. Applies the same camera-relative drop `d²/2R` as the ocean and clouds, blends biome texture arrays, and uses the ocean's tonemap and haze. |
| Queries | `query` | `height_at`, `normal_at`, `sphere_contact`, `segment_intersect`, `material_at`, all on the same triangles that are drawn. |

### Elevation

The elevation is built up in stages:

1. A domain warp.
2. Continents, with a sea bias set so that `ocean_fraction` of the planet lies below sea level.
3. Ridged mountains, masked into ranges.
4. Hills, then detail.
5. Stamps.

Detail is band-limited to the chunk's vertex spacing. Fields with nothing shorter than about 25 km are sampled on a fixed 512 m world lattice and interpolated, which costs under 1 cm of error and saves about half the time.

### Biomes

Temperature and humidity come from low-frequency noise. Each planet preset sets their mean and spread, which is how you bias a planet (desert, forest, ice, …).

On top of that:

- Altitude cools the temperature at 6.5 °C/km, so peaks get snow.
- Slopes steeper than about 30–48° turn to rock.
- On planets with an ocean, ground up to 6 m above sea level becomes beach.
- Paint masks are applied last.

### Hand-made content

The hand-made pieces are listed in a level YAML; `levels/demo.yaml` documents the format.

- **Stamps** are 16-bit grayscale PNG heightmaps.
- **Paints** are colour PNGs that force biomes.

Each stamp or paint is placed by latitude and longitude, or by chart metres, with a size and rotation. Stamps blend in one of these modes: `replace`, `add`, `max`, `min`, or `flatten` (level to the ground height at the stamp's centre).

The demo reloads the level when you press **L**.

## Textures

Put PBR sets in `textures/<biome>/` for the six biomes: snow, rock, grass, forest, desert and beach. Files are recognised by name (`*Color*` / `*_diff*` / `*albedo*`, `*Normal*` / `*_nor_gl*`, `*Displacement*` / `*height*`, `*Rough*`, `*AO*` / `*AmbientOcclusion*`).

The ambientCG and Poly Haven naming schemes both work, and a DirectX normal map is detected from `dx` in its name. Any biome or map that is missing falls back to a procedural placeholder. Tile sizes are set in `textures/textures.yaml`.

## Paths to SpaceFlight

- **Collision:** physics stays flat, like the ocean's collision plane. Panda 1.10 has no heightfield solid, so a small bridge would run after the traverser and call the existing `CollisionSystem` handlers with `TerrainQuery` results:
  - ships: `sphere_contact` at the hitbox radius
  - bot sensors: `sphere_contact` at 30/50/100 m, served from coarse levels
  - lasers: `segment_intersect`

  The hit point and normal are exactly what `sensor_into_obstacle` and `ship_into_terrain_pushback` read today.
- **Orbit / KSP-style:** the planet data is already sphere-based, so an orbital renderer would sample the same planet on a cube-sphere instead of the flat chart. The flat-chart shortcuts are the detail octaves being band-limited by chart spacing and the 512 m interpolation lattice; both are purely a matter of how the planet is sampled. The real work is the game's physics going spherical.
- **Known effect:** ships and props are not dropped. A bot flying low 30 km away looks about 70 m too high. The ocean has the same issue today.
