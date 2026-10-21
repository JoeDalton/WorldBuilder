# WorldBuilder

Procedurally build whole worlds in Panda3D !
This is also a full vibe-coding experiment for me

| Package | What it is |
|---|---|
| `world_builder.terrain` | Procedural terrain on a true spherical planet, from orbit to the ground: noise heightmaps, climate-driven biomes, hand-made stamps and biome paint, cube-sphere quadtree LOD generated in background processes, collision queries |
| `world_builder.atmosphere` | Single-scattering atmospheres (sky colour, sunsets, aerial perspective, the limb from space), applied by every shader |
| `world_builder.sky` | A star field (procedural, with tunable nebulae, or a painted cube map) and the sun, seen through the air |
| `world_builder.clouds` | Volumetric billboard clouds: a flat field, or a whole planet's, following a weather map (billboards near, a deck far, one pattern) |
| `world_builder.ocean` | A camera-locked reflective ocean with an optional geometric swell |
| `world_builder.depth` | A near/far camera split, for views from the ground to orbit |

The ocean and the flat cloud field curve with the planet by a camera-relative
drop, `d²/2R`, so they meet at the horizon (`world_builder.PLANET_RADIUS_M` by
default). The terrain and the planet clouds are real geometry around the
planet's centre; `scripts/demo_terrain.py` shows how to place the ocean
under the camera next to them.

## Try it

```
python scripts/demo_terrain.py            # planet, moon, atmosphere, clouds, ocean: orbit to ground
python scripts/terrain_shots.py           # the same from fixed poses, to PNG, with frame times
python scripts/flight_bench.py            # frame times in scripted fast flights
python -m world_builder.terrain.preview   # globe and top-down terrain maps, to PNG
python scripts/demo_clouds.py             # the flat cloud field around a smiley
```

The controls are listed at the top of each demo.

## Install

Python 3.12 to 3.14; depends on Panda3D 1.10, numpy, scipy and PyYAML.

For development, as for SpaceFlight:

```
pip install poetry invoke poetry-dynamic-versioning
invoke develop      # poetry install + pre-commit hooks
invoke test
```

From another project, depend on a tagged release's zip; the install then needs
no git (Poetry):

```toml
world_builder = { url = "https://github.com/JoeDalton/WorldBuilder/archive/refs/tags/0.1.0.zip" }
```

The version is read from git tags; a tag's zip carries it in
`.git_archival.json`, which GitHub fills in when it builds the archive.

## Documentation

`docs/source/`: [terrain](docs/source/terrain.md),
[atmosphere, sky and planet clouds](docs/source/atmosphere.md),
[ocean](docs/source/ocean.md), [clouds](docs/source/clouds.md),
[performance](docs/source/performance.md) and its
[backlog](docs/source/performance_backlog.md),
[lighting on objects](docs/source/object_lighting.md). Build the HTML, with
the API reference, with `invoke doc`. Changes are listed in the
[changelog](CHANGELOG.md).
