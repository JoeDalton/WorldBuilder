# WorldBuilder

Procedurally build whole worlds in Panda3D !

| Package | What it is |
|---|---|
| `world_builder.terrain` | Procedural terrain on a spherical planet: noise heightmaps, climate-driven biomes, hand-made stamps and biome paint, quadtree LOD generated in background processes, collision queries |
| `world_builder.ocean` | A camera-locked reflective ocean with an optional geometric swell |
| `world_builder.clouds` | Volumetric billboard clouds |

All three curve with the planet by the same camera-relative drop, `d²/2R`, so
they meet at the horizon (`world_builder.PLANET_RADIUS_M`).

## Try it

```
python scripts/demo_terrain.py            # terrain flythrough, with the ocean
python -m world_builder.terrain.preview   # top-down terrain maps, to PNG
python scripts/demo_clouds.py             # cloud flythrough
```

The controls are listed at the top of each demo.

## Install

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

`docs/source/`: [terrain](docs/source/terrain.md), [ocean](docs/source/ocean.md),
[clouds](docs/source/clouds.md). Build the HTML with `invoke doc`.
