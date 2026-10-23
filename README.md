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

Python 3.12 to 3.14; depends on Panda3D 1.10.16, numpy, scipy and PyYAML.

For development, as for SpaceFlight:

```
pip install poetry invoke poetry-dynamic-versioning
invoke develop      # poetry install + pre-commit hooks
invoke test
```

From another project, depend on the wheel attached to a tagged release; the
install then needs no git. There is one wheel per platform, each carrying the
compiled [collision library](docs/source/native_build.md), so pick it with
markers (Poetry):

```toml
world_builder = [
  { url = "https://github.com/JoeDalton/WorldBuilder/releases/download/X.Y.Z/world_builder-X.Y.Z-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl", markers = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
  { url = "https://github.com/JoeDalton/WorldBuilder/releases/download/X.Y.Z/world_builder-X.Y.Z-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl", markers = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
  { url = "https://github.com/JoeDalton/WorldBuilder/releases/download/X.Y.Z/world_builder-X.Y.Z-py3-none-win_amd64.whl", markers = "sys_platform == 'win32'" },
  { url = "https://github.com/JoeDalton/WorldBuilder/releases/download/X.Y.Z/world_builder-X.Y.Z-py3-none-macosx_11_0_universal2.whl", markers = "sys_platform == 'darwin'" },
]
```

Not the tag's source zip (`archive/refs/tags/...zip`): GitHub regenerates those,
and a lock file's hash of one can stop matching. The release's sdist has no
compiled library: without it, `Terrain.enable_collision` raises (everything
else works). The wheels pin Panda3D to 1.10.16, the version the library is
built against.

The version is read from git tags. Pushing a tag `X.Y.Z` runs CI, which builds
the wheel of each platform, installs it in a clean venv on Python 3.12 to 3.14
and runs the tests against it, then attaches the wheels and the sdist to the
`X.Y.Z` release.

## Documentation

`docs/source/`: [terrain](docs/source/terrain.md),
[atmosphere, sky and planet clouds](docs/source/atmosphere.md),
[ocean](docs/source/ocean.md), [clouds](docs/source/clouds.md),
[the native collision library](docs/source/native_build.md),
[performance](docs/source/performance.md) and its
[backlog](docs/source/performance_backlog.md),
[lighting on objects](docs/source/object_lighting.md). Build the HTML, with
the API reference, with `invoke doc`. Changes are listed in the
[changelog](CHANGELOG.md).
