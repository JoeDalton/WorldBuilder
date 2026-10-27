# Quality settings

Every cost-versus-quality trade in WorldBuilder is a *knob*. A game built on
it can show each knob on its own in a settings menu, or offer them bundled
as four presets: LOW, MID, HIGH and ULTRA. It can do both without naming a
single knob in its own code: the knobs describe themselves as plain data, so
knobs WorldBuilder adds later appear in the game's menu unchanged.

- [`world_builder.quality`](../../src/world_builder/quality.py) holds the
  shared parts:
  - `QualityLevel`, the four levels, cheapest first, each with a `label`;
  - `Knob`, what a menu needs to show one field;
  - `QualitySettings`, what every settings class has in common.
- Each drawn part has a frozen settings dataclass beside the code that reads
  it: `TerrainSettings`, `AtmosphereSettings`, `SkySettings`,
  `CloudSettings` and `OceanSettings`.
- [`world_builder.settings`](../../src/world_builder/settings.py)'s
  `WorldSettings` bundles them, one section each.

This is the only way to set what a part costs. There are no quality
arguments besides `settings=`, no one-purpose setters, and nothing read from
a game's own configuration objects.

## How a knob applies

Each object takes `settings=` when it is built and exposes `.settings` and
`set_settings(new)`. A setting whose value is unchanged costs nothing, so a
menu's "apply" button may call `set_settings` freely. Each knob says how a
change takes effect:

| Mode | What happens |
|---|---|
| `live` | At once: a shader input, a budget or a cache size. Lower caches evict on the next update, never a chunk or tile the drawn view needs. |
| `rebuild` | The object rebuilds a part of itself. The clouds re-plan their shells and regenerate their tiles (a few seconds of refill). The sea makes its reflection buffer again. The sky makes a new Milky Way texture. |
| `restart` | Only a newly built object takes it: the terrain's texture size and worker count. `set_settings` logs that it will apply at the next start. |

## A game's settings menu, without naming a knob

```python
from world_builder.settings import WorldSettings

settings = WorldSettings.from_dict(config.get("world"))  # never raises

# Build each WorldBuilder object with its section, found by its type.
atmospheres = Atmospheres(settings.settings_for(Atmospheres))
terrain = Terrain(render, spec, atmospheres=atmospheres,
                  settings=settings.settings_for(Terrain))
...
parts = [terrain, atmospheres, sky, clouds, ocean]

# The menu, from plain data (JSON-ready):
schema = WorldSettings.schema()
levels = schema["levels"]                  # [{"id": "low", "label": "Low"}, ...]
for section in schema["sections"]:         # {"id", "label", "knobs"}
    for knob in section["knobs"]:
        # knob["id"], ["label"], ["help"], ["unit"]
        # knob["kind"]: "bool", "int", "float" or "choice"
        # knob["range"] and ["step"] (a slider), or knob["choices"] (labels)
        # knob["apply"]: "live", "rebuild" or "restart" (say so in the menu)
        # knob["advanced"]: a menu may hide it
        # knob["default"] and knob["presets"]: {"low": ..., ...}
        ...

# The player moves a slider, picks a preset, or one part's preset:
settings = settings.with_value(section["id"], knob["id"], value)  # clamped
settings = settings.with_level("low")
settings = settings.with_section_level("clouds", "mid")
settings.apply(*parts)                     # each object takes its section
config["world"] = settings.to_dict()
```

`to_dict()` stores the format's version, the preset, and what differs from
it, choices by their label:

```json
{"version": 1, "preset": "mid",
 "overrides": {"terrain": {"detail_far": 3000.0},
               "ocean": {"foam_noise": false}}}
```

To show which preset each part is at, use `section_level(section)`, which
gives the level or None for custom. `is_custom()` covers the whole world.
Presets are distinct within each section, so a level is always
recognisable.

**Strict construction, lenient loading.** Building a settings object by hand
is strict: a value out of its range or choices raises `ValueError`.
`from_dict`, `from_overrides` and `with_value`, which read saved or typed
data, never raise; each problem is logged as a warning:

- an unknown preset is HIGH;
- an unknown section or knob is skipped;
- a value of the wrong type keeps the preset's;
- a value out of range is clamped.

Limits that involve two fields are applied where the settings are used: the
ocean's wave cut-off is never sharper than its full-strength limit, and the
terrain's GPU chunk cap is never above its RAM one.

## Staying compatible as WorldBuilder changes

- **Ids are the contract.** Section and knob ids are what saved settings are
  keyed by, and the only names a game could ever need.
  - A knob WorldBuilder adds appears in the schema, takes its preset value
    in old saves, and is in the menu with no change to the game.
  - A knob it removes is skipped when an old save is loaded (with a warning).
  - A knob it renames lists its old id in its class's `RENAMED`, so old
    saves keep their value.
- **Labels are display text**, not keys: a game that translates its menu
  should translate by id.
- **The saved format has a version** (`FORMAT_VERSION`, now 1). Data
  without one reads as version 1.
- **Objects find their own section** (`SETTINGS_TYPE` on each class:
  `Terrain`, `Atmospheres`, `Sky`, `SphericalClouds`, `CloudField` /
  `Clouds`, `Ocean`), so `apply()` and `settings_for()` work for any mix of
  them.

What the schema does not say:

- **Dependencies between knobs.** Foam detail means nothing with foam off,
  and the planet-cloud knobs (overlap, octaves, shells, tile budgets) mean
  nothing to the flat cloud field. A menu shows them all.
- **Which knobs matter to a given scene.** A game without an ocean simply
  never builds the ocean, but its section is still in the schema.

## Every knob

The field defaults are HIGH, so `TerrainSettings()` is HIGH. The ids,
labels and ranges below are what the schema gives.

### `terrain`: Terrain (`TerrainSettings`)

| Id | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `min_level` | Finest detail dropped | live | 0 to 8 levels | 3 | 1 | 0 | 0 |
| `split_factor` | Detail distance | live | 1.25 to 3 x | 1.5 | 1.75 | 2 | 2.5 |
| `detail_far` | Texture distance | live | 500 to 10000 m | 1500 | 2500 | 4000 | 6000 |
| `anisotropy` | Texture filtering | live | 1x, 2x, 4x, 8x, 16x | 2x | 4x | 8x | 16x |
| `textured` | Textures | live | on / off | on | on | on | on |
| `reflected` | In reflections | live | on / off | off | on | on | on |
| `node_cache_size` (advanced) | Chunks kept on the GPU | live | 300 to 4000 chunks | 800 | 1000 | 1200 | 1800 |
| `store_size` (advanced) | Chunks kept in memory | live | 500 to 6000 chunks | 1200 | 1600 | 2000 | 3000 |
| `uploads_per_frame` (advanced) | Uploads per frame | live | 1 to 32 chunks | 4 | 6 | 6 | 8 |
| `upload_ms_per_frame` (advanced) | Upload time per frame | live | 0.5 to 16 ms | 2.5 | 3 | 4 | 6 |
| `select_every` (advanced) | LOD pass period | live | 1 to 60 frames | 15 | 15 | 15 | 15 |
| `select_every_on_upload` (advanced) | LOD pass period while loading | live | 1 to 15 frames | 3 | 3 | 3 | 3 |
| `texture_size` | Texture size | restart | 256, 512, 1024, 2048, 4096 | 512 | 1024 | 2048 | 2048 |
| `workers` (advanced) | Generation processes | restart | 0 to 64 | 0 | 0 | 0 | 0 |

### `atmosphere`: Air (`AtmosphereSettings`)

| Id | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `steps` | Quality | live | 2 to 16 samples | 4 | 6 | 8 | 12 |
| `vertex_steps` (advanced) | Quality on stars and clouds | live | 1 to 8 samples | 2 | 3 | 4 | 8 |

### `sky`: Sky (`SkySettings`)

| Id | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `star_fraction` | Stars | live | 0.05 to 1 x | 0.4 | 0.7 | 1 | 1 |
| `glow_size` (advanced) | Milky Way detail | rebuild | 64, 128, 256, 512 | 128 | 256 | 256 | 512 |

### `clouds`: Clouds (`CloudSettings`)

| Id | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `count_scale` | Billboards | rebuild | 0.1 to 4 x | 0.25 | 0.5 | 1 | 2 |
| `march_scale` | Lighting | live | 0.25 to 2 x | 0.44 | 0.67 | 1 | 1.5 |
| `overlap_near` (advanced) | Overlap, near | rebuild | 0.5 to 4 | 1.5 | 2 | 2.5 | 2.5 |
| `overlap_far` (advanced) | Overlap, far | rebuild | 0.5 to 4 | 1 | 1.5 | 2.5 | 2.5 |
| `shell_octaves` | Detail | rebuild | 4-3-3-2, 4-4-3-3, 4-4-4-3, 4 | 4-3-3-2 | 4-4-3-3 | 4-4-4-3 | 4 |
| `shells` (advanced) | Billboard distance | rebuild | 1 to 6 shells | 4 | 4 | 4 | 4 |
| `reflected` | In reflections | live | on / off | off | on | on | on |
| `half_resolution` | Half-resolution clouds | live | on / off | off | off | off | off |
| `uploads_per_frame` (advanced) | Tile uploads per frame | live | 1 to 32 | 3 | 3 | 3 | 3 |
| `resorts_per_frame` (advanced) | Tile re-sorts per frame | live | 1 to 32 | 4 | 4 | 4 | 4 |
| `tile_cache` (advanced) | Tiles kept | live | 0 to 2000 tiles | 160 | 160 | 160 | 160 |
| `resort_frames` (advanced) | Re-sort period (flat clouds) | live | 1 to 60 frames | 8 | 8 | 8 | 8 |

### `ocean`: Sea (`OceanSettings`)

| Id | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `wave_iterations` | Wave detail | live | 1 to 64 octaves | 16 | 24 | 36 | 48 |
| `aa_full_px` | Wave sharpness | live | 2 to 32 px | 12 | 10 | 8 | 6 |
| `aa_gone_px` (advanced) | Wave cut-off | live | 2 to 32 px | 6 | 5 | 4 | 3 |
| `rough_fresnel` | Distant water roughness | live | on / off | off | on | on | on |
| `reflection_blur` | Reflection blur | live | 0 to 1 | 0 | 0.25 | 0.25 | 0.25 |
| `depth_colour` | Shallow water colour | live | on / off | on | on | on | on |
| `foam` | Shoreline foam | live | on / off | on | on | on | on |
| `foam_noise` (advanced) | Foam detail | live | on / off | off | on | on | on |
| `swell_by_depth` | Swell calms in shallows | live | on / off | on | on | on | on |
| `reflection_scale` | Reflection quality | rebuild | 0.1 to 1 x | 0.25 | 0.35 | 0.5 | 0.75 |
| `reflection_fov_scale` (advanced) | Reflection margin | live | 1 to 2 x | 1.1 | 1.1 | 1.1 | 1.1 |

### `asteroids`: Asteroids (`AsteroidSettings`)

| Id | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `min_pixels` | Smallest rock drawn | live | 0.5 to 8 px | 4 | 3 | 2 | 1.5 |
| `lod_error_px` | Rock detail | live | 0.25 to 6 px | 3 | 2 | 1 | 0.5 |
| `max_lod` | Finest rock mesh | rebuild | 16, 32, 64 | 16 | 32 | 32 | 64 |
| `normal_size` | Rock detail maps | rebuild | 64, 128, 256 px | 64 | 128 | 128 | 256 |
| `instance_budget` | Rocks drawn (in view) | live | 1000 to 200000 | 15000 | 40000 | 80000 | 150000 |
| `held_budget` (advanced) | Rocks kept (every direction, CPU) | live | 1000 to 2000000 | 80000 | 150000 | 300000 | 600000 |
| `detail_far` | Rock texture distance | live | 500 to 10000 m | 1000 | 2000 | 3000 | 5000 |
| `texture_size` | Rock texture size | restart | 256, 512, 1024, 2048 px | 256 | 512 | 1024 | 2048 |
| `terrain_min_radius` (advanced) | Detailed rocks from | live | 50 to 5000 m | 200 | 200 | 200 | 200 |
| `max_terrain_rocks` | Detailed rocks | live | 0 to 32 | 0 | 4 | 8 | 16 |

### `shadows`: Shadows (`ShadowSettings`)

| Id | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `cloud_shadow_steps` | Cloud shadows | live | 0 to 16 samples | 0 | 4 | 8 | 12 |
| `terrain_map_size` | Mountain shadows | rebuild | off, 256, 512, 1024, 2048 px | 256 | 512 | 1024 | 2048 |
| `local_map_size` | Nearby ridges' shadows | rebuild | off, 512, 1024, 2048 nodes | off | 512 | 1024 | 2048 |
| `local_map_spacing` (advanced) | Nearby ridges' detail | rebuild | 20 to 200 m | 50 | 50 | 50 | 50 |
| `cloud_map_size` | Cloud shadow maps | rebuild | off, 512, 1024, 2048 px | off | 512 | 1024 | 2048 |

## Notes

- **The terrain's finest detail** (`min_level`) counts levels dropped below
  the finest. Level 0 is about 2 m of vertex spacing at any planet radius,
  so the knob means the same on every planet.
  - MID drops one level (about 3.6 m) and LOW three (about 14 m).
  - Collision queries follow it, because they read the drawn grid, so the
    ground a ship hits stays the one on screen.
- **The air's steps** are a uniform, not a `#define`. They are written in
  place on every bound node, so they change live everywhere. The change also
  made HIGH *faster*: the loop is no longer unrolled, and the frames
  measured 13.3–13.7 ms instead of 15.8 ms at the sea-level night pose. The
  images are unchanged.
- **The flat cloud field** takes the same `CloudSettings`, but only its
  billboard count, sun march, reflection and re-sort period. The other
  fields belong to the planet's clouds.
- **The sea's reflection** is sized from the main camera's display region
  (`app.cam`). A game that renders its scene at a lower resolution than the
  window gets a reflection sized to that resolution, with nothing to set.

## Measured

`flight_bench.py --flight low` (300 m up at about 1,500 m/s, offscreen
1600×900, GTX 1070), before MID and LOW started dropping LOD levels:

| Preset | fps | Frame median | Frame p95 |
|---|---|---|---|
| `main` before this work | 51.4 | 18.7 ms | 28.8 ms |
| HIGH | 49.8 | 19.5 ms | 29.2 ms |
| LOW | 93.2 | 10.4 ms | 15.6 ms |

LOW's visible trades: the sea reflects only the sky (no terrain and no
clouds in it), there are fewer and larger cloud billboards, the textures
stop at 1.5 km, and the ground is three levels coarser.

## The demo

`scripts/demo_terrain.py --quality low|mid|high|ultra` (or a
`--<section>-quality` override per part) starts at a preset. In flight:

- F1–F5 cycle one part's preset: terrain, air, sky, clouds, ocean.
- F6 cycles them all.

`terrain_shots.py` and `flight_bench.py` take the same arguments
(`add_world_arguments`, `settings_from_args`).
