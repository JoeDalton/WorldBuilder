# Quality settings

Every cost-versus-quality trade in WorldBuilder is a *knob*. A game built on
it can show each knob on its own in a settings menu, or offer them bundled
as four presets: LOW, MID, HIGH and ULTRA.

- [`world_builder.quality`](../../src/world_builder/quality.py) holds the
  shared parts:
  - `QualityLevel`, the four levels. `CloudQuality` is the same enum, under
    the name the clouds gave it first.
  - `Knob`, what a menu needs to show one field.
  - `QualitySettings`, what every settings class has in common.
- Each drawn part has a frozen settings dataclass beside the code that reads
  it: `TerrainSettings`, `AtmosphereSettings`, `SkySettings`,
  `CloudSettings` and `OceanSettings`.
- [`world_builder.settings`](../../src/world_builder/settings.py)'s
  `WorldSettings` bundles them, one section each.

## How a knob applies

Each object takes `settings=` when it is built and exposes `.settings` and
`set_settings(new)`. A setting whose value is unchanged costs nothing, so a
menu's "apply" button may call `set_settings` freely. Each field's
`Knob.apply` says how a change takes effect:

| Mode | What happens |
|---|---|
| `live` | At once: a shader input, a budget or a cache size. Lower caches evict on the next update, never a chunk or tile the drawn view needs. |
| `rebuild` | The object rebuilds a part of itself. The clouds re-plan their shells and regenerate their tiles (a few seconds of refill). The ocean makes its reflection buffer or swell mesh again. The sky makes a new Milky Way texture. |
| `restart` | Only a newly built object takes it: the terrain's texture size and worker count. `set_settings` logs that it will apply at the next start. |

The objects' own one-purpose setters keep `.settings` true:
`Terrain.set_min_level`, `set_textured`, `set_reflected`, `Ocean.set_foam`
and `set_wave_iterations`.

## A game's settings menu

```python
from world_builder.settings import WorldSettings

settings = WorldSettings.from_dict(config.get("world"))  # never raises

# Build the world from it.
atmospheres = Atmospheres(settings.atmosphere)
terrain = Terrain(render, spec, atmospheres=atmospheres, settings=settings.terrain)
sky = Sky(render, Starfield(), atmospheres, settings=settings.sky)
clouds = SphericalClouds(render, radius, executor, settings=settings.clouds)
ocean = Ocean(game, settings=settings.ocean)

# The menu: a preset selector, then one row per knob.
for section, name, knob, value in settings.knobs():
    if knob.advanced and not show_advanced:
        continue
    # knob.label, knob.unit, knob.help; knob.range (a slider) or
    # knob.choices (labels -> values); knob.apply ("restart": say so)
    ...

settings = settings.with_value("terrain", "detail_far", 2500.0)  # clamped
settings = settings.with_level("low")                            # a preset
terrain.set_settings(settings.terrain)  # and atmospheres, sky, clouds, ocean
config["world"] = settings.to_dict()
```

`to_dict()` stores the preset and what differs from it, choices by their
label:

```json
{"preset": "mid", "overrides": {"terrain": {"detail_far": 3000.0},
                                "ocean": {"seabed_cascades": "low"}}}
```

`is_custom()` tells whether the settings still equal their preset. That is
how the demo's HUD shows "custom".

**Strict construction, lenient loading.** Building a settings object by hand
is strict: a value out of its range or choices raises `ValueError`.
`from_dict` / `from_overrides` / `with_value`, which read saved or typed
data, never raise; each problem is logged as a warning:

- an unknown preset is HIGH;
- an unknown section or knob is skipped;
- a value of the wrong type keeps the preset's;
- a value out of range is clamped.

Limits that involve two fields are applied where the settings are used: the
ocean's wave cut-off is never sharper than its full-strength limit, and the
terrain's GPU chunk cap is never above its RAM one.

## Every knob

The field defaults are HIGH, so `TerrainSettings()` is HIGH. The labels and
ranges below are the `Knob` metadata a menu reads.

### `terrain`: `TerrainSettings`

| Knob | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `min_level` | Terrain finest detail dropped | live | 0 to 8 levels | 1 | 0 | 0 | 0 |
| `split_factor` | Terrain detail distance | live | 1.25 to 3 x | 1.5 | 1.75 | 2 | 2.5 |
| `detail_far` | Terrain texture distance | live | 500 to 10000 m | 1500 | 2500 | 4000 | 6000 |
| `anisotropy` | Terrain texture filtering | live | 1x, 2x, 4x, 8x, 16x | 2x | 4x | 8x | 16x |
| `textured` | Terrain textures | live | on / off | on | on | on | on |
| `reflected` | Terrain in reflections | live | on / off | off | on | on | on |
| `node_cache_size` (advanced) | Terrain chunks kept on the GPU | live | 300 to 4000 chunks | 800 | 1000 | 1200 | 1800 |
| `store_size` (advanced) | Terrain chunks kept in memory | live | 500 to 6000 chunks | 1200 | 1600 | 2000 | 3000 |
| `uploads_per_frame` (advanced) | Terrain uploads per frame | live | 1 to 32 chunks | 4 | 6 | 6 | 8 |
| `upload_ms_per_frame` (advanced) | Terrain upload time per frame | live | 0.5 to 16 ms | 2.5 | 3 | 4 | 6 |
| `select_every` (advanced) | Terrain LOD pass period | live | 1 to 60 frames | 15 | 15 | 15 | 15 |
| `select_every_on_upload` (advanced) | Terrain LOD pass period while loading | live | 1 to 15 frames | 3 | 3 | 3 | 3 |
| `texture_size` | Terrain texture size | restart | 256, 512, 1024, 2048, 4096 | 512 | 1024 | 2048 | 2048 |
| `workers` (advanced) | Terrain generation processes | restart | 0 to 64 | 0 | 0 | 0 | 0 |

### `atmosphere`: `AtmosphereSettings`

| Knob | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `steps` | Air quality | live | 2 to 16 samples | 4 | 6 | 8 | 12 |
| `vertex_steps` (advanced) | Air quality on stars and clouds | live | 1 to 8 samples | 2 | 3 | 4 | 8 |

### `sky`: `SkySettings`

| Knob | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `star_fraction` | Stars | live | 0.05 to 1 x | 0.4 | 0.7 | 1 | 1 |
| `glow_size` (advanced) | Milky Way detail | rebuild | 64, 128, 256, 512 | 128 | 256 | 256 | 256 |

### `clouds`: `CloudSettings`

| Knob | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `count_scale` | Cloud billboards | rebuild | 0.1 to 4 x | 0.25 | 0.5 | 1 | 2 |
| `march_scale` | Cloud lighting | live | 0.25 to 2 x | 0.44 | 0.67 | 1 | 1.5 |
| `overlap_near` (advanced) | Cloud overlap, near | rebuild | 0.5 to 4 | 1.5 | 2 | 2.5 | 2.5 |
| `overlap_far` (advanced) | Cloud overlap, far | rebuild | 0.5 to 4 | 1 | 1.5 | 2.5 | 2.5 |
| `shell_octaves` | Cloud detail | rebuild | 4-3-3-2, 4-4-3-3, 4-4-4-3, 4 | 4-3-3-2 | 4-4-3-3 | 4-4-4-3 | 4 |
| `shells` (advanced) | Cloud billboard distance | rebuild | 1 to 7 shells | 4 | 4 | 4 | 4 |
| `reflected` | Clouds in reflections | live | on / off | off | on | on | on |
| `uploads_per_frame` (advanced) | Cloud tile uploads per frame | live | 1 to 32 | 3 | 3 | 3 | 3 |
| `resorts_per_frame` (advanced) | Cloud tile re-sorts per frame | live | 1 to 32 | 4 | 4 | 4 | 4 |
| `tile_cache` (advanced) | Cloud tiles kept | live | 0 to 2000 tiles | 160 | 160 | 160 | 160 |
| `resort_frames` (advanced) | Flat clouds' re-sort period | live | 1 to 60 frames | 8 | 8 | 8 | 8 |

### `ocean`: `OceanSettings`

| Knob | Label | Applies | Range or choices | LOW | MID | HIGH | ULTRA |
|---|---|---|---|---|---|---|---|
| `wave_iterations` | Wave detail | live | 1 to 64 octaves | 16 | 24 | 36 | 48 |
| `aa_full_px` | Wave sharpness | live | 2 to 32 px | 12 | 10 | 8 | 6 |
| `aa_gone_px` (advanced) | Wave cut-off | live | 2 to 32 px | 6 | 5 | 4 | 3 |
| `rough_fresnel` | Distant water roughness | live | on / off | off | on | on | on |
| `reflection_blur` | Reflection blur | live | 0 to 1 | 0 | 0.25 | 0.25 | 0.25 |
| `seabed_cascades` (advanced) | Seabed detail | live | low, mid, high, ultra | low | mid | high | ultra |
| `depth_colour` | Shallow water colour | live | on / off | on | on | on | on |
| `foam` | Shoreline foam | live | on / off | on | on | on | on |
| `foam_noise` (advanced) | Foam detail | live | on / off | off | on | on | on |
| `swell_by_depth` | Swell calms in shallows | live | on / off | on | on | on | on |
| `reflection_scale` | Reflection quality | rebuild | 0.1 to 1 x | 0.25 | 0.35 | 0.5 | 0.75 |
| `reflection_fov_scale` (advanced) | Reflection margin | live | 1 to 2 x | 1.4 | 1.4 | 1.4 | 1.4 |
| `swell_grid_subdivs` (advanced) | Swell mesh detail | rebuild | 64 to 1024 quads | 256 | 384 | 512 | 512 |

## Notes

- **The terrain's finest detail** (`min_level`) counts levels dropped below
  the finest. Level 0 is about 2 m of vertex spacing at any planet radius,
  so the knob means the same on every planet. A menu can show the metres
  with `terrain.grid.spacing(level)`.
  - Collision queries follow it, because they read the drawn grid, so the
    ground a ship hits stays the one on screen. That is why only LOW drops a
    level.
- **The air's steps** were a `#define` (8). They are now a uniform written in
  place on every bound node, so they change live everywhere. The change also
  made HIGH *faster*: the loop is no longer unrolled, and the frames
  measured 13.3–13.7 ms instead of 15.8 ms at the sea-level night pose. The
  images are unchanged.
- **The clouds' presets** are built from the old tables (`QUALITY_SCALES`,
  `VOLUME_FRACTIONS`, `SHELL_OCTAVES`, now in `clouds/cloud.py`), so each
  knob is independent while the presets look exactly as the old levels did.
  `at_quality` still works, through the same helpers (`scale_count`,
  `scale_march`).
- **The flat cloud field** takes the same `CloudSettings`, but only its
  billboard count, sun march, reflection and re-sort period. The other
  fields belong to the planet's clouds.
- **Compatibility.** These still work as in 0.1.0:
  - `CloudQuality` and its values;
  - `CloudField(quality=)`, and its read of
    `game.app.graphics_settings.config["clouds"]["quality"]`;
  - `Ocean(wave_iterations=, reflection_scale=)` and its read of
    `config["render"]["reflection_scale"]`;
  - `Terrain(min_level=)` and `SphericalClouds(quality=, shells=,
    volume_fractions=)`.

  The ocean's `OceanQuality` / `OCEAN_QUALITY` / `set_quality`, never
  released, are now `OceanSettings` / `OceanSettings.preset` /
  `set_settings`.

## Measured

`flight_bench.py --flight low` (300 m up at about 1,500 m/s, offscreen
1600×900, GTX 1070):

| Preset | fps | Frame median | Frame p95 |
|---|---|---|---|
| `main` before this work | 51.4 | 18.7 ms | 28.8 ms |
| HIGH | 49.8 | 19.5 ms | 29.2 ms |
| LOW | 93.2 | 10.4 ms | 15.6 ms |

`terrain_shots.py` frame times, offscreen:

| Pose | HIGH | LOW |
|---|---|---|
| ground | 14.2 ms | 6.8 ms |
| coast | 14.2 ms | 7.0 ms |
| sunset | 14.3 ms | 6.3 ms |
| beach | 16.7 ms | 7.4 ms |

LOW's visible trades: the sea reflects only the sky (no terrain and no
clouds in it), there are fewer and larger cloud billboards, and the textures
stop at 1.5 km.

The demo (`scripts/demo_terrain.py --quality low|mid|high|ultra`) cycles the
whole world's preset with **H**, the clouds' with F5 and the ocean's with
F11. `terrain_shots.py` and `flight_bench.py` take `--quality`, with
`--cloud-quality` and `--ocean-quality` as overrides.
