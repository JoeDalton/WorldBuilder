# World files

A whole world, in one line of a game:

```python
from world_builder import load_world

world = load_world(app, "my_world.yaml")      # once: lens, bodies, fields, sky, sun, lights
...
world.update(camera_in_the_world)             # every frame, once the camera is placed
...
world.destroy()
```

[`world_builder.world_file`](../../src/world_builder/world_file.py) reads the
file, builds the [`World`](world.md) it describes, puts its depth
camera on it and returns a `GameWorld`: `.world` (the `World`), `.depth`,
`.spec` (what the file defined), `.body(name)`, `.field(name)`, `update` and
`destroy`. The camera stays the game's: it sits at render's origin, and the
game gives `update` its position in the world frame (float64, metres).
`GameWorld.views` is a list of more cameras (a rear-view mirror's camera
NodePath): while one is in it, the asteroid fields and rings are drawn for it
too (their rocks are held and drawn for every view), with no second update.
The terrain, sea, trees and clouds are the main camera's alone. Add the
camera when the mirror is shown, remove it when it is not.
An app without a window (`window-type none`: a game's headless runs) gets
no depth camera (`.depth` is None) and no seas, which both draw through the
window; the rest is built and updated as usual.

`load_world(app, source, *, keep_outs=None, settings=None, backend="process",
clock=None, collision_mask=...)` takes the file, or a `WorldSpec` (what `parse_world` returns: a
game can change it and load it, no file in between), and what is about running
the world, not defining it: the room a level keeps clear in some fields
(`keep_outs={"field name": [KeepOut, ...]}`, added to the file's; an unknown
field name is a `ConfigError`), every terrain's into mask for collisions (`collision_mask`: bodies' ground and
fields' rocks are all solid from the start, nothing to enable), the quality
(`WorldSettings`), where chunks are generated, the game's clock, and any other
`World` argument that is the game's (`lights`, `light_scale`, `light_period`).
`parse_world(path)` reads a file without a window (to check it, or to read what
a world is made of); `world_to_yaml(spec)` writes one back.

**A world file is a definition, nothing else.** It has every knob of every
object (no presets), and none of what is about running it: no quality level, no
camera, no clock. A game keeps its worlds wherever it likes; the shipped ones
are in `src/world_builder/worlds/`:

```
python scripts/demo_world.py                                  # three_planets.yaml
python scripts/demo_world.py src/world_builder/worlds/deep_space.yaml
python scripts/demo_world.py my_world.yaml --body planet --lat 12 --lon 40 --alt 900
```

## Keys

Each key is a field of the same name of a look or a spec of the library, so
every knob is there, and a field added to one later is a key with no change to
the loader ([`config.py`](../../src/world_builder/config.py)). An omitted key
takes the field's default; an unknown key, a missing required one, a wrong
type or a value the library refuses is a `ValueError` naming where it is
(`bodies[1].planet.elevation.plates: 4.5 is not a whole number`).

```yaml
sun: {direction: [x, y, z], color: [r, g, b], angular_radius: ..}   # toward the sun
sky: {stars: {...}, star_intensity, star_brightness, nebula: {...}, sun_radiance,
      exposure, star_hpr, star_adaptation}        # SkyLook; absent or null = no skybox
space_ambient: [r, g, b]                          # sky light away from every body
bodies:
  - name: planet                    # what GameWorld.body(name) finds it by
    position: [x, y, z]             # metres, world frame
    mass: 5.97e24                   # or density: kg/m^3 (the mass of that sphere);
                                    # exactly one. The gravity constant is the game's
    planet: {...}                   # PlanetSpec: every field, elevation, climate, stamps, ...
    atmosphere: {...}               # AtmosphereParams (composition, heights), or null: airless
    clouds: {...}                   # CloudLook, or null (clouds need air)
    ocean: {...}                    # OceanLook, or null (needs planet.has_ocean)
    ambient: [r, g, b]
    exposure: 0.9
fields:
  - name: ring
    position: [x, y, z]
    axis: [x, y, z]                 # which way the field's z axis (a ring's normal)
                                    # points; or orientation: [w, x, y, z]
    spec: {...}                     # FieldSpec: shape, density, sizes, spin, shapes, ...
    ambient, exposure, far, haze, keep_outs
```

Conventions of the few types that are not plain mappings of fields:

- **Union**: a field's `shape` and a spin's `axis` say which with `type`
  (`box`, `sphere`, `annulus`; `isotropic`, `around_axis`).
- **Distribution**: `{kind: power_law, q: 3.0, lo: 50, hi: 2000}`, the
  factory of that name with its arguments (`uniform`, `normal`, `lognormal`,
  `power_law`, `u_shaped`, `table`).
- **Enums** are their values (`cloud_type: cumulus`, a pin's `preset:
  hurricane`).
- **Global and pinned weather**: `clouds.weather` is a `WeatherSpec` (seed,
  time, systems, season, zonal cover, jets, cirrus, ...) with its `pins`:
  `{lat, lon, preset, heading_deg}`. Its planet is the body's, never written
  in the file. `clouds.genera` has the genera and their `features`; its weather
  is the cloud's own unless it gives one.
- **Stamps and paints** (`planet.stamps`, `paints`, `density_paints`) place
  an image by latitude, longitude and size in metres (`placement: {lat, lon,
  width, height, rotation}`); image paths are relative to the world file's
  folder.
- **Stars**: `sky.stars` is a `Starfield`'s arguments (`seed`, `stars`,
  `galactic_pole`, ...); null is the default one. `nebula.palette` is a list of
  colours or a palette name (`violet`, `emerald`, ...).
- Numbers: PyYAML reads `1e6` as text (it wants `1.0e+6`); the loader accepts
  both. Quote hex colours (`"#808080"`).

## Links between files

Any block may be `{file: other.yaml}`: the content of that file, in its place.
The file must be **in the same folder as the world file** (a plain file name;
a path is an error), wherever that is, so a game's folder of worlds moves as a
whole. Keys beside `file` replace the file's own, so one body definition serves
many places:

```yaml
bodies:
  - {name: planet, file: kerbin.yaml, position: [0, 0, 0]}
  - {name: moon,   file: moon.yaml,   position: [-1.9e6, -1.1e6, 2.09e7]}
fields:
  - {name: ring, file: ring.yaml, position: [6.0e7, 1.5e7, 0], axis: [0.25, 0.1, 1.0]}
```

Links nest; a circle is an error.

## The shipped worlds

- `sky.yaml` (the sky, linked by both worlds), `stamps/` (the images
  `kerbin.yaml` places).
- `three_planets.yaml`: `kerbin.yaml` (a 600 km earthlike planet: air, cumulus
  and cirrus, a sea, four hand-made stamps and three paints), `moon.yaml` (an airless 200 km
  moon 21,000 km away), `ice.yaml` (a cold dry planet 3000 km across, with air
  and clouds) and `ring.yaml` (a ring of ice rocks, ringlets and a gap, from 5,000
  to 10,000 km, tilted, around the ice planet).
- `deep_space.yaml`: `asteroid_cube.yaml`, a 15 km cube of rocks in open space,
  with a clear corridor to fly through.

They are written out in full (the library keeps no presets: every planet and
field is defined in a file), and
`earth.yaml`, `desert.yaml` and `forest.yaml` are further planets to link from
a world of your own. `world_builder.load_planet("desert")` reads one planet and
`load_field("ring")` one field (a name of this folder, or a path; each file is
read once). The tests read these files too: `earthlike.yaml` is the plain
default planet, `tiny.yaml` the 20 km one (`tiny_world.yaml` is it alone under
the sun, for the tree scripts).

## Where the code is

- `config.py`: the reader and writer of the looks (`from_config`,
  `to_config`, `dump_yaml`), strict, with the links.
- `world_file.py`: the world file's keys, `load_world` and `GameWorld`.
- `World` and the looks (`BodyLook`, `FieldLook`, ...) are what a world file
  builds; a script that varies a planet by its command line may still build
  them directly.
