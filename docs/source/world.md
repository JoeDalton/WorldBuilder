# A whole world

[`world_builder.world`](../../src/world_builder/world.py)'s `World` builds and
updates every WorldBuilder part a game draws around its camera:
- each body's terrain, air, planet clouds and sea;
- the sky and the sun;
- Panda3D lights for everything else.

It does the wiring that `scripts/demo_terrain.py` does by hand.

The split is between what the world looks like and how it works:
- **The game decides the look:** which bodies there are and where, the sun,
  the colours, the clouds and the swell. It passes these as small frozen
  dataclasses.
- **`World` does the work:** building the parts, binding them to the air,
  sharing one worker pool, and updating them every frame.

```python
world = World(
    app,
    bodies=[
        BodyLook(
            "planet",
            spec,
            atmosphere=AtmosphereParams(),
            clouds=CloudLook(altitude=2200.0),
            ocean=OceanLook(swell_amplitude=20.0),
        ),
        BodyLook("moon", moon_spec, position=moon_centre, ambient=(0.025,) * 3),
    ],
    sun=SunLook(direction=toward_the_sun),
    sky=SkyLook(),
    settings=world_settings,
)
depth = SplitDepthCameras(app)  # after the world: see "Order" below

# Every frame, once the camera is placed:
depth.update(*world.depth_ranges(origin_world))
world.update(origin_world, far=depth.far)

world.destroy()
```

`scripts/demo_world.py` is the terrain demo rebuilt this way.

## Looks

Each look field is an argument of the same name of the part it configures,
with the same default. A test checks this, so the looks never drift from the
parts.

- **`BodyLook`**: one planet or moon.
  - `name`: what `World.body(name)` finds it by.
  - `spec`: a preset, or a level file's.
  - `position`: its centre, in the world frame.
  - `atmosphere`: `AtmosphereParams`, or None for an airless body.
  - `clouds`: a `CloudLook`, or None. Clouds need air.
  - `ocean`: an `OceanLook`, or None. An ocean needs `spec.has_ocean`.
    Without one, the terrain still draws its sea, as flat water only.
  - `ambient` and `exposure`: the terrain's sky light and tonemap.
- **`CloudLook`**: `SphericalClouds`' arguments: altitude, cloud type,
  weather, seeds, shading.
- **`OceanLook`**: `Sea`'s arguments: water colour, wind, swell, foam,
  `near_band` (the altitudes over which the near look fades into the far
  one) and `anchor_range` (how far the waves stay anchored before
  re-anchoring). The other fields are the world's own:
  - `tuned_elevation`: the sun elevation the water colour was tuned in;
  - `haze_color` and `haze_distance`: ignored (the atmosphere hazes the
    sea), kept so older looks still build.
- **`SunLook`**: the direction toward the sun and its colour above any air.
- **`SkyLook`**: `Sky`'s arguments: stars, nebula, sun disc, exposure.

Quality is not part of the look. It comes in as `settings=`, a
`WorldSettings` (see [Quality settings](quality.md)), and
`World.set_settings` changes it live.

## Frames

- **World frame.** Float64 metres, with render's axes: nothing is rotated.
  Body positions and the sun's direction are given in it.
- **Render frame.** Float32. The game keeps render's origin near the camera.
- **`origin_world`.** Where render's origin is in the world frame. Only the
  game knows it:
  - a game that keeps its player at the origin passes the player's world
    position;
  - the demo, whose camera sits at the origin, passes the camera's.

  `World` adds the camera's own render position, read from `app.camera`, to
  get the camera's world position. Everything is placed around that.

## Every frame

`World.update(origin_world, far=None)`. `far` is the far camera's far
distance this frame; the sky is drawn just inside it. It defaults to
`depth_ranges()`'s far value.

In order, it updates:
1. each body's terrain, placed and detailed around the camera, its collision
   node following;
2. the air (`Atmospheres.update`);
3. the sky;
4. each body's clouds, with the angle one pixel spans, taken from the
   camera's own output (a window, or a game's offscreen buffer);
5. Panda3D's sun and ambient lights, set to the light at the camera
   (`World.light_at`): the air reddens the sun, and an airless body's
   shadow hides it. Their colours are multiplied by `light_scale`. A game
   whose models use Panda3D's fixed-function lighting scales them down, since
   that lighting has no tone map and clips the physical noon sun (about 2.5)
   to white (see [Lighting on objects](object_lighting.md));
6. each body's sea (`Sea.update`):
   - its swell, drifted to the game time and handed to the terrain, its
     query and its collision solid;
   - its near look's share by altitude (`near_band`), which also turns the
     reflection pass on or off;
   - its frame, the tangent plane at the camera's nadir, with the waves
     anchored to the ground through a chart;
   - while the near look shows, its light, relative to the daylight its
     colour was tuned in.

## Order with split depth

Build the world before `SplitDepthCameras`. The sea's reflection camera
copies the app's lens once when it is built, and the split cameras then
shorten that lens to the near range every frame.

A game's own translucent volumes, such as shield bubbles, go to
`depth.update(split, far, translucent=...)`. The split then never cuts one:
the two cameras overlap by 2%, and they would blend a volume twice in that
band (see `world_builder.depth`).

## Other members

- **`depth_ranges(origin_world)`**: split and far distances over every body
  (`SplitDepthCameras.system_ranges`).
- **`loading`**: True while any terrain waits for ground or any clouds for
  tiles. Wait for it to stay False for a moment before trusting the ground
  (see the terrain's loading phase in [Terrain](terrain.md)).
- **`enable_collision(into_mask, bodies=None, **kw)`**: makes bodies' drawn
  terrain solid to the game's collision traverser.
- **`set_sun(direction)`**: moves the sun; every part follows.
- **`body(name)`**: a body's terrain, clouds and sea, for tools and debug
  views.
- **`destroy()`**: removes every part and the lights, and stops the worker
  pool.
