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
            mass=5.97e24,
        ),
        BodyLook("moon", moon_spec, position=moon_centre, ambient=(0.025,) * 3, mass=7.35e22),
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

**Open space.** A world may have no body: `World(app, [], sun=..., sky=...,
space_ambient=...)` is the sky, the sun and the Panda3D lights alone, for a
space level. Its light everywhere is the bare sun and `space_ambient` (the
sky light away from any body), `loading` is always False, and `depth_ranges`
gives a fixed split (`SPACE_SPLIT_M`, 2 km) and far plane (`SPACE_FAR_M`,
1,000 km). An asteroid field (`world_builder.asteroids`) can share its worker
pool (`executor=world.executor`).

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
  weather, seeds, shading. Its `exposure` (0.9) feeds the sky's tonemap, as
  the sky's and the terrain's do; keep them equal, or hazed clouds stop
  matching the sky behind them.
- **`OceanLook`**: `Sea`'s arguments: water colour, wind, swell, foam,
  `near_band` (the altitudes over which the near look fades into the far
  one) and `anchor_range` (how far the waves stay anchored before
  re-anchoring). The other fields are the world's own:
  - `tuned_elevation`: the sun elevation the water colour was tuned in;
  - `haze_color` and `haze_distance`: ignored (the atmosphere hazes the
    sea), kept so older looks still build.
- **`FieldLook`**: an asteroid field or a planet's ring
  (`AsteroidField`'s arguments, see [Asteroid fields](asteroids.md)):
  `name` (what `World.field(name)` finds it by), `spec`, `position`,
  `orientation` (a ring lies in its field's xy plane), `keep_outs`,
  `exposure`, `far`, `haze`. Its `ambient` defaults to the world's
  `space_ambient`. Given as `World(fields=[...])`; it takes the world's pool,
  `asteroids` settings, sun, clock (the rocks turn with the game's time) and
  the bodies' textures. Its rocks are massless.
- **`SunLook`**: the direction toward the sun, its colour above any air,
  and its `angular_radius` (radians): the disc the sky draws and every
  eclipse's penumbra ([Shadows](shadows.md)).
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
   node following, then each asteroid field (its rocks in range, drawn and
   solid, its haze);
2. the air (`Atmospheres.update`) and the bodies as sun occluders
   (`World.shadows`, a `Shadows`: each body eclipses the sun for the others,
   see [Shadows](shadows.md)); with the terrain, each body's mountain
   shadow maps (`Body.shadow_map`, the global one and the local one
   around the camera) collect what the pool baked; after the clouds, their
   shadow maps (`Body.cloud_maps`) are drawn again if they must be;
3. the sky;
4. each body's clouds, with the angle one pixel spans, taken from the
   camera's own output (a window, or a game's offscreen buffer);
5. Panda3D's sun and ambient lights, set to the light at the camera
   (`World.light_at`), and each tracked model's at the model: the air reddens the sun, an airless body's shadow
   hides it, and the other bodies eclipse it. Their colours are multiplied by `light_scale`. A game
   whose models use Panda3D's fixed-function lighting scales them down, since
   that lighting has no tone map and clips the physical noon sun (about 2.5)
   to white (see [Lighting on objects](object_lighting.md));
   a model tracked with `shader=True` takes its nearest body's model shader
   inputs instead (`Body.models`);
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
shorten that lens to the near range every frame. Then hand the split
cameras to the world, `world.use_split_depth(depth)`: half-resolution
clouds (`CloudSettings.half_resolution`) mirror them, and stay at full
resolution without them.

Load `transform-cache false` in the game's PRC (before ShowBase opens its
window). The world moves around the camera, so every drawn node's net
transform is new every frame and Panda's cache of composed transforms only
fills up for its garbage collection to empty: about 1 ms a frame in low
flight (see [performance](performance.md) §19). The library sets no PRC of
its own; the demos do.

A game's own translucent volumes, such as shield bubbles, go to
`depth.update(split, far, translucent=...)`. The split then never cuts one:
the two cameras overlap by 2%, and they would blend a volume twice in that
band (see `world_builder.depth`).

## Other members

- **`depth_ranges(origin_world)`**: split and far distances over every body
  (`SplitDepthCameras.system_ranges`).
- **`gravity_field(g_constant=1.0)`**: the bodies' gravity as point masses
  (`BodyLook.mass`, keyword only); `acceleration(points)` is the raw vector
  for N points at once (numpy, or native: 4-9x faster), `-g` is "up". The
  field is capped inside a body, at its surface value.
- **`loading`**: True while any terrain waits for ground, any clouds for
  tiles, or any field for its prototypes and the rocks in range. Wait for it to stay False for a moment before trusting the ground
  (see the terrain's loading phase in [Terrain](terrain.md)).
- **`enable_collision(into_mask, bodies=None, **kw)`**: makes bodies' drawn
  terrain solid to the game's collision traverser.
- **`set_sun(direction)`**: moves the sun; every part follows.
- **`light_at(point)`, `lights_at(points)`**: the light (direct, ambient)
  at one point or many, each by its nearest body, in the world's shadows;
  **`sun_visibility(points)`** is those shadows' share alone (the
  eclipses, the clouds', the mountains': see [Shadows](shadows.md)).
  `lights_at(points, local=False)` leaves the clouds' and the mountains'
  out, for what a shader shades per pixel (the near sea).
- **`track_light(node, cast=False, shader=False)`, `untrack_light(node)`**:
  a model lit by the light where it is, not at the camera: its own Panda3D
  sun and ambient lights (the world's off on it), set at every `update` in
  one vectorised call, times `light_scale`. Place the model before
  `update`. With `cast=True` it casts shadows on the terrain, the sea and
  the shaded models (`World.casters`, see [Shadows](shadows.md)). With
  `shader=True` the world's model shader draws it: lit and tonemapped as
  the terrain, every world shadow per pixel, the air in front (see
  [Lighting on objects](object_lighting.md)); `Body.models` holds its
  inputs near each body.
- **`body(name)`**: a body's terrain, clouds and sea, for tools and debug
  views.
- **`field(name)`**: an asteroid field, to make it solid
  (`enable_collision`) or find clear spots (`scatter.find_clear`).
- **`destroy()`**: removes every part and the lights, and stops the worker
  pool.
