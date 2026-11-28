# A whole world

[`world_builder.world`](../../src/world_builder/world.py)'s `World` builds and
updates every WorldBuilder part a game draws around its camera:
- each body's terrain, air, planet clouds and sea;
- the sky and the sun;
- Panda3D lights for everything else.

A game does not build it by hand: it writes its world in a
[world file](world_file.md) and loads it with `world_builder.load_world`, which
builds this `World` and its depth camera in one line. This page is what
that builds, for code that needs to reach into it (`GameWorld.world`), or that
builds a world itself, for instance one varied from its own settings.

The split is between what the world looks like and how it works:
- **The game decides the look:** which bodies there are and where, the sun,
  the colours, the clouds and the swell. It passes these as small frozen
  dataclasses.
- **`World` does the work:** building the parts, binding them to the air,
  sharing one worker pool, and updating them every frame.

The same world, built in code instead of read from a file:

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
depth = ReversedDepthCamera(app)  # after the world: see "Order" below
world.use_depth_camera(depth)

# Every frame, once the camera is placed:
depth.update(world.far_distance(origin_world))
world.update(origin_world, far=depth.far)

world.destroy()
```

`scripts/demo_world.py` flies a world file's world.

**Open space.** A world may have no body: `World(app, [], sun=..., sky=...,
space_ambient=...)` is the sky, the sun and the Panda3D lights alone, for a
space level. Its light everywhere is the bare sun and `space_ambient` (the
sky light away from any body), `loading` is always False, and `far_distance`
is at least `SPACE_FAR_M` (1,000 km), farther where a field's rocks are drawn
farther (see below). An asteroid field (`world_builder.asteroids`) can share its worker
pool (`executor=world.executor`).

## Looks

Each look field is an argument of the same name of the part it configures,
with the same default. A test checks this, so the looks never drift from the
parts. (A world *file* has no defaults: see `world_file.md`.)

- **`BodyLook`**: one planet or moon.
  - `name`: what `World.body(name)` finds it by.
  - `spec`: a planet (a preset, or built from a world file's `planet` block).
  - `position`: its centre, in the world frame.
  - `atmosphere`: `AtmosphereParams`, or None for an airless body.
  - `clouds`: a `CloudLook`, or None. Clouds need air.
  - `ocean`: an `OceanLook`, or None. An ocean needs `spec.has_ocean`.
    Without one, the terrain still draws its sea, as flat water only.
  - `ambient` and `exposure`: the terrain's sky light and tonemap.
- **`CloudLook`**: `PlanetClouds`' arguments: altitude, cloud type,
  `cirrus` (True: the weather's high cirrus above the low clouds, drawn by
  the march's high layer; default False), weather, seeds, shading,
  `genera`. Its `exposure` (0.9) feeds the sky's tonemap, as the sky's and
  the terrain's do; keep them equal, or hazed clouds stop matching the sky
  behind them. The cirrus's slab (11-12.5 km) must clear the low clouds'
  and stay inside the air, or the world refuses it.
- **`OceanLook`**: `Sea`'s arguments: water colour, wind, swell, foam,
  `near_band` (the altitudes over which the near look fades into the far
  one) and `anchor_range` (how far the waves stay anchored before
  re-anchoring). Its `tuned_elevation` is the world's own: the sun
  elevation the water colour was tuned in.
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
  - `demo_world.py`, whose camera sits at the origin, passes the camera's.

  `World` adds the camera's own render position, read from `app.camera`, to
  get the camera's world position. Everything is placed around that.

## Every frame

`World.update(origin_world, far=None)`. `far` is the view's far distance this
frame (`ReversedDepthCamera.far`). It defaults to `far_distance()`.

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
5. each body's sea (`Sea.update`):
   - its swell, drifted to the game time and handed to the terrain, its
     query and its collision solid;
   - its near look's share by altitude (`near_band`), which also turns the
     reflection pass on or off;
   - its frame, the tangent plane at the camera's nadir, with the waves
     anchored to the ground through a chart;
6. Panda3D's sun and ambient lights, set to the light at the camera
   (`World.light_at`), and each tracked model's at the model: the air reddens the sun, an airless body's shadow
   hides it, and the other bodies eclipse it. Their colours are multiplied by `light_scale`. A game
   whose models use Panda3D's fixed-function lighting scales them down, since
   that lighting has no tone map and clips the physical noon sun (about 2.5)
   to white (see [Lighting on objects](object_lighting.md));
   a model tracked with `shader=True` takes its nearest body's model shader
   inputs instead (`Body.models`). With them, each near sea's light, relative
   to the daylight its colour was tuned in. `World(light_period=N)` sets
   these lights every N frames, their reckoning spread over the frames
   before in `LIGHT_STAGES` stages (one a frame at 3: no frame takes it
   whole); they are reckoned at once when the sun moves or the tracked
   models change;
7. the casters' shadow map (`World.casters`), fitted to the tracked models
   that cast (see [Shadows](shadows.md)).

## Order and the depth camera

Build the world before its `ReversedDepthCamera` (`world_builder.depth`): the
sea's reflection camera copies the app's lens once when it is built. Then hand
the camera to the world, `world.use_depth_camera(depth)`: the sky, the seas'
reflections and the fields' haze take its depth convention, and the clouds'
raymarch draws its passes with it.

The camera draws the view with a reversed, infinite projection (near at 1,
infinity at 0) and keeps a float depth buffer's precision at every range: a
planet from orbit and a rock under the camera in one view, one cull and one
draw (#57; the split cameras it replaced made two). A window's framebuffer has
no float depth, so it draws into a scene buffer of its own, presented in the
window (or in the game's post-process buffer, where the app's camera drew)
under the other regions; `scene_buffer=False` for an output that has float
depth already (`depth.float_depth_fbprops()`). Load `gl-depth-zero-to-one #t`
in the game's PRC before ShowBase opens its window: OpenGL's 0-1 depth keeps
the float precision. macOS's OpenGL has no clip control: there the depth keeps
about a 24-bit range's precision (far bodies' depths may fight). The near
plane stays at 1 m everywhere: one following the altitude cut the cockpit and
close ships. It culls to the world's far distance (`depth.frustum_bounds`:
six planes, exact in every model's frame; Panda's own frustum stops at
10,000 km, and a box's corners that far out cost its planes metres of
precision near the camera). The world's `render` settings (render scale,
MSAA, FXAA, FSR 1) set its scene buffer's size and samples, and the passes
that show it in the window. See
[Render scale, anti-aliasing and FSR 1](render_pipeline.md).

Load `transform-cache false` in the game's PRC (before ShowBase opens its
window). The world moves around the camera, so every drawn node's net
transform is new every frame and Panda's cache of composed transforms only
fills up for its garbage collection to empty: about 1 ms a frame in low
flight (see [performance](performance.md) §19). The library sets no PRC of
its own; `scripts/tools.window_config` does.

A game's own translucent volumes, such as shield bubbles, need nothing: one
camera draws them once.

## Other members

- **`far_distance(origin_world)`**: how far the view reaches, over every body
  (`depth.system_far`), and every asteroid field's rocks too
  (`AsteroidField.reach`: a shorter far plane cut the biggest while they were
  still a few pixels wide): the camera's `update` argument.
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
  sun and ambient lights (the world's off on it), set by `update` in one
  vectorised call (every `light_period` frames), times `light_scale`. Place the model before
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
