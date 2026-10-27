# Shadows

What hides the sun from a point. Every shader that lights something by the
sun multiplies its sunlight by a product of independent visibility terms,
and a CPU twin of each term lights what the CPU lights (Panda3D's lights,
the sea's near look). Issues #11 and #13; the plan is `SHADOWS_PLAN.md`.

| Term | Casters | Receivers | Status |
|---|---|---|---|
| Eclipse | spherical bodies (planets, moons) | terrain, far water, clouds, air, CPU light | done |
| Cloud shadow | the planet's clouds | terrain, sea, objects | done |
| Terrain shadow, global | mountains | terrain, sea, clouds, objects | done |
| Terrain shadow, local | nearby ridges | terrain, sea, clouds, objects | done |
| Ring shadow | rings | planet, clouds, air, objects | planned (I) |
| Object shadow | tracked models | terrain, sea, models | planned (H) |

The planet's own shadow is not one of them: it is the air's
(`atmoSunlight`, `atmoShadow`, see [the atmosphere](atmosphere.md)) or an
airless body's day/night line, as before.

## The sun

The sun is a direction plus an angular radius, so the disc the sky draws is
its physical size: one value, `SunLook.angular_radius`
(`shadows.SUN_ANGULAR_RADIUS`, 0.55°, about twice the real sun's so that the
disc reads at any resolution). `Sky(sun_angular_radius=)` draws it and
`Shadows(sun_angular_radius)` shapes every penumbra with it.

## `Shadows` and `shadows.glsl`

`world_builder.shadows.Shadows` holds the occluders as shader inputs shared
by every node bound, as `Atmospheres` does: uniform arrays rewritten in
place once a frame.

- `add(radius, position, reach=None)`: a spherical body; its index is what
  receivers name as their own body. `reach` is how far from its centre what
  it holds goes (its air's top, its highest ground): see [Cost](#cost).
- `set_sun(direction)`: the sun, for which body's shadow can reach which.
- `set_atmosphere(atmosphere, body)`: which body owns an atmosphere
  (`uAtmoBody`), so a planet's air is not eclipsed by the planet itself.
- `bind(node, body=None)`: the arrays, plus `uShadowSelf`, the node's own
  body (a body never eclipses itself; None for the sky).
- `update(camera_world)`: each centre minus the camera, in float64, into
  `uShadowBodies` (xyz, and the radius in w).
- `eclipse(points_world, sun_direction, exclude=None)`: the CPU twin,
  vectorised.

`bind_defaults(node)` gives "nothing casts" inputs. `atmosphere.glsl`
includes `shadows.glsl`, so every owner of a shader that sees through air
(terrain, sky, planet and flat clouds) calls it next to the atmosphere's
`bind_defaults`, and every shader links without a `Shadows`.

## Eclipses

`eclipseVisibility(rel, L, self)` loops over the bodies, skipping `self`.
For each, from the point at `rel` (relative to the camera, render frame):

- the body's angular radius `asin(R / d)` and the angle between its centre
  and the sun, `atan(across, along)` (precise near 0, where `acos` is not);
- `discCoverage(sun, body, angle)`: the covered share of the sun's disc,
  the overlap of two circles over the sun's area (angles taken as flat
  lengths, a uniform disc: no limb darkening);
- the visibility is the product of `1 - coverage`.

A body clear of the sun is rejected before any inverse trigonometry,
exactly: their angle is past the sum of the radii when
`across · (1 - tan b · tan a) ≥ along · (tan b + tan a)`, from the two
tangents. A point inside a body sees no sun.

The penumbra is physical: it widens with the distance to the body, and there
is an umbra only where the body looks wider than the sun (otherwise an
annular eclipse).

### Receivers

- **Terrain** (`terrain_frag.glsl`): land and far water, per pixel, at
  `vRelPos`. An airless moon inside the planet's shadow goes dark: a lunar
  eclipse.
- **Cloud billboards** (`cloud_sphere.vert`): per vertex, at the
  billboard's centre, into `vSunTint` and `vTwilight`.
- **Cloud deck** (`cloud.frag`): per fragment.
- **The air** (`atmoSegment`): each sample's direct sunlight, at the
  sample. The sky darkens, and so does the aerial perspective over the
  ground.
- **Sky light**: the terrain's sky light, the clouds' sky term and the air's
  multiply scattered fill are multiplied by
  `mix(ECLIPSE_SKY_FLOOR, 1, visibility)` (0.15: the air outside the shadow
  still glows).
- **CPU light** (`World.light_at`, so Panda3D's lights and the sea's near
  look): the same eclipse, at the camera or the sea under it.

## Cloud shadows

The planet's clouds shade the ground, the far water and the near sea, read
straight from their density field: the shadow depends on the world position
only, so it lines up with the clouds drawn, and it moves with the sun.

- **The march** (`cloud_field.glsl`). The clouds' own sun march is now a call
  to `slabOpticalDepth(f, start, dir, stepLength, steps)`: samples along a
  straight stretch of the field, through the two lowest octaves calibrated
  onto the full field, the weather fetched once at the start.
  `cloudShadowDepth(f, p, dir, maxSteps)` intersects the sun ray from `p`
  with the slab's two spheres (`planetRadius + base`, `+ thickness`) and
  marches the chord between them in steps of about the layer's
  `sunStepLength`, at most `maxSteps` (a long, low chord takes longer steps
  rather than more of them). The shadow is `exp2(-depth)`.
- **The field.** The deck layer's (`SphericalClouds._deck_layer`, the full
  field with per-metre extinction, and the cumulus's `shadow_strength`), so
  the shadow matches the clouds as drawn.
- **Inputs.** `SphericalClouds.bind_shadow_inputs(node)` gives a node (the
  planet's terrain root) the field's inputs and `uCloudShadowLayer`, its
  layer: -1 until the weather is made, or after `remove`.
  `clouds.field.cloud_shadow_defaults` (the terrain binds it) is "no clouds
  overhead". `World` binds each body's clouds to its terrain; the
  hand-built demo does the same.
- **Receivers.** The terrain's sunlight (`terrain_frag.glsl`, at the planet
  frame position `uAtmoCam[own] + vRelPos`: clouds come with air), so land
  and far water. The near sea takes a per-pixel `sunShade` in `seaShade`,
  which dims the sun's share of its light (`Sea.set_light(sky, sun=...)`):
  the CPU's light on the sea leaves the clouds out. The sky light is not
  dimmed: under a cloud the air all round is still lit.
- **CPU twin.** `noise.cloud_shadow_on_sphere`, and
  `SphericalClouds.sun_visibility(points, sun, steps)`, which
  `World.sun_visibility` and `lights_at` multiply in: Panda3D's lights and
  the tracked models darken under a cloud.
- **Knob.** `ShadowSettings.cloud_shadow_steps` (the `shadows` section of
  `WorldSettings`, live): 0 (off) at LOW, 4, 8, 12 at ULTRA.
- **Far away**, where a pixel spans several clouds, the march would
  sparkle; **under a grazing sun**, the chord is kilometres long and its
  steps longer than the clouds, so the march would band (each height of
  ground on a slope stepping onto other clouds: stripes along the
  contours). In both cases `cloudShadow` fades it, over the second octave's
  cell, to the shadow's mean for the weather's coverage and the sun's
  elevation (`noise.ground_shadow_table`, made with the deck's lighting, in
  its texture's third channel), which costs one fetch. A long, low sun's
  shadow is a smear anyway.
- **Cost.** The ground's steps are twice the clouds' own
  (`CLOUD_SHADOW_STEP_SCALE`: within 0.004 of a 10 m march on average), an
  exact early-out skips clear weather, and pixels the sun cannot light skip
  the march. The CPU side keeps each point's shade per 25 m cell
  (`SHADE_CELL`), so a model at rest costs nothing. Still, marching per
  pixel costs, on a GTX 1070 at 1600×900 (median frame ms, Part B then
  HIGH, MID): ground 10.95, 12.70, 11.98; high 6.01, 7.37, 6.78; stratosphere
  4.29, 5.15, 5.10; sea_surface 8.23, 9.27, 9.02; orbit 2.62, 2.84. A
  shadow map (the march once per texel of a map around the camera, one
  fetch per pixel) is the way to most of it back: see the
  [performance backlog](performance_backlog.md).

## Mountain shadows (the global map)

`world_builder.terrain.shadow_map`: each body's mountains' shadow, at
planetary scale, as a cube map over the body (its own frame) baked for one
sun.

- **The ceiling.** Per texel, the lowest altitude from which the sun clears
  every occluder toward it, and the distance to the occluder that sets it.
  Shadow on a heightfield only lessens with altitude, so that says it all:
  a receiver at altitude `a` clears the occluder by `(a - ceiling) /
  distance` radians, and `terrainShadow` (shadows.glsl, one fetch) is a
  smoothstep of that over the sun's angular radius: the physical penumbra,
  sharp near the occluder, soft far from it.
- **The bake** (`ceiling_cube`). A point's ceiling only depends on the
  point a step toward the sub-solar point: `ceiling(P) = max(height(P'),
  ceiling(P')) - gain`, the gain the sun's ray climbs over the step
  (closed form, curvature included). On a grid aligned with the sun
  (columns along great circles through the sub-solar point, rows half a
  texel apart in angle from it) `P'` is the node a row up, exactly: one
  pass over the rows, every column at once, ~2.3 s at 1024 in numpy, no
  native code. Interpolating at each step instead (a sweep over the cube's
  own texels) rounded the ceiling's crest down every step: shadows ~20 %
  short. Against a march toward the sun over the same ground: within ~40 m
  at 1024.
- **Heights** (`height_cube`): `Planet.sample_dirs` at the texel centres,
  band-limited to two texels, stamps in, the sea flat at 0. Six jobs (a
  face each, ~2.5 us a texel: ~16 s of CPU at 1024 in all), once per body
  and size.
- **The near gap.** The nearest occluder counted is two texels away (1.8 km
  at 1024 on a 600 km planet): a texel's own slope never shadows it, and a
  ridge nearer than that casts nothing here. Part E's local map covers
  that; the demo's 750 m mesas show only the far ends of their shadows.
- **`ShadowMap`** (one per body: `Body.shadow_map` in `World`). Bakes in
  the worker pool: the heights once, the ceiling for each sun
  (`set_sun`; the last map stays until the new one is in, so a swinging
  sun gets a new map every ~2-3 s). One half-float cube texture in km,
  mipmapped (read across a whole face from orbit, it thrashed the texture
  cache without), uploaded in place: `bind(node)` once. `loading` until
  the first map (`World.loading` counts it), `current` once the map is the
  sun's (the shots wait for it).
- **Receivers.** The terrain (land, far water) per pixel at its planet-frame
  position (a varying from terrain.vert: an airless moon has its craters'
  shadows too) and its drawn height; the near sea through `sunShade`; the
  clouds (billboards per vertex, the deck per fragment: peaks between a
  low sun and the clouds). The CPU twin, `ShadowMap.visibility` (kept per
  25 m cell until the next bake), is in `World.sun_visibility` and
  `lights_at`.
- **Knob.** `ShadowSettings.terrain_map_size` (rebuild): 256 at LOW, 512,
  1024, 2048 at ULTRA (16 s of heights and ~9 s a bake), off.
- **Cost.** One cube fetch per pixel: 0.0-0.3 ms (orbit, where every pixel
  is terrain, the most; stratosphere 5.39 → 5.62 ms, orbit 2.85 → 3.08,
  moon_orbit 1.74 → 1.98, ground within noise).

The clouds' `grazingLight` (the sun's path below a cloud's altitude dims it
toward the terminator) is kept: it stands for the haze and the clouds on
that path, which the map does not hold, and the map only darkens the
clouds behind real peaks.

## Nearby ridges (the local map)

`shadow_map.LocalShadowMap`, owned by each body's `ShadowMap` (`local`):
the occluders nearer than the global map's reach (`global_reach`, two
global texels: 2.3 km at 1024), at 50 m, over a square chart around the
camera (1024 nodes: 51 km).

- **The bake** (`local_ceiling`): the global map's pass over a grid aligned
  with the sun, over the window that covers the square, counting only the
  occluders between two local nodes (100 m) and the global reach. The two
  maps never count one occluder twice, and they combine by the lesser
  share (`min`, not the plan's product: the higher of two horizons hides
  the sun, a product would darken their penumbrae twice). ~0.9 s at 1024;
  within ~10 m of a march over the same ground. With the sun within the
  window's reach of the zenith there is no room for the grid, and the
  shadows are short: no local map.
- **Its ground** is sampled (`_local_heights`, four jobs, ~2.6 s of CPU at
  1024) once the camera has moved a quarter of the square from its centre,
  and only below 20 km (`LOCAL_MAX_ALTITUDE`: higher, its detail is under
  a pixel, and the last map stays). The ground does not depend on the sun:
  a new sun only bakes again. The last map stays until the next is in, and
  where they overlap their shadows are the same: no pop.
- **The shaders** (`localShadow`): the chart's coordinates from the
  planet-frame direction (the angle from its centre by `atan`, precise
  near it), one fetch of a float texture (mipmapped), faded out over the
  band within the global reach of its edges (the ground beyond is
  unknown). `terrainShadow` is the lesser of the two maps.
- **Cost.** ~0.3-0.5 ms near the ground (ground 13.06 → 13.84 ms, half of
  it the fetch); its CPU twin is kept per 25 m cell with the global one's.
- **Knobs.** `ShadowSettings.local_map_size` (off at LOW, 512, 1024, 2048)
  and `local_map_spacing` (50 m).

The water barely shows a mountain's shadow at a low sun: its light is
mostly the sky's, and a level surface takes little direct sunlight then
(`mesa_above`).

## The cloud shadow maps

`clouds.shadow_map.CloudShadowMaps`: the clouds' shadow marched once per
texel instead of once per pixel of ground (it was 0.8-1.7 ms at HIGH).

- **What a texel holds.** The march depends on the sun's ray alone, so it
  is stored by where the ray enters the slab's base: each texel is a node
  of an azimuthal equidistant chart on the base's sphere around the camera,
  and `cloud_shadow_map.frag` runs `cloudShadowDepth` from there. A pixel
  of ground below the slab finds its ray's entry point (analytic,
  `cloudShadowChord`) and fetches it (`cloudMapsDepth`).
- **Two maps**: near, 60 m texels (1024: 61 km wide), and far, 500 m
  (512: 256 km), each faded into the next over the last eighth of its
  half-width; beyond them (and for ground inside the slab, whose ray does
  not start on the base) the march per pixel, and the shadow's mean
  farther still, as before.
- **Drawn again** (one frame of its buffer, `one_shot`) when the camera
  has moved a quarter of a map, the sun has turned, or the steps or the
  clouds' layer changed. With the sun still, a fetch per pixel; with it
  swinging, a march per texel each frame.
- **Cost.** 0.3-1.1 ms less than the march per pixel near the ground
  (sea_surface 11.65 → 10.57, ground 13.50 → 13.12, clouds_noon 10.63 →
  10.08); what remains is the lookup (two fetches and the chart's maths).
  The shadows match the march within a texel's blur.
- **Knob.** `ShadowSettings.cloud_map_size` (off at LOW, 512, 1024, 2048):
  the near map's size, the far one half of it (at least 256). `World`
  makes them for a body with clouds and binds them to its terrain
  (`Body.cloud_maps`); the hand-built demo the same.

## Long cloud shadows across the deck

A cloud's fragment marches toward the sun for its own few hundred metres
(`sun_steps × sun_step_length`: 270 m for cumulus, 1.9 km for a storm
tower), enough for its own form. With a low sun the clouds kilometres
upwind shade it too, and that march misses them.

- **The far segment.** `cloud_field.glsl`'s `cloudFarDepth`, per vertex
  in `cloud_sphere.vert`, from the billboard's centre: from where the
  fragment's march stops along the sun's ray to the slab's top (or the
  reach), in steps of 600 m (`CLOUD_FAR_STEP`, at most 12), through
  `slabOpticalDepth`. It multiplies `vSunTint`. The fragment's march does
  not change; only the slowly varying far part sits in the vertex (the
  march per vertex rejected before lit far billboards flat because it held
  the near part too).
- **When it does anything.** With a high sun the ray leaves the slab
  within the fragment's march and the segment returns at once; it acts
  only as the sun gets low. The far deck's averaged table
  (`deck_light_table`) already holds a low sun's long path and is left as
  it is. CPU twin: `noise.far_cloud_depth`.
- **Look.** At dusk the banks' bases and the deck's sides away from the
  sun go blue-grey, and the clouds' reflections in the sea darken with
  them. A billboard takes one value, so where the deck's distant rim
  already shows the billboards' edges, the darker ones stand out a little
  more.
- **Cost.** None measurable: median frame ms on / off, `dusk_in_clouds`
  12.82 / 12.84, `dusk_above_clouds` 13.69 / 13.73; `flight_bench.py
  --flight low` (mean / p95), alternating, on 9.1 / 15.2 and 9.5 / 15.9,
  off 9.6 / 16.0 and 9.6 / 15.3.
- **Knob.** `ShadowSettings.cloud_far_reach` (0 at LOW, 2 km, 4 km, 8 km),
  live.

## Objects

Panda3D's lights on `render` are the light at the camera, wrong for a model
elsewhere (a ship in an eclipse the camera is outside of). `World` lights a
tracked model where it is:

- `World.sun_visibility(points)`: the product of the CPU shadow terms at
  each point, by its nearest body: the eclipses, the planet's clouds' and
  its mountains' shadows (part E's local map will join them).
- `World.lights_at(points)`: the light at many points in one vectorised call
  (the air's `Atmospheres.light_at`, `in_shadow` and `twilight` take arrays
  of points), times that visibility; `light_at(point)` is its single-point
  form.
- `World.track_light(node, cast=False)`: the node gets its own
  `DirectionalLight` and `AmbientLight`
  (`world_builder.lights.TrackedLights`), the world's off on it; `update`
  sets them from `lights_at` at the nodes' positions, times `light_scale`.
  `untrack_light` puts back the node's own lights. `cast` is kept for the
  caster map (part H).

The demos track the test balls. Their light only differs from the
camera's over the scale of a shadow's edge: the moon's penumbra is ~400 km
wide, so from 150 m away it looks the same. `tests/test_world.py` checks a
ball in an umbra goes dark while the camera outside it keeps the sun.

## Models drawn with every shadow (part G)

`World.track_light(node, shader=True)`: the model is drawn by the world's
model shader (`world_builder.objects`, see
[Lighting on objects](object_lighting.md)), lit and tonemapped as the
terrain is, and shadowed per pixel, not by one CPU value at its centre:

- `model_light.glsl`'s `modelSunlight`: the eclipses, the body's air, its
  clouds' shadow and its mountains' (the maps, in the body's own axes:
  `uPlanetAxes`, from `Terrain.rotation`);
- on an airless body, its own horizon: `ownShadow`, the sun's elevation
  against the horizon's dip, the limb a straight edge across the sun's
  disc (`discCoverage` cancels away in float32 for a disc half the sky
  wide), none under the sea-level radius (a crater's floor is not inside
  the body). CPU twin `Shadows.own_shadow`.

Each body has a `ModelLighting` (`Body.models`) bound as its terrain is;
shaded nodes take their nearest body's every `update`. The read-back
tests (`tests/test_objects.py`) draw the sunlight over a turned planet
with air, in a moon's penumbra, under a hand-made ceiling map, and over an
airless moon's terminator, against the CPU; and the whole shader on a card
against the CPU's tonemap.

Poses: `ball_noon`, `ball_sunset`, `ball_eclipse`, `ball_cloud_edge`,
`ball_mountain_edge` (see [Demo and shots](#demo-and-shots));
`--panda-lights` draws the balls by Panda3D's lighting, to compare.

## Demo and shots

F7 in the terrain demos cycles the shadows' preset (`--shadows-quality` at
start): LOW leaves the clouds' shadows out and makes a coarse mountain map.

Mountain shadow poses: `mountains_low` (clouds_grazing's peaks, the sun 6°
up in the south-east), `mountains_orbit` (1 radius up, the sun 8° up),
`moon_terminator` (the moon's craters at its terminator). Each pose's line
prints the last bake's time.

Long cloud shadows: `dusk_in_clouds` (2.7 km, in the deck, looking north)
and `dusk_above_clouds` (4.5 km, over it), the sun 4° up in the west.

The test balls (part G): `ball_noon` (the ground view, the sun 75° up),
`ball_sunset` and `ball_eclipse` (3.7 km up, over the cloud deck, out of
its shadow: the sun 3° up in the west, and 0.4° off the moon's centre),
`ball_cloud_edge` and `ball_mountain_edge` (the balls moved onto the
nearest edge of a cloud's or a mesa's shadow around the start, where the
CPU twin leaves half the sun, seen from 150 m across the sun). A cloud's
shadow edge is hundreds of metres wide: a ball on it is dimmed evenly.

`scripts/demo_terrain.py` puts the moon 21,000 km out: from the planet's
ground it looks 0.56° wide, as wide as the sun, so a total eclipse has an
umbra a few km across inside a penumbra some 400 km wide. Swing the sun
behind the moon with J / K and I / U: the shadow slides.

`scripts/terrain_shots.py` can aim a pose's sun at a body (`AimedSun`: the
direction from the point under the camera to the body, turned some degrees
north). Its eclipse poses:

- `eclipse_ground`: the moonrise view, the sun 0.6° off the moon's centre:
  a partial eclipse;
- `eclipse_total`: the sun behind the moon's centre, the camera in the
  umbra;
- `eclipse_orbit`: 3 radii over the point under the moon: its shadow on the
  planet at noon;
- `lunar_eclipse`: the `moon_orbit` view, the sun behind the planet,
  0.8° off: the planet's umbra edge across the moon.

## Cost

Most of the time no body's shadow is anywhere near another body, and the
eclipse must cost nothing then. `Shadows.set_sun` works out, per body, which
other bodies' penumbra cones can reach what it holds (its `reach`: its air's
top, its highest ground; `World` sets it), as bits in `uShadowCasters`.
`eclipseVisibility` returns at once for a body with none. In the air's march
the eclipse is gathered in a loop of its own before the sample loop, only
when `eclipseCanReach` says a shadow can reach that air: inside the sample
loop, its code alone (never run) cost ~0.5 ms a frame at the ground poses.
The CPU twin rejects bodies clear of the sun before any trigonometry (26 µs
a call, from 73).

Measured on a GTX 1070, 1600×900, HIGH, median frame ms (CPU and GPU in
series), `terrain_shots.py`, before (two runs) and after:

| Pose | Before | After |
|---|---|---|
| ground | 10.24 / 10.50 | 10.74 |
| coast | 11.82 / 12.12 | 12.11 |
| high | 5.70 / 5.76 | 6.08 |
| stratosphere | 3.90 / 3.99 | 4.21 |
| orbit | 2.27 / 2.29 | 2.46 |
| in_clouds | 10.81 / 11.18 | 11.11 |
| sunset | 12.14 / 12.65 | 12.42 |
| beach | 14.79 / 15.45 | 15.60 |
| eclipse_ground | | 13.31 |
| eclipse_total | | 13.31 |
| eclipse_orbit | | 3.83 |
| lunar_eclipse | | 1.76 |

The views without an eclipse cost at most ~0.2–0.3 ms more (high,
stratosphere, orbit), most within run-to-run noise. With an eclipse in view
the air's gather loop runs: roughly 2.5 ms more than the nearest view
without one at the ground (`eclipse_ground` against `moonrise`, whose sun
differs), 1.4 ms from orbit (`eclipse_orbit` against `orbit`).

`flight_bench.py --flight low` (mean / p95 frame ms), alternating runs:
before 8.4 / 11.8 and 8.5 / 11.9, after 8.7 / 12.1 and 8.5 / 11.8. The
`World` demo (`--demo world`), all flights: low 8.4 / 11.6 before, 8.5 /
11.9 after; cruise 1.6 / 5.2 and 1.7 / 5.3; descent 3.3 / 7.9 and 3.5 / 7.5.

The views without an eclipse render as before (`compare_shots.py`), but
for the moon's new distance (the moon in `moonrise` and `in_clouds`, a few
star pixels from the farther far plane) and the HUD.
