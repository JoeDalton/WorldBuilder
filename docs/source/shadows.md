# Shadows

What hides the sun from a point. Every shader that lights something by the
sun multiplies its sunlight by a product of independent visibility terms,
and a CPU twin of each term lights what the CPU lights (Panda3D's lights,
the sea's near look). Issues #11 and #13; the plan is `SHADOWS_PLAN.md`.

| Term | Casters | Receivers | Status |
|---|---|---|---|
| Eclipse | spherical bodies (planets, moons) | terrain, far water, clouds, air, CPU light | done |
| Cloud shadow | the planet's clouds | terrain, sea, objects | done |
| Terrain shadow | mountains | terrain, sea, clouds, objects | planned (D, E) |
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

## Objects

Panda3D's lights on `render` are the light at the camera, wrong for a model
elsewhere (a ship in an eclipse the camera is outside of). `World` lights a
tracked model where it is:

- `World.sun_visibility(points)`: the product of the CPU shadow terms at
  each point, by its nearest body (so far the eclipse; the cloud and terrain
  terms join it in later parts).
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

The demos track the test ball. Its light only differs from the camera's
over the scale of a shadow's edge: the moon's penumbra is ~400 km wide, so
from 150 m away it looks the same. `tests/test_world.py` checks a ball in an
umbra goes dark while the camera outside it keeps the sun.

## Demo and shots

F7 in the terrain demos cycles the shadows' preset (`--shadows-quality` at
start): LOW leaves the clouds' shadows out.

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
