# Shadows

What hides the sun from a point. Every shader that lights something by the
sun multiplies its sunlight by a product of independent visibility terms,
and a CPU twin of each term lights what the CPU lights (Panda3D's lights,
the sea's near look). Issues #11 and #13; the plan is `SHADOWS_PLAN.md`.

| Term | Casters | Receivers | Status |
|---|---|---|---|
| Eclipse | spherical bodies (planets, moons) | terrain, far water, clouds, air, CPU light | done |
| Cloud shadow | the planet's clouds | terrain, sea, objects | planned (C) |
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

## Demo and shots

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
