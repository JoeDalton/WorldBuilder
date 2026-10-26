# Asteroid fields

[`world_builder.asteroids`](https://github.com/JoeDalton/WorldBuilder/tree/main/src/world_builder/asteroids/)
makes fields of rocks, from a few kilometres across to a planet's ring
(issue #14). No list of rocks is stored: any part of a field is generated from
its spec when asked for, and is the same every time. That is what lets a field
be as large as a ring, drawn and collided with only around the camera.

```
python scripts/demo_asteroids.py            # fly through a field (controls in the script)
python -m world_builder.asteroids.preview   # each preset's rocks from above and the side, and their statistics, to PNG
python scripts/asteroid_gallery.py          # rock prototypes at each level of detail, to PNG
python scripts/asteroid_bench.py --field cube --offscreen   # frame times in fixed flights
pytest tests/test_asteroid_*.py
```

What exists so far is the field's definition, its rocks, their shapes, their
drawing and their collision (phases 1 to 4 of the plan); see
[Phases](#phases). Improvements found but not made are in the
[backlog](asteroids_backlog.md).

## A field

A `FieldSpec` (`asteroids/spec.py`) is a frozen, picklable dataclass. Lengths
are metres, angles radians, rates radians per second, all in the **field
frame**: the field's own centre and axes.

| Part | Class | What it sets |
|---|---|---|
| Extent | `Box(half_size, edge)`, `Sphere(radius, edge)` | Where rocks are. The density fades to nothing over `edge` inside the surface. |
| Density | `Density(count_per_km3, clumping)` | Rocks per km³ where densest. |
| Clumping | `Clumping(scale, coverage, softness, octaves)` | Patchiness: rocks gather where a 3D noise is high. |
| Sizes | a `Distribution` | The rocks' bounding radii. |
| Spin | `SpinSpec(rate, axis, size_exponent, tumble)` | How they turn. |
| Shapes | `ShapeSpec(prototypes)` | How many distinct shapes. |
| Material | `MaterialSpec(ice_fraction)` | Each rock's share of ice. |
| Keep-outs | `KeepOut(centre, radius, end=None)` | Balls or capsules no rock enters. |

`FieldSpec.summary()` gives what a spec amounts to: the expected rock count,
the mean spacing and the share of volume filled where densest, and the
median, mean and largest radius.

`spec.PRESETS` has two examples. `cube` is SpaceFlight's missions 1 and 2
setting: a 15 km cube of rock, power-law sizes from 50 m to 2 km, patchy,
with a 300 m corridor through it. `icy_ball` is a 10 km ball of rock and ice
spinning about one axis, some of it tumbling.

### Sizes and other variables

A rock's `radius` is its bounding radius: no part of it reaches farther. Sizes,
spin rates, cone angles and ice fractions are each a `Distribution`
(`asteroids/distributions.py`): a family truncated to [lo, hi].

| Family | Use |
|---|---|
| `uniform(lo, hi)` | |
| `normal(mean, sd, lo, hi)` | |
| `lognormal(median, sigma, lo, hi)` | One typical size with a spread. |
| `power_law(q, lo, hi)` | Density ∝ x^-q: few big, many small. q = 3.5 is a collisional cascade, as in rings and belts. |
| `u_shaped(a, lo, hi, log=True)` | beta(a, a) across the interval, by default across log x: mostly small and large, few between. |
| `table(xs, pdf)` | Any shape, piecewise linear. |
| `constant(value)` | |

A field draws from each one's inverse CDF tabulated at 1,025 points, by
linear interpolation. numpy and the native scatter do the same arithmetic on
the same table, so they draw the same values.

### Clumping

The idea of "asteroids where a noise is above a threshold" is kept, but it
sets the density, not the rocks: a rock's chance to exist is multiplied by
`smoothstep(t - softness, t + softness, fbm(x))`. `scale` is the noise's
largest wavelength, the clusters' spacing. `coverage` is the share of space
populated; the threshold `t` is derived from it through the measured
distribution of the noise. `softness` is how gradually a cluster's edge
fades, in noise units: 0.02 is sharp, 0.3 very soft.

The rocks' sizes are not tied to the noise, so the density, the clumping and
the size distribution are three independent knobs.

### Spin and tumble

Each rock turns at a constant rate about a fixed axis.

- **Rate** is drawn for the smallest rocks, and scales as (radius /
  smallest)^-`size_exponent`: big rocks turn slower. At the default 1.5, a
  2 km rock turns about 250 times slower than a 50 m one, as in SpaceFlight's
  current field.
- **Axes** are `Isotropic()`, or `AroundAxis(axis, kappa)`: von
  Mises–Fisher around one direction, for a ring's rocks turning with its
  orbit. kappa 10 keeps most within about 25° of it, 100 within about 8°.
- **Tumble** (`Tumble(fraction, cone_angle, ratio)`): a share of the rocks
  move as free symmetric tops. Their body z axis sweeps a cone about the
  fixed axis while they turn about it. `ratio` is I1/I3 - 1 for moments of
  inertia I1 = I2 and I3 about body z; it is positive for an elongated rock.

`asteroids/motion.py` has the closed form: q(t) = rot(axis, precession · t) ·
q0 · rot(z, spin · t), the solution of Euler's equations for a free
symmetric top (`tests/test_asteroid_motion.py` integrates them to check). A
rock that only spins has spin = 0. Time is always given by the caller, from
the game's clock, so a paused game stops the rocks.

## Scattering

`asteroids/scatter.py` gives the rocks of any cells, boxes or neighbourhoods
of a field: `Scatter.rocks_in_cells`, `rocks_in_box`, `rocks_near`,
`is_clear`, `find_clear`. Each rock is a `ROCK_DTYPE` record: id, size
class, prototype, centre, radius, q0, axis, precession, spin, ice.

- **Size classes.** Radii [lo, 2 lo), [2 lo, 4 lo), and so on up to the
  sizes' hi. Class k scatters in cubic cells of edge `cell_factor` × its
  smallest radius (8 by default). Big rocks are sparse in big cells, so each
  class can be asked for out to its own distance: a 2 km rock is seen from
  much farther than a 50 m one.
- **Candidates.** A cell's 64-bit hash (splitmix64 chained over its
  coordinates, from the class's seed) draws a Poisson count of candidates.
  Each candidate's position, radius, prototype, ice, spin and q0 then come
  from the hash of (cell, candidate, slot). Nothing depends on which other
  cells are asked for, or in what order.
- **Existence.** A candidate exists with probability = the density there:
  the extent's profile times the clumping. Candidates inside a keep-out do
  not exist.
- **Hard core (Matérn type II).** A rock is dropped where it overlaps an
  existing candidate that ranks above it: larger, or as large with a smaller
  id. The test is against candidates, not against survivors, so it stays
  local: a cell's rocks depend on its 27 neighbours in its own class and in
  every larger one, nothing farther. No two rocks overlap.
  - At the densities of a game field this removes a few percent: 4 % of the
    `cube` preset.
  - Near packing it removes most, because a candidate goes wherever any
    larger one overlaps it, kept or not. A field whose rocks fill more than
    about a tenth of its volume gets far fewer rocks than its density says.
- **Keep-outs.** Static ones are in the spec. `Scatter.with_keep_outs` adds
  more at runtime, such as around a level's spawn points and routes.
  `find_clear(point, radius, search)` finds the nearest point where a ball
  touches no rock, for ships placed relative to the player.

The native library runs the same steps (`native/scatter.c`, about 100x
faster): the 3,500 rocks of the `cube` preset take 12 ms. numpy runs without
it. The two give the same rocks: every choice uses the same arithmetic on the
same values, and the clumping noise goes through the same per-point fbm.
Without the native noise as well, the clumping's last bits differ, and a rock
on the very edge of existing may differ.

## Rock shapes

Every rock is one of the field's prototypes (`ShapeSpec.prototypes`, 32 by
default), turned and scaled. A prototype is a radius function of direction
(`asteroids/shape.py`), in units of the rock's bounding radius:

- an ellipsoid, semi-axes (short, middle, 1), its long axis along body z (a
  tumbling rock's symmetry axis): `elongation` draws short / long;
- lumpy fbm noise (`lumps`, `lump_scale`) and faint ridged noise (`ridges`);
- craters with raised rims (`craters` per prototype, `crater_size` their
  angular radius, `crater_depth` over their diameter);
- scaled so its farthest point is 1, measured on a fixed sampling with a 2 %
  margin, so the shape does not depend on the quality settings.

A radius function has no overhangs, arches or caves.

The workers build each prototype (`asteroids/prototype.py`,
`generate_prototype` on a `ChunkExecutor` channel, about 180 ms each):

- **Meshes**, one per level of detail: a cube-sphere grid of 4, 8, 16 and 32
  quads per face side (64 at ULTRA), the terrain's face frames, equal-angle
  grid, diagonal and winding. A level samples the shape band-limited to four
  times its spacing, so a coarse mesh is a smoothed rock, not an aliased one.
  The faces share no vertices, but seam vertices come from bit-identical
  directions, so the mesh has no cracks.
- **Normals** by central differences about each vertex, in a tangent frame
  of the direction alone: the same on both sides of a seam.
- **Errors**: each level's largest radial distance from the finest mesh (how
  far it can be seen without showing) and from the full detail (the normal
  map's). With the defaults: about 0.08, 0.05, 0.03 and 0 of the radius from
  the finest; the finest is about 0.025 from the full detail.
- **Radius grid**: the finest mesh's vertex radii, for collision on the
  triangles drawn.
- **Normal map**: the full-detail shape's body-frame normals per texel of
  each face's equal-angle grid (6 × 128² RGB8 per prototype). Drawn at every
  level, it lends coarse meshes the fine detail.

`asteroids/mesh.py` turns them into Panda Geoms (one index buffer per level,
shared) and one normal-map texture array (six layers per prototype).

## Drawing

`AsteroidField(app, spec, clock, settings=, ...)` (`asteroids/field.py`)
draws a field around the camera; `update(camera_world, camera_render,
cameras=)` every frame once the camera is placed. It hangs under render and
is placed relative to the camera, as the terrain is. The field's own frame
sits at `position` with `orientation` (world from field) in the world.

- **Prototypes** are built by the terrain's worker pool (its own channel)
  when the field is made: about 2.5 s for 32. `loading` is True until they
  are all in; nothing is drawn before. A game's loading phase waits for it.
- **Rocks in range** (`asteroids/population.py`). A rock is drawn while its
  diameter covers `min_pixels` on screen. Each size class holds the cells
  within its largest rock's drawing distance: small rocks near the camera,
  big ones far. Each rock in those cells holds a slot of the instance table
  until its cell leaves the range.
- **The instance table** is a float texture of four texels per slot: the
  rock's centre relative to its cell's corner, its radius, its orientation at
  an epoch, its precession and spin, its cell and its class, prototype and
  ice. The camera is given per class the same way (its cell, its offset in
  it), so a rock's position relative to the camera is a difference of small
  numbers: float32 holds at any distance from the field's centre. Rows are
  uploaded at most every three frames; a rock is drawn once its row is on
  the GPU (new ones are at the edge of their range, a few frames late is
  unseen).
- **The visible pass**, per frame: big enough on screen, inside some
  camera's view (`cameras=`: the main one, and a rear-view mirror), at the
  coarsest level of detail whose error stays under `lod_error_px` pixels,
  with 15 % hysteresis. The slots drawn go in an index texture, grouped by
  (prototype, level): one GeomNode each, drawn with
  `set_instance_count`. Nothing per rock goes through the scene graph.
- **Budget.** Past `instance_budget` rocks in range, the smallest size drawn
  grows by 10 % a frame until they fit, and shrinks back when well under.
- **Spin** is computed in the vertex shader at the game's time (`clock=`,
  never Panda's): q(t) as in `motion.py`, from the table's orientation at
  the epoch. The epoch moves every 1,024 s, rewriting the table, so float32
  time stays precise in a long game. A paused clock freezes every rock.
- **Fade.** Rocks fade out by dithering over the last quarter of their
  drawing distance.
- **Shading** (`asteroids/shaders/asteroid.frag`): the prototype's normal map
  (the full-detail shape) turned with the rock, the terrain's rock and snow
  textures blended by the rock's ice (`terrain_material.glsl`: rock
  projected from the sides on steep flanks), lit as the terrain lights its
  ground (`object_light.glsl`), no shadows. Then either the world's exposure
  and ACES tonemap, or, with `tonemap=False`, a plain display encoding to
  match models lit by Panda's fixed-function lights.
- **Costs** are the `asteroids` section of the [quality settings](quality.md).

### Measured

`scripts/asteroid_bench.py`, offscreen at 1600×900 on a GTX 1070, each frame
waiting for the GPU, two runs each (October 2026). The legacy field is
SpaceFlight's for missions 1 and 2: 2,520 nodes instancing one 10,000-triangle
glTF, each with a collision sphere, 520 of them turned every frame. The new
one is the `cube` preset, about 3,400 rocks, at HIGH.

| Flight | Legacy p95 | Field p95 | Legacy mean | Field mean |
|---|---|---|---|---|
| cruise (through the field, 250 m/s) | 19.9–20.5 ms | 3.6–4.0 ms | 17.9–18.0 ms | 2.8–3.0 ms |
| sweep (turning at its centre) | 10.1–10.3 ms | 3.0–3.4 ms | 8.7 ms | 2.2–2.4 ms |
| outside (the whole field in view) | 35.3–37.0 ms | 4.1–4.7 ms | 33.4–33.7 ms | 3.2–3.4 ms |

The field's own main-thread work is about 1–1.5 ms a frame, most of it the
visible pass in numpy (see the [backlog](asteroids_backlog.md)).

## Collision

`field.enable_collision(into_mask, name="asteroids", owner_tag="owner",
max_distance=20_000, samples=5)` makes the rocks solid to a game's own
`CollisionTraverser` (traversing render). It adds one into-only
`CollisionNode` holding a native `CollisionAsteroids` solid
(`terrain/native/asteroid_collision.cpp`, in the same library as the
terrain's).

- **Which rocks.** Those within `max_distance` of the camera, registered and
  dropped as the camera moves. Farther rocks are drawn but not solid: a ship
  flying far from the player passes through them, as with the terrain.
- **On what.** Each prototype's finest mesh: spheres (ships, ordnance,
  sensors) and segments (lasers) are tested in each rock's own frame, turned
  to the time it was drawn at, against those very triangles (the radial ray
  under a direction meets one triangle, the meshes' diagonal). A segment is
  marched within the rock's bounding sphere and bisected; a sphere takes the
  nearest of a footprint of samples, refined on its triangle's plane, as the
  terrain's test does.
- **What the game gets.** Ordinary `CollisionEntry` objects, at most one per
  collider per frame: the deepest sphere contact, the first segment hit, with
  the surface point and the triangle's normal.
- **The owner** (`AsteroidSurface`, the node's python tag):
  `material_at(point, relative_to)` gives "snow" for a rock more than half
  ice, else "rock"; `velocity_at(point, relative_to)` gives how fast the
  surface moves there (the rock's angular velocity crossed with the point's
  offset from its centre), for a game whose handlers take the obstacle's
  motion into account.
- **Where the node goes.** Beside the field's root, in the field's axes, at an
  origin snapped to 4 km around the camera: its float32 coordinates stay
  small at any distance from the field's centre.

**Cost:** one `traverse` with SpaceFlight's load (64 ship spheres, a sixth
of their sensors, 300 lasers within 5 km of the camera) takes 2.4 ms over
the `cube` preset's 3,400 rocks, against 32 ms for the legacy field's 2,520
collision spheres (`scripts/asteroid_bench.py --collision`).

## Phases

The design (issue #14) builds the rest on this:

1. The field's definition and its rocks. Done.
2. The rocks' shapes: prototypes from the terrain's noise, at several
   resolutions, generated in the terrain's worker processes. Done.
3. Drawing: hardware instancing, one draw per prototype and level of detail,
   spin in the vertex shader, the terrain's materials and lighting
   (`terrain_material.glsl`, `object_light.glsl`). Done.
4. Collision: a native Panda collision solid for the rocks near the camera,
   on the drawn shapes. Done.
5. SpaceFlight's missions 1 and 2 on a field.
6. Rocks above 200 m (`terrain_min_radius`) turn into real terrain bodies
   near the camera, with the terrain's level of detail.
7. Rings: an annulus with a radial profile, and a far band for the rocks too
   small to draw.
